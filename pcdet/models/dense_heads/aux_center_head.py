import torch
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, radar_utils
from .center_head import CenterHead


class AuxCenterHead(CenterHead):
    """
    CenterHead used as an auxiliary training head on the BEV of one branch (INPUT_KEY) before fusion.

    HEAD_ORDER may be any subset of center, center_z, dim, rot, vel: the regression targets are cut to the
    matching columns. It never predicts boxes, the detector only runs it in training mode, and its losses are
    weighted by AUX_LOSS_WEIGHT and logged under LOSS_PREFIX/.
    """
    TARGET_COLUMNS = {'center': [0, 1], 'center_z': [2], 'dim': [3, 4, 5], 'rot': [6, 7], 'vel': [8, 9]}

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 **kwargs):
        super().__init__(
            model_cfg=model_cfg, input_channels=input_channels, num_class=num_class, class_names=class_names,
            grid_size=grid_size, point_cloud_range=point_cloud_range, voxel_size=voxel_size,
            predict_boxes_when_training=False
        )
        self.input_key = model_cfg.INPUT_KEY
        self.loss_prefix = model_cfg.LOSS_PREFIX
        self.aux_loss_weight = model_cfg.get('AUX_LOSS_WEIGHT', 1.0)
        # run in fp32 under AMP: the camera heatmap convs grew large enough to overflow fp16 (inf BN statistics)
        self.force_fp32 = model_cfg.get('FORCE_FP32', False)
        self.target_columns = [c for name in self.separate_head_cfg.HEAD_ORDER for c in self.TARGET_COLUMNS[name]]
        assert len(model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']) == len(self.target_columns), \
            f'{self.loss_prefix}: code_weights must have one weight per column of {self.separate_head_cfg.HEAD_ORDER}'

    def forward(self, data_dict):
        assert self.training, 'auxiliary heads only run in training mode'
        x = data_dict[self.input_key]
        with torch.autocast(device_type=x.device.type, enabled=torch.is_autocast_enabled() and not self.force_fp32):
            x = self.shared_conv(x.float() if self.force_fp32 else x)
            pred_dicts = [head(x) for head in self.heads_list]

        # assign_targets relabels gt_boxes in place, and the next heads still need the original labels
        target_dict = self.assign_targets(
            data_dict['gt_boxes'].clone(), feature_map_size=x.size()[2:],
            feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
        )
        target_dict['target_boxes'] = [t[..., self.target_columns] for t in target_dict['target_boxes']]
        self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        return data_dict

    def get_heatmap_loss(self):
        """CenterHead.get_loss for an empty HEAD_ORDER (heatmap only, nothing to regress)."""
        tb_dict = {}
        loss = 0
        for idx, pred_dict in enumerate(self.forward_ret_dict['pred_dicts']):
            hm_loss = self.hm_loss_func(self.sigmoid(pred_dict['hm']), self.forward_ret_dict['target_dicts']['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            loss += hm_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_loss(self):
        loss, tb_dict = super().get_loss() if len(self.target_columns) > 0 else self.get_heatmap_loss()
        loss = loss * self.aux_loss_weight
        tb_dict = {f'{self.loss_prefix}/{"loss" if key == "rpn_loss" else key}': val for key, val in tb_dict.items()}
        tb_dict[f'{self.loss_prefix}/loss'] = loss.item()
        return loss, tb_dict


class RadarAuxCenterHead(AuxCenterHead):
    """
    Radar auxiliary head (HEAD_ORDER [center, vel]) with a Doppler consistency loss: for every radar point in a GT box
    (enlarged by BOX_EXTRA_WIDTH), the velocity this head predicts at the box center, projected on the point's line of
    sight, should equal the point's measured (ego-motion compensated) radial velocity:
        SmoothL1(v_pred . ray_xy - v_r), averaged per box, then over the boxes with radar points.
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, num_class=num_class,
                         class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
                         voxel_size=voxel_size)
        assert 'vel' in self.separate_head_cfg.HEAD_ORDER
        self.doppler_cfg = model_cfg.DOPPLER_LOSS
        # global label (1-based) -> index of the head predicting that class, -1 for none
        head_of_label = torch.full((num_class + 1,), -1, dtype=torch.long)
        for head_idx, cur_class_names in enumerate(self.class_names_each_head):
            for name in cur_class_names:
                head_of_label[class_names.index(name) + 1] = head_idx
        self.register_buffer('head_of_label', head_of_label, persistent=False)

    def forward(self, data_dict):
        self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes'].clone()  # before any head relabels them
        self.forward_ret_dict['radar_points'] = data_dict['radar_points']
        return super().forward(data_dict)

    @staticmethod
    def box_index_of_points(points, boxes):
        """(N,) index of the first box (N, 7) containing each point, -1 if none."""
        if points.is_cuda:
            return roiaware_pool3d_utils.points_in_boxes_gpu(
                points[None, :, 0:3].contiguous(), boxes[None].contiguous()).long()[0]
        in_boxes = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3].contiguous(), boxes.contiguous()).bool()
        first = in_boxes.float().argmax(dim=0)
        return torch.where(in_boxes.any(dim=0), first, torch.full_like(first, -1))

    def get_doppler_loss(self):
        pred_vel = torch.stack([pred_dict['vel'] for pred_dict in self.forward_ret_dict['pred_dicts']])  # (H, B, 2, Y, X)
        gt_boxes = self.forward_ret_dict['gt_boxes']  # (B, M, 7 + 2 + 1)
        radar_points = self.forward_ret_dict['radar_points']  # (P, 1 + len(RADAR_FEATURES)), batch index first
        num_heads, batch_size, _, size_y, size_x = pred_vel.shape
        num_max_boxes = gt_boxes.shape[1]
        extra_width = gt_boxes.new_tensor(self.doppler_cfg.BOX_EXTRA_WIDTH)

        point_list, box_list = [], []
        for bs_idx in range(batch_size):
            cur_boxes = gt_boxes[bs_idx]
            cur_points = radar_points[radar_points[:, 0] == bs_idx, 1:]
            valid = (cur_boxes[:, 3] > 0) & (cur_boxes[:, -1] > 0)
            if cur_points.shape[0] == 0 or not valid.any():
                continue
            large_boxes = box_utils.enlarge_box3d(cur_boxes[valid, 0:7], extra_width=extra_width)
            box_idx = self.box_index_of_points(cur_points, large_boxes)
            in_box = box_idx >= 0
            point_list.append(cur_points[in_box])
            box_list.append(bs_idx * num_max_boxes + valid.nonzero()[:, 0][box_idx[in_box]])
        if len(point_list) == 0:
            return pred_vel.sum() * 0.0, 0
        points, flat_box = torch.cat(point_list), torch.cat(box_list)
        boxes = gt_boxes.view(-1, gt_boxes.shape[-1])[flat_box]

        head_idx = self.head_of_label[boxes[:, -1].long()]
        # the BEV cell of the box center, as in assign_target_of_single_head
        coord_x = (boxes[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        coord_y = (boxes[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
        cell_x = torch.clamp(coord_x, min=0, max=size_x - 0.5).long()
        cell_y = torch.clamp(coord_y, min=0, max=size_y - 0.5).long()
        keep = head_idx >= 0
        if not keep.any():
            return pred_vel.sum() * 0.0, 0
        points, flat_box = points[keep], flat_box[keep]
        vel = pred_vel[head_idx[keep], flat_box // num_max_boxes, :, cell_y[keep], cell_x[keep]]  # (N, 2)

        ray_xy = points[:, [radar_utils.RX, radar_utils.RY]]
        pred_v_r = (vel * ray_xy.to(vel.dtype)).sum(dim=-1)
        point_loss = F.smooth_l1_loss(pred_v_r.float(), points[:, radar_utils.VR].float(), reduction='none',
                                      beta=self.doppler_cfg.get('BETA', 1.0))

        # mean per box, then over boxes, so large objects with many returns do not dominate
        _, box_ids = torch.unique(flat_box, return_inverse=True)
        num_boxes = int(box_ids.max().item()) + 1
        box_loss = point_loss.new_zeros(num_boxes).index_add_(0, box_ids, point_loss)
        box_count = point_loss.new_zeros(num_boxes).index_add_(0, box_ids, torch.ones_like(point_loss))
        return (box_loss / box_count).mean(), num_boxes

    def get_loss(self):
        loss, tb_dict = super().get_loss()
        doppler_loss, num_boxes = self.get_doppler_loss()
        doppler_loss = doppler_loss * self.doppler_cfg.WEIGHT
        tb_dict[f'{self.loss_prefix}/doppler_loss'] = doppler_loss.item()
        tb_dict[f'{self.loss_prefix}/doppler_num_boxes'] = num_boxes
        tb_dict[f'{self.loss_prefix}/loss'] = tb_dict[f'{self.loss_prefix}/loss'] + doppler_loss.item()
        return loss + doppler_loss, tb_dict
