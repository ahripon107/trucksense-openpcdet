import torch
import torch.nn.functional as F
from torch import nn

from .depth_lss import DepthLSSTransform


class LSSTransform(DepthLSSTransform):
    """
        Camera-only LSS (as in BEVDet): depth is predicted from image features alone,
        without the lidar depth input used by DepthLSSTransform.

        Options (all off by default, so BEVDet configs are unchanged):
            OUTPUT_KEY: batch_dict key of the camera BEV (default spatial_features)
            DEPTH_BINS: {MODE: LID | UD, MIN, MAX, NUM} depth bins instead of DBOUND; LID bins grow linearly with
                depth. The frustum then also sits at the feature cell centers ((j + 0.5) * stride, pixel-edge
                convention), where camera_depth_targets measures the depth of a cell.
            DEPTH_LOSS: {TARGET_KEY, WEIGHT} BEVDepth loss: BCE of the predicted depth distribution vs the one-hot
                bin of the lidar depth target, averaged over the cells with a target (needs DEPTH_BINS)
    """
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        in_channel = self.model_cfg.IN_CHANNEL
        self.dtransform = None
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)

        self.output_key = self.model_cfg.get('OUTPUT_KEY', 'spatial_features')
        self.depth_loss_cfg = self.model_cfg.get('DEPTH_LOSS', None)
        if self.model_cfg.get('DEPTH_BINS', None) is not None:
            self.register_buffer('depth_bin_edges', self.get_depth_bin_edges(self.model_cfg.DEPTH_BINS),
                                 persistent=False)
        assert self.depth_loss_cfg is None or self.model_cfg.get('DEPTH_BINS', None) is not None, \
            'DEPTH_LOSS needs DEPTH_BINS'
        self.forward_ret_dict = {}

    @staticmethod
    def get_depth_bin_edges(bins_cfg):
        """(NUM + 1,) depth bin edges [m] from MIN to MAX."""
        i = torch.arange(bins_cfg.NUM + 1, dtype=torch.float64)
        if bins_cfg.MODE == 'LID':
            edges = bins_cfg.MIN + (bins_cfg.MAX - bins_cfg.MIN) * i * (i + 1) / (bins_cfg.NUM * (bins_cfg.NUM + 1))
        elif bins_cfg.MODE == 'UD':
            edges = bins_cfg.MIN + (bins_cfg.MAX - bins_cfg.MIN) * i / bins_cfg.NUM
        else:
            raise NotImplementedError(bins_cfg.MODE)
        return edges.float()

    def create_frustum(self):
        bins_cfg = self.model_cfg.get('DEPTH_BINS', None)
        if bins_cfg is None:
            return super().create_frustum()

        iH, iW = self.image_size
        fH, fW = self.feature_size
        edges = self.get_depth_bin_edges(bins_cfg)
        ds = ((edges[:-1] + edges[1:]) / 2).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = ((torch.arange(fW, dtype=torch.float) + 0.5) * iW / fW).view(1, 1, fW).expand(D, fH, fW)
        ys = ((torch.arange(fH, dtype=torch.float) + 0.5) * iH / fH).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        if self.training and self.depth_loss_cfg is not None:
            self.forward_ret_dict['depth_logits'] = x[:, : self.D]
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def get_loss(self):
        """BEVDepth depth loss, (loss, tb_dict)."""
        logits = self.forward_ret_dict['depth_logits']  # (B * N, D, fH, fW)
        target = self.forward_ret_dict['depth_target'].flatten(0, 1).to(logits.device)  # (B * N, fH, fW)
        assert target.shape == (logits.shape[0], *logits.shape[2:]), \
            f'depth targets {tuple(target.shape)} vs depth logits {tuple(logits.shape)}: check the DOWNSAMPLE of ' \
            f'camera_depth_targets against the stride of the lifted image features'
        edges = self.depth_bin_edges
        valid = (target >= edges[0]) & (target < edges[-1])
        bin_idx = torch.bucketize(target[valid], edges[1:-1], right=True)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            prob = logits.permute(0, 2, 3, 1)[valid].float().softmax(dim=-1)  # (M, D)
            onehot = F.one_hot(bin_idx, self.D).float()
            loss = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), onehot, reduction='sum') / max(1, prob.shape[0])
        loss = loss * self.depth_loss_cfg.WEIGHT
        tb_dict = {'depth/loss': loss.item(), 'depth/num_cells': int(prob.shape[0])}
        if prob.shape[0] > 0:
            tb_dict['depth/bin_acc'] = (prob.argmax(dim=-1) == bin_idx).float().mean().item()
        return loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                OUTPUT_KEY (tensor): bev features from image modality
        """
        x = batch_dict['image_fpn']
        x = x[0]
        BN, C, H, W = x.size()
        num_cams = batch_dict['camera_intrinsics'].shape[1]
        img = x.view(BN // num_cams, num_cams, C, H, W)

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots,
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )
        x = self.get_cam_feats(img)
        if self.training and self.depth_loss_cfg is not None:
            self.forward_ret_dict['depth_target'] = batch_dict[self.depth_loss_cfg.TARGET_KEY]
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        # convert bev features from (b, c, x, y) to (b, c, y, x)
        x = x.permute(0, 1, 3, 2)
        batch_dict[self.output_key] = x
        return batch_dict
