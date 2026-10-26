import numpy as np

from .. import backbones_2d, dense_heads
from ..backbones_2d import fuser, map_to_bev
from ..backbones_3d import vfe
from .detector3d_template import Detector3DTemplate


class RadarFusionCenterPoint(Detector3DTemplate):
    """
    CenterPoint with a radar pillar branch fused in BEV:

        lidar: VFE -> BACKBONE_3D -> MAP_TO_BEV ---------------> spatial_features ----\
        radar: RADAR_VFE -> RADAR_MAP_TO_BEV -> RADAR_BACKBONE_2D -> radar_spatial_features_2d -> FUSER
                                                                       FUSER -> BACKBONE_2D -> DENSE_HEAD

    AUX_HEADS.LIDAR / AUX_HEADS.RADAR are CenterHeads on each branch's BEV before fusion, run in training only.
    The lidar modules keep the names of the lidar-only CenterPoint, so --pretrained_model loads its checkpoint.
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'radar_vfe', 'radar_map_to_bev', 'radar_backbone_2d',
            'lidar_aux_head', 'radar_aux_head',
            'fuser', 'backbone_2d', 'dense_head',
        ]
        self.check_bev_grids(dataset)
        self.module_list = self.build_networks()
        self.aux_heads = [m for m in [self.lidar_aux_head, self.radar_aux_head] if m is not None]

    def check_bev_grids(self, dataset):
        """The radar BEV must have exactly the lidar BEV's grid: same range (one POINT_CLOUD_RANGE) and cell size."""
        assert dataset.radar_voxel_size is not None, 'DATA_PROCESSOR needs transform_radar_points_to_voxels'
        lidar_stride = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        radar_cfg = self.model_cfg.RADAR_BACKBONE_2D
        radar_stride = radar_cfg.LAYER_STRIDES[0] / radar_cfg.UPSAMPLE_STRIDES[0]
        lidar_cell = np.array(dataset.voxel_size[:2]) * lidar_stride
        radar_cell = np.array(dataset.radar_voxel_size[:2]) * radar_stride
        assert np.allclose(lidar_cell, radar_cell), f'lidar BEV cell {lidar_cell} != radar BEV cell {radar_cell}'
        assert np.all(dataset.grid_size[:2] % lidar_stride == 0) and \
            np.all(dataset.radar_grid_size[:2] / radar_stride == dataset.grid_size[:2] / lidar_stride)

    def build_radar_vfe(self, model_info_dict):
        radar_vfe = vfe.__all__[self.model_cfg.RADAR_VFE.NAME](
            model_cfg=self.model_cfg.RADAR_VFE,
            num_point_features=self.dataset.radar_num_point_features,
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=self.dataset.radar_voxel_size,
            grid_size=self.dataset.radar_grid_size,
            depth_downsample_factor=None
        )
        model_info_dict['module_list'].append(radar_vfe)
        model_info_dict['num_radar_point_features'] = radar_vfe.get_output_feature_dim()
        return radar_vfe, model_info_dict

    def build_radar_map_to_bev(self, model_info_dict):
        radar_map_to_bev = map_to_bev.__all__[self.model_cfg.RADAR_MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.RADAR_MAP_TO_BEV,
            grid_size=self.dataset.radar_grid_size
        )
        model_info_dict['module_list'].append(radar_map_to_bev)
        model_info_dict['num_radar_bev_features'] = radar_map_to_bev.num_bev_features
        return radar_map_to_bev, model_info_dict

    def build_radar_backbone_2d(self, model_info_dict):
        radar_backbone_2d = backbones_2d.__all__[self.model_cfg.RADAR_BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.RADAR_BACKBONE_2D,
            input_channels=model_info_dict['num_radar_bev_features']
        )
        model_info_dict['module_list'].append(radar_backbone_2d)
        model_info_dict['num_radar_bev_features'] = radar_backbone_2d.num_bev_features
        return radar_backbone_2d, model_info_dict

    def build_aux_head(self, head_cfg, input_channels, model_info_dict):
        aux_head = dense_heads.__all__[head_cfg.NAME](
            model_cfg=head_cfg,
            input_channels=input_channels,
            num_class=self.num_class if not head_cfg.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['module_list'].append(aux_head)
        return aux_head, model_info_dict

    def build_lidar_aux_head(self, model_info_dict):
        if self.model_cfg.get('AUX_HEADS', {}).get('LIDAR', None) is None:
            return None, model_info_dict
        return self.build_aux_head(self.model_cfg.AUX_HEADS.LIDAR, model_info_dict['num_bev_features'], model_info_dict)

    def build_radar_aux_head(self, model_info_dict):
        if self.model_cfg.get('AUX_HEADS', {}).get('RADAR', None) is None:
            return None, model_info_dict
        return self.build_aux_head(self.model_cfg.AUX_HEADS.RADAR, model_info_dict['num_radar_bev_features'],
                                   model_info_dict)

    def build_fuser(self, model_info_dict):
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER,
            lidar_channels=model_info_dict['num_bev_features'],
            radar_channels=model_info_dict['num_radar_bev_features']
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = fuser_module.num_bev_features
        return fuser_module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if not self.training and any(cur_module is aux_head for aux_head in self.aux_heads):
                continue
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        for aux_head in self.aux_heads:
            loss_aux, tb_dict_aux = aux_head.get_loss()
            tb_dict.update(tb_dict_aux)
            loss = loss + loss_aux
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
