import numpy as np
import torch

from .. import dense_heads
from ..backbones_2d import fuser
from .bevfusion import BevFusion
from .detector3d_template import Detector3DTemplate
from .radar_fusion_centerpoint import RadarFusionCenterPoint


class RadarCameraFusionCenterPoint(RadarFusionCenterPoint):
    """
    RadarFusionCenterPoint + a camera LSS branch, fused in BEV after the radar:

        lidar + radar (as in RadarFusionCenterPoint) -> FUSER -> spatial_features --------------------\
        camera: IMAGE_BACKBONE -> NECK -> VTRANSFORM (LSS) -> camera_spatial_features -> CAMERA_FUSER -> BACKBONE_2D
                                                                                                         -> DENSE_HEAD
    AUX_HEADS.CAMERA is a CenterHead on the camera BEV (training only, like the lidar / radar aux heads), and
    VTRANSFORM.DEPTH_LOSS supervises the camera depth with lidar.

    Training options:
        MODALITY_DROPOUT: {RADAR: p, CAMERA: p} per sample, the radar / camera BEV is zeroed before its fuser
        FREEZE_MODULES: module names whose parameters are not trained and whose BNs stay in eval mode
        TRAIN_LOSSES: subset of rpn, aux_lidar, aux_radar, aux_camera, depth (default: all); in training only the
            modules these losses need are run
    """
    AUX_HEAD_OF_LOSS = {'aux_lidar': 'lidar_aux_head', 'aux_radar': 'radar_aux_head', 'aux_camera': 'camera_aux_head'}
    MODULES_OF_LOSS = {
        'aux_lidar': ['vfe', 'backbone_3d', 'map_to_bev_module', 'lidar_aux_head'],
        'aux_radar': ['radar_vfe', 'radar_map_to_bev', 'radar_backbone_2d', 'radar_aux_head'],
        'aux_camera': ['image_backbone', 'neck', 'vtransform', 'camera_aux_head'],
        'depth': ['image_backbone', 'neck', 'vtransform'],
    }

    def __init__(self, model_cfg, num_class, dataset):
        Detector3DTemplate.__init__(self, model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'radar_vfe', 'radar_map_to_bev', 'radar_backbone_2d',
            'image_backbone', 'neck', 'vtransform',
            'lidar_aux_head', 'radar_aux_head', 'camera_aux_head',
            'fuser', 'camera_fuser', 'backbone_2d', 'dense_head',
        ]
        self.check_bev_grids(dataset)
        self.check_camera_bev_grid(dataset)
        self.module_list = self.build_networks()
        self.aux_heads = [m for m in [self.lidar_aux_head, self.radar_aux_head, self.camera_aux_head] if m is not None]

        aux_head_names = list(self.AUX_HEAD_OF_LOSS.values())
        self.MODULES_OF_LOSS = dict(self.MODULES_OF_LOSS,
                                    rpn=[name for name in self.module_topology if name not in aux_head_names])
        self.train_losses = list(model_cfg.get('TRAIN_LOSSES', ['rpn', 'aux_lidar', 'aux_radar', 'aux_camera', 'depth']))
        self.train_losses = [name for name in self.train_losses if name != 'depth' or self.has_depth_loss()]
        self.train_losses = [name for name in self.train_losses
                             if name not in self.AUX_HEAD_OF_LOSS or getattr(self, self.AUX_HEAD_OF_LOSS[name]) is not None]
        assert set(self.train_losses) <= set(self.MODULES_OF_LOSS), self.train_losses
        train_names = set(sum([self.MODULES_OF_LOSS[name] for name in self.train_losses], []))
        self.train_modules = [getattr(self, name) for name in self.module_topology
                              if name in train_names and getattr(self, name) is not None]

        self.frozen_module_names = list(model_cfg.get('FREEZE_MODULES', []))
        for name in self.frozen_module_names:
            for param in getattr(self, name).parameters():
                param.requires_grad_(False)

    def has_depth_loss(self):
        return self.vtransform is not None and getattr(self.vtransform, 'depth_loss_cfg', None) is not None

    def check_camera_bev_grid(self, dataset):
        """The camera BEV must have exactly the lidar BEV's grid: same range and cell size."""
        vt_cfg = self.model_cfg.VTRANSFORM
        lidar_stride = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        lidar_cell = np.array(dataset.voxel_size[:2]) * lidar_stride
        camera_cell = np.array([vt_cfg.XBOUND[2], vt_cfg.YBOUND[2]]) * vt_cfg.DOWNSAMPLE
        assert np.allclose(lidar_cell, camera_cell), f'lidar BEV cell {lidar_cell} != camera BEV cell {camera_cell}'
        pc_range = dataset.point_cloud_range
        assert np.allclose([vt_cfg.XBOUND[0], vt_cfg.XBOUND[1], vt_cfg.YBOUND[0], vt_cfg.YBOUND[1]],
                           [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]), \
            f'VTRANSFORM XBOUND / YBOUND must span POINT_CLOUD_RANGE {pc_range}'

    build_image_backbone = BevFusion.build_image_backbone
    build_neck = BevFusion.build_neck

    def build_vtransform(self, model_info_dict):
        vtransform_module, model_info_dict = BevFusion.build_vtransform(self, model_info_dict)
        if vtransform_module is not None:
            model_info_dict['num_camera_bev_features'] = self.model_cfg.VTRANSFORM.OUT_CHANNEL
        return vtransform_module, model_info_dict

    def build_camera_aux_head(self, model_info_dict):
        if self.model_cfg.get('AUX_HEADS', {}).get('CAMERA', None) is None:
            return None, model_info_dict
        return self.build_aux_head(self.model_cfg.AUX_HEADS.CAMERA, model_info_dict['num_camera_bev_features'],
                                   model_info_dict)

    def build_camera_fuser(self, model_info_dict):
        camera_fuser = fuser.__all__[self.model_cfg.CAMERA_FUSER.NAME](
            model_cfg=self.model_cfg.CAMERA_FUSER,
            lidar_channels=model_info_dict['num_bev_features'],
            radar_channels=model_info_dict['num_camera_bev_features']
        )
        model_info_dict['module_list'].append(camera_fuser)
        model_info_dict['num_bev_features'] = camera_fuser.num_bev_features
        return camera_fuser, model_info_dict

    def train(self, mode=True):
        super().train(mode)
        for name in getattr(self, 'frozen_module_names', []):
            getattr(self, name).eval()
        return self

    def modality_dropout(self, batch_dict, key, prob):
        """Zero the BEV batch_dict[key] of each sample with probability prob."""
        if prob <= 0:
            return
        bev = batch_dict[key]
        keep = (torch.rand(bev.shape[0], device=bev.device) >= prob).to(bev.dtype)
        batch_dict[key] = bev * keep.view(-1, 1, 1, 1)

    def forward(self, batch_dict):
        if self.training:
            dropout_cfg = self.model_cfg.get('MODALITY_DROPOUT', {})
            for cur_module in self.train_modules:
                if cur_module is self.fuser:
                    self.modality_dropout(batch_dict, self.fuser.radar_key, dropout_cfg.get('RADAR', 0.0))
                elif cur_module is self.camera_fuser:
                    self.modality_dropout(batch_dict, self.camera_fuser.radar_key, dropout_cfg.get('CAMERA', 0.0))
                batch_dict = cur_module(batch_dict)

            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            for cur_module in self.module_list:
                if any(cur_module is aux_head for aux_head in self.aux_heads):
                    continue
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss = 0

        if 'rpn' in self.train_losses:
            loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
            tb_dict.update({'loss_rpn': loss_rpn.item(), **tb_dict_rpn})
            loss = loss + loss_rpn
        for name, head_name in self.AUX_HEAD_OF_LOSS.items():
            if name in self.train_losses:
                loss_aux, tb_dict_aux = getattr(self, head_name).get_loss()
                tb_dict.update(tb_dict_aux)
                loss = loss + loss_aux
        if 'depth' in self.train_losses:
            loss_depth, tb_dict_depth = self.vtransform.get_loss()
            tb_dict.update(tb_dict_depth)
            loss = loss + loss_depth
        return loss, tb_dict, disp_dict
