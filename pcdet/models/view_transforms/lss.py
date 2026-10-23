import torch
from torch import nn

from .depth_lss import DepthLSSTransform


class LSSTransform(DepthLSSTransform):
    """
        Camera-only LSS (as in BEVDet): depth is predicted from image features alone,
        without the lidar depth input used by DepthLSSTransform.
    """
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        in_channel = self.model_cfg.IN_CHANNEL
        self.dtransform = None
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features (tensor): bev features from image modality
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
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        # convert bev features from (b, c, x, y) to (b, c, y, x)
        x = x.permute(0, 1, 3, 2)
        batch_dict['spatial_features'] = x
        return batch_dict
