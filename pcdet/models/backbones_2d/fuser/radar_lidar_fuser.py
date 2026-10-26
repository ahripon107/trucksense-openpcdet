import torch
from torch import nn


class RadarLidarBEVFuser(nn.Module):
    """
    Fuses the radar BEV into the lidar BEV: concat(lidar, radar) -> NUM_FILTERS conv3x3-BN-ReLU -> conv3x3-BN back to the
    lidar channels, added to the lidar BEV. The last BN starts with zero scale, so the fused BEV starts out as the
    lidar BEV and a model initialised from a lidar-only checkpoint starts with its accuracy.
    """
    def __init__(self, model_cfg, lidar_channels, radar_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.lidar_key = model_cfg.get('LIDAR_KEY', 'spatial_features')
        self.radar_key = model_cfg.get('RADAR_KEY', 'radar_spatial_features_2d')

        layers = []
        c_in = lidar_channels + radar_channels
        for c_out in model_cfg.NUM_FILTERS:
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
            ])
            c_in = c_out
        last_bn = nn.BatchNorm2d(lidar_channels, eps=1e-3, momentum=0.01)
        nn.init.zeros_(last_bn.weight)
        layers.extend([nn.Conv2d(c_in, lidar_channels, 3, padding=1, bias=False), last_bn])
        self.fuse = nn.Sequential(*layers)
        self.num_bev_features = lidar_channels

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features: (B, C_lidar, H, W) lidar BEV
                radar_spatial_features_2d: (B, C_radar, H, W) radar BEV, same grid and extent
        Returns:
            batch_dict:
                spatial_features: (B, C_lidar, H, W) fused BEV (the lidar BEV stays in lidar_spatial_features)
        """
        lidar_bev = batch_dict[self.lidar_key]
        radar_bev = batch_dict[self.radar_key]
        assert lidar_bev.shape[0] == radar_bev.shape[0] and lidar_bev.shape[2:] == radar_bev.shape[2:], \
            f'lidar BEV {tuple(lidar_bev.shape)} and radar BEV {tuple(radar_bev.shape)} must share the grid'
        fused = lidar_bev + self.fuse(torch.cat([lidar_bev, radar_bev.to(lidar_bev.dtype)], dim=1))
        batch_dict['lidar_' + self.lidar_key] = lidar_bev
        batch_dict[self.lidar_key] = fused
        return batch_dict
