"""
Radar point layout shared by the dataset, the augmentations, the GT sampler and the radar losses, and the
geometric transforms that must keep the radar velocity consistent with the moved points.

Columns of 'radar_points' (N, len(RADAR_FEATURES)):
  x, y, z: position; rcs [dBsm]; vx_comp, vy_comp: xy of the ego-motion compensated radial velocity vector
  (v_r * ray); v_r: compensated radial velocity [m/s] (> 0 moving away); time_lag: t_ref - t_sweep [s];
  ray_x/y/z: unit line of sight from the sensor at t_sweep; range: distance to that sensor; sensor_id
"""
import numpy as np

from . import common_utils

RADAR_FEATURES = ['x', 'y', 'z', 'rcs', 'vx_comp', 'vy_comp', 'v_r', 'time_lag',
                  'ray_x', 'ray_y', 'ray_z', 'range', 'sensor_id']
RADAR_COL = {name: i for i, name in enumerate(RADAR_FEATURES)}
VX, VY, VR = RADAR_COL['vx_comp'], RADAR_COL['vy_comp'], RADAR_COL['v_r']
RX, RY, RZ, RANGE = RADAR_COL['ray_x'], RADAR_COL['ray_y'], RADAR_COL['ray_z'], RADAR_COL['range']


def flip_along_x(radar_points):
    """y -> -y, for the position, the velocity vector and the ray."""
    radar_points[:, [1, VY, RY]] = -radar_points[:, [1, VY, RY]]
    return radar_points


def flip_along_y(radar_points):
    """x -> -x, for the position, the velocity vector and the ray."""
    radar_points[:, [0, VX, RX]] = -radar_points[:, [0, VX, RX]]
    return radar_points


def rotate_along_z(radar_points, angle):
    """Rotate the position, the velocity vector and the ray by the same angle (common_utils convention)."""
    def rot(cols):
        xyz = np.zeros((radar_points.shape[0], 3), dtype=radar_points.dtype)
        xyz[:, :len(cols)] = radar_points[:, cols]
        return common_utils.rotate_points_along_z(xyz[np.newaxis], np.array([angle]))[0][:, :len(cols)]

    radar_points[:, 0:3] = rot([0, 1, 2])
    radar_points[:, [VX, VY]] = rot([VX, VY])
    radar_points[:, [RX, RY]] = rot([RX, RY])
    return radar_points


def scale(radar_points, noise_scale):
    """Scaling the world scales positions, distances and velocities (as for the GT box velocity); rays are unit."""
    radar_points[:, [0, 1, 2, VX, VY, VR, RANGE]] *= noise_scale
    return radar_points


def translate(radar_points, noise_translate):
    radar_points[:, 0:3] += noise_translate.reshape(1, 3)
    return radar_points


def move_object_points(radar_points, offset, box_velocity):
    """
    Move radar points of one object by offset (3,), keeping the sensor where it was, and recompute their radial
    velocity for the new line of sight: v_r' = v_r + v_box . (ray' - ray). This keeps the measured residual
    (micro-Doppler, noise) and swaps the geometric part. box_velocity: (2,) [vx, vy] of the object.
    """
    if radar_points.shape[0] == 0:
        return radar_points
    pos = radar_points[:, 0:3].astype(np.float64)
    ray = radar_points[:, [RX, RY, RZ]].astype(np.float64)
    sensor = pos - radar_points[:, RANGE:RANGE + 1] * ray
    new_pos = pos + np.asarray(offset, dtype=np.float64).reshape(1, 3)
    new_rng = np.maximum(np.linalg.norm(new_pos - sensor, axis=1), 1e-3)
    new_ray = (new_pos - sensor) / new_rng[:, None]
    v_r = radar_points[:, VR] + (new_ray[:, :2] - ray[:, :2]) @ np.asarray(box_velocity, dtype=np.float64)

    radar_points[:, 0:3] = new_pos
    radar_points[:, [RX, RY, RZ]] = new_ray
    radar_points[:, RANGE] = new_rng
    radar_points[:, VR] = v_r
    radar_points[:, VX] = v_r * new_ray[:, 0]
    radar_points[:, VY] = v_r * new_ray[:, 1]
    return radar_points
