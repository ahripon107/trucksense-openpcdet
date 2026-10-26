"""
MAN TruckScenes cameras for the nuScenes-format conversion.

The converted dataset only exports CAMERA_LEFT_FRONT (as CAM_FRONT), so the 4 cameras are read from the original
TruckScenes data. add_camera_infos() gives every converted info (matched by the sample token) info['cams'] in the
layout NuScenesDataset.load_camera_info() expects, for the keyframe image of each camera:
  * sensor2lidar_rotation / sensor2lidar_translation: camera -> ego(t_camera) -> global -> virtual LIDAR_TOP of the
    sample (points @ R.T + T), so each camera uses its own ego pose (the back cameras fire ~35 ms off the lidar);
  * camera_intrinsics: the stored 3x3 pinhole matrix (f = 640, principal point (960, 520) on 1980 x 943 images; the
    images are undistorted, which the lidar/GT-box reprojection confirms);
  * data_path: image path relative to the TruckScenes root (CAMERA_CONFIG.DATA_PATH).
"""
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from .truckscenes_radar import _pose_matrix, iter_json_array

CAMERA_CHANNELS = ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK']


def add_camera_infos(data_path, truckscenes_path, truckscenes_version, info_names, suffix='_withcam'):
    """Write <info><suffix>.pkl next to every info file in info_names, with info['cams'] for the 4 cameras."""
    meta = Path(truckscenes_path) / truckscenes_version
    sensors = {r['token']: r['channel'] for r in json.load(open(meta / 'sensor.json'))}
    calibs = {r['token']: r for r in json.load(open(meta / 'calibrated_sensor.json'))}
    channel_of_calib = {tok: sensors[r['sensor_token']] for tok, r in calibs.items()}

    keyframes = {}
    for r in tqdm(iter_json_array(meta / 'sample_data.json'), desc='camera sample_data', dynamic_ncols=True):
        ch = channel_of_calib[r['calibrated_sensor_token']]
        if ch in CAMERA_CHANNELS and r['is_key_frame']:
            keyframes[(r['sample_token'], ch)] = (r['filename'], r['timestamp'], r['ego_pose_token'],
                                                  r['calibrated_sensor_token'], r['width'], r['height'])
    pose_tokens = {v[2] for v in keyframes.values()}
    ego_pose = {}
    for r in tqdm(iter_json_array(meta / 'ego_pose.json'), desc='ego_pose', dynamic_ncols=True):
        if r['token'] in pose_tokens:
            ego_pose[r['token']] = (r['translation'], r['rotation'])

    for info_name in info_names:
        src = Path(data_path) / info_name
        with open(src, 'rb') as f:
            infos = pickle.load(f)
        n_missing = 0
        for info in tqdm(infos, desc=f'camera infos {info_name}', dynamic_ncols=True):
            ref_from_global = info['ref_from_car'] @ info['car_from_global']
            info['cams'] = {}
            for ch in CAMERA_CHANNELS:
                rec = keyframes.get((info['token'], ch))
                if rec is None:
                    n_missing += 1
                    continue
                filename, timestamp, pose_tok, calib_tok, width, height = rec
                calib = calibs[calib_tok]
                translation, rotation = ego_pose[pose_tok]
                cam_to_ref = ref_from_global @ _pose_matrix(translation, rotation) @ \
                    _pose_matrix(calib['translation'], calib['rotation'])
                info['cams'][ch] = {
                    'data_path': filename,
                    'type': ch,
                    'timestamp': timestamp,
                    'image_size': (width, height),
                    'sensor2ego_translation': calib['translation'],
                    'sensor2ego_rotation': calib['rotation'],
                    'ego2global_translation': translation,
                    'ego2global_rotation': rotation,
                    'sensor2lidar_rotation': cam_to_ref[:3, :3],
                    'sensor2lidar_translation': cam_to_ref[:3, 3],
                    'camera_intrinsics': np.array(calib['camera_intrinsic']),
                }
        assert n_missing == 0, f'{n_missing} camera keyframes missing in {info_name}'

        dst = src.with_name(src.stem + suffix + '.pkl')
        with open(dst, 'wb') as f:
            pickle.dump(infos, f)
        print(f'{info_name}: {len(infos)} samples with {len(CAMERA_CHANNELS)} cameras -> {dst}')
