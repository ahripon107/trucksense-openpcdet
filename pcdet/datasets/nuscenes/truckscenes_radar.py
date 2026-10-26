"""
MAN TruckScenes radar for the nuScenes-format conversion.

The converted dataset only holds the merged lidar, so the 6 radars are read from the original
TruckScenes sensor data. add_radar_infos() attaches, to every converted info (matched by the
sample token, which the converter keeps), the keyframe and the past sweeps of each radar with
  * the transform sensor -> ego(t_sweep) -> global -> ego(t_ref) -> virtual LIDAR_TOP, built from the
    info's own ref_from_car / car_from_global, so radar lands in exactly the lidar frame;
  * the sensor's own velocity (ego chassis velocity + yaw_rate x mount position) in the sensor frame,
    used to compensate the radial velocity for ego motion.
TruckScenes vrel_xyz is the raw Doppler velocity as a vector along the ray (not ego-compensated):
on static points vrel . ray == -v_sensor . ray.

Radar points are returned as (N, len(RADAR_FEATURES)) float32 in the virtual lidar frame (columns: radar_utils).
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from ...utils.radar_utils import RADAR_COL, RADAR_FEATURES

RADAR_CHANNELS = ['RADAR_LEFT_FRONT', 'RADAR_RIGHT_FRONT', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_SIDE',
                  'RADAR_LEFT_BACK', 'RADAR_RIGHT_BACK']  # index = sensor_id



def iter_json_array(path, chunk_size=1 << 24):
    """Yield the records of a top-level JSON array one by one, without loading the whole file."""
    decoder = json.JSONDecoder()
    skip = re.compile(r'[\s,]*')
    with open(path, 'r') as f:
        buf = f.read(chunk_size)
        pos = buf.index('[') + 1
        while True:
            pos = skip.match(buf, pos).end()
            if pos >= len(buf) or buf[pos] != ']':
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except (json.JSONDecodeError, IndexError):
                    more = f.read(chunk_size)
                    if not more:
                        if buf[pos:].strip() in ('', ']'):
                            return
                        raise
                    buf, pos = buf[pos:] + more, 0
                    continue
                yield obj
                pos = end
            else:
                return


def read_radar_pcd(path):
    """TruckScenes radar .pcd (binary): (N, 7) float32 [x, y, z, vrel_x, vrel_y, vrel_z, rcs] in the sensor frame."""
    buf = open(path, 'rb').read()
    start = buf.index(b'DATA binary\n') + len(b'DATA binary\n')
    header = buf[:start].decode('ascii')
    assert 'FIELDS x y z vrel_x vrel_y vrel_z rcs' in header, f'{path}: unexpected radar fields'
    return np.frombuffer(buf, dtype=np.float32, offset=start).reshape(-1, 7)


def radar_filename(channel, timestamp, is_key_frame):
    folder = 'samples' if is_key_frame else 'sweeps'
    return f'{folder}/{channel}/{channel}_{timestamp}.pcd'


def _pose_matrix(translation, rotation):
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def add_radar_infos(data_path, truckscenes_path, truckscenes_version, info_names, max_sweeps=12):
    """
    Write <info>_withradar.pkl next to every info file in info_names.

    info['radar'] = {
        'sensor_id': (K,) int8, 'timestamp': (K,) int64, 'is_key_frame': (K,) bool,  # file of each sweep
        'transform': (K, 4, 4) float32  sensor -> virtual lidar frame of this sample,
        'time_lag': (K,) float32, 'sensor_velocity': (K, 3) float32  sensor velocity in the sensor frame,
        'sweep_index': (K,) int8  0 = keyframe, 1 = previous sweep, ...,
    }
    info['radar_sensor_origins'] = (len(RADAR_CHANNELS), 3) float32 radar mounts in the virtual lidar frame.
    K = up to max_sweeps (keyframe + past sweeps) per radar. TruckScenes allows 11 past radar sweeps at test time.
    """
    meta = Path(truckscenes_path) / truckscenes_version
    sensors = {r['token']: r['channel'] for r in json.load(open(meta / 'sensor.json'))}
    calibs = {r['token']: r for r in json.load(open(meta / 'calibrated_sensor.json'))}
    channel_of_calib = {tok: sensors[r['sensor_token']] for tok, r in calibs.items()}
    radar_calib = {RADAR_CHANNELS.index(ch): calibs[tok] for tok, ch in channel_of_calib.items() if ch in RADAR_CHANNELS}
    assert len(radar_calib) == len(RADAR_CHANNELS), 'expected one calibration per radar'
    car_from_sensor = {sid: _pose_matrix(c['translation'], c['rotation']) for sid, c in radar_calib.items()}

    # sample_data is 1.7 GB of JSON: stream it and keep only the radar records
    radar_sd, keyframe_of = {}, {}
    for r in tqdm(iter_json_array(meta / 'sample_data.json'), desc='radar sample_data', dynamic_ncols=True):
        ch = channel_of_calib[r['calibrated_sensor_token']]
        if ch not in RADAR_CHANNELS:
            continue
        sid = RADAR_CHANNELS.index(ch)
        assert r['filename'] == radar_filename(ch, r['timestamp'], r['is_key_frame']), r['filename']
        radar_sd[r['token']] = (sid, r['timestamp'], r['is_key_frame'], r['prev'],
                                r['ego_pose_token'], r['ego_motion_chassis_token'])
        if r['is_key_frame']:
            keyframe_of[(r['sample_token'], sid)] = r['token']
    pose_tokens = {v[4] for v in radar_sd.values()}
    motion_tokens = {v[5] for v in radar_sd.values()}

    ego_pose = {}
    for r in tqdm(iter_json_array(meta / 'ego_pose.json'), desc='ego_pose', dynamic_ncols=True):
        if r['token'] in pose_tokens:
            ego_pose[r['token']] = (r['translation'], r['rotation'])
    ego_motion = {}
    for r in tqdm(iter_json_array(meta / 'ego_motion_chassis.json'), desc='ego_motion_chassis', dynamic_ncols=True):
        if r['token'] in motion_tokens:
            ego_motion[r['token']] = (r['vx'], r['vy'], r['vz'], r['yaw_rate'])

    for info_name in info_names:
        src = Path(data_path) / info_name
        with open(src, 'rb') as f:
            infos = pickle.load(f)
        n_missing = 0
        for info in tqdm(infos, desc=f'radar infos {info_name}', dynamic_ncols=True):
            ref_from_global = info['ref_from_car'] @ info['car_from_global']
            ref_time_us = int(round(info['timestamp'] * 1e6))
            cols = {k: [] for k in ['sensor_id', 'timestamp', 'is_key_frame', 'transform', 'time_lag',
                                    'sensor_velocity', 'sweep_index']}
            for sid in range(len(RADAR_CHANNELS)):
                token = keyframe_of.get((info['token'], sid))
                if token is None:
                    n_missing += 1
                    continue
                R_cs, t_cs = car_from_sensor[sid][:3, :3], car_from_sensor[sid][:3, 3]
                for k in range(max_sweeps):
                    _, ts, is_key, prev, pose_tok, motion_tok = radar_sd[token]
                    translation, rotation = ego_pose[pose_tok]
                    tf = ref_from_global @ _pose_matrix(translation, rotation) @ car_from_sensor[sid]
                    vx, vy, vz, yaw_rate = ego_motion[motion_tok]
                    # velocity of the mount point (ego frame): v_ego + omega_z x t_cs, then into the sensor frame
                    v_mount = np.array([vx - yaw_rate * t_cs[1], vy + yaw_rate * t_cs[0], vz])
                    cols['sensor_id'].append(sid)
                    cols['timestamp'].append(ts)
                    cols['is_key_frame'].append(is_key)
                    cols['transform'].append(tf)
                    cols['time_lag'].append((ref_time_us - ts) * 1e-6)
                    cols['sensor_velocity'].append(R_cs.T @ v_mount)
                    cols['sweep_index'].append(k)
                    if prev == '' or prev not in radar_sd:
                        break
                    token = prev
            info['radar'] = {
                'sensor_id': np.array(cols['sensor_id'], dtype=np.int8),
                'timestamp': np.array(cols['timestamp'], dtype=np.int64),
                'is_key_frame': np.array(cols['is_key_frame'], dtype=bool),
                'transform': np.array(cols['transform'], dtype=np.float32).reshape(-1, 4, 4),
                'time_lag': np.array(cols['time_lag'], dtype=np.float32),
                'sensor_velocity': np.array(cols['sensor_velocity'], dtype=np.float32).reshape(-1, 3),
                'sweep_index': np.array(cols['sweep_index'], dtype=np.int8),
            }
            info['radar_sensor_origins'] = np.stack([
                (info['ref_from_car'] @ car_from_sensor[sid][:, 3])[:3] for sid in range(len(RADAR_CHANNELS))
            ]).astype(np.float32)

        dst = src.with_name(src.stem + '_withradar.pkl')
        with open(dst, 'wb') as f:
            pickle.dump(infos, f)
        print(f'{info_name}: {len(infos)} samples, {n_missing} missing radar keyframes -> {dst}')


def load_radar_points(info, truckscenes_path, max_sweeps):
    """Radar points of all radars, keyframe + (max_sweeps - 1) past sweeps each, as (N, len(RADAR_FEATURES))."""
    radar = info['radar']
    out = []
    for i in np.nonzero(radar['sweep_index'] < max_sweeps)[0]:
        sid = int(radar['sensor_id'][i])
        path = Path(truckscenes_path) / radar_filename(RADAR_CHANNELS[sid], radar['timestamp'][i],
                                                       radar['is_key_frame'][i])
        pts = read_radar_pcd(path)
        pts = pts[np.isfinite(pts).all(axis=1)]
        xyz, vrel, rcs = pts[:, 0:3].astype(np.float64), pts[:, 3:6], pts[:, 6]
        rng = np.linalg.norm(xyz, axis=1)
        keep = rng > 1e-3
        xyz, vrel, rcs, rng = xyz[keep], vrel[keep], rcs[keep], rng[keep]
        ray_s = xyz / rng[:, None]
        # raw Doppler along the ray, plus the sensor's own motion along the ray => radial velocity in the world
        v_r = np.sum(vrel * ray_s, axis=1) + ray_s @ radar['sensor_velocity'][i].astype(np.float64)

        tf = radar['transform'][i].astype(np.float64)
        ray = ray_s @ tf[:3, :3].T
        cur = np.empty((len(xyz), len(RADAR_FEATURES)), dtype=np.float32)
        cur[:, 0:3] = xyz @ tf[:3, :3].T + tf[:3, 3]
        cur[:, RADAR_COL['rcs']] = rcs
        cur[:, RADAR_COL['vx_comp']] = v_r * ray[:, 0]
        cur[:, RADAR_COL['vy_comp']] = v_r * ray[:, 1]
        cur[:, RADAR_COL['v_r']] = v_r
        cur[:, RADAR_COL['time_lag']] = radar['time_lag'][i]
        cur[:, RADAR_COL['ray_x']:RADAR_COL['ray_z'] + 1] = ray
        cur[:, RADAR_COL['range']] = rng
        cur[:, RADAR_COL['sensor_id']] = sid
        out.append(cur)
    if len(out) == 0:
        return np.zeros((0, len(RADAR_FEATURES)), dtype=np.float32)
    return np.concatenate(out, axis=0)
