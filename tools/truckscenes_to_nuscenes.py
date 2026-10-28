"""
Convert MAN TruckScenes into a nuScenes-format dataset that OpenPCDet's NuScenesDataset
(and nuscenes-devkit) can consume unmodified.

What the conversion does
  * Merges the 6 lidars (LIDAR_LEFT/RIGHT/TOP_FRONT/TOP_LEFT/TOP_RIGHT/REAR) of every lidar
    frame into a single virtual LIDAR_TOP, written as nuScenes-style .pcd.bin
    (float32 x, y, z, intensity[0-255], ring). "ring" holds the source lidar index (0-5).
    Each sensor's points go sensor -> ego(t_sensor) -> global -> ego(t_ref), so the
    up-to-50 ms firing offsets between lidars are compensated with the ego poses.
    LIDAR_TOP_FRONT (fires closest to the sample timestamp) is the reference clock.
  * The virtual LIDAR_TOP is axis-aligned with the ego frame and placed VIRTUAL_LIDAR_HEIGHT
    above the ego origin (which sits on the ground), so stock nuScenes z ranges
    ([-5, 3] in sensor frame) still cover the ground and tall trucks.
  * Writes the 13 nuScenes tables (log/map are synthetic placeholders; no map rasters).
  * Drops annotations of vehicle.train, the only category that is not one of the 12 TruckScenes
    detection classes (truckscenes.eval.detection.utils.category_to_detection_name). The other
    TruckScenes-only categories (vehicle.other, vehicle.ego_trailer, static_object.traffic_sign)
    are kept; OpenPCDet maps them with map_name_from_truckscenes_to_detection.
  * Renames scenes to nuScenes split names (scene-0061, ...) following the official
    TruckScenes train/val split, because OpenPCDet and the nuScenes eval select scenes by
    hard-coded nuScenes names. The original name is kept in scene['description'].
  * Only CAMERA_LEFT_FRONT keyframes are exported (as CAM_FRONT), because OpenPCDet's info
    builder requires a CAM_FRONT entry; other cameras and radars are not exported.
    Images are not copied: their sample_data filename is a path relative to <out>/<version>
    that points back into the original TruckScenes sensor data.

Usage (openmmlab env, needs python-lzf):
  python truckscenes_to_nuscenes.py \
      --meta  man-truckscenes_metadata_v1.2-mini/man-truckscenes/v1.2-mini \
      --data  man-truckscenes_sensordata_v1.2-mini/man-truckscenes \
      --out   openpcdet --version v1.0-mini
  --data takes several roots (e.g. sensordata01/man-truckscenes sensordata02/man-truckscenes ...);
  each sample_data filename is resolved to the root that contains it.
  --crop drops merged points outside POINT_CLOUD_RANGE (x/y +-51.2 m, z [-5, 3] in the virtual lidar frame).
"""
import argparse
import gc
import json
import os
import struct
import uuid
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import lzf
import numpy as np
from PIL import Image
from nuscenes.utils import splits as nusc_splits
from pyquaternion import Quaternion

LIDARS = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']
REF_LIDAR = 'LIDAR_TOP_FRONT'
FRONT_CAMERA = 'CAMERA_LEFT_FRONT'  # exported as CAM_FRONT (optical axis ~7 deg left of ego x)
VIRTUAL_LIDAR_HEIGHT = 1.84  # nuScenes LIDAR_TOP mounting height
MAX_SYNC_DIFF_US = 60000  # max timestamp gap when grouping lidars into one frame
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # used by --crop
DROP_CATEGORIES = {'vehicle.train'}

# Official TruckScenes splits (truckscenes-devkit/src/truckscenes/utils/splits.py)
TRUCKSCENES_SPLITS = {
    'v1.0-mini': {
        'train': [
            'scene-3f542f89ec5241b6a4e30ca743adcf34-29', 'scene-b201f6e737e34c2d87830fc1e2911516-6',
            'scene-6be74229febb4aad9e7ae1a37217b231-4', 'scene-a6a87db5125846bda72ccfc9931ee153-11',
            'scene-b13542304b244af585a646082c57a079-1', 'scene-23160c4bff134cdd86b96ee822b4aca9-14',
            'scene-fb64d203d417452f830ca73efed80c41-11', 'scene-3195e3bfd1de4f9ba82f874790c7856c-15',
        ],
        'val': ['scene-13f4b71b1bd04a9e88747ad8f58a3f67-4', 'scene-0044384af3d8494e913fb8b14915239e-3'],
    },
}
NUSC_SPLITS = {
    'v1.0-mini': {'train': nusc_splits.mini_train, 'val': nusc_splits.mini_val},
    'v1.0-trainval': {'train': nusc_splits.train, 'val': nusc_splits.val},
}


def get_truckscenes_splits(version):
    if version in TRUCKSCENES_SPLITS:
        return TRUCKSCENES_SPLITS[version]
    from truckscenes.utils import splits as ts_splits  # pip install truckscenes-devkit
    return {'train': ts_splits.train, 'val': ts_splits.val}


def read_pcd(path):
    """Read a PCD file (ascii / binary / binary_compressed) into a structured numpy array."""
    buf = open(path, 'rb').read()
    data_start = buf.index(b'\n', buf.index(b'\nDATA ') + 1) + 1
    header = {}
    for line in buf[:data_start].decode('ascii').splitlines():
        if line and not line.startswith('#'):
            key, *vals = line.split()
            header[key] = vals
    num_points = int(header['POINTS'][0])
    dtype = np.dtype([(f, {'F': 'f', 'U': 'u', 'I': 'i'}[t] + s)
                      for f, t, s in zip(header['FIELDS'], header['TYPE'], header['SIZE'])])
    fmt = header['DATA'][0]
    body = buf[data_start:]
    if fmt == 'binary':
        return np.frombuffer(body, dtype=dtype, count=num_points)
    if fmt == 'ascii':
        return np.loadtxt(body.decode().splitlines(), dtype=dtype, ndmin=1)
    assert fmt == 'binary_compressed', fmt
    comp_size, raw_size = struct.unpack('<II', body[:8])
    raw = lzf.decompress(body[8:8 + comp_size], raw_size)
    # binary_compressed stores the fields column by column
    out = np.empty(num_points, dtype=dtype)
    offset = 0
    for name in dtype.names:
        nbytes = dtype[name].itemsize * num_points
        out[name] = np.frombuffer(raw, dtype=dtype[name], count=num_points, offset=offset)
        offset += nbytes
    return out


def pose_matrix(rec):
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rec['rotation']).rotation_matrix
    mat[:3, 3] = rec['translation']
    return mat


def merge_frame(job):
    """job: (out_path, ref_ego_pose, [(lidar_idx, pcd_path, calib, ego_pose), ...], crop)"""
    out_path, ref_pose, sources, crop = job
    if os.path.exists(out_path):
        return out_path, None
    global_to_ref = np.linalg.inv(pose_matrix(ref_pose))
    global_to_ref[2, 3] -= VIRTUAL_LIDAR_HEIGHT
    chunks = []
    for lidar_idx, pcd_path, calib, ego_pose in sources:
        pcd = read_pcd(pcd_path)
        xyz = np.stack([pcd['x'], pcd['y'], pcd['z']], axis=1).astype(np.float64)
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        tf = global_to_ref @ pose_matrix(ego_pose) @ pose_matrix(calib)
        xyz = xyz @ tf[:3, :3].T + tf[:3, 3]
        pts = np.empty((len(xyz), 5), dtype=np.float32)
        pts[:, :3] = xyz
        pts[:, 3] = np.clip(pcd['intensity'][valid], 0, 1) * 255.0
        pts[:, 4] = lidar_idx
        chunks.append(pts)
    points = np.concatenate(chunks)
    if crop:
        lo, hi = np.array(POINT_CLOUD_RANGE[:3]), np.array(POINT_CLOUD_RANGE[3:])
        points = points[((points[:, :3] >= lo) & (points[:, :3] <= hi)).all(axis=1)]
    tmp = out_path + '.tmp'
    points.tofile(tmp)
    os.replace(tmp, out_path)
    return out_path, len(points)


def new_token():
    return uuid.uuid4().hex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True, help='TruckScenes metadata dir (contains sample.json ...)')
    parser.add_argument('--data', required=True, nargs='+',
                        help='TruckScenes sensor data root(s) (each contains samples/, sweeps/)')
    parser.add_argument('--out', required=True, help='output DATA_PATH; data goes to <out>/<version>')
    parser.add_argument('--version', default='v1.0-mini', choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--crop', action='store_true', help='drop merged points outside POINT_CLOUD_RANGE')
    args = parser.parse_args()

    meta, data_roots = Path(args.meta), [Path(d) for d in args.data]

    def resolve(filename):
        """Path of a sample_data filename in whichever sensor data root contains it."""
        for root in data_roots:
            if (root / filename).exists():
                return root / filename
        raise FileNotFoundError(f'{filename} not found in any of {args.data}')

    out_root = Path(args.out) / args.version
    table_dir = out_root / args.version
    for d in [table_dir, out_root / 'samples' / 'LIDAR_TOP', out_root / 'sweeps' / 'LIDAR_TOP', out_root / 'maps']:
        d.mkdir(parents=True, exist_ok=True)

    load = lambda name: json.load(open(meta / f'{name}.json'))
    scenes, samples, sample_data = load('scene'), load('sample'), load('sample_data')
    calibs = {r['token']: r for r in load('calibrated_sensor')}
    sensors = {r['token']: r for r in load('sensor')}
    ego_poses = {r['token']: r for r in load('ego_pose')}
    categories, instances, annotations = load('category'), load('instance'), load('sample_annotation')

    sample_by_token = {s['token']: s for s in samples}
    channel_of = lambda sd: sensors[calibs[sd['calibrated_sensor_token']]['sensor_token']]['channel']

    # ---------------- scenes / logs / map ----------------
    ts_split, nu_split = get_truckscenes_splits(args.version), NUSC_SPLITS[args.version]
    assert len(ts_split['train']) <= len(nu_split['train']) and len(ts_split['val']) <= len(nu_split['val'])
    rename = dict(zip(ts_split['train'], nu_split['train']))
    rename.update(zip(ts_split['val'], nu_split['val']))

    out_scenes, out_logs = [], []
    for scene in scenes:
        if scene['name'] not in rename:
            print(f'skip scene not in {args.version} train/val split: {scene["name"]}')
            continue
        log_token = new_token()
        first_ts = sample_by_token[scene['first_sample_token']]['timestamp']
        out_logs.append({
            'token': log_token, 'logfile': scene['name'], 'vehicle': 'man-truck',
            'date_captured': datetime.fromtimestamp(first_ts / 1e6, tz=timezone.utc).strftime('%Y-%m-%d'),
            'location': 'truckscenes',
        })
        out_scenes.append({
            'token': scene['token'], 'log_token': log_token, 'nbr_samples': scene['nbr_samples'],
            'first_sample_token': scene['first_sample_token'], 'last_sample_token': scene['last_sample_token'],
            'name': rename[scene['name']],
            'description': f'{scene["name"]};{scene["description"]}',
        })
    kept_scenes = {s['token'] for s in out_scenes}
    out_map = [{'token': new_token(), 'log_tokens': [l['token'] for l in out_logs],
                'category': 'semantic_prior', 'filename': 'maps/placeholder.png'}]
    Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(out_root / 'maps' / 'placeholder.png')  # devkit asserts it exists

    # ---------------- lidar frames ----------------
    lidar_sds = defaultdict(lambda: defaultdict(list))  # scene -> channel -> [sd]
    for sd in sample_data:
        ch = channel_of(sd)
        if ch in LIDARS:
            scene_token = sample_by_token[sd['sample_token']]['scene_token']
            if scene_token in kept_scenes:
                lidar_sds[scene_token][ch].append(sd)
    for per_ch in lidar_sds.values():
        for lst in per_ch.values():
            lst.sort(key=lambda r: r['timestamp'])

    lidar_sensor = {'token': new_token(), 'channel': 'LIDAR_TOP', 'modality': 'lidar'}
    lidar_calib = {'token': new_token(), 'sensor_token': lidar_sensor['token'],
                   'translation': [0.0, 0.0, VIRTUAL_LIDAR_HEIGHT], 'rotation': [1.0, 0.0, 0.0, 0.0],
                   'camera_intrinsic': []}

    out_sd, out_poses, jobs, n_missing = [], {}, [], 0
    for scene_token, per_ch in lidar_sds.items():
        stamps = {ch: [r['timestamp'] for r in per_ch[ch]] for ch in LIDARS}
        keyframes = {ch: {r['sample_token']: r for r in per_ch[ch] if r['is_key_frame']} for ch in LIDARS}
        for ref in per_ch[REF_LIDAR]:
            sources = []
            for lidar_idx, ch in enumerate(LIDARS):
                if ref['is_key_frame'] and ref['sample_token'] in keyframes[ch]:
                    match = keyframes[ch][ref['sample_token']]
                else:
                    i = bisect_left(stamps[ch], ref['timestamp'])
                    cands = [per_ch[ch][j] for j in (i - 1, i) if 0 <= j < len(stamps[ch])]
                    match = min(cands, key=lambda r: abs(r['timestamp'] - ref['timestamp'])) if cands else None
                if match is None or abs(match['timestamp'] - ref['timestamp']) > MAX_SYNC_DIFF_US:
                    n_missing += 1
                    continue
                sources.append((lidar_idx, str(resolve(match['filename'])),
                                calibs[match['calibrated_sensor_token']], ego_poses[match['ego_pose_token']]))
            folder = 'samples' if ref['is_key_frame'] else 'sweeps'
            filename = f'{folder}/LIDAR_TOP/{Path(ref["filename"]).stem.replace(REF_LIDAR, "LIDAR_TOP")}.pcd.bin'
            pose = ego_poses[ref['ego_pose_token']]
            out_poses[pose['token']] = pose
            out_sd.append({
                'token': ref['token'], 'sample_token': ref['sample_token'], 'ego_pose_token': pose['token'],
                'calibrated_sensor_token': lidar_calib['token'], 'timestamp': ref['timestamp'],
                'fileformat': 'pcd', 'is_key_frame': ref['is_key_frame'], 'height': 0, 'width': 0,
                'filename': filename, 'prev': ref['prev'], 'next': ref['next'],
            })
            jobs.append((str(out_root / filename), pose, sources, args.crop))
    if n_missing:
        print(f'WARNING: {n_missing} lidar scans had no partner within {MAX_SYNC_DIFF_US / 1e3:.0f} ms')

    # ---------------- CAM_FRONT (keyframes only) ----------------
    # OpenPCDet's info builder always reads sample['data']['CAM_FRONT'], even for lidar-only training.
    cam_sensor = {'token': new_token(), 'channel': 'CAM_FRONT', 'modality': 'camera'}
    cam_calib = None
    for sd in sample_data:
        if not sd['is_key_frame'] or channel_of(sd) != FRONT_CAMERA:
            continue
        if sample_by_token[sd['sample_token']]['scene_token'] not in kept_scenes:
            continue
        if cam_calib is None:
            cam_calib = dict(calibs[sd['calibrated_sensor_token']], token=new_token(), sensor_token=cam_sensor['token'])
        filename = os.path.relpath(resolve(sd['filename']).resolve(), out_root.resolve())
        pose = ego_poses[sd['ego_pose_token']]
        out_poses[pose['token']] = pose
        out_sd.append({
            'token': sd['token'], 'sample_token': sd['sample_token'], 'ego_pose_token': pose['token'],
            'calibrated_sensor_token': cam_calib['token'], 'timestamp': sd['timestamp'],
            'fileformat': sd['fileformat'], 'is_key_frame': True, 'height': sd['height'], 'width': sd['width'],
            'filename': filename, 'prev': '', 'next': '',
        })

    # ---------------- samples / annotations ----------------
    out_samples = [{'token': s['token'], 'timestamp': s['timestamp'], 'prev': s['prev'], 'next': s['next'],
                    'scene_token': s['scene_token']} for s in samples if s['scene_token'] in kept_scenes]
    kept_samples = {s['token'] for s in out_samples}
    drop_cat = {c['token'] for c in categories if c['name'] in DROP_CATEGORIES}
    out_categories = [c for c in categories if c['token'] not in drop_cat]
    out_instances = [i for i in instances if i['category_token'] not in drop_cat]
    kept_instances = {i['token'] for i in out_instances}
    out_anns = [a for a in annotations if a['instance_token'] in kept_instances and a['sample_token'] in kept_samples]
    print(f'annotations: kept {len(out_anns)} / {len(annotations)} (dropped categories: {sorted(DROP_CATEGORIES)})')

    tables = {
        'attribute': load('attribute'), 'visibility': load('visibility'), 'category': out_categories,
        'instance': out_instances, 'sample_annotation': out_anns, 'sample': out_samples,
        'sample_data': out_sd, 'ego_pose': list(out_poses.values()), 'sensor': [lidar_sensor, cam_sensor],
        'calibrated_sensor': [lidar_calib, cam_calib], 'scene': out_scenes, 'log': out_logs, 'map': out_map,
    }
    for name, rows in tables.items():
        with open(table_dir / f'{name}.json', 'w') as f:
            json.dump(rows, f, indent=0)
        print(f'{name}.json: {len(rows)} records')

    # ---------------- point clouds ----------------
    # free the big trainval tables before forking the workers (jobs keep the records they need)
    del sample_data, annotations, ego_poses, out_sd, out_anns, out_poses, tables
    gc.collect()
    print(f'merging {len(jobs)} lidar frames with {args.workers} workers ...')
    n_points = []
    with Pool(args.workers) as pool:
        for i, (_, n) in enumerate(pool.imap_unordered(merge_frame, jobs, chunksize=4)):
            if n is not None:
                n_points.append(n)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f'  {i + 1}/{len(jobs)}')
    if n_points:
        print(f'points per merged frame: mean {np.mean(n_points):.0f}, min {min(n_points)}, max {max(n_points)}')
    print(f'done -> {out_root}')


if __name__ == '__main__':
    main()
