"""
Checks of the camera data pipeline (dataset_configs/man_nuscenes_radar_camera_dataset.yaml).

    cd tools && python tests/test_camera_data.py [test_name ...]

CPU only. Needs the *_withradar_withcam infos and the TruckScenes camera keyframes.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR.parent))

from pcdet.utils import box_utils, common_utils  # noqa: E402

DATA_CFG_FILE = 'cfgs/dataset_configs/man_nuscenes_radar_camera_dataset.yaml'
CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 'pedestrian',
               'motorcycle', 'bicycle', 'traffic_cone', 'barrier', 'animal', 'traffic_sign']
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load_data_cfg(**image_overrides):
    import os
    import yaml
    os.chdir(TOOLS_DIR)
    data_cfg = EasyDict(yaml.safe_load(open(DATA_CFG_FILE)))
    data_cfg.BALANCED_RESAMPLING = False
    data_cfg.CAMERA_CONFIG.IMAGE.update(image_overrides)
    return data_cfg


def build_dataset(data_cfg, training):
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    return NuScenesDataset(dataset_cfg=data_cfg, class_names=CLASS_NAMES, training=training,
                           logger=common_utils.create_logger())


def raw_projection(cam, xyz):
    """Pixel (u, v) and depth of lidar-frame points in the raw image, from the info's calibration only."""
    R, T = cam['sensor2lidar_rotation'], cam['sensor2lidar_translation']
    in_cam = (xyz - T) @ R  # inverse of points @ R.T + T
    uv = in_cam @ np.asarray(cam['camera_intrinsics']).T
    return uv[:, :2] / uv[:, 2:3], in_cam[:, 2]


def augmented_projection(sample, c, xyz_aug):
    """Pixel of augmented lidar-frame points in augmented image c, from the matrices the model receives."""
    lidar_aug = sample['lidar_aug_matrix']
    orig = (xyz_aug - lidar_aug[:3, 3]) @ np.linalg.inv(lidar_aug[:3, :3]).T
    l2i = np.asarray(sample['lidar2image'][c])
    q = orig @ l2i[:3, :3].T + l2i[:3, 3]
    uv = q[:, :2] / q[:, 2:3]
    img_aug = np.asarray(sample['img_aug_matrix'][c])
    return uv @ img_aug[:2, :2].T + img_aug[:2, 3], q[:, 2]


def marker_boxes(info, depth=18.0):
    """One small 'car' box per camera, on the ray through a pixel off the image center, at the given depth."""
    boxes = []
    for k, cam in enumerate(info['cams'].values()):
        K = np.asarray(cam['camera_intrinsics'])
        u, v = 700 + 150 * k, 560
        p_cam = np.linalg.inv(K) @ np.array([u, v, 1.0]) * depth
        p = cam['sensor2lidar_rotation'] @ p_cam + cam['sensor2lidar_translation']
        boxes.append([*p, 0.5, 0.5, 0.5, 0.3, 1.0, 2.0])
    return np.array(boxes, dtype=np.float32)


def patch_with_markers(dataset, idx, radius=7):
    """Replace sample idx's GT with marker boxes and its images with black images with a white dot at each marker."""
    from pcdet.datasets.nuscenes import nuscenes_dataset as nd
    info = copy.deepcopy(dataset.infos[idx])
    boxes = marker_boxes(info)
    info['gt_boxes'], info['gt_names'] = boxes, np.array(['car'] * len(boxes))
    info['num_lidar_pts'] = np.full(len(boxes), 100)
    dataset.infos[idx] = info
    images = {}
    for cam in info['cams'].values():
        w, h = cam['image_size']
        im = Image.new('RGB', (w, h))
        draw = ImageDraw.Draw(im)
        uv, depth = raw_projection(cam, boxes[:, :3])
        for (u, v), d in zip(np.round(uv), depth):  # PIL truncates float coordinates: 559.99999 would draw at 559
            if d > 0 and 0 <= u < w and 0 <= v < h:
                draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 255, 255))
        images[cam['data_path']] = im
    orig_open = nd.Image.open
    nd.Image.open = lambda path, *a, **k: images[str(Path(path).relative_to(dataset.camera_data_path))].copy()
    return boxes, orig_open


def bright_blobs(img_tensor):
    """Centroids (u, v) of the white dots in a normalized (3, H, W) image."""
    from scipy import ndimage
    img = img_tensor.numpy().transpose(1, 2, 0) * STD + MEAN
    labels, n = ndimage.label(img.mean(axis=2) > 0.5)
    centers = np.array(ndimage.center_of_mass(np.ones(labels.shape), labels, range(1, n + 1)))[:, ::-1].reshape(-1, 2)
    return centers + 0.5  # pixel index -> the pixel-edge convention of img_aug_matrix


def lss_geometry(sample, c, uv, depth):
    """The view transform's own get_geometry: augmented image pixel + depth -> augmented lidar frame."""
    from pcdet.models.view_transforms.depth_lss import DepthLSSTransform
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    frustum = torch.tensor([[uv[0], uv[1], depth]], dtype=torch.float32).view(1, 1, 1, 3)
    obj = type('Geom', (), {'frustum': frustum})()
    cam2lidar = t(sample['camera2lidar'][c])
    img_aug = t(sample['img_aug_matrix'][c])
    lidar_aug = t(sample['lidar_aug_matrix'])
    pts = DepthLSSTransform.get_geometry(
        obj, cam2lidar[None, None, :3, :3], cam2lidar[None, None, :3, 3], t(sample['camera_intrinsics'][c])[None, None, :3, :3],
        img_aug[None, None, :3, :3], img_aug[None, None, :3, 3],
        extra_rots=lidar_aug[None, :3, :3], extra_trans=lidar_aug[None, :3, 3])
    return pts.view(3).numpy()


def test_markers_land_where_the_matrices_say():
    """Global 3D aug + image resize/crop/flip/rotation: dots, matrices and LSS geometry must agree."""
    data_cfg = load_data_cfg()
    dataset = build_dataset(data_cfg, training=True)
    idx = 1500
    boxes, orig_open = patch_with_markers(dataset, idx)
    errs, offsets, geo_errs, n_flip = [], [], [], 0
    try:
        for seed in range(12):
            np.random.seed(seed)
            sample = dataset[idx]
            # the box centers went through the actual 3D augmentation; lidar_aug_matrix must describe it
            aug = sample['lidar_aug_matrix']
            assert np.allclose(boxes[:, :3] @ aug[:3, :3].T + aug[:3, 3], sample['gt_boxes'][:, :3], atol=2e-3)
            n_flip += sum(info[2] for info in sample['img_process_infos'])
            for c, img in enumerate(sample['camera_imgs']):
                blobs = bright_blobs(img)
                uv, depth = augmented_projection(sample, c, sample['gt_boxes'][:, :3])
                h, w = img.shape[1:]
                visible = (depth > 0) & (uv[:, 0] > 8) & (uv[:, 0] < w - 8) & (uv[:, 1] > 8) & (uv[:, 1] < h - 8)
                assert len(blobs) == visible.sum(), (seed, c, len(blobs), visible.sum())
                for k in np.nonzero(visible)[0]:
                    nearest = blobs[np.argmin(np.linalg.norm(blobs - uv[k], axis=1))]
                    sign = -1 if sample['img_process_infos'][c][2] else 1  # flips mirror the u offset
                    offsets.append(((nearest - uv[k]) * [sign, 1]))
                    errs.append(np.linalg.norm(nearest - uv[k]))
                    geo = lss_geometry(sample, c, uv[k], depth[k])
                    geo_errs.append(np.linalg.norm(geo - sample['gt_boxes'][k, :3]))
    finally:
        from pcdet.datasets.nuscenes import nuscenes_dataset as nd
        nd.Image.open = orig_open
    bias = np.mean(offsets, axis=0)
    print(f'  {len(errs)} dots, {n_flip} flipped images: pixel error median {np.median(errs):.2f} max {np.max(errs):.2f} px, '
          f'bias ({bias[0]:+.2f}, {bias[1]:+.2f}) px; LSS geometry error max {1000 * np.max(geo_errs):.1f} mm')
    assert n_flip > 0 and len(errs) >= 24
    # dots are a few pixels wide and rotations resample them (nearest), so single dots move by < 1 px; no bias
    assert np.max(errs) < 1.0 and np.abs(bias).max() < 0.2, 'image augmentation and img_aug_matrix disagree'
    assert np.max(geo_errs) < 0.02, 'view-transform geometry does not invert the projection'


def test_real_sample_shapes_and_timing():
    for training in [True, False]:
        data_cfg = load_data_cfg()
        dataset = build_dataset(data_cfg, training=training)
        t = time.time()
        sample = dataset[7]
        dt = time.time() - t
        h, w = data_cfg.CAMERA_CONFIG.IMAGE.FINAL_DIM
        stride = data_cfg.DATA_PROCESSOR[-2].DOWNSAMPLE
        assert len(sample['camera_imgs']) == 4 and tuple(sample['camera_imgs'][0].shape) == (3, h, w)
        assert sample['camera_depth_targets'].shape == (4, h // stride, w // stride)
        for key in ['radar_points', 'radar_voxels', 'voxels', 'lidar2image', 'camera2lidar', 'img_aug_matrix']:
            assert key in sample, key
        filled = (sample['camera_depth_targets'] > 0).mean(axis=(1, 2))
        print(f'  training={training}: {dt * 1000:.0f} ms/sample, depth-target cells filled per camera {np.round(filled, 2)}')
        assert (filled > 0.05).all()
        batch = dataset.collate_batch([sample, sample])
        assert batch['camera_imgs'].shape == (2, 4, 3, h, w) and batch['camera_depth_targets'].shape == (2, 4, h // stride, w // stride)
        assert batch['img_aug_matrix'].shape == (2, 4, 4, 4) and batch['lidar2image'].shape == (2, 4, 4, 4)


def test_configurable_image_size():
    data_cfg = load_data_cfg(FINAL_DIM=[384, 832], RESIZE_LIM_TRAIN=[0.42, 0.48], RESIZE_LIM_TEST=[0.42, 0.42])
    dataset = build_dataset(data_cfg, training=True)
    sample = dataset[3]
    assert tuple(sample['camera_imgs'][0].shape) == (3, 384, 832)
    assert sample['camera_depth_targets'].shape == (4, 48, 104)


def test_depth_targets_match_the_point_cloud():
    """Lift each filled depth cell back with the LSS geometry: it must land on a (current-sweep) lidar point."""
    from scipy.spatial import cKDTree
    data_cfg = load_data_cfg()
    dataset = build_dataset(data_cfg, training=True)
    stride = data_cfg.DATA_PROCESSOR[-2].DOWNSAMPLE
    rel = []
    for seed in range(3):
        np.random.seed(seed)
        sample = dataset[100 + seed]
        pts = sample['points']
        tree = cKDTree(pts[pts[:, -1] <= 1e-4, :3])
        rng = np.random.default_rng(seed)
        for c in range(4):
            target = sample['camera_depth_targets'][c]
            cells = np.argwhere(target > 0)
            for i, j in cells[rng.choice(len(cells), 40, replace=False)]:
                d = target[i, j]
                # the nearest point lies somewhere in the cell: search within the cell's footprint at that depth
                geo = lss_geometry(sample, c, ((j + 0.5) * stride, (i + 0.5) * stride), d)
                cell_size = d * stride / 640 / 0.485 * 1.5
                rel.append(tree.query(geo)[0] / cell_size)
    print(f'  nearest-point distance / cell footprint: median {np.median(rel):.2f}, 95% {np.percentile(rel, 95):.2f}')
    assert np.median(rel) < 0.5 and np.percentile(rel, 95) < 1.0


def test_gt_box_overlay_after_augmentation(out_path=str(TOOLS_DIR.parent / 'output' / 'camera_aug_overlay.jpg')):
    """Writes the augmented images with the augmented GT boxes drawn through the matrices, for a visual check."""
    data_cfg = load_data_cfg()
    dataset = build_dataset(data_cfg, training=True)
    np.random.seed(4)
    sample = dataset[1500]
    corners = box_utils.boxes_to_corners_3d(sample['gt_boxes'][:, :7])
    tiles = []
    for c, img in enumerate(sample['camera_imgs']):
        im = Image.fromarray(((img.numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        for bc in corners:
            uv, depth = augmented_projection(sample, c, bc)
            if (depth < 1).any() or depth.mean() > 80:
                continue
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=(0, 255, 0), width=2)
        tiles.append(im)
    w, h = tiles[0].size
    grid = Image.new('RGB', (2 * w, 2 * h))
    for k, im in enumerate(tiles):
        grid.paste(im, ((k % 2) * w, (k // 2) * h))
    grid.save(out_path)
    print(f'  wrote {out_path} (flips {[i[2] for i in sample["img_process_infos"]]}, '
          f'rotations {[round(i[3], 1) for i in sample["img_process_infos"]]})')


if __name__ == '__main__':
    names = sys.argv[1:] or [n for n in list(globals()) if n.startswith('test_')]
    failed = []
    for name in names:
        t = time.time()
        print(f'{name} ...')
        try:
            globals()[name]()
            print(f'  ok ({time.time() - t:.1f} s)')
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(name)
    print('FAILED: ' + ', '.join(failed) if failed else 'all passed')
    sys.exit(1 if failed else 0)
