"""
Checks of the radar fusion pipeline (RadarFusionCenterPoint, cbgs_voxel0075_res3d_centerpoint_radar_man_dataset.yaml).

    cd tools && python tests/test_radar_fusion.py [test_name ...]

Runs on CPU: the lidar branch (spconv) is replaced by a stub BEV in the model tests. The data tests read the
TruckScenes radar and the converted dataset, so they need the *_withradar infos and the radar GT database.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR.parent))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils, radar_utils  # noqa: E402
from pcdet.utils.radar_utils import RADAR_COL as C  # noqa: E402

CFG_FILE = 'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_man_dataset.yaml'
LIDAR_CKPT = TOOLS_DIR.parent / 'output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_man_dataset/v3_150m/ckpt/checkpoint_epoch_4.pth'

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # CenterHead puts its class mappings on the GPU


def load_cfg(small_range=False):
    """small_range: +-76.8 m instead of +-153.6 m, so a CPU forward/backward fits in a few GB."""
    import os
    os.chdir(TOOLS_DIR)
    config = copy.deepcopy(cfg)
    cfg_from_yaml_file(CFG_FILE, config)
    if small_range:
        config.DATA_CONFIG.POINT_CLOUD_RANGE = [-76.8, -76.8, -4.0, 76.8, 76.8, 4.0]
    return config


def build_dataset(config, training, logger=None):
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    data_cfg = copy.deepcopy(config.DATA_CONFIG)
    data_cfg.BALANCED_RESAMPLING = False
    return NuScenesDataset(dataset_cfg=data_cfg, class_names=config.CLASS_NAMES, training=training,
                           root_path=None, logger=logger or common_utils.create_logger())


def synthetic_scene(num_points=400, seed=0):
    """Radar points with exact geometry: 3 sensors, 2 moving boxes, v_r = v_box . ray (no noise)."""
    rng = np.random.default_rng(seed)
    sensors = np.array([[5.0, 1.4, 2.0], [5.0, -1.4, 2.0], [-3.0, 1.2, 1.0]])
    boxes = np.array([[20.0, 5.0, 0.5, 4.5, 2.0, 1.6, 0.3, 8.0, 3.0],
                      [-15.0, -8.0, 0.8, 10.0, 2.5, 3.5, -1.2, -5.0, 12.0]], dtype=np.float32)
    box_of = rng.integers(0, 2, num_points)
    pos = boxes[box_of, :3] + rng.uniform(-1, 1, (num_points, 3)) * boxes[box_of, 3:6] / 2
    sid = rng.integers(0, 3, num_points)
    rel = pos - sensors[sid]
    rng_ = np.linalg.norm(rel, axis=1)
    ray = rel / rng_[:, None]
    v_r = (ray[:, :2] * boxes[box_of, 7:9]).sum(1)
    pts = np.zeros((num_points, len(radar_utils.RADAR_FEATURES)), dtype=np.float32)
    pts[:, 0:3] = pos
    pts[:, C['rcs']] = rng.normal(0, 10, num_points)
    pts[:, C['vx_comp']], pts[:, C['vy_comp']], pts[:, C['v_r']] = v_r * ray[:, 0], v_r * ray[:, 1], v_r
    pts[:, [C['ray_x'], C['ray_y'], C['ray_z']]] = ray
    pts[:, C['range']] = rng_
    pts[:, C['sensor_id']] = sid
    return pts, boxes, box_of


def check_radar_geometry(pts, boxes, box_of, atol=1e-3):
    ray = pts[:, [C['ray_x'], C['ray_y'], C['ray_z']]]
    assert np.allclose(np.linalg.norm(ray, axis=1), 1, atol=atol), 'rays must stay unit vectors'
    assert np.allclose(pts[:, C['vx_comp']], pts[:, C['v_r']] * ray[:, 0], atol=atol)
    assert np.allclose(pts[:, C['vy_comp']], pts[:, C['v_r']] * ray[:, 1], atol=atol)
    # the measured radial velocity still matches the (transformed) box velocity along the (transformed) ray
    assert np.allclose(pts[:, C['v_r']], (ray[:, :2] * boxes[box_of, 7:9]).sum(1), atol=atol), 'Doppler mismatch'
    # all points of one sensor still see it at one place: position - range * ray
    sensor_pos = pts[:, 0:3] - pts[:, C['range']:C['range'] + 1] * ray
    for s in np.unique(pts[:, C['sensor_id']]):
        cur = sensor_pos[pts[:, C['sensor_id']] == s]
        assert np.allclose(cur, cur[0], atol=atol * 10), 'implied sensor positions diverged'


def test_global_augmentations_keep_radar_consistent():
    config = load_cfg()
    aug_cfg = copy.deepcopy(config.DATA_CONFIG.DATA_AUGMENTOR)
    aug_cfg.AUG_CONFIG_LIST = [c for c in aug_cfg.AUG_CONFIG_LIST if c.NAME != 'gt_sampling']
    augmentor = DataAugmentor(Path('.'), aug_cfg, config.CLASS_NAMES, logger=common_utils.create_logger())
    for seed in range(20):
        np.random.seed(seed)
        pts, boxes, box_of = synthetic_scene(seed=seed)
        data_dict = {'points': pts[:, :5].copy(), 'radar_points': pts.copy(), 'gt_boxes': boxes.copy(),
                     'gt_names': np.array(['car', 'truck']), 'gt_boxes_mask': np.array([True, True])}
        out = augmentor.forward(data_dict)
        check_radar_geometry(out['radar_points'], out['gt_boxes'], box_of)
        # the radar points moved exactly like the lidar points (same first 3 columns before)
        assert np.allclose(out['radar_points'][:, :3], out['points'][:, :3], atol=1e-4)
    # each op on its own, including flips that are always on
    for name, fn in [('flip_x', lambda p, b: (radar_utils.flip_along_x(p), flip_boxes(b, 'x'))),
                     ('flip_y', lambda p, b: (radar_utils.flip_along_y(p), flip_boxes(b, 'y')))]:
        pts, boxes, box_of = synthetic_scene(seed=1)
        p, b = fn(pts.copy(), boxes.copy())
        check_radar_geometry(p, b, box_of)


def flip_boxes(boxes, axis):
    from pcdet.datasets.augmentor import augmentor_utils
    b, _ = getattr(augmentor_utils, f'random_flip_along_{axis}')(boxes, np.zeros((1, 3)), enable=True)
    return b


def test_move_object_points():
    pts, boxes, box_of = synthetic_scene(seed=3)
    same = radar_utils.move_object_points(pts.copy(), np.zeros(3), boxes[0, 7:9])
    assert np.allclose(same, pts, atol=1e-4), 'zero offset must not change the points'
    m = box_of == 0
    offset = np.array([7.0, -4.0, 0.2])
    moved = radar_utils.move_object_points(pts[m].copy(), offset, boxes[0, 7:9])
    new_boxes = boxes.copy()
    new_boxes[0, :3] += offset
    check_radar_geometry(moved, new_boxes, np.zeros(m.sum(), dtype=int))


def test_radar_loading_on_real_data():
    """Compensated v_r ~ 0 on the static world, and radar matches the GT box velocities, on val samples."""
    config = load_cfg()
    dataset = build_dataset(config, training=False)
    rng = np.random.default_rng(0)
    static, resid, t_load = [], [], []
    for idx in rng.choice(len(dataset.infos), 40, replace=False):
        info = dataset.infos[idx]
        t = time.time()
        pts = dataset.get_radar_with_sweeps(idx)
        t_load.append(time.time() - t)
        assert pts.shape[1] == len(radar_utils.RADAR_FEATURES) and np.isfinite(pts).all()
        assert set(np.unique(pts[:, C['sensor_id']]).astype(int)) <= set(range(6))
        # the challenge limit counts sweeps (11 past); dropped scans can make 7 sweeps span ~0.65 s
        for sid in range(6):
            num_sweeps = len(np.unique(pts[pts[:, C['sensor_id']] == sid, C['time_lag']]))
            assert num_sweeps <= config.DATA_CONFIG.RADAR.MAX_SWEEPS, num_sweeps
        if np.linalg.norm(info['radar']['sensor_velocity'][0]) > 5:
            static.append(np.median(np.abs(pts[:, C['v_r']])))
        b = info['gt_boxes'][:, :9].astype(np.float32)
        b = b[np.isfinite(b).all(1) & (np.linalg.norm(b[:, 7:9], axis=1) > 3)]
        key = pts[pts[:, C['time_lag']] < 0.03]
        if len(b) == 0 or len(key) == 0:
            continue
        big = b[:, :7].copy()
        big[:, 3:6] += 1.0
        inb = np.asarray(roiaware_pool3d_utils.points_in_boxes_cpu(key[:, :3], big)).astype(bool)
        for j in range(len(b)):
            p = key[inb[j]]
            if len(p) > 0:
                resid.append(np.median(np.abs(p[:, [C['ray_x'], C['ray_y']]] @ b[j, 7:9] - p[:, C['v_r']])))
    print(f'  load {1000 * np.median(t_load):.0f} ms/sample, static |v_r| {np.median(static):.2f} m/s, '
          f'moving-box Doppler residual {np.median(resid):.2f} m/s ({len(resid)} boxes)')
    assert np.median(static) < 0.5, 'ego-motion compensation is off'
    assert np.median(resid) < 2.5, 'radar velocity does not match the GT box velocities'


def test_training_sample_with_gt_sampling():
    """Full training pipeline: radar keys, pasted objects carry radar, Doppler still matches after augmentation."""
    config = load_cfg()
    dataset = build_dataset(config, training=True)
    resid, num_radar, pasted_with_radar = [], [], 0
    for idx in range(8):
        np.random.seed(idx)
        sample = dataset[idx * 997 % len(dataset)]
        for key in ['radar_points', 'radar_voxels', 'radar_voxel_coords', 'radar_voxel_num_points']:
            assert key in sample, key
        rp = sample['radar_points']
        assert rp.shape[1] == len(radar_utils.RADAR_FEATURES)
        assert sample['radar_voxels'].shape[2] == len(config.DATA_CONFIG.DATA_PROCESSOR[-1].USED_FEATURES)
        pcr = dataset.point_cloud_range
        assert (rp[:, 0] >= pcr[0]).all() and (rp[:, 0] <= pcr[3]).all()
        num_radar.append(len(rp))
        b = sample['gt_boxes']
        moving = np.linalg.norm(b[:, 7:9], axis=1) > 3
        if not moving.any():
            continue
        big = b[moving, :7].copy()
        big[:, 3:6] += 1.0
        inb = np.asarray(roiaware_pool3d_utils.points_in_boxes_cpu(rp[:, :3], big)).astype(bool)
        for j, box in enumerate(b[moving]):
            p = rp[inb[j] & (rp[:, C['time_lag']] < 0.03)]
            if len(p) > 0:
                resid.append(np.median(np.abs(p[:, [C['ray_x'], C['ray_y']]] @ box[7:9] - p[:, C['v_r']])))
    sampler = dataset.data_augmentor.data_augmentor_queue[0]
    infos = [x for v in sampler.db_infos.values() for x in v]
    pasted_with_radar = np.mean([x['num_radar_points_in_gt'] > 0 for x in infos])
    print(f'  radar points/sample {np.median(num_radar):.0f}, db objects with radar {100 * pasted_with_radar:.0f}%, '
          f'moving-box Doppler residual after augmentation {np.median(resid):.2f} m/s ({len(resid)} boxes)')
    assert np.median(resid) < 2.5


def test_collate_batch():
    from pcdet.datasets.dataset import DatasetTemplate
    s = lambda n: {'radar_points': np.ones((n, 13), np.float32), 'radar_voxels': np.ones((n, 10, 8), np.float32),
                   'radar_voxel_coords': np.ones((n, 3), np.int32), 'radar_voxel_num_points': np.ones(n, np.int32)}
    batch = DatasetTemplate.collate_batch([s(3), s(0), s(5)])
    assert batch['radar_points'].shape == (8, 14) and batch['radar_voxel_coords'].shape == (8, 4)
    assert (batch['radar_points'][:3, 0] == 0).all() and (batch['radar_points'][3:, 0] == 2).all()
    assert batch['radar_voxels'].shape == (8, 10, 8) and batch['radar_voxel_num_points'].shape == (8,)


class StubLidarBranch(torch.nn.Module):
    """Stands in for VFE + BACKBONE_3D + MAP_TO_BEV (spconv needs a GPU): a learnable 256-channel BEV."""
    def __init__(self, channels, ny, nx):
        super().__init__()
        self.bev = torch.nn.Parameter(torch.rand(1, channels, ny, nx))

    def forward(self, batch_dict):
        batch_dict['spatial_features'] = torch.relu(self.bev).expand(batch_dict['batch_size'], -1, -1, -1).clone()
        return batch_dict


def build_cpu_model(config, dataset):
    from pcdet.models import build_network
    model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
    ny, nx = dataset.grid_size[1] // 8, dataset.grid_size[0] // 8
    stub = StubLidarBranch(256, ny, nx)
    identity = lambda batch_dict: batch_dict
    model.vfe.forward = stub.forward
    model.backbone_3d.forward = identity
    model.map_to_bev_module.forward = identity
    model.add_module('stub', stub)
    return model


def to_torch(batch):
    for key, val in batch.items():
        if isinstance(val, np.ndarray) and key not in ['frame_id', 'metadata']:
            batch[key] = torch.from_numpy(val).float()
    return batch


def test_model_train_and_eval_on_cpu():
    torch.manual_seed(0)
    config = load_cfg(small_range=True)
    dataset = build_dataset(config, training=True)
    batch = to_torch(dataset.collate_batch([dataset[0], dataset[123]]))
    assert batch['radar_points'][:, 0].unique().tolist() == [0, 1]
    model = build_cpu_model(config, dataset)
    model.train()

    # aux heads must not relabel the gt_boxes the main head sees
    labels_before = batch['gt_boxes'][..., -1].clone()
    seen = {}
    orig_dense_forward = model.dense_head.forward
    def dense_forward(batch_dict):
        seen['labels'] = batch_dict['gt_boxes'][..., -1].clone()
        seen['fused'] = batch_dict['spatial_features'].detach().clone()
        seen['lidar'] = batch_dict['lidar_spatial_features'].detach().clone()
        return orig_dense_forward(batch_dict)
    model.dense_head.forward = dense_forward

    ret_dict, tb_dict, _ = model(dict(batch))
    loss = ret_dict['loss']
    assert torch.isfinite(loss)
    assert torch.equal(seen['labels'], labels_before), 'an aux head changed the gt labels'
    assert torch.equal(seen['fused'], seen['lidar']), 'the fuser must start as identity on the lidar BEV'
    for key in ['loss_rpn', 'aux_lidar/loss', 'aux_lidar/hm_loss_head_0', 'aux_lidar/loc_loss_head_0',
                'aux_radar/loss', 'aux_radar/hm_loss_head_0', 'aux_radar/doppler_loss', 'aux_radar/doppler_num_boxes']:
        assert key in tb_dict, key
    assert tb_dict['aux_radar/doppler_num_boxes'] > 0
    print('  ' + ', '.join(f'{k}={v:.3f}' for k, v in tb_dict.items() if not k.split('/')[-1].startswith(('hm_', 'loc_'))))
    loss.backward()
    for name in ['radar_vfe', 'radar_backbone_2d', 'radar_aux_head', 'lidar_aux_head', 'fuser']:
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert len(grads) > 0 and sum(g.abs().sum() for g in grads) > 0, f'no gradient reaches {name}'

    # inference: aux heads are skipped and boxes carry velocity
    model.eval()
    def fail(*args, **kwargs):
        raise AssertionError('aux head called at inference')
    model.lidar_aux_head.forward = fail
    model.radar_aux_head.forward = fail
    model.dense_head.forward = orig_dense_forward
    del batch, ret_dict, loss, seen
    eval_dataset = build_dataset(config, training=False)
    model.dataset = eval_dataset
    eval_batch = to_torch(eval_dataset.collate_batch([eval_dataset[0]]))
    eval_batch.pop('gt_boxes')  # the recall record uses the GPU-only 3D IoU
    import pcdet.models.model_utils.model_nms_utils as nms_utils
    orig_nms = nms_utils.class_agnostic_nms
    nms_utils.class_agnostic_nms = lambda box_scores, box_preds, nms_config, score_thresh=None: (
        torch.argsort(box_scores, descending=True)[:nms_config.NMS_POST_MAXSIZE], box_scores)  # nms_gpu is GPU-only
    try:
        with torch.no_grad():
            pred_dicts, _ = model(dict(eval_batch))
    finally:
        nms_utils.class_agnostic_nms = orig_nms
    assert pred_dicts[0]['pred_boxes'].shape[-1] == 9, 'predicted boxes must carry vx, vy'


def test_doppler_loss_is_zero_for_the_true_velocity():
    from easydict import EasyDict
    from pcdet.models.dense_heads.aux_center_head import RadarAuxCenterHead
    config = load_cfg()
    head_cfg = copy.deepcopy(config.MODEL.AUX_HEADS.RADAR)
    head_cfg.DOPPLER_LOSS.BOX_EXTRA_WIDTH = [0.5, 0.5, 0.5]
    pcr = np.array(config.DATA_CONFIG.POINT_CLOUD_RANGE)
    head = RadarAuxCenterHead(head_cfg, 16, len(config.CLASS_NAMES), config.CLASS_NAMES, np.array([2048, 2048, 40]),
                              pcr, [0.15, 0.15, 0.2])
    pts, boxes, box_of = synthetic_scene(seed=5)
    labels = np.array([1, 2], dtype=np.float32)  # car, truck
    gt = torch.from_numpy(np.concatenate([boxes, labels[:, None]], 1))[None]
    radar = torch.from_numpy(np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1))
    pred_dicts = [{'vel': torch.zeros(1, 2, 256, 256)} for _ in head.class_names_each_head]
    for box, label in zip(boxes, labels):
        h = int(head.head_of_label[int(label)])
        cx = int((box[0] - pcr[0]) / 0.15 / 8)
        cy = int((box[1] - pcr[1]) / 0.15 / 8)
        pred_dicts[h]['vel'][0, :, cy, cx] = torch.from_numpy(box[7:9])
    head.forward_ret_dict = {'pred_dicts': pred_dicts, 'gt_boxes': gt, 'radar_points': radar}
    loss, num_boxes = head.get_doppler_loss()
    assert num_boxes == 2 and loss.item() < 1e-6, (loss.item(), num_boxes)
    pred_dicts[0]['vel'].zero_()
    assert head.get_doppler_loss()[0].item() > 0.1


def test_pretrained_lidar_checkpoint_loads():
    if not LIDAR_CKPT.exists():
        print(f'  skipped: {LIDAR_CKPT} not found')
        return
    from pcdet.models import build_network
    config = load_cfg()
    dataset = build_dataset(config, training=False)
    model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
    ckpt = torch.load(LIDAR_CKPT, map_location='cpu', weights_only=False)['model_state']
    state = model.state_dict()
    lidar_keys = [k for k in ckpt if not k.startswith('global_step')]
    missing = [k for k in lidar_keys if k not in state or state[k].shape != ckpt[k].shape]
    assert not missing, f'checkpoint weights without a matching model weight: {missing[:5]}'
    model.load_params_from_file(str(LIDAR_CKPT), logger=common_utils.create_logger(), to_cpu=True)
    loaded = model.state_dict()
    assert all(torch.equal(loaded[k].cpu(), ckpt[k].cpu()) for k in lidar_keys)
    new = sorted({k.split('.')[0] for k in state if k not in ckpt})
    print(f'  loaded {len(lidar_keys)} lidar weights; new modules: {new}')
    # radar_map_to_bev (the scatter) has no weights
    assert set(new) == {'radar_vfe', 'radar_backbone_2d', 'lidar_aux_head', 'radar_aux_head', 'fuser'}, new


def test_result_json_has_velocity_and_radar_meta():
    from pcdet.datasets.nuscenes import nuscenes_utils
    from pyquaternion import Quaternion
    det = {'boxes_lidar': np.array([[10, 2, 0, 4, 2, 1.5, 0.3, 5.0, -2.0]]), 'score': np.array([0.9]),
           'pred_labels': np.array([1])}
    boxes = nuscenes_utils.boxes_lidar_to_nusenes(det)
    assert np.allclose(boxes[0].velocity[:2], [5.0, -2.0])

    class FakeNusc:  # lidar frame = ego frame rotated by 90 deg in the world
        def get(self, table, token):
            return {'sample': {'data': {'LIDAR_TOP': 'sd'}},
                    'sample_data': {'calibrated_sensor_token': 'cs', 'ego_pose_token': 'ep'},
                    'calibrated_sensor': {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 1.84], 'sensor_token': 's'},
                    'sensor': {},
                    'ego_pose': {'rotation': list(Quaternion(axis=[0, 0, 1], angle=np.pi / 2)), 'translation': [100, 200, 0]},
                    }[table]

        def get_sample_data_path(self, token):
            return ''
    glob = nuscenes_utils.lidar_nusc_box_to_global(FakeNusc(), boxes, 'tok')
    assert np.allclose(glob[0].velocity[:2], [2.0, 5.0], atol=1e-6), glob[0].velocity
    config = load_cfg()
    assert config.DATA_CONFIG.get('RADAR', None) is not None  # evaluation() sets meta.use_radar from this


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
