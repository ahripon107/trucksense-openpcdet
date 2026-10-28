"""
Checks of the Phase 2 model (RadarCameraFusionCenterPoint, lidar + radar + 4 cameras) on the GPU:
    cd tools && python tests/test_camera_fusion.py
"""
import copy
import os
import sys
from pathlib import Path

import torch

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TOOLS_DIR.parent), str(TOOLS_DIR)]

from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

CFG_B = 'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_man_dataset.yaml'
CFG_A = 'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_warmup_man_dataset.yaml'
LIDAR_CKPT = TOOLS_DIR.parent / 'output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_man_dataset/v3_150m/ckpt/checkpoint_epoch_4.pth'
LOGGER = common_utils.create_logger()


def load_cfg(cfg_file):
    os.chdir(TOOLS_DIR)
    config = copy.deepcopy(cfg)
    cfg_from_yaml_file(cfg_file, config)
    return config


def build_dataset(config, training):
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    data_cfg = copy.deepcopy(config.DATA_CONFIG)
    data_cfg.BALANCED_RESAMPLING = False
    return NuScenesDataset(dataset_cfg=data_cfg, class_names=config.CLASS_NAMES, training=training,
                           root_path=None, logger=LOGGER)


def build_model(config, dataset):
    from pcdet.models import build_network
    model = build_network(config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(str(LIDAR_CKPT), logger=LOGGER, to_cpu=False)
    return model.cuda()


def get_batch(dataset, indices):
    from pcdet.models import load_data_to_gpu
    batch = dataset.collate_batch([dataset[i] for i in indices])
    load_data_to_gpu(batch)
    return batch


def test_depth_bins_and_loss():
    """LID edges span [MIN, MAX] with growing widths; a one-hot logit on the target bin gives a ~0 loss."""
    from pcdet.models.view_transforms.lss import LSSTransform
    bins = EasyDict(MODE='LID', MIN=1.0, MAX=153.6, NUM=120)
    edges = LSSTransform.get_depth_bin_edges(bins)
    widths = edges[1:] - edges[:-1]
    assert abs(edges[0] - 1.0) < 1e-5 and abs(edges[-1] - 153.6) < 1e-3 and (widths[1:] > widths[:-1]).all()

    vt_cfg = load_cfg(CFG_B).MODEL.VTRANSFORM
    vt = LSSTransform(vt_cfg).train()
    fH, fW = vt_cfg.FEATURE_SIZE
    # the frustum sits at the feature cell centers, at the depth bin centers
    assert torch.allclose(vt.frustum[0, 0, :2, 0], torch.tensor([4.0, 12.0]))
    assert torch.allclose(vt.frustum[:, 0, 0, 2], (edges[:-1] + edges[1:]) / 2)

    target = torch.zeros(1, 2, fH, fW)
    target[0, 0, 3, 5], target[0, 1, 10, 7], target[0, 1, 0, 0] = 2.5, 80.0, 200.0  # the last is out of range
    logits = torch.full((2, vt.D, fH, fW), -30.0)
    for cam, y, x, d in [(0, 3, 5, 2.5), (1, 10, 7, 80.0)]:
        logits[cam, torch.bucketize(torch.tensor(d), edges[1:-1], right=True), y, x] = 30.0
    vt.forward_ret_dict = {'depth_logits': logits, 'depth_target': target}
    loss, tb_dict = vt.get_loss()
    assert tb_dict['depth/num_cells'] == 2 and tb_dict['depth/bin_acc'] == 1.0 and loss.item() < 1e-3, tb_dict


def test_stage_a_freezes_and_starts_as_the_lidar_model():
    """Stage A from the lidar checkpoint: Swin weights loaded, frozen weights / BN statistics unchanged by a train
    step, the trainable modules get gradients, and in eval the fused BEV is exactly the lidar BEV."""
    config = load_cfg(CFG_A)
    train_set = build_dataset(config, training=True)
    model = build_model(config, train_set)
    model.train()
    assert all(not m.training for name in model.frozen_module_names for m in getattr(model, name).modules())
    frozen = {k: v.clone() for k, v in model.state_dict().items() if k.split('.')[0] in model.frozen_module_names}

    batch = get_batch(train_set, range(2))
    with torch.autocast('cuda'):
        ret_dict, tb_dict, _ = model(batch)
    ret_dict['loss'].backward()
    assert 'loss_rpn' not in tb_dict and {'depth/loss', 'aux_camera/loss', 'aux_radar/loss', 'aux_lidar/loss'} <= set(tb_dict)
    for name in ['image_backbone', 'neck', 'vtransform', 'camera_aux_head', 'radar_backbone_2d', 'lidar_aux_head']:
        assert all(p.grad is not None for p in getattr(model, name).parameters() if p.requires_grad), name
    assert all(p.grad is None for name in model.frozen_module_names for p in getattr(model, name).parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())

    val_set = build_dataset(config, training=False)
    model.eval()
    with torch.no_grad():
        batch = get_batch(val_set, range(2))
        pred_dicts, _ = model(batch)
    assert torch.equal(batch['spatial_features'], batch['lidar_spatial_features'])
    assert batch['camera_spatial_features'].shape == batch['spatial_features'].shape[:1] + (80, 256, 256)
    assert all(len(p['pred_boxes']) > 0 for p in pred_dicts)


def test_stage_b_train_step_with_modality_dropout():
    config = load_cfg(CFG_B)
    config.MODEL.MODALITY_DROPOUT = {'RADAR': 1.0, 'CAMERA': 1.0}
    train_set = build_dataset(config, training=True)
    model = build_model(config, train_set).train()
    batch = get_batch(train_set, range(2))
    with torch.autocast('cuda'):
        ret_dict, tb_dict, _ = model(batch)
    ret_dict['loss'].backward()
    assert {'loss_rpn', 'depth/loss', 'aux_camera/loss', 'aux_radar/loss', 'aux_lidar/loss'} <= set(tb_dict)
    assert batch['radar_spatial_features_2d'].abs().sum() == 0 and batch['camera_spatial_features'].abs().sum() == 0
    assert torch.isfinite(ret_dict['loss'])


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'these checks run the real model on the GPU'
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASSED {name}')
