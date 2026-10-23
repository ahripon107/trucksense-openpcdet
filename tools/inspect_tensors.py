"""
Run one sample through a detector module by module and inspect what each stage writes into batch_dict.

For every stage (vfe, backbone_3d, map_to_bev_module, image_backbone, neck, vtransform, backbone_2d, dense_head, ...)
it prints the shape / dtype / statistics of each new or updated entry and saves figures:
    - camera inputs, image feature maps (per camera), LSS depth (expected depth per camera)
    - sparse 3D voxels / features (BEV occupancy), dense BEV feature maps with GT boxes overlaid
    - CenterHead heatmaps per head, final predictions vs GT in BEV

Usage (from tools/):
    python inspect_tensors.py --cfg_file cfgs/nuscenes_models/bevdet_tiny_swint_256x704.yaml \
        --ckpt ../output/.../checkpoint_epoch_24.pth --index 0
Without --ckpt the model is randomly initialised (pretrained image backbone weights still load),
which is enough to check shapes and data flow.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


def parse_config():
    parser = argparse.ArgumentParser(description='inspect intermediate tensors')
    parser.add_argument('--cfg_file', type=str, required=True, help='model config')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load (default: random init)')
    parser.add_argument('--index', type=int, default=0, help='sample index in the split')
    parser.add_argument('--split', choices=['val', 'train'], default='val',
                        help='val = no augmentation; train = with training augmentation')
    parser.add_argument('--out_dir', type=str, default=None, help='where to save figures')
    parser.add_argument('--save_tensors', action='store_true', help='also save every stage output as .pt')
    parser.add_argument('--no_plot', action='store_true', help='only print the summary')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    if args.out_dir is None:
        args.out_dir = f'../output/inspect/{Path(args.cfg_file).stem}/{args.split}_{args.index}'
    return args


# ----------------------------------------------------------------------------- summary

def is_sparse(x):
    return spconv is not None and isinstance(x, spconv.SparseConvTensor)


def describe(x):
    if torch.is_tensor(x):
        s = f'{str(tuple(x.shape)):<26} {str(x.dtype).replace("torch.", ""):<8}'
        if x.numel() > 0 and (x.is_floating_point() or x.dtype in (torch.int32, torch.int64)):
            xf = x.detach().float()
            finite = torch.isfinite(xf)
            s += (f' min {xf[finite].min().item():10.4g}  max {xf[finite].max().item():10.4g}'
                  f'  mean {xf[finite].mean().item():10.4g}  std {xf[finite].std().item() if finite.sum() > 1 else 0:10.4g}'
                  f'  nonzero {100 * (xf != 0).float().mean().item():5.1f}%')
            if not finite.all():
                s += f'  !! {(~finite).sum().item()} non-finite'
        return s
    if is_sparse(x):
        dense_cells = int(np.prod(x.spatial_shape)) * x.batch_size
        return (f'SparseConvTensor features {tuple(x.features.shape)} spatial_shape {list(x.spatial_shape)}'
                f'  active {x.indices.shape[0]} ({100 * x.indices.shape[0] / dense_cells:.2f}% of grid)  '
                f'| features: {describe(x.features)}')
    if isinstance(x, np.ndarray):
        return f'{str(x.shape):<26} np.{x.dtype}'
    return f'{type(x).__name__}'


def print_entry(key, val, indent='    '):
    if isinstance(val, dict):
        print(f'{indent}{key}: dict')
        for k, v in val.items():
            print_entry(k, v, indent + '    ')
    elif isinstance(val, (list, tuple)) and len(val) > 0 and (torch.is_tensor(val[0]) or is_sparse(val[0]) or isinstance(val[0], dict)):
        print(f'{indent}{key}: {type(val).__name__}[{len(val)}]')
        for i, v in enumerate(val[:8]):
            print_entry(f'[{i}]', v, indent + '    ')
    else:
        print(f'{indent}{key:<28} {describe(val)}')


# ----------------------------------------------------------------------------- plotting helpers

def box_corners_bev(boxes):
    """boxes (N, >=7) [x, y, z, dx, dy, dz, heading] -> (N, 4, 2)"""
    x, y, dx, dy, h = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    template = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) / 2
    corners = template[None] * np.stack([dx, dy], -1)[:, None]
    cos, sin = np.cos(h)[:, None], np.sin(h)[:, None]
    rx = corners[..., 0] * cos - corners[..., 1] * sin + x[:, None]
    ry = corners[..., 0] * sin + corners[..., 1] * cos + y[:, None]
    return np.stack([rx, ry], -1)


def draw_boxes(ax, boxes, pc_range, shape_hw, color, label=None):
    """Draw boxes on a BEV map whose rows are y (origin lower) and columns are x."""
    if boxes is None or len(boxes) == 0:
        return
    H, W = shape_hw
    sx = W / (pc_range[3] - pc_range[0])
    sy = H / (pc_range[4] - pc_range[1])
    for i, c in enumerate(box_corners_bev(boxes)):
        c = np.vstack([c, c[:1]])
        ax.plot((c[:, 0] - pc_range[0]) * sx - 0.5, (c[:, 1] - pc_range[1]) * sy - 0.5, color=color, lw=0.8,
                label=label if i == 0 else None)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)


def to_bev_image(x):
    """(C, H, W) -> (H, W) mean |activation|"""
    return x.detach().float().abs().mean(0).cpu().numpy()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'        -> {path}')


class Plotter:
    def __init__(self, out_dir, batch_dict, pc_range):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pc_range = pc_range
        self.num_cams = batch_dict['camera_imgs'].shape[1] if 'camera_imgs' in batch_dict else None
        gt = batch_dict.get('gt_boxes', None)
        self.gt = None if gt is None else gt[0][gt[0].abs().sum(-1) > 0].cpu().numpy()
        self.counter = 0

    def path(self, stage, key):
        self.counter += 1
        return self.out_dir / f'{self.counter:02d}_{stage}_{key.replace("/", "_")}.png'

    def bev(self, stage, key, x, title=None):
        img = to_bev_image(x)
        fig, ax = plt.subplots(figsize=(7, 7))
        nz = img[img > 0]
        vmax = np.percentile(nz, 99) if nz.size > 0 else None  # a few strong cells would wash out the rest
        ax.imshow(img, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
        draw_boxes(ax, self.gt, self.pc_range, img.shape, 'red', 'GT')
        ax.set_title(title or f'{stage}: {key}  {tuple(x.shape)}  (mean |act|, GT in red)', fontsize=9)
        ax.set_xlabel('x cells'); ax.set_ylabel('y cells')
        save(fig, self.path(stage, key))

    def per_camera(self, stage, key, maps, title, cmap='viridis', vmin=None, vmax=None):
        """maps: (N, H, W) numpy or (N, H, W, 3)"""
        n = len(maps)
        cols = 3 if n >= 3 else n
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.2 * rows + 0.6), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis('off')
            if i < n:
                im = ax.imshow(maps[i], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f'cam {i}', fontsize=8)
        if maps[0].ndim == 2:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        fig.suptitle(title, fontsize=9)
        fig.savefig(self.path(stage, key), dpi=120)
        plt.close(fig)
        print(f'        -> {self.out_dir / f"{self.counter:02d}_{stage}_{key}.png"}')

    def image_features(self, stage, key, x):
        """(B*N, C, H, W) image features -> per camera mean |act| for sample 0"""
        maps = x[:self.num_cams].detach().float().abs().mean(1).cpu().numpy()
        self.per_camera(stage, key, maps, f'{stage}: {key}  {tuple(x.shape)}  (mean |act|, sample 0)')

    def sparse(self, stage, key, x):
        idx = x.indices[x.indices[:, 0] == 0]
        feats = x.features[x.indices[:, 0] == 0].detach().float().abs().mean(1)
        shape = list(x.spatial_shape)
        if len(shape) == 3:      # (z, y, x)
            img = torch.zeros(shape[1], shape[2], device=feats.device)
            img.index_put_((idx[:, 2].long(), idx[:, 3].long()), feats, accumulate=True)
        else:                    # (y, x)
            img = torch.zeros(shape[0], shape[1], device=feats.device)
            img.index_put_((idx[:, 1].long(), idx[:, 2].long()), feats, accumulate=True)
        self.bev(stage, key, img[None], f'{stage}: {key} sparse {shape}  (sum over z of mean |feat|, GT in red)')

    def points(self, stage, key, pts):
        p = pts[pts[:, 0] == 0].cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(p[:, 1], p[:, 2], s=0.05, c=p[:, 3], cmap='gray')
        if self.gt is not None:
            for c in box_corners_bev(self.gt):
                c = np.vstack([c, c[:1]]); ax.plot(c[:, 0], c[:, 1], 'r', lw=0.8)
        ax.set_xlim(self.pc_range[0], self.pc_range[3]); ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.set_aspect('equal'); ax.set_title(f'input points {tuple(pts.shape)} (sample 0, GT in red)', fontsize=9)
        save(fig, self.path(stage, key))

    def camera_inputs(self, imgs):
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[:, None, None]
        maps = [((im * std + mean).clamp(0, 1)).permute(1, 2, 0).cpu().numpy() for im in imgs[0]]
        self.per_camera('input', 'camera_imgs', maps, f'input camera_imgs {tuple(imgs.shape)} (sample 0, denormalised)')

    def center_heatmaps(self, stage, pred_dicts, class_names_each_head):
        n = len(pred_dicts)
        fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(4.5 * int(np.ceil(n / 2)), 9), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis('off')
            if i >= n:
                continue
            hm = pred_dicts[i]['hm'][0].detach().float()
            hm = hm if (hm.min() >= 0 and hm.max() <= 1) else hm.sigmoid()
            img = hm.max(0)[0].cpu().numpy()
            ax.imshow(img, origin='lower', cmap='magma', vmin=0, vmax=1)
            if self.gt is not None:
                names = class_names_each_head[i]
                mask = np.isin(self.gt[:, -1].astype(int) - 1, [self.class_names.index(c) for c in names])
                draw_boxes(ax, self.gt[mask], self.pc_range, img.shape, 'cyan')
            ax.set_title(f'head {i} {class_names_each_head[i]}  max {img.max():.2f}', fontsize=8)
        fig.suptitle(f'{stage}: CenterHead heatmaps (max over classes, sigmoid; GT of that head in cyan)', fontsize=9)
        fig.savefig(self.path(stage, 'heatmaps'), dpi=120)
        plt.close(fig)
        print(f'        -> {self.out_dir / f"{self.counter:02d}_{stage}_heatmaps.png"}')

    def predictions(self, stage, pred, score_thresh):
        boxes = pred['pred_boxes'].detach().cpu().numpy()
        scores = pred['pred_scores'].detach().cpu().numpy()
        keep = scores >= score_thresh
        fig, ax = plt.subplots(figsize=(7, 7))
        if self.gt is not None:
            for i, c in enumerate(box_corners_bev(self.gt)):
                c = np.vstack([c, c[:1]]); ax.plot(c[:, 0], c[:, 1], 'r', lw=1, label='GT' if i == 0 else None)
        for i, c in enumerate(box_corners_bev(boxes[keep])):
            c = np.vstack([c, c[:1]]); ax.plot(c[:, 0], c[:, 1], 'b', lw=0.8, label='pred' if i == 0 else None)
        ax.set_xlim(self.pc_range[0], self.pc_range[3]); ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.set_aspect('equal'); ax.legend(loc='upper right')
        ax.set_title(f'final predictions (score >= {score_thresh}): {keep.sum()} / {len(scores)}, GT: '
                     f'{0 if self.gt is None else len(self.gt)}', fontsize=9)
        save(fig, self.path(stage, 'predictions'))


# ----------------------------------------------------------------------------- main

def changed_keys(before, after):
    return [k for k, v in after.items() if k not in before or before[k] is not v]


def main():
    args = parse_config()
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                     logger=logger, training=args.split == 'train')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    if args.ckpt is not None:
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    else:
        logger.info('No --ckpt given: using randomly initialised weights')
    model.cuda().eval()

    batch_dict = dataset.collate_batch([dataset[args.index]])
    load_data_to_gpu(batch_dict)
    print(f'\nsample {args.index} ({args.split}), frame {batch_dict["frame_id"][0]}')

    plotter = None if args.no_plot else Plotter(args.out_dir, batch_dict, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    if plotter is not None:
        plotter.class_names = cfg.CLASS_NAMES

    print('\n[input]')
    for k, v in batch_dict.items():
        if k in ('frame_id', 'metadata', 'image_paths', 'img_process_infos', 'ori_shape'):
            continue
        print_entry(k, v)
    if plotter is not None:
        if 'camera_imgs' in batch_dict:
            plotter.camera_inputs(batch_dict['camera_imgs'])
        if 'points' in batch_dict and not cfg.DATA_CONFIG.get('CAMERA_CONFIG', {}).get('USE_CAMERA', False):
            plotter.points('input', 'points', batch_dict['points'])

    # LSS depth isn't kept in batch_dict, so capture it from the depth net
    depth_store = {}
    vtransform = getattr(model, 'vtransform', None)
    if vtransform is not None and plotter is not None:
        vtransform.depthnet.register_forward_hook(lambda m, i, o: depth_store.update(out=o.detach()))

    stage_names = [n for n in model.module_topology if getattr(model, n, None) is not None]
    assert len(stage_names) == len(model.module_list)
    saved = {}
    with torch.no_grad():
        for name, module in zip(stage_names, model.module_list):
            before = dict(batch_dict)
            batch_dict = module(batch_dict)
            keys = changed_keys(before, batch_dict)
            print(f'\n[{name}] {type(module).__name__}  -> {len(keys)} new/updated keys')
            for k in keys:
                print_entry(k, batch_dict[k])
                if args.save_tensors:
                    saved[f'{name}/{k}'] = batch_dict[k]

            if plotter is None:
                continue
            for k in keys:
                v = batch_dict[k]
                if is_sparse(v):
                    plotter.sparse(name, k, v)
                elif k == 'multi_scale_3d_features' or k == 'multi_scale_2d_features':
                    for sk, sv in v.items():
                        if is_sparse(sv):
                            plotter.sparse(name, f'{k}.{sk}', sv)
                        elif torch.is_tensor(sv) and sv.dim() == 4:
                            plotter.bev(name, f'{k}.{sk}', sv[0])
                elif k in ('image_features', 'image_fpn'):
                    for i, f in enumerate(v):
                        plotter.image_features(name, f'{k}[{i}]', f)
                elif torch.is_tensor(v) and v.dim() == 4 and k.startswith('spatial_features'):
                    plotter.bev(name, k, v[0])
            if name == 'vtransform' and 'out' in depth_store:
                d = depth_store['out'][:plotter.num_cams, :vtransform.D].float().softmax(1)
                bins = torch.arange(*vtransform.dbound, device=d.device)[:vtransform.D].view(1, -1, 1, 1)
                plotter.per_camera(name, 'expected_depth', (d * bins).sum(1).cpu().numpy(),
                                   f'vtransform: expected depth (m) per camera, {vtransform.D} bins '
                                   f'{list(vtransform.dbound)}', cmap='turbo_r')
                plotter.per_camera(name, 'depth_confidence', d.max(1)[0].cpu().numpy(),
                                   'vtransform: max depth-bin probability (1 = confident)', cmap='magma', vmin=0, vmax=1)
            if name == 'dense_head' and hasattr(module, 'forward_ret_dict') and 'pred_dicts' in module.forward_ret_dict:
                plotter.center_heatmaps(name, module.forward_ret_dict['pred_dicts'],
                                        cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD)

        if 'final_box_dicts' in batch_dict or not model.training:
            pred_dicts, _ = model.post_processing(batch_dict)
            print(f'\n[post_processing] {len(pred_dicts[0]["pred_boxes"])} boxes, '
                  f'top scores {pred_dicts[0]["pred_scores"][:5].cpu().numpy().round(3)}')
            if plotter is not None:
                plotter.predictions('post', pred_dicts[0], score_thresh=0.3)

    if args.save_tensors:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        to_save = {k: (v.dense() if is_sparse(v) else v) for k, v in saved.items()}
        torch.save(to_save, out / 'tensors.pt')
        print(f'\nsaved stage outputs -> {out / "tensors.pt"}')
    if plotter is not None:
        print(f'\nfigures in {Path(args.out_dir).resolve()}')


if __name__ == '__main__':
    main()
