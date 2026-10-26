import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image


class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
            # images may live outside the converted dataset (TruckScenes: the original sensor data)
            self.camera_data_path = Path(self.camera_config.get('DATA_PATH', self.root_path))
        else:
            self.use_camera = False

        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        # info['sweeps'] is ordered newest first; at inference take the most recent ones (deterministic, and
        # TruckScenes allows at most 6 past lidar sweeps at test time), in training keep the random pick
        if self.training:
            sweep_ids = np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False)
        else:
            sweep_ids = range(min(max_sweeps - 1, len(info['sweeps'])))
        for k in sweep_ids:
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_radar_with_sweeps(self, index):
        """
        Ego-motion compensated radar points of the 6 TruckScenes radars, keyframe + RADAR.MAX_SWEEPS - 1 past sweeps
        each, in the lidar frame; needs the *_withradar infos (columns: utils/radar_utils.RADAR_FEATURES).
        """
        from .truckscenes_radar import load_radar_points
        radar_cfg = self.dataset_cfg.RADAR
        return load_radar_points(self.infos[index], radar_cfg.DATA_PATH, max_sweeps=radar_cfg.MAX_SWEEPS)

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            # the integer image size makes the real scale differ slightly per axis; record that one
            img_process_infos.append([(newW / W, newH / H), crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            # img_aug_matrix (resize x * s, flip W - x) treats pixel i as [i, i + 1]; intrinsics put pixel centers on
            # integers, so 0.5 moves them to the same convention (else up to ~1 px error after a flip)
            camera_intrinsics[:2, 2] += self.camera_config.get('PIXEL_CENTER_OFFSET', 0.0)
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.camera_data_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }
        if self.dataset_cfg.get('RADAR', None) is not None:
            input_dict['radar_points'] = self.get_radar_with_sweeps(index)

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        if 'TRUCKSCENES' in self.dataset_cfg:
            # the TruckScenes eval and submission server reject samples with > 500 boxes; the cap in the
            # head config is per head, so keep the 500 highest-scoring boxes over all heads
            for token, boxes in nusc_annos['results'].items():
                nusc_annos['results'][token] = sorted(boxes, key=lambda b: -b['detection_score'])[:500]
        nusc_annos['meta'] = {
            'use_camera': self.use_camera,
            'use_lidar': True,
            'use_radar': self.dataset_cfg.get('RADAR', None) is not None,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        if 'TRUCKSCENES' in self.dataset_cfg:
            return self.evaluation_truckscenes(res_path, output_path)

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def evaluation_truckscenes(self, res_path, output_path):
        """
        Official MAN TruckScenes detection eval (truckscenes-devkit). It reads the GT from the original
        TruckScenes tables; the converted dataset keeps the TruckScenes sample tokens and global frame.
        """
        import json
        from truckscenes import TruckScenes
        from truckscenes.eval.detection.config import config_factory
        from truckscenes.eval.detection.evaluate import DetectionEval
        from . import nuscenes_utils

        ts_cfg = self.dataset_cfg.TRUCKSCENES
        trucksc = TruckScenes(version=ts_cfg.VERSION, dataroot=ts_cfg.DATA_PATH, verbose=True)
        eval_version = ts_cfg.get('EVAL_CONFIG', 'detection_cvpr_2024')
        ts_eval = DetectionEval(
            trucksc,
            config=config_factory(eval_version),
            result_path=res_path,
            eval_set={'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        ts_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)['all']

        return nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            if torch.cuda.is_available():
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()
            elif gt_boxes.shape[0] == 0:
                box_idxs_of_pts = np.full(points.shape[0], -1)
            else:
                # CPU fallback: like the GPU kernel, each point goes to the first box containing it (-1 if none)
                in_boxes = roiaware_pool3d_utils.points_in_boxes_cpu(
                    points[:, 0:3], gt_boxes[:, 0:7]
                ).astype(bool)  # (num_boxes, num_points)
                box_idxs_of_pts = np.where(in_boxes.any(axis=0), in_boxes.argmax(axis=0), -1)

            for i in range(gt_boxes.shape[0]):
                if 'gt_is_ego' in info and info['gt_is_ego'][i]:
                    continue
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


_radar_db_dataset = None


def _radar_points_of_gt_boxes(args):
    """Worker of create_radar_groundtruth_database: [(gt_idx, radar points relative to the box center)] of a sample."""
    from ...utils import box_utils
    idx, gt_indices, extra_width = args
    dataset = _radar_db_dataset
    radar_points = dataset.get_radar_with_sweeps(idx)
    gt_boxes = dataset.infos[idx]['gt_boxes'][:, 0:7].astype(np.float32)
    out = []
    if len(radar_points) == 0 or len(gt_boxes) == 0:
        return idx, [(i, radar_points[:0]) for i in gt_indices]
    large_boxes = box_utils.enlarge_box3d(gt_boxes, extra_width=extra_width).numpy()
    near = box_utils.points_near_boxes_bev(radar_points[:, 0:2], large_boxes)
    box_idxs_of_pts = np.full(len(radar_points), -1)
    if near.any():
        # like the lidar database: each point goes to the first box containing it
        in_boxes = np.asarray(roiaware_pool3d_utils.points_in_boxes_cpu(
            radar_points[near, 0:3], large_boxes)).astype(bool)
        box_idxs_of_pts[near] = np.where(in_boxes.any(axis=0), in_boxes.argmax(axis=0), -1)
    for i in gt_indices:
        pts = radar_points[box_idxs_of_pts == i].copy()
        pts[:, :3] -= gt_boxes[i, :3]
        out.append((i, pts))
    return idx, out


def create_radar_groundtruth_database(dataset, lidar_db_info_path, extra_width=(1.0, 1.0, 2.0), num_workers=4):
    """
    Radar counterpart of the lidar GT database, for gt_sampling of the radar fusion models. Every lidar db_info
    (same sample index and gt index) gets the radar points of its box, enlarged by extra_width (radar returns
    scatter around the object): all of them go into one flat float32 file and each db_info stores its
    'radar_offset' rows. A pasted object then brings the radar returns it really had (possibly none).
    dataset must hold the *_withradar train infos in the order the lidar database was built from.
    """
    import multiprocessing
    from ...utils.radar_utils import RADAR_FEATURES
    global _radar_db_dataset

    max_sweeps = dataset.dataset_cfg.RADAR.MAX_SWEEPS
    with open(lidar_db_info_path, 'rb') as f:
        all_db_infos = pickle.load(f)
    db_infos_of_sample = {}
    for name, db_infos in all_db_infos.items():
        for db_info in db_infos:
            info = dataset.infos[db_info['image_idx']]
            assert info['gt_names'][db_info['gt_idx']] == name and \
                np.allclose(info['gt_boxes'][db_info['gt_idx']], db_info['box3d_lidar'], equal_nan=True), \
                'the infos are not the ones the lidar GT database was built from'
            db_infos_of_sample.setdefault(db_info['image_idx'], {})[db_info['gt_idx']] = db_info

    stem = Path(lidar_db_info_path).stem
    data_save_path = dataset.root_path / f'gt_database_radar_{max_sweeps}sweeps.bin'
    db_info_save_path = dataset.root_path / f'{stem}_radar{max_sweeps}sweeps.pkl'
    _radar_db_dataset = dataset
    offset, num_feat = 0, len(RADAR_FEATURES)
    jobs = [(idx, sorted(gt.keys()), tuple(extra_width)) for idx, gt in sorted(db_infos_of_sample.items())]
    with open(data_save_path, 'wb') as f_data, multiprocessing.get_context('fork').Pool(num_workers) as pool:
        for idx, per_box in tqdm(pool.imap(_radar_points_of_gt_boxes, jobs, chunksize=8), total=len(jobs),
                                 desc='radar gt database', dynamic_ncols=True):
            for gt_idx, pts in per_box:
                assert pts.shape[1] == num_feat
                pts.astype(np.float32).tofile(f_data)
                db_info = db_infos_of_sample[idx][gt_idx]
                db_info['radar_offset'] = (offset, offset + len(pts))
                db_info['num_radar_points_in_gt'] = len(pts)
                offset += len(pts)
    _radar_db_dataset = None

    for name, db_infos in all_db_infos.items():
        num_with_radar = sum(x['num_radar_points_in_gt'] > 0 for x in db_infos)
        print('Radar database %s: %d / %d objects with radar points' % (name, num_with_radar, len(db_infos)))
    for db_infos in all_db_infos.values():
        for db_info in db_infos:
            db_info['radar_path'] = str(data_save_path.relative_to(dataset.root_path))
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
    print(f'{offset} radar points ({offset * num_feat * 4 / 1e9:.2f} GB) -> {data_save_path}, infos -> {db_info_save_path}')


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False, truckscenes=False):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam,
        name_map=(nuscenes_utils.map_name_from_truckscenes_to_detection if truckscenes
                  else nuscenes_utils.map_name_from_general_to_detection)
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


def add_camera_infos(version, data_path, max_sweeps=10):
    """
    Add the 6-camera calibration ('cams') to existing info files without regenerating
    infos or the gt database. Writes nuscenes_infos_{max_sweeps}sweeps_{split}_withcam.pkl.
    """
    from nuscenes.nuscenes import NuScenes
    from . import nuscenes_utils
    data_path = data_path / version
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    for split in ['train', 'val']:
        src = data_path / f'nuscenes_infos_{max_sweeps}sweeps_{split}.pkl'
        if not src.exists():
            continue
        with open(src, 'rb') as f:
            infos = pickle.load(f)
        for info in tqdm(infos, desc=f'add_cams_{split}', dynamic_ncols=True):
            sample = nusc.get('sample', info['token'])
            lidar_sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
            lidar_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
            l2e_t = np.array(lidar_cs_rec['translation'])[None]
            l2e_r_mat = Quaternion(lidar_cs_rec['rotation']).rotation_matrix
            e2g_t = lidar_pose_rec['translation']
            e2g_r_mat = Quaternion(lidar_pose_rec['rotation']).rotation_matrix

            info['cams'] = dict()
            for cam in camera_types:
                cam_token = sample['data'][cam]
                _, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = nuscenes_utils.obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info['cams'].update({cam: cam_info})
        dst = data_path / f'nuscenes_infos_{max_sweeps}sweeps_{split}_withcam.pkl'
        with open(dst, 'wb') as f:
            pickle.dump(infos, f)
        print(f'{split} sample: {len(infos)} -> {dst}')


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    parser.add_argument('--lidar_db_info', type=str, default='nuscenes_dbinfos_10sweeps_withvelo.pkl',
                        help='lidar GT database the radar GT database is paired with')
    parser.add_argument('--radar_sweeps', type=int, default=12, help='radar keyframe + past sweeps kept in the infos')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        # DATA_PATH may be absolute or relative to tools/, matching how train.py resolves it
        data_root = (ROOT_DIR / 'tools' / dataset_cfg.DATA_PATH).resolve()
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=data_root,
            save_path=data_root,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam,
            truckscenes='TRUCKSCENES' in dataset_cfg
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=data_root,
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
    elif args.func == 'add_camera_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        data_root = (ROOT_DIR / 'tools' / dataset_cfg.DATA_PATH).resolve()
        add_camera_infos(version=args.version, data_path=data_root, max_sweeps=dataset_cfg.MAX_SWEEPS)
    elif args.func == 'add_radar_infos':
        from .truckscenes_radar import add_radar_infos
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        data_root = (ROOT_DIR / 'tools' / dataset_cfg.DATA_PATH).resolve() / args.version
        add_radar_infos(
            data_path=data_root, truckscenes_path=dataset_cfg.TRUCKSCENES.DATA_PATH,
            truckscenes_version=dataset_cfg.TRUCKSCENES.VERSION,
            info_names=[f'nuscenes_infos_{dataset_cfg.MAX_SWEEPS}sweeps_{split}.pkl' for split in ['train', 'val']],
            max_sweeps=args.radar_sweeps
        )
    elif args.func == 'add_truckscenes_camera_infos':
        # 4 TruckScenes cameras from the original tables, on top of the radar infos
        from .truckscenes_camera import add_camera_infos as add_truckscenes_camera_infos
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        data_root = (ROOT_DIR / 'tools' / dataset_cfg.DATA_PATH).resolve() / args.version
        add_truckscenes_camera_infos(
            data_path=data_root, truckscenes_path=dataset_cfg.TRUCKSCENES.DATA_PATH,
            truckscenes_version=dataset_cfg.TRUCKSCENES.VERSION,
            info_names=[f'nuscenes_infos_10sweeps_{split}_withradar.pkl' for split in ['train', 'val']]
        )
    elif args.func == 'create_radar_gt_database':
        # --cfg_file: the radar model config (DATA_CONFIG with RADAR and the *_withradar INFO_PATH); run from tools/
        from ...config import cfg_from_yaml_file, cfg
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        cfg_from_yaml_file(args.cfg_file, cfg)
        dataset_cfg = cfg.DATA_CONFIG if 'DATA_CONFIG' in cfg else cfg
        dataset_cfg.VERSION = args.version
        data_root = (ROOT_DIR / 'tools' / dataset_cfg.DATA_PATH).resolve()
        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None, root_path=data_root,
            logger=common_utils.create_logger(), training=True
        )
        create_radar_groundtruth_database(
            nuscenes_dataset, nuscenes_dataset.root_path / args.lidar_db_info
        )
