# TruckScenes 3D Object Detection

3D object detection on [MAN TruckScenes](https://www.man.eu/truckscenes/) for the TruckScenes
detection challenge. TruckScenes is converted to the nuScenes format and CenterPoint models are trained on it:
lidar-only baselines, and **TriSense-CenterPoint**, which fuses lidar, the 6 radars and 4 cameras. All models are
scored with the official truckscenes-devkit evaluation.

TriSense-CenterPoint scores **NDS 0.5542 / mAP 0.5050 on the TruckScenes test set** (see
[Challenge submission](#challenge-submission)).

The code is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Environment

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.12.1 (CUDA 13.0) |
| spconv | spconv-cu126 2.3.8 |
| nuscenes-devkit | 1.2.0 |
| truckscenes-devkit | 1.2.0 |
| numpy | 1.26.4 |

```shell
pip install -r requirements.txt
pip install nuscenes-devkit truckscenes-devkit
python setup.py develop
```

## Data

### 1. Convert TruckScenes to nuScenes format

The converter is `tools/truckscenes_to_nuscenes.py` (it needs `pip install python-lzf`). It:

- merges the 6 lidars of each frame into one virtual `LIDAR_TOP`, motion-compensated with the ego poses;
- renames scenes to nuScenes split names, following the official TruckScenes train/val split;
- writes the nuScenes tables, dropping only `vehicle.train` annotations.

```shell
python tools/truckscenes_to_nuscenes.py \
    --meta  <truckscenes>/man-truckscenes/v1.2-trainval \
    --data  <sensordata-root-1>/man-truckscenes <sensordata-root-2>/man-truckscenes ... \
    --out   /root/truckscenes-dataset/man_nuscenes/openpcdet --version v1.0-trainval
```

### 2. Build the infos and the GT database

```shell
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/man_nuscenes_dataset.yaml --version v1.0-trainval
```

This writes `nuscenes_infos_10sweeps_{train,val}.pkl` (20907 train / 2995 val samples), the GT
database `gt_database_10sweeps_withvelo/` (about 35 GB) and its index
`nuscenes_dbinfos_10sweeps_withvelo.pkl` to `<DATA_PATH>/v1.0-trainval/`.

The `TRUCKSCENES` section of the dataset config does two things:

- it switches the class mapping to the 12 TruckScenes classes;
- it points the evaluation at the original TruckScenes tables.

```yaml
DATA_PATH: '/root/truckscenes-dataset/man_nuscenes/openpcdet'     # converted dataset
TRUCKSCENES:
    DATA_PATH: '/root/truckscenes-dataset/man-truckscenes'        # original TruckScenes (GT for eval)
    VERSION: 'v1.2-trainval'
    EVAL_CONFIG: 'detection_cvpr_2024'
```

### 3. Test split (for challenge submissions)

The test split has no annotations, so only the infos are built (no GT database). Convert it with
`--version v1.0-test`, which maps the test scenes to the nuScenes test scene names; sample tokens stay the
original TruckScenes ones. The info commands run from `tools/`, because the test config's `_BASE_CONFIG_` is
relative to it:

```shell
python tools/truckscenes_to_nuscenes.py \
    --meta <truckscenes-test>/man-truckscenes/v1.2-test --data <truckscenes-test>/man-truckscenes \
    --out  /root/truckscenes-dataset/man_nuscenes/openpcdet --version v1.0-test
cd tools
for f in create_nuscenes_infos add_radar_infos add_truckscenes_camera_infos; do
    python -m pcdet.datasets.nuscenes.nuscenes_dataset --func $f \
        --cfg_file cfgs/dataset_configs/man_nuscenes_test_dataset.yaml --version v1.0-test
done
```

This writes `nuscenes_infos_10sweeps_test{,_withradar,_withradar_withcam}.pkl` (149 scenes, 5960 samples) to
`<DATA_PATH>/v1.0-test/`. The paths of the original test data are set in
`tools/cfgs/dataset_configs/man_nuscenes_test_dataset.yaml`.

### Classes

These are the 12 TruckScenes detection classes:

`car, truck, bus, trailer, other_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier, animal, traffic_sign`

- `vehicle.construction` and `vehicle.other` map to `other_vehicle`.
- The truck's own trailer (`vehicle.ego_trailer`) is labelled `trailer`, but it is kept out of the GT database used for GT-sampling augmentation.

## Models

| Config | Backbone | Range (x/y) | Voxel | Sweeps |
|---|---|---|---|---|
| `tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_man_dataset.yaml` | VoxelResBackBone8x (3D sparse) | ±153.6 m | 0.15 × 0.15 × 0.2 m | 5 |
| `tools/cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_man_dataset.yaml` | PillarRes18BackBone8x (2D pillars) | ±153.6 m | 0.15 × 0.15 m | 5 |

Both configs share these settings:

- **Heads:** 7 CenterPoint heads: `[car]`, `[truck, other_vehicle]`, `[bus, trailer]`, `[barrier]`, `[motorcycle, bicycle]`, `[pedestrian, traffic_cone, animal]` and `[traffic_sign]`.
- **Training:** CBGS class-balanced resampling, GT sampling, and 20 epochs with a one-cycle Adam schedule.
- **Range:** The 153.6 m range covers the challenge's 150 m evaluation distance for vehicles.

## Radar–lidar fusion

`tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_man_dataset.yaml` (`RadarFusionCenterPoint`)
adds a radar pillar branch to the voxel CenterPoint and fuses the two BEV maps before `BACKBONE_2D`.

- **Radar data:** the converted dataset has no radar, so the 6 radars are read from the original TruckScenes data
  (`RADAR.DATA_PATH`). Each sample gets the keyframe plus `RADAR.MAX_SWEEPS - 1` past sweeps of each radar (20 Hz;
  the challenge allows 11 past sweeps), in `data_dict['radar_points']`. TruckScenes `vrel` is the raw Doppler along the ray, so
  the radial velocity is compensated with the ego chassis velocity and yaw rate. Columns:
  `pcdet/utils/radar_utils.py`.
- **Augmentation:** flip, rotation, scaling and translation move the radar points and their velocity vectors and rays.
  GT sampling pastes each object's own radar returns from a radar GT database and recomputes their radial velocity for the
  pasted position.
- **Model:** `RadarPillarVFE` → `RadarPointPillarScatter` (0.6 m pillars) → `RadarBEVBackbone` (stride 2) gives the
  lidar BEV grid (256 × 256, 1.2 m cells). `RadarLidarBEVFuser` adds convs over the concatenated maps to the lidar BEV.
  Training-only aux CenterHeads: lidar `[center, center_z, dim, rot]` and radar `[center, vel]` + a Doppler consistency loss,
  logged as `train/aux_lidar/*` and `train/aux_radar/*`.

Build the radar infos (about 3 min) and the radar GT database (about 5 min, 0.9 GB), then train starting from the lidar model:

```shell
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func add_radar_infos \
    --cfg_file tools/cfgs/dataset_configs/man_nuscenes_dataset.yaml --version v1.0-trainval
cd tools
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_radar_gt_database \
    --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_man_dataset.yaml --version v1.0-trainval
python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_man_dataset.yaml \
    --extra_tag v1 --workers 8 --batch_size 4 --use_amp \
    --pretrained_model ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_man_dataset/v3_150m/ckpt/checkpoint_epoch_4.pth
```

`python tests/test_radar_fusion.py` (from `tools/`, CPU, about 30 s) checks the compensation, the augmentations, the
GT sampling, the model losses and the checkpoint loading.

### Cameras (Phase 2, data side)

`tools/cfgs/dataset_configs/man_nuscenes_radar_camera_dataset.yaml` adds the 4 cameras (`CAMERA_LEFT_FRONT`,
`CAMERA_RIGHT_FRONT`, `CAMERA_LEFT_BACK`, `CAMERA_RIGHT_BACK`) to the lidar + radar data, for the BEVFusion-style models.

- The images are read in place from the original TruckScenes data (`CAMERA_CONFIG.DATA_PATH`). Each camera's
  camera → lidar transform uses that camera's ego pose, because the back cameras fire about 35 ms off the lidar.
- Image size is set in the config: `CAMERA_CONFIG.IMAGE.FINAL_DIM` (448 × 960 by default; 384 × 832 is a smaller option)
  and `RESIZE_LIM_*`. Image flip and rotation come from `imgaug`. `img_aug_matrix` records the exact per-axis resize,
  and `PIXEL_CENTER_OFFSET: 0.5` moves the intrinsics to its pixel-edge convention.
- `camera_depth_targets` stores the nearest current-sweep lidar depth of each image feature cell (`DOWNSAMPLE`), for
  depth supervision.
- No GT sampling: pasted lidar and radar objects have no image content.

```shell
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func add_truckscenes_camera_infos \
    --cfg_file tools/cfgs/dataset_configs/man_nuscenes_dataset.yaml --version v1.0-trainval   # ~2 min, 0.6 GB
cd tools && python tests/test_camera_data.py
```

The test projects 3D points through the full augmented pipeline and checks that they land on the drawn image pixels
(< 1 px, no bias) and that the view transform's geometry maps them back. It also writes
`output/camera_aug_overlay.jpg`, which shows the GT boxes on the augmented images.

## Lidar–radar–camera fusion (TriSense-CenterPoint)

`tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_man_dataset.yaml`
(`RadarCameraFusionCenterPoint`) is the radar–lidar model above plus a camera branch: Swin-T + FPN → a camera-only
LSS view transform onto the shared 256 × 256 / 1.2 m BEV grid → a camera fuser added to the lidar + radar BEV.

Training has two stages, both from `tools/`:

1. **Stage A (warm-up)**, `..._radar_camera_warmup_man_dataset.yaml`, starting from the lidar-only checkpoint. The
   lidar model is frozen; the camera and radar branches train on their own aux losses (camera heatmap + depth, radar
   aux + Doppler). The fusers stay at zero init, so the model still predicts exactly like the lidar model.
2. **Stage B (joint)**, `..._radar_camera_man_dataset.yaml`, starting from the stage A checkpoint.

```shell
python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_warmup_man_dataset.yaml \
    --extra_tag v1 --workers 16 --batch_size 4 --use_amp \
    --pretrained_model ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_man_dataset/v3_150m/ckpt/checkpoint_epoch_4.pth
python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_man_dataset.yaml \
    --extra_tag v1 --workers 16 --batch_size 4 --use_amp \
    --pretrained_model ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_warmup_man_dataset/v1/ckpt/checkpoint_epoch_1.pth
```

Inputs at inference: keyframe + 4 past lidar sweeps, keyframe + 6 past radar sweeps, keyframe images of the 4
cameras. No test-time augmentation, future frames, map or external data.

## Training

Run from `tools/`:

```shell
python train.py --cfg_file cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_man_dataset.yaml \
    --extra_tag v1_150m --workers 8 --batch_size 14 --use_amp
```

### Memory and speed
- **Host RAM limits workers × batch size.** Each worker holds the batch it is building, and the main process holds prefetched batches. On a 30-core, 88 GB machine, 8 workers with batch 14 uses about 14 GB.
- **`--num_threads` (default 8)** sets the torch CPU threads of the main process. Torch's default of one thread per core competes with the dataloader workers and nearly doubles the step time.
- **Keep `EVAL_INTERVAL: 0`** and evaluate checkpoints separately (see below). The per-epoch eval starts a second set of workers next to the training ones and can run the machine out of memory.

### Resuming
Run the same command with the same `--extra_tag`. Training resumes from the newest file in `ckpt/`, which is either `latest_model.pth` (saved every 5 minutes) or `checkpoint_epoch_N.pth`.

**Use the same `--batch_size` as the original run.** The position within the epoch is recomputed as the iteration count modulo the iterations per epoch. A different batch size skips part of the epoch and shifts the learning-rate schedule.

If the main process is OOM-killed, its dataloader workers can survive as orphans. Free their RAM with:

```shell
ps -eo pid,ppid,comm | awk '$2==1 && $3=="pt_data_worker" {print $1}' | xargs -r kill -9
```

## Evaluation

```shell
python test.py --cfg_file cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_man_dataset.yaml \
    --extra_tag v1_150m --eval_tag ep5 --batch_size 4 --workers 4 \
    --ckpt ../output/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_man_dataset/v1_150m/ckpt/checkpoint_epoch_5.pth
```

- Scoring uses the official `truckscenes.eval.detection.evaluate.DetectionEval` with `detection_cvpr_2024`. It reports mAP, the TP errors and NDS, and filters by class-specific distances: 150 m for vehicles, 75 m for the other classes.
- At inference, each sample uses the current frame plus the most recent `MAX_SWEEPS - 1` sweeps. Training picks the past sweeps at random.
- Each sample keeps at most its 500 highest-scoring boxes, as the evaluation server requires.
- Results are written to `output/.../eval/epoch_N/val/<eval_tag>/final_result/data/` (`results_nusc.json` and `metrics_summary.json`).

### Validation results so far

| Model | Epoch | mAP | NDS |
|---|---|---|---|
| Voxel CenterPoint (lidar) | 3 / 20 | 0.357 | 0.437 |
| Pillar CenterPoint (lidar) | 5 / 20 | 0.322 | 0.425 |
| Voxel CenterPoint (lidar, `v3_150m`, start of the fusion model) | 4 | 0.334 | 0.433 |
| TriSense-CenterPoint, stage B | 3 | 0.482 | 0.504 |
| TriSense-CenterPoint, stage B | 5 | 0.497 | 0.545 |
| TriSense-CenterPoint, stage B (submitted) | 6 | 0.492 | 0.548 |

Velocity error is still the largest TP error: mAVE is about 0.82 for the fusion model (1.3–1.7 for lidar-only),
against about 0.25 for CenterPoint on nuScenes.

## Challenge submission

You run inference yourself and upload one JSON file of detections to the
[evaluation server](https://huggingface.co/spaces/TruckScenes/MANTruckScenesDetectionChallenge).
You don't submit model weights.

### Running inference on the test split

Build the test infos first ([Test split](#3-test-split-for-challenge-submissions)), then from `tools/`:

```shell
python test.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_test_man_dataset.yaml \
    --batch_size 4 --workers 16 --extra_tag v1 \
    --ckpt ../output/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_radar_camera_man_dataset/v1/ckpt/checkpoint_epoch_6.pth
```

There is no ground truth, so `test.py` only writes `results_nusc.json` (about 37 min on an A100; about 195 MB,
54 MB zipped). It keeps the 500 highest-scoring boxes per sample and writes the full `meta` block. The `use_*`
flags come from the config; the method info comes from `DATA_CONFIG.SUBMISSION_META` in the test config:

```yaml
SUBMISSION_META:
    method_name: 'TriSense-CenterPoint'
    authors: ''
    affiliation: ''
    description: '...'
    code_url: ''
    paper_url: ''
```

Check the file with the devkit's own loader before uploading:

```shell
python -c "from truckscenes.eval.common.loaders import load_prediction; \
from truckscenes.eval.detection.data_classes import DetectionBox; \
r, m = load_prediction('results_nusc.json', 500, DetectionBox, verbose=True); print(len(m), m['method_name'])"
```

### Rules

From truckscenes-devkit `eval/detection/README.md` (the evaluation server installs `truckscenes-devkit`):

- At inference, use at most about 0.5 s of past data: 6 past lidar sweeps, 11 past radar sweeps, 6 past camera
  images. Training has no limit.
- Submit at most 500 boxes per sample.
- The JSON file must have fewer than 1 million lines.
- Include every test sample token, even with an empty list.
- `meta` must have all 13 fields: `use_camera`, `use_lidar`, `use_radar`, `use_map`, `use_external`,
  `use_future_frames`, `use_tta` (bools) and `method_name`, `authors`, `affiliation`, `description`, `code_url`,
  `paper_url` (strings; empty if not available). A submission with fewer fields is rejected.
- `attribute_name` must be valid for the class, and empty for barrier and traffic_cone.
- Submit at most 5 times per day. Each submission must come from a different model, not just a different epoch.
- Only one result can be selected for the public leaderboard.

### Test results

TriSense-CenterPoint, stage B epoch 6, submitted 2026-11-02:

| mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|
| 0.5050 | 0.5542 | 0.373 | 0.359 | 0.276 | 0.868 | 0.108 |

| car | truck | bus | trailer | other_vehicle | pedestrian | motorcycle | bicycle | traffic_cone | barrier | animal | traffic_sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.842 | 0.588 | 0.401 | 0.652 | 0.254 | 0.584 | 0.598 | 0.136 | 0.720 | 0.565 | 0.000 | 0.719 |

Weakest conditions: night (NDS 0.00), tunnels (0.22), fog (0.28), glare (0.29), against 0.56 on regular roads.

## License

Apache 2.0, inherited from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). See [LICENSE](LICENSE).
