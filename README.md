# TruckScenes 3D Object Detection

Lidar-only 3D object detection on [MAN TruckScenes](https://www.man.eu/truckscenes/) for the
TruckScenes detection challenge. Two CenterPoint models are trained on TruckScenes after it is
converted to the nuScenes format, and both are scored with the official truckscenes-devkit evaluation.

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

The converter, `truckscenes_to_nuscenes.py`, is kept outside this repository. It:

- merges the 6 lidars of each frame into one virtual `LIDAR_TOP`, motion-compensated with the ego poses;
- renames scenes to nuScenes split names, following the official TruckScenes train/val split;
- writes the nuScenes tables, dropping only `vehicle.train` annotations.

```shell
python truckscenes_to_nuscenes.py \
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
| Voxel CenterPoint | 3 / 20 | 0.357 | 0.437 |
| Pillar CenterPoint | 5 / 20 | 0.322 | 0.425 |

Velocity error is still high: mAVE is 1.3–1.7, against about 0.25 for CenterPoint on nuScenes.

## Challenge submission

You run inference yourself and upload one JSON file of detections to the
[evaluation server](https://huggingface.co/spaces/TruckScenes/MANTruckScenesDetectionChallenge).
You don't submit model weights.

The rules (truckscenes-devkit `eval/detection/README.md`):

- At inference, use at most 6 past lidar sweeps (about 0.5 s). Training has no limit.
- Submit at most 500 boxes per sample.
- The JSON file must have fewer than 1 million lines.
- Include every test sample token, even with an empty list.
- `meta` must declare `use_camera`, `use_lidar`, `use_radar`, `use_map`, `use_external`, `use_future_frames`, `use_tta`, `method_name` and `authors`.
- Submit at most 5 times per day. Each submission must come from a different model, not just a different epoch.
- Only one result can be selected for the public leaderboard.

Still missing for a test submission:

- converting the TruckScenes test split;
- the extra `meta` fields (`use_future_frames`, `use_tta`, `method_name`, `authors`).

## License

Apache 2.0, inherited from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). See [LICENSE](LICENSE).
