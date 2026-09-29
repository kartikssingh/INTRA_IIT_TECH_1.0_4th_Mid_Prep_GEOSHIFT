# GeoShift: Overnight Training Pipeline

A single-script training pipeline for multi-band satellite tile classification under geographic shift. It trains a ResNet-50 (and optionally an EfficientNet-B0) with region-grouped cross-validation, ensembles them, and writes a Kaggle-style `submission.csv`. It is designed to run unattended overnight, with a GPU thermal watchdog and safety submissions written after each stage.

## Results

| Split | F1 score |
|-------|----------|
| Training dataset | **0.89** |
| Kaggle test dataset | **0.86415** |

## Models

| Stage | Model | Strategy |
|-------|-------|----------|
| 1 | ResNet-50 (ImageNet V2 weights) | Full fine-tune, backbone frozen for the first epoch (main model) |
| 2 | EfficientNet-B0 (ImageNet V1 weights) | Full fine-tune, run only if time allows |

The first conv layer is expanded to accept the extra input channels. The pretrained RGB weights are copied, and the extra channels are initialised with the mean of the RGB filters.

## Key Features

- **Region-grouped CV:** `GroupKFold` on the region code parsed from each `Id` (regex `GEO_(R\d+)_`), so validation folds contain regions unseen in training. This mimics the geographic shift at test time.
- **NDVI feature:** one extra channel computed as `(NIR - Red) / (NIR + Red)`. The script assumes **Red = channel index 2** and **NIR = channel index 3**. Change `add_ndvi` if your band order differs.
- **Per-tile normalisation:** each tile is standardised per channel, then bicubically resized to 224x224.
- **Augmentation:** random horizontal and vertical flips, random 90° rotations, and mild per-channel gain jitter.
- **Class imbalance handling:** inverse-square-root class weights (capped at `CLASS_WEIGHT_CAP`) plus label smoothing.
- **TTA:** 4-way flip test-time augmentation (none, H, V, H+V) for out-of-fold and test predictions.
- **OOM recovery:** on a CUDA out-of-memory error the fold is retried once with half the batch size.
- **Time-aware staging:** stage 2 is skipped if too much of the wall-clock budget has already been used.
- **Thermal watchdog:** a background thread polls `nvidia-smi` and kills the process if the GPU stays too hot.

## Requirements

- Python 3.9+
- An NVIDIA GPU with `nvidia-smi` on the PATH (the watchdog only starts on CUDA; the script also runs on CPU, but very slowly)
- Packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`
- Internet access on first run to download the pretrained torchvision weights

```bash
pip install torch torchvision numpy pandas scikit-learn
```

## Input Data

Place these files in the working directory (or change `DATA_DIR`):

| File | Description |
|------|-------------|
| `train.csv` | Must contain `Id` and `label` columns |
| `train_images.npy` | Array of tiles, shaped NHWC or NCHW (auto-detected and converted to NHWC) |
| `test.csv` | Test metadata |
| `test_images.npy` | Test tiles in the same format as training |
| `sample_submission.csv` | Template whose `label` column gets overwritten |

Labels are expected to be integers `0..K-1`. The number of classes is inferred as `max(label) + 1`.

## Usage

```bash
python train.py
```

Progress is logged with an elapsed-time prefix, for example `[01:23:45] ...`. To keep a persistent log:

```bash
nohup python train.py > train.log 2>&1 &
```

## Outputs

| File | Written when |
|------|--------------|
| `submission_resnet50_only.csv` | After stage 1 (safety net) |
| `submission.csv` | End of run (final ensemble) |
| `oof_<arch>.npy` | Out-of-fold probabilities per architecture (`resnet`, `effnet`) |
| `test_<arch>.npy` | Fold-averaged test probabilities per architecture |
| `fold_f1s_<arch>.json` | Per-fold macro F1, mean and std |

A final training report (per-model fold F1s, ensemble weights, OOF macro F1, test prediction class distribution, total wall clock) is printed at the end.

## Ensemble Logic

1. Compute each model's mean CV macro F1.
2. Drop any model scoring below 60% of the best model's mean F1.
3. Weight the remaining models proportionally to their mean F1.
4. Weighted-average the probabilities and take the `argmax` as the final prediction.

If only ResNet-50 completes, the final submission is simply its predictions.

## Configuration

All settings are constants at the top of the script.

| Setting | Default | Notes |
|---------|---------|-------|
| `IMG_SIZE` | 224 | Input resolution |
| `N_FOLDS` | 5 | Capped at the number of unique regions |
| `WALL_CLOCK_BUDGET_H` | 8.0 | Used for the stage-2 skip decision only |
| `TEMP_LIMIT_C` | 82 | GPU abort threshold |
| `TEMP_CHECK_SEC` | 10 | Watchdog polling interval |
| `RN50_EPOCHS` / `RN50_BATCH` | 12 / 32 | ResNet-50 |
| `RN50_LR_HEAD` / `RN50_LR_BACKBONE` | 1e-3 / 1e-4 | Separate learning rates |
| `EFF_EPOCHS` / `EFF_BATCH` | 12 / 32 | EfficientNet-B0 |
| `EFF_LR_HEAD` / `EFF_LR_BACKBONE` | 1e-3 / 1e-4 | Separate learning rates |
| `LABEL_SMOOTHING` | 0.05 | |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `CLASS_WEIGHT_CAP` | 3.0 | Upper bound on class weights |
| `WARMUP_EPOCHS` | 1 | Epochs with the backbone frozen |

Optimiser is AdamW with cosine annealing. The best epoch (by validation macro F1) is kept for each fold.

## Thermal Watchdog

- Reads GPU temperature via `nvidia-smi` every `TEMP_CHECK_SEC` seconds.
- If the temperature is at or above `TEMP_LIMIT_C` on **2 consecutive checks**, the process exits immediately with code **42** (`os._exit`).
- Because the exit is a hard kill, in-progress folds are lost. Only outputs from already completed stages remain on disk.
- If `nvidia-smi` is unavailable, the watchdog silently does nothing.

## Time Budget Behaviour

The budget check happens only **between stages**, not inside them. Stage 2 (EfficientNet-B0) starts only if elapsed time is under `WALL_CLOCK_BUDGET_H - 3` hours (5h by default). A stage that starts just before the cutoff can still run past the nominal budget, so leave headroom.

## Notes

- Best-epoch selection uses the same validation fold that is later used for OOF predictions, so OOF scores are slightly optimistic.
- Runtime depends heavily on GPU, dataset size and image count. With 5 folds and 12 epochs per fold, each architecture takes a large share of the total time.
- Random seeds are set (`SEED = 42`), but `cudnn.benchmark = True` and multi-worker data loading mean runs are not bit-for-bit reproducible.
