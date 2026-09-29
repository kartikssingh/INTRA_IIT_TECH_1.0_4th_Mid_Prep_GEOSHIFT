# GeoShift: Multi-Band Satellite Tile Classification

A single-script training pipeline for classifying multi-band satellite image tiles under **geographic shift**, where the test tiles come from regions that differ from the ones used in training. The pipeline fine-tunes a ResNet-50 as its main model, optionally adds an EfficientNet-B0, evaluates both with region-grouped cross-validation, combines them into a weighted ensemble, and writes a Kaggle-style `submission.csv`.

---

## Table of Contents

1. [Results](#results)
2. [Pipeline Overview](#pipeline-overview)
3. [Models](#models)
4. [Data Format](#data-format)
5. [Preprocessing](#preprocessing)
6. [Training Details](#training-details)
7. [Validation Strategy](#validation-strategy)
8. [Inference and TTA](#inference-and-tta)
9. [Ensembling](#ensembling)
10. [Requirements and Installation](#requirements-and-installation)
11. [Usage](#usage)
12. [Outputs](#outputs)
13. [Configuration Reference](#configuration-reference)
14. [Time Budget Behaviour](#time-budget-behaviour)
15. [Troubleshooting](#troubleshooting)
16. [Notes (P.S.)](#notes-ps)

---

## Results

| Split | F1 score |
|-------|----------|
| Training dataset | **0.89** |
| Kaggle test dataset | **0.86415** |

The gap between the training score and the Kaggle score is small (about 0.026), which suggests the region-grouped validation setup does a reasonable job of preparing the model for unseen regions.

---

## Pipeline Overview

```
train.csv + train_images.npy          test.csv + test_images.npy
            |                                     |
            v                                     v
   channel-order fix (NCHW -> NHWC)      channel-order fix (NCHW -> NHWC)
            |                                     |
            v                                     v
       add NDVI channel                      add NDVI channel
            |                                     |
            v                                     |
  parse region from Id  ->  GroupKFold            |
            |                                     |
            v                                     |
  +-----------------------------+                 |
  | STAGE 1: ResNet-50 (5 folds)| ---- test preds averaged over folds
  +-----------------------------+                 |
            | writes submission_resnet50_only.csv |
            v                                     |
  +--------------------------------+              |
  | STAGE 2: EfficientNet-B0       | ---- test preds averaged over folds
  | (only if time budget allows)   |              |
  +--------------------------------+              |
            |                                     |
            v                                     v
       weighted ensemble by mean CV macro F1
            |
            v
      submission.csv
```

---

## Models

| Stage | Model | Weights | Strategy |
|-------|-------|---------|----------|
| 1 | ResNet-50 | ImageNet V2 | Full fine-tune with the backbone frozen for the first epoch (main model) |
| 2 | EfficientNet-B0 | ImageNet V1 | Full fine-tune with the backbone frozen for the first epoch; runs only if the time budget allows |

### Adapting pretrained models to more than 3 channels

Satellite tiles have more bands than the RGB images ImageNet models expect (the original bands plus one NDVI channel). For both architectures the first convolution is replaced with a new one that accepts `in_channels` inputs:

- The pretrained RGB filters are copied into the first three input channels.
- Every extra channel is initialised with the **mean of the three RGB filters**, so the extra bands start with a sensible, non-random response.
- The final classification layer is replaced with a fresh linear layer sized to the number of classes.

---

## Data Format

Place these files in the working directory (or change `DATA_DIR` in the script):

| File | Description |
|------|-------------|
| `train.csv` | Must contain an `Id` column and an integer `label` column |
| `train_images.npy` | Array of tiles in NHWC or NCHW layout (auto-detected) |
| `test.csv` | Test metadata |
| `test_images.npy` | Test tiles in the same layout as the training tiles |
| `sample_submission.csv` | Template; its `label` column is overwritten with predictions |

Additional assumptions:

- Labels are integers `0..K-1`. The number of classes is inferred as `max(label) + 1`.
- Each `Id` contains a region code matching `GEO_R<number>_` (for example `GEO_R12_...`). Ids that do not match are assigned to a fallback region `R00`.
- Band order places **Red at index 2** and **NIR at index 3**, which the NDVI calculation depends on.

---

## Preprocessing

1. **Layout fix:** if the array looks channels-first (second dimension is at most 8 and smaller than the last dimension), it is transposed to NHWC.
2. **NDVI channel:** computed as `(NIR - Red) / (NIR + Red + 1e-6)` and appended as an extra channel.
3. **Per-tile standardisation:** each tile is normalised per channel using its own mean and standard deviation. This makes the model less sensitive to brightness and sensor differences between regions.
4. **Resize:** each tile is resized to 224x224 with bicubic interpolation.
5. **Augmentation (training only):**
   - random horizontal flip (50%)
   - random vertical flip (50%)
   - random rotation by 0, 90, 180 or 270 degrees
   - mild per-channel gain jitter of about ±5% (30% of samples)

---

## Training Details

| Aspect | Setting |
|--------|---------|
| Optimiser | AdamW, weight decay `1e-4` |
| Learning rates | Head `1e-3`, backbone `1e-4` (separate parameter groups) |
| Scheduler | Cosine annealing; restarted for the remaining epochs when the backbone is unfrozen |
| Warm-up | Backbone frozen for `WARMUP_EPOCHS` (1) so the new head can settle first |
| Loss | Cross-entropy with class weights and label smoothing `0.05` |
| Class weights | Inverse square root of class frequency, normalised, capped at `3.0` |
| Batch size | 32 (halved automatically on a CUDA out-of-memory error, minimum 4) |
| Epochs | 12 per fold for both architectures |
| Model selection | The epoch with the best validation macro F1 is kept for each fold |
| Data loading | 2 worker processes, pinned memory on CUDA |

---

## Validation Strategy

The pipeline uses **`GroupKFold`** with the region code as the group:

- Every tile from a region lands in the same fold, so each validation fold contains regions the model has never seen during training.
- This mirrors the real task, where test tiles come from different geography, and gives a more honest estimate than random splitting.
- The number of folds is `min(N_FOLDS, number of unique regions)`, with `N_FOLDS = 5` by default.
- The validation regions for each fold are printed at the start of a run.

The metric throughout is **macro F1**, which weighs every class equally and is therefore sensitive to rare classes.

---

## Inference and TTA

Predictions use 4-way flip test-time augmentation: no flip, horizontal, vertical, and both. The softmax probabilities of the four views are averaged.

- **Out-of-fold (OOF) predictions** are produced for each validation fold using the model trained without that fold.
- **Test predictions** are produced by every fold's model and averaged across folds, giving a 5-model average per architecture.

---

## Ensembling

1. Compute each architecture's mean CV macro F1.
2. Drop any architecture scoring below **60% of the best** architecture's mean F1.
3. Weight the remaining architectures proportionally to their mean F1.
4. Take the weighted average of their probabilities and use `argmax` as the final label.
5. Report the ensemble's macro F1 on the OOF predictions.

If only ResNet-50 completes (for example, stage 2 is skipped or fails), the final submission is simply the ResNet-50 predictions.

---

## Requirements and Installation

- Python 3.9 or newer
- A CUDA-capable GPU is strongly recommended (the script runs on CPU, but very slowly)
- Packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`
- Internet access on the first run to download the pretrained torchvision weights

```bash
pip install torch torchvision numpy pandas scikit-learn
```

---

## Usage

```bash
python train.py
```

Every log line is prefixed with the elapsed time, for example `[01:23:45] ...`. To keep a persistent log and run in the background:

```bash
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

---

## Outputs

| File | Written when |
|------|--------------|
| `submission_resnet50_only.csv` | After stage 1 (a safety net if later stages fail) |
| `submission.csv` | End of the run (final ensemble) |
| `oof_<arch>.npy` | Out-of-fold probabilities per architecture (`resnet`, `effnet`) |
| `test_<arch>.npy` | Fold-averaged test probabilities per architecture |
| `fold_f1s_<arch>.json` | Per-fold macro F1 with mean and standard deviation |

At the end, a training report is printed with per-model fold F1s, ensemble members and weights, the ensemble OOF macro F1, the predicted class distribution on the test set, and the total wall-clock time.

---

## Configuration Reference

All settings are constants at the top of `train.py`.

| Setting | Default | Notes |
|---------|---------|-------|
| `DATA_DIR` | `"."` | Folder containing the input files |
| `IMG_SIZE` | 224 | Input resolution |
| `N_FOLDS` | 5 | Capped at the number of unique regions |
| `SEED` | 42 | Random seed |
| `WALL_CLOCK_BUDGET_H` | 8.0 | Used only for the stage-2 skip decision |
| `RN50_EPOCHS` / `RN50_BATCH` | 12 / 32 | ResNet-50 |
| `RN50_LR_HEAD` / `RN50_LR_BACKBONE` | 1e-3 / 1e-4 | ResNet-50 learning rates |
| `EFF_EPOCHS` / `EFF_BATCH` | 12 / 32 | EfficientNet-B0 |
| `EFF_LR_HEAD` / `EFF_LR_BACKBONE` | 1e-3 / 1e-4 | EfficientNet-B0 learning rates |
| `LABEL_SMOOTHING` | 0.05 | Loss smoothing |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `CLASS_WEIGHT_CAP` | 3.0 | Upper bound on class weights |
| `WARMUP_EPOCHS` | 1 | Epochs with the backbone frozen |
| `LOG_EVERY` | 100 | Batches between progress log lines |

---

## Time Budget Behaviour

The budget check happens only **between stages**, not inside them. Stage 2 (EfficientNet-B0) starts only if the elapsed time is under `WALL_CLOCK_BUDGET_H - 3` hours (5 hours by default). A stage that starts just before the cutoff can still run past the nominal budget, so leave headroom when choosing the value.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| CUDA out of memory | The script retries once with half the batch size. If it still fails, lower `RN50_BATCH` / `EFF_BATCH` manually. |
| Wrong or odd NDVI values | Your band order differs from Red at index 2 and NIR at index 3. Edit `add_ndvi`. |
| All tiles fall into one fold or fewer folds than expected | The `Id` format does not match `GEO_R<number>_`, so tiles get the fallback region `R00`. Adjust `extract_region`. |
| Very slow training | Confirm that `DEVICE` is `cuda`. Data loading uses only 2 workers, so a fast CPU or more workers may help. |
| Weights fail to download | The first run needs internet access for the torchvision weights. |
| Stage 2 skipped | Stage 1 used too much of the budget. Raise `WALL_CLOCK_BUDGET_H` or reduce `RN50_EPOCHS`. |

---

## Notes (P.S.)

**On the results**

- The 0.89 score was measured on the training dataset and 0.86415 on the Kaggle test set. A small gap like this is a good sign, but it is one data point, and Kaggle's public and private leaderboards can differ.
- Training-side scores come from out-of-fold validation, where the best epoch is chosen using the same fold that is scored. This makes them slightly optimistic, and the Kaggle number is the more trustworthy of the two.

**On model choices**

- ResNet-50 is the main model and carries the result. EfficientNet-B0 is an optional second opinion whose value depends on whether it scores close enough to ResNet-50 to pass the 60% ensemble filter.
- A DINOv2 frozen-backbone variant was tried during development and did not train successfully, so it has been removed from the pipeline. If you want to revisit it, the likely issue is that a frozen ImageNet-style backbone adapts poorly to multi-band satellite input when the extra channels are never trained.
- The extra input channels are initialised from the mean of the RGB filters. This is a simple heuristic and works well here, but learning those filters properly (by unfreezing the backbone) is what gives the gains.

**On reproducibility**

- Seeds are fixed (`SEED = 42`), but `cudnn.benchmark = True` and multi-worker data loading mean two runs will not match bit for bit. Expect small differences in F1 between runs.
- Fold assignments are deterministic given the same data, since `GroupKFold` does not shuffle.

**On limitations**

- Per-tile normalisation removes absolute brightness information. This helps across regions but may discard signal if absolute reflectance matters for some classes.
- Resizing to 224x224 with bicubic interpolation changes the effective resolution. Very small tiles are upsampled, which adds no new information.
- The 60% ensemble filter and F1-proportional weights are simple rules. Weights tuned on the OOF predictions could squeeze out a little more, at the risk of overfitting.

**Ideas for improvement**

- Train more seeds or more epochs of ResNet-50 and average them, since it is the strongest model.
- Try a larger backbone such as ResNet-101 or ConvNeXt, or a satellite-pretrained backbone.
- Add mixup or CutMix, and stronger colour or spectral augmentation, to improve robustness across regions.
- Use more spectral indices (for example NDWI or NDBI) alongside NDVI.
- Tune ensemble weights on OOF predictions and consider threshold or prior adjustment for rare classes.
- Add pseudo-labelling on the test set to adapt to the new regions.