"""
GeoShift — overnight training.
Main horse: ResNet-50 (fine-tuned).
Second:     EfficientNet-B0 (only if wall-clock budget allows).

Adds a thermal watchdog: if GPU temp exceeds TEMP_LIMIT_C, training aborts.
Safety submissions written after every stage, so you never lose everything.
"""

import os
import re
import gc
import json
import time
import threading
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "."
TRAIN_CSV   = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMAGES= os.path.join(DATA_DIR, "train_images.npy")
TEST_CSV    = os.path.join(DATA_DIR, "test.csv")
TEST_IMAGES = os.path.join(DATA_DIR, "test_images.npy")
SAMPLE_SUB  = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_SUB     = os.path.join(DATA_DIR, "submission.csv")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
N_FOLDS    = 5
SEED       = 42
WALL_CLOCK_BUDGET_H = 8.0

# Thermal watchdog
TEMP_LIMIT_C    = 82    # if GPU temp >= this, abort training
TEMP_CHECK_SEC  = 10    # check every 10 seconds

# ResNet-50
RN50_EPOCHS      = 12
RN50_BATCH       = 32
RN50_LR_HEAD     = 1e-3
RN50_LR_BACKBONE = 1e-4

# EfficientNet-B0
EFF_EPOCHS       = 12
EFF_BATCH        = 32
EFF_LR_HEAD      = 1e-3
EFF_LR_BACKBONE  = 1e-4

# Regularisation
LABEL_SMOOTHING  = 0.05
WEIGHT_DECAY     = 1e-4
CLASS_WEIGHT_CAP = 3.0
WARMUP_EPOCHS    = 1

LOG_EVERY = 100

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

START_TIME = time.time()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg):
    elapsed = time.time() - START_TIME
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)


def hours_elapsed():
    return (time.time() - START_TIME) / 3600.0


def clear_gpu():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_report(tag=""):
    if DEVICE.type != "cuda":
        return
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"[GPU {tag}] alloc={a:.2f}GB reserved={r:.2f}GB total={t:.2f}GB")


# ---------------------------------------------------------------------------
# Thermal watchdog — runs in background thread
# ---------------------------------------------------------------------------
class ThermalAbort(Exception):
    pass


_WATCHDOG_STOP = threading.Event()


def _read_gpu_temp():
    """Return current GPU temp in °C, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def _thermal_watchdog():
    """Background thread: check temp every TEMP_CHECK_SEC. Abort if too hot."""
    over_limit_count = 0
    while not _WATCHDOG_STOP.is_set():
        t = _read_gpu_temp()
        if t is not None:
            if t >= TEMP_LIMIT_C:
                over_limit_count += 1
                log(f"[watchdog] GPU temp {t}°C >= {TEMP_LIMIT_C}°C "
                    f"(hit #{over_limit_count})")
                if over_limit_count >= 2:      # require 2 consecutive hits
                    log(f"[watchdog] ABORTING: sustained temp >= {TEMP_LIMIT_C}°C")
                    os._exit(42)               # hard exit, kills all threads
            else:
                over_limit_count = 0
        _WATCHDOG_STOP.wait(TEMP_CHECK_SEC)


def start_watchdog():
    if DEVICE.type != "cuda":
        return
    th = threading.Thread(target=_thermal_watchdog, daemon=True)
    th.start()
    log(f"[watchdog] Started. Limit={TEMP_LIMIT_C}°C, check every "
        f"{TEMP_CHECK_SEC}s (2 consecutive hits required).")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def extract_region(id_str):
    m = re.search(r"GEO_(R\d+)_", id_str)
    return m.group(1) if m else "R00"


def to_nhwc(arr):
    if arr.ndim == 4 and arr.shape[1] <= 8 and arr.shape[1] < arr.shape[-1]:
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


def add_ndvi(images):
    images = images.astype(np.float32)
    red = images[..., 2]
    nir = images[..., 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.concatenate([images, ndvi[..., None]], axis=-1)


def normalize_per_tile(img):
    mean = img.reshape(-1, img.shape[-1]).mean(axis=0)
    std  = img.reshape(-1, img.shape[-1]).std(axis=0) + 1e-6
    return (img - mean) / std


class TileDataset(Dataset):
    def __init__(self, images, labels=None, train=True):
        self.images = images
        self.labels = labels
        self.train  = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        img = normalize_per_tile(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                            mode="bicubic", align_corners=False).squeeze(0)

        if self.train:
            if np.random.rand() < 0.5:
                img = torch.flip(img, dims=[2])
            if np.random.rand() < 0.5:
                img = torch.flip(img, dims=[1])
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, dims=[1, 2])
            if np.random.rand() < 0.3:
                gain = 1.0 + 0.1 * (torch.rand(img.shape[0], 1, 1) - 0.5)
                img = img * gain

        if self.labels is not None:
            return img, self.labels[idx]
        return img


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def get_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)
    w = w / w.sum() * num_classes
    w = np.clip(w, None, CLASS_WEIGHT_CAP)
    log(f"[weights] counts={counts.tolist()}")
    log(f"[weights] final ={[round(float(x), 4) for x in w]}")
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_resnet50(in_channels, num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    old = m.conv1
    new = nn.Conv2d(in_channels, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        rgb = old.weight
        new.weight[:, :3] = rgb
        mean_w = rgb.mean(dim=1, keepdim=True)
        for c in range(in_channels - 3):
            new.weight[:, 3 + c] = mean_w[:, 0]
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_efficientnet_b0(in_channels, num_classes):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    old = m.features[0][0]
    new = nn.Conv2d(in_channels, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        rgb = old.weight
        new.weight[:, :3] = rgb
        mean_w = rgb.mean(dim=1, keepdim=True)
        for c in range(in_channels - 3):
            new.weight[:, 3 + c] = mean_w[:, 0]
    m.features[0][0] = new
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


# ---------------------------------------------------------------------------
# Forward wrapper / head helpers
# ---------------------------------------------------------------------------
def forward_logits(model, x, arch):
    return model(x)


def head_module(model, arch):
    return model.fc if arch == "resnet" else model.classifier[1]


def head_name(arch):
    # Parameter-name prefix of the classification head.
    # (EfficientNet's SE blocks contain "fc1"/"fc2", so match "classifier".)
    return "fc." if arch == "resnet" else "classifier"


# ---------------------------------------------------------------------------
# Training — one fold
# ---------------------------------------------------------------------------
def train_fold(fold, tr_imgs, tr_labs, val_imgs, val_labs,
               num_classes, in_channels, arch, batch_size,
               epochs, lr_head, lr_backbone, warmup, tta_flips):
    log("=" * 70)
    log(f"[{arch} FOLD {fold}] train={len(tr_imgs)} val={len(val_imgs)} "
        f"bs={batch_size} epochs={epochs}")

    if arch == "resnet":
        model = build_resnet50(in_channels, num_classes).to(DEVICE)
    elif arch == "effnet":
        model = build_efficientnet_b0(in_channels, num_classes).to(DEVICE)
    else:
        raise ValueError(arch)

    gpu_report(f"{arch}-f{fold}-built")

    class_weights = get_class_weights(tr_labs, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=LABEL_SMOOTHING)

    for n, p in model.named_parameters():
        if head_name(arch) not in n:
            p.requires_grad = False
    log(f"[{arch} FOLD {fold}] backbone frozen for {warmup} epoch(s)")

    def make_optimizer():
        head_p = [p for n, p in model.named_parameters()
                  if head_name(arch) in n and p.requires_grad]
        bb_p = [p for n, p in model.named_parameters()
                if head_name(arch) not in n and p.requires_grad]
        groups = [{"params": head_p, "lr": lr_head}]
        if bb_p:
            groups.append({"params": bb_p, "lr": lr_backbone})
        return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

    optimizer = make_optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(TileDataset(tr_imgs, tr_labs, True),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(TileDataset(val_imgs, val_labs, False),
                            batch_size=batch_size, shuffle=False,
                            num_workers=2,
                            pin_memory=(DEVICE.type == "cuda"))

    log(f"[{arch} FOLD {fold}] batches/epoch={len(train_loader)} "
        f"val_batches={len(val_loader)}")

    best_f1, best_state = -1.0, None

    for epoch in range(epochs):
        if epoch == warmup:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = make_optimizer()
            rem = max(1, epochs - warmup)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=rem)
            log(f"[{arch} FOLD {fold}] backbone UNFROZEN at epoch {epoch+1}")

        model.train()
        run_loss, nb = 0.0, 0
        for bi, (imgs, labs) in enumerate(train_loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out  = forward_logits(model, imgs, arch)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            run_loss += loss.item(); nb += 1
            if (bi + 1) % LOG_EVERY == 0:
                log(f"[{arch} FOLD {fold}] ep {epoch+1}/{epochs} "
                    f"b {bi+1}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = run_loss / max(nb, 1)

        model.eval()
        preds_all, labs_all = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                out  = forward_logits(model, imgs, arch)
                preds_all.extend(out.argmax(dim=1).cpu().numpy())
                labs_all.extend(labs.numpy())

        f1 = f1_score(labs_all, preds_all, average="macro")
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  <-- BEST"
        log(f"[{arch} FOLD {fold}] ep {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} val_f1={f1:.4f}{marker}")

    log(f"[{arch} FOLD {fold}] DONE. best_val_f1={best_f1:.4f}")
    model.load_state_dict(best_state)
    return model, best_f1


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def predict_tta(model, images, batch_size, arch, flips):
    model.eval()
    loader = DataLoader(TileDataset(images, None, False),
                        batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    out_list = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            nc = head_module(model, arch).out_features
            probs_sum = torch.zeros(imgs.size(0), nc, device=DEVICE)
            for fh, fv in flips:
                x = imgs
                if fh: x = torch.flip(x, dims=[3])
                if fv: x = torch.flip(x, dims=[2])
                out = forward_logits(model, x, arch)
                probs_sum += F.softmax(out, dim=1)
            probs_sum /= len(flips)
            out_list.append(probs_sum.cpu().numpy())
    return np.concatenate(out_list, axis=0)


# ---------------------------------------------------------------------------
# Full architecture runner (5-fold CV)
# ---------------------------------------------------------------------------
def run_architecture(arch, train_images, labels, test_images,
                     num_classes, in_channels, splits,
                     batch_size, epochs, lr_head, lr_backbone, warmup):
    tta_flips = [(False, False), (True, False), (False, True), (True, True)]
    oof  = np.zeros((len(train_images), num_classes), dtype=np.float32)
    test = np.zeros((len(test_images), num_classes), dtype=np.float32)
    fold_f1s = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        log(f"\n[{arch}] ====== FOLD {fold} / {len(splits)} ======")
        tr_imgs, tr_labs = train_images[tr_idx], labels[tr_idx]
        va_imgs, va_labs = train_images[val_idx], labels[val_idx]

        try:
            model, f1 = train_fold(fold, tr_imgs, tr_labs, va_imgs, va_labs,
                                   num_classes, in_channels, arch, batch_size,
                                   epochs, lr_head, lr_backbone, warmup, tta_flips)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_gpu()
                new_bs = max(4, batch_size // 2)
                log(f"[{arch}] OOM at bs={batch_size} → retry bs={new_bs}")
                model, f1 = train_fold(fold, tr_imgs, tr_labs, va_imgs, va_labs,
                                       num_classes, in_channels, arch, new_bs,
                                       epochs, lr_head, lr_backbone, warmup,
                                       tta_flips)
                batch_size = new_bs
            else:
                raise

        fold_f1s.append(f1)
        log(f"[{arch}] Fold {fold} F1 = {f1:.4f}")

        oof[val_idx] = predict_tta(model, va_imgs, batch_size, arch, tta_flips)
        test += predict_tta(model, test_images, batch_size, arch, tta_flips)

        del model
        clear_gpu()

    test /= len(splits)
    log(f"\n[{arch}] Fold F1s: {[round(f, 4) for f in fold_f1s]}")
    log(f"[{arch}] Mean CV macro F1: {np.mean(fold_f1s):.4f} "
        f"(± {np.std(fold_f1s):.4f})")

    np.save(f"oof_{arch}.npy", oof)
    np.save(f"test_{arch}.npy", test)
    with open(f"fold_f1s_{arch}.json", "w") as fh:
        json.dump({"fold_f1s": fold_f1s,
                   "mean": float(np.mean(fold_f1s)),
                   "std":  float(np.std(fold_f1s))}, fh, indent=2)

    return oof, test, fold_f1s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("GeoShift overnight training (thermal-limited)")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name}  VRAM={p.total_memory/1024**3:.2f}GB")

    # ---- Thermal watchdog --------------------------------------------------
    start_watchdog()

    # ---- Data --------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    log(f"train_df {train_df.shape}  test_df {test_df.shape}")

    train_images = add_ndvi(to_nhwc(np.load(TRAIN_IMAGES)))
    test_images  = add_ndvi(to_nhwc(np.load(TEST_IMAGES)))
    in_channels  = train_images.shape[-1]

    labels      = train_df["label"].values.astype(np.int64)
    num_classes = int(labels.max()) + 1
    log(f"in_channels={in_channels}  num_classes={num_classes}")

    groups = np.array([extract_region(i) for i in train_df["Id"].values])
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(train_images, labels, groups))
    for f, (_, va) in enumerate(splits):
        log(f"  fold {f}: val regions = {sorted(np.unique(groups[va]).tolist())}")

    # ---- STAGE 1: ResNet-50 -----------------------------------------------
    log("\n" + "#" * 70)
    log("# STAGE 1: ResNet-50 (main horse)")
    log("#" * 70)
    oof_rn, test_rn, f1_rn = run_architecture(
        "resnet", train_images, labels, test_images,
        num_classes, in_channels, splits,
        batch_size=RN50_BATCH, epochs=RN50_EPOCHS,
        lr_head=RN50_LR_HEAD, lr_backbone=RN50_LR_BACKBONE,
        warmup=WARMUP_EPOCHS)

    sub_df = pd.read_csv(SAMPLE_SUB)
    sub_df["label"] = test_rn.argmax(axis=1)
    sub_df.to_csv("submission_resnet50_only.csv", index=False)
    log("[stage1] Wrote submission_resnet50_only.csv (safety net)")

    models_ran = {"resnet": {"oof": oof_rn, "test": test_rn, "f1": f1_rn}}

    # ---- STAGE 2: EfficientNet-B0 -----------------------------------------
    if hours_elapsed() < WALL_CLOCK_BUDGET_H - 3.0:
        log("\n" + "#" * 70)
        log("# STAGE 2: EfficientNet-B0 (time permitting)")
        log("#" * 70)
        try:
            oof_e, test_e, f1_e = run_architecture(
                "effnet", train_images, labels, test_images,
                num_classes, in_channels, splits,
                batch_size=EFF_BATCH, epochs=EFF_EPOCHS,
                lr_head=EFF_LR_HEAD, lr_backbone=EFF_LR_BACKBONE,
                warmup=WARMUP_EPOCHS)
            models_ran["effnet"] = {"oof": oof_e, "test": test_e, "f1": f1_e}
        except Exception as e:
            log(f"[stage2] EffNet FAILED: {type(e).__name__}: {e}")
    else:
        log("[stage2] Skipping EffNet — budget check failed.")

    # ---- Ensemble ----------------------------------------------------------
    log("\n" + "#" * 70)
    log("# ENSEMBLE")
    log("#" * 70)
    means = {a: float(np.mean(v["f1"])) for a, v in models_ran.items()}
    for a, m in means.items():
        log(f"  {a}: mean CV F1 = {m:.4f}")

    best_mean = max(means.values())
    keep = {a for a, m in means.items() if m >= 0.6 * best_mean}
    log(f"  Keeping for ensemble: {sorted(keep)}")

    wsum = sum(means[a] for a in keep)
    weights = {a: means[a] / wsum for a in keep}
    log(f"  Ensemble weights: "
        f"{ {a: round(w, 3) for a, w in weights.items()} }")

    oof_ens = np.zeros_like(next(iter(models_ran.values()))["oof"])
    test_ens = np.zeros_like(next(iter(models_ran.values()))["test"])
    for a in keep:
        oof_ens  += weights[a] * models_ran[a]["oof"]
        test_ens += weights[a] * models_ran[a]["test"]

    oof_ens_f1 = f1_score(labels, oof_ens.argmax(axis=1), average="macro")
    log(f"  Ensemble OOF macro F1 = {oof_ens_f1:.4f}")

    final_preds = test_ens.argmax(axis=1)

    # ---- Submission --------------------------------------------------------
    sub_df = pd.read_csv(SAMPLE_SUB)
    sub_df["label"] = final_preds
    sub_df.to_csv(OUT_SUB, index=False)
    log(f"[submit] Wrote {OUT_SUB}")

    # ---- Final report ------------------------------------------------------
    print("=" * 58)
    print("GeoShift — Training Report")
    print("=" * 58)
    print(f"Device: {DEVICE}   Folds: {n_splits}   "
          f"in_ch: {in_channels}   classes: {num_classes}")
    print("-" * 58)
    for a in ["resnet", "effnet"]:
        if a in models_ran:
            f1s = models_ran[a]["f1"]
            print(f"{a:8s} fold F1s: " + ", ".join(f"{f:.4f}" for f in f1s))
            print(f"{'':8s} mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    print("-" * 58)
    print(f"Ensemble members: {sorted(keep)}")
    print(f"Ensemble weights: { {a: round(weights[a], 3) for a in keep} }")
    print(f"Ensemble OOF macro F1: {oof_ens_f1:.4f}")
    print("-" * 58)
    print("Test predictions distribution:")
    uniq, cnt = np.unique(final_preds, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  class {u}: {c}")
    print("-" * 58)
    print(f"Submission written to: {OUT_SUB}")
    print(f"Total wall clock: {hours_elapsed():.2f} h")
    print("=" * 58)

    # Stop watchdog cleanly
    _WATCHDOG_STOP.set()


if __name__ == "__main__":
    main()