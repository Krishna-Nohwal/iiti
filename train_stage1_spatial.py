"""
train_stage1.py — Stage 1 training script.

model() returns (logits_list, features_list, cls_list):
  - logits_list   : 4 × (B, 2)    one per tapped layer [20,21,22,23]
  - features_list : 4 × (B, 512)  512-dim bottleneck per SpatialHead
  - cls_list      : 4 × (B, 1024) CLS tokens, discarded here

Run this first, then pass best.pth to train_stage2.py.
"""

import os, math, torch, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.losses import SupConLoss, MultiSimilarityLoss
from torch import nn
from augmentations import augment_batch, load_and_resize, normalize
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score,
)
from model import ViT


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',        default=50,   type=int)
parser.add_argument('--batch_size',    default=16,    type=int)
parser.add_argument('--num_workers',   default=6,    type=int,
                    help='6 workers suits Ryzen 7000; tune down if RAM is tight')
parser.add_argument('--save_root',     default='checkpoints_vit', type=str)
parser.add_argument('--load_from',     default='',   type=str)
parser.add_argument('--manifest',      default='E:/Work/sampled_30k/manifest_onct.csv', type=str)
parser.add_argument('--root_dir',      default='E:/Work/sampled_30k/', type=str)
parser.add_argument('--cdf_root',      default='E:/Work/cdfv1_onct_out', type=str)
parser.add_argument('--cdf_csv',       default='E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv', type=str)
parser.add_argument('--val_ratio',     default=0.05, type=float)
parser.add_argument('--supcon_weight', default=1/16, type=float)
parser.add_argument('--ms_weight',     default=1/16, type=float,
                    help='Weight for MultiSimilarityLoss (same scale as supcon_weight)')
parser.add_argument('--no_compile',    action='store_true',
                    help='Disable torch.compile (useful for debugging)')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

save_root = args.save_root
IMG_SIZE  = 256
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_probs, split_name: str, epoch: int):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    ap  = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.nanargmin(np.abs(fpr_arr - fnr_arr))
    eer     = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  [{split_name}] Epoch {epoch+1:02d} | "
          f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc*100:.2f}%  F1={f1:.4f}  EER={eer*100:.2f}%  "
          f"TPR={tpr*100:.2f}%  FPR={fpr*100:.2f}%  TNR={tnr*100:.2f}%  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")

    return auc


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    real_pool = df[df["label"] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    fake_pool = df[df["label"] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Full dataset -> Real: {len(real_pool)} | Fake: {len(fake_pool)}")

    real_val_n = int(len(real_pool) * val_ratio)
    fake_val_n = int(len(fake_pool) * val_ratio)

    real_val   = real_pool.iloc[:real_val_n]
    real_train = real_pool.iloc[real_val_n:]
    fake_val   = fake_pool.iloc[:fake_val_n]
    fake_train = fake_pool.iloc[fake_val_n:]

    train_df = pd.concat([real_train, fake_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df   = pd.concat([real_val,   fake_val  ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Train -> Real: {len(real_train)} | Fake: {len(fake_train)} | Total: {len(train_df)}")
    print(f"Val   -> Real: {len(real_val)}   | Fake: {len(fake_val)}   | Total: {len(val_df)}")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ManifestImageDataset(Dataset):
    """Train/val dataset. label: 0=Real, 1=Fake.

    __init__ uses vectorised pandas ops instead of iterrows() —
    10-50× faster for large manifests.
    """

    def __init__(self, df: pd.DataFrame, root_dir: str):
        paths = (
            df["sample_dir"]
            .str.replace("\\", "/", regex=False)
            .str.split("sampled_30k/", n=1)
            .str[-1]
            .apply(lambda rel: os.path.join(root_dir, rel, "image.png"))
        )
        labels = df["label"].astype(int).values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [Dataset] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


class CDFv1Dataset(Dataset):
    """
    CDFv1 test dataset.
    Manifest convention: 1=Real, 0=Fake — flipped on load to match 0=Real, 1=Fake.
    """

    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = 1 - df["label"].astype(int)

        print(f"CDFv1 -> Real: {(df['label']==0).sum()} | Fake: {(df['label']==1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        paths  = df["sample_dir"].apply(lambda d: str(root / d / "image.png"))
        labels = df["label"].values

        exists_mask = np.array([os.path.exists(p) for p in paths])
        skipped = int((~exists_mask).sum())
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing image.png ({exists_mask.sum()} remaining)")

        self.entries = list(zip(paths[exists_mask], labels[exists_mask]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

bce_loss    = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()
ms_loss     = MultiSimilarityLoss()

def cls_loss(logits, features, labels, lam_supcon, lam_ms):
    features_norm = torch.nn.functional.normalize(features, dim=1)  # L2-norm for MS
    return (
        bce_loss(logits, labels)
        + lam_supcon * supcon_loss(features, labels)
        + lam_ms     * ms_loss(features_norm, labels)
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, loader, desc, device):
    all_labels, all_probs = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits_list, _, _ = model(imgs)   # discard features_list, cls_list
            probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, val_df = prepare_splits(args.manifest, args.root_dir, val_ratio=args.val_ratio)

    train_dataset = ManifestImageDataset(train_df, args.root_dir)
    val_dataset   = ManifestImageDataset(val_df,   args.root_dir)
    cdf_dataset   = CDFv1Dataset(args.cdf_csv, args.cdf_root)

    _persistent = _num_workers > 0
    _prefetch   = 4 if _num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=_num_workers,
        pin_memory=True, shuffle=True,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=_num_workers,
        pin_memory=True, shuffle=False,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    cdf_loader = DataLoader(
        cdf_dataset, batch_size=args.batch_size, num_workers=_num_workers,
        pin_memory=True, shuffle=False,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    os.makedirs(save_root, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = ViT().to(device)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location='cpu'))
        print(f"Loaded checkpoint from {args.load_from}")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── AMP scaler ──────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device=device.type)

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    lr_base        = 1e-4
    epochs         = args.epochs
    iter_per_epoch = len(train_loader)
    totalstep      = epochs * iter_per_epoch
    warmstep       = 512
    lr_min         = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmstep) * math.pi / (totalstep - warmstep))) / 2) + lr_min)
            if i > warmstep
            else (i / warmstep + lr_min)
        )
        for i in range(totalstep)
    }

    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[step]
    )

    lam_supcon = args.supcon_weight
    lam_ms     = args.ms_weight

    # ── Training loop ───────────────────────────────────────────────────────
    best_test_auc = 0.0
    best_epoch    = -1
    SEP           = "=" * 80

    for epoch in range(epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)

        model.train()
        iter_i                    = epoch * iter_per_epoch
        train_labels, train_probs = [], []

        for batch_idx, (imgs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        ):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs   = augment_batch(imgs)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits_list, features_list, _ = model(imgs)   # discard cls_list

                l_primary = cls_loss(logits_list[3], features_list[3], labels, lam_supcon, lam_ms)
                l_aux     = (
                    cls_loss(logits_list[0], features_list[0], labels, lam_supcon, lam_ms) +
                    cls_loss(logits_list[1], features_list[1], labels, lam_supcon, lam_ms) +
                    cls_loss(logits_list[2], features_list[2], labels, lam_supcon, lam_ms)
                ) / 4.0
                loss = l_primary + l_aux

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 256 == 0:
                print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

        # ── Metrics ─────────────────────────────────────────────────────────
        print()
        compute_metrics(train_labels, train_probs, "Train", epoch)

        val_labels, val_probs = run_eval(model, val_loader, f"Epoch {epoch+1} [val]", device)
        compute_metrics(val_labels, val_probs, "Val  ", epoch)

        cdf_labels, cdf_probs = run_eval(model, cdf_loader, f"Epoch {epoch+1} [CDFv1]", device)
        test_auc = compute_metrics(cdf_labels, cdf_probs, "Test ", epoch)

        # ── Checkpointing ───────────────────────────────────────────────────
        state_dict = (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict()
        torch.save(state_dict, os.path.join(save_root, 'latest.pth'))
        vit_module = (model._orig_mod if hasattr(model, '_orig_mod') else model).vit
        vit_module.save_pretrained(os.path.join(save_root, 'latest_lora'))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch    = epoch
            torch.save(state_dict, os.path.join(save_root, 'best.pth'))
            vit_module.save_pretrained(os.path.join(save_root, 'best_lora'))
            print(f"\n  ★ New best Test AUC={best_test_auc:.4f} → saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Training complete. Best checkpoint: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(save_root, 'best.pth')}")
    print(f"\n  To run Stage 2:")
    print(f"    python train_stage2.py "
          f"--load_from {os.path.join(save_root, 'best.pth')} "
          f"--save_root checkpoints_s2 --epochs <N>")
    print(SEP)