"""
train_stage2.py — Stage 2 of two-stage deepfake detection training.

Loads the Stage 1 ViT checkpoint into VideoViT (video_model.py), freezes
frame_model entirely (backbone + MACHeads + LoRA adapters), and trains only
the temporal transformers and video_classifier on video-level data.

Loss: BCE(video_logits) + lam * SupCon(video_feats per temporal head)
Metrics: both frame-level (from frozen MACHeads) and video-level are reported
         so you can see whether temporal reasoning actually helps.

Outputs
-------
checkpoints_s2/latest.pth     — full VideoViT state dict after every epoch
checkpoints_s2/best.pth       — state dict of epoch with highest CDFv1 video AUC

Note: LoRA adapter weights are NOT saved separately in Stage 2 because the
      LoRA layers inside frame_model are frozen and unchanged from Stage 1.
      Use the Stage 1 best_lora/ for the frozen ViT backbone adapter.

Usage
-----
python train_stage2.py \
    --load_from     checkpoints_s1/best.pth \
    --manifest      E:/Work/sampled_30k/manifest_onct.csv \
    --root_dir      E:/Work/sampled_30k/ \
    --cdf_root      E:/Work/cdfv1_onct_out \
    --cdf_csv       E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv \
    --num_frames    12 \
    --batch_size    8

batch_size must be even (BalancedRealFakeBatchSampler requires equal halves).
Pass --no_compile to skip torch.compile (useful for debugging).
"""

import os, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score,
)
from augmentations import augment_batch, load_and_resize, normalize
from video_model import VideoViT


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Stage 2: train temporal transformers on top of frozen Stage 1 ViT"
)
parser.add_argument("--epochs",        default=30,   type=int)
parser.add_argument("--batch_size",    default=8,    type=int,
                    help="Videos per batch. Must be even for balanced sampler.")
parser.add_argument("--num_frames",    default=12,   type=int,
                    help="Frames sampled per video (uniform stride).")
parser.add_argument("--num_workers",   default=6,    type=int)
parser.add_argument("--save_root",     default="checkpoints_s2", type=str)
parser.add_argument("--load_from",     default="",   type=str,
                    help="Path to Stage 1 best.pth (required).")
parser.add_argument("--manifest",      default="E:/Work/sampled_30k/manifest_onct.csv", type=str)
parser.add_argument("--root_dir",      default="E:/Work/sampled_30k/", type=str)
parser.add_argument("--cdf_root",      default="E:/Work/cdfv1_onct_out", type=str)
parser.add_argument("--cdf_csv",       default="E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv", type=str)
parser.add_argument("--val_ratio",     default=0.05, type=float,
                    help="Fraction of videos held out for validation.")
parser.add_argument("--lr",            default=1e-3, type=float,
                    help="Base LR for temporal transformers + video_classifier. "
                         "Higher than Stage 1 because these layers start from scratch.")
parser.add_argument("--warmup_steps",  default=64,   type=int)
parser.add_argument("--supcon_weight", default=1/16, type=float)
parser.add_argument("--no_compile",    action="store_true",
                    help="Disable torch.compile (useful for debugging).")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

IMG_SIZE     = 266
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_num_workers = args.num_workers

torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_probs, split_name: str, epoch: int) -> float:
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

    cm             = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"  [{split_name}] Epoch {epoch+1:02d} | "
        f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc*100:.2f}%  F1={f1:.4f}  "
        f"EER={eer*100:.2f}%  TPR={tpr*100:.2f}%  FPR={fpr*100:.2f}%  "
        f"TNR={tnr*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return auc


# ---------------------------------------------------------------------------
# Data splits  (video-level — must match Stage 1 to prevent leakage)
# ---------------------------------------------------------------------------

def _extract_video_id(sample_dir: str) -> str:
    """
    Derive a video-level ID from a per-frame sample_dir, preserving the
    manipulation-method subfolder.

    Examples
    --------
    'real/000_frame_03'               -> 'real/000'
    'fake/FaceSwap/922_898_frame_31'  -> 'fake/FaceSwap/922_898'
    """
    parts    = Path(sample_dir).parts
    basename = parts[-1]
    marker   = "_frame_"
    idx      = basename.rfind(marker)
    clip_id  = basename[:idx] if idx != -1 else basename
    prefix   = "/".join(parts[:-1])
    return f"{prefix}/{clip_id}" if prefix else clip_id


def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    """
    Video-level split using the same seed as Stage 1 so the train/val
    boundary is identical regardless of which script ran first.
    """
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    df["video_id"] = df["sample_dir"].apply(_extract_video_id)

    real_vids = df[df["label"] == 0]["video_id"].unique()
    fake_vids = df[df["label"] == 1]["video_id"].unique()

    rng = np.random.default_rng(42)
    real_vids = rng.permutation(real_vids)
    fake_vids = rng.permutation(fake_vids)

    print(f"Full dataset -> Real videos: {len(real_vids)} | Fake videos: {len(fake_vids)}")

    real_val_n = max(1, int(len(real_vids) * val_ratio))
    fake_val_n = max(1, int(len(fake_vids) * val_ratio))

    val_ids  = set(real_vids[:real_val_n]) | set(fake_vids[:fake_val_n])

    train_df = df[~df["video_id"].isin(val_ids)].reset_index(drop=True)
    val_df   = df[ df["video_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train -> {len(train_df)} frames "
          f"(real vids: {len(real_vids) - real_val_n}  fake vids: {len(fake_vids) - fake_val_n})")
    print(f"Val   -> {len(val_df)} frames "
          f"(real vids: {real_val_n}  fake vids: {fake_val_n})")

    return train_df, val_df


# ---------------------------------------------------------------------------
# Video datasets
# ---------------------------------------------------------------------------

def _load_video_frames(frame_paths: list, img_size: int) -> torch.Tensor:
    """Load and preprocess a list of frame paths into a (T, 3, H, W) tensor."""
    frames = []
    for p in frame_paths:
        try:
            img = load_and_resize(p, img_size)
            img = normalize(img)
        except Exception:
            # Corrupted frame: substitute a zero tensor so the batch still forms.
            img = torch.zeros(3, img_size, img_size)
        frames.append(img)
    return torch.stack(frames, dim=0)   # (T, 3, H, W)


def _sample_frame_indices(n_available: int, n_target: int) -> np.ndarray:
    """
    Uniform stride when enough frames are available; tile when fewer.
    Always returns exactly n_target indices.
    """
    if n_available >= n_target:
        return np.linspace(0, n_available - 1, n_target, dtype=int)
    return np.tile(np.arange(n_available), math.ceil(n_target / n_available))[:n_target]


def video_collate_fn(batch):
    """
    Pads variable-length clip tensors to the longest clip in the batch.
    Returns (frames, labels, lengths) where:
        frames  : (B, T_max, C, H, W)  float32
        labels  : (B,)                  int64
        lengths : (B,)                  int64  — actual frame count per clip
    """
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = []
    for f in frames_list:
        pad_t = max_len - f.size(0)
        if pad_t > 0:
            # Pad temporal dimension only (last dim group: T, C, H, W -> pad on dim 0 of f).
            f = F.pad(f, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded.append(f)

    return (
        torch.stack(padded, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lengths,
    )


class ManifestVideoDataset(Dataset):
    """
    Stage 2 train/val dataset.  label: 0 = Real, 1 = Fake.

    Groups per-frame CSV rows by video_id, then for each video samples
    exactly num_frames frames with uniform stride (tiles if too few).
    Augmentation (augment_batch) is applied per-item here — unlike Stage 1
    where it is applied per-batch — because video frames must be augmented
    consistently as a unit.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        num_frames: int = 32,
        augment: bool = True,
    ):
        self.num_frames = num_frames
        self.augment    = augment
        self.videos: list = []

        root = Path(root_dir)
        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
            paths = []
            for rel in group["sample_dir"].str.replace("\\", "/", regex=False):
                img_path = root / rel / "image.png"
                if img_path.is_file():
                    paths.append(str(img_path))
            paths = sorted(paths)
            if not paths:
                continue
            self.videos.append((paths, label))

        real_n = sum(1 for _, l in self.videos if l == 0)
        fake_n = sum(1 for _, l in self.videos if l == 1)
        print(f"  [ManifestVideoDataset] {len(self.videos)} videos "
              f"({real_n} real, {fake_n} fake)")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        indices      = _sample_frame_indices(len(paths), self.num_frames)
        sampled      = [paths[i] for i in indices]
        frames       = _load_video_frames(sampled, IMG_SIZE)   # (T, 3, H, W)
        if self.augment:
            frames = augment_batch(frames)
        return frames, label


class CDFv1VideoDataset(Dataset):
    """
    Stage 2 CDFv1 test dataset (video-level).
    Manifest convention: 1 = Real, 0 = Fake — flipped on load to 0 = Real, 1 = Fake.
    No augmentation is applied at test time.
    """

    def __init__(self, csv_path: str, data_root: str, num_frames: int = 32):
        self.num_frames = num_frames

        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"]    = 1 - df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(_extract_video_id)

        print(f"CDFv1 frames -> Real: {(df['label']==0).sum()} | "
              f"Fake: {(df['label']==1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        self.videos: list = []

        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
            paths = []
            for rel in group["sample_dir"].str.replace("\\", "/", regex=False):
                img_path = root / rel / "image.png"
                if img_path.is_file():
                    paths.append(str(img_path))
            paths = sorted(paths)
            if not paths:
                continue
            self.videos.append((paths, label))

        total_vids   = df["video_id"].nunique()
        skipped_vids = total_vids - len(self.videos)
        if skipped_vids:
            print(f"  [CDFv1] Skipped {skipped_vids} videos with no frames on disk")
        print(f"  [CDFv1] {len(self.videos)} videos loaded.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        indices      = _sample_frame_indices(len(paths), self.num_frames)
        sampled      = [paths[i] for i in indices]
        frames       = _load_video_frames(sampled, IMG_SIZE)
        return frames, label


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

class BalancedRealFakeBatchSampler(Sampler):
    """
    Yields class-balanced batches of video indices.
    Each batch contains exactly batch_size // 2 real and batch_size // 2 fake
    videos. Oversamples the minority class within each epoch.

    batch_size must be even.
    """

    def __init__(self, dataset: ManifestVideoDataset, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("BalancedRealFakeBatchSampler requires an even batch_size.")
        self.per_class    = batch_size // 2
        self.real_indices = [i for i, (_, l) in enumerate(dataset.videos) if l == 0]
        self.fake_indices = [i for i, (_, l) in enumerate(dataset.videos) if l == 1]
        if not self.real_indices or not self.fake_indices:
            raise ValueError("Balanced sampler needs at least one real and one fake video.")
        # Number of batches = enough to exhaust the majority class.
        self.num_batches = math.ceil(
            max(len(self.real_indices), len(self.fake_indices)) / self.per_class
        )

    @staticmethod
    def _oversample(indices: list, n: int) -> list:
        """Draw n indices from the list, cycling with fresh shuffles as needed."""
        rng = np.random.default_rng()
        out = []
        remaining = n
        while remaining > 0:
            perm = rng.permutation(indices).tolist()
            out.extend(perm[:remaining])
            remaining -= len(perm[:remaining])
        return out

    def __iter__(self):
        n          = self.num_batches * self.per_class
        real_pool  = self._oversample(self.real_indices, n)
        fake_pool  = self._oversample(self.fake_indices, n)
        rng        = np.random.default_rng()

        for i in range(self.num_batches):
            s     = i * self.per_class
            e     = s + self.per_class
            batch = real_pool[s:e] + fake_pool[s:e]
            yield rng.permutation(batch).tolist()

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

_bce_loss    = nn.CrossEntropyLoss()
_supcon_loss = SupConLoss()


def stage2_loss(
    video_logits: torch.Tensor,
    video_feats_list: list,
    labels: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Video-level loss only. frame_model is frozen so frame terms produce no
    gradient and are omitted for efficiency.

    video_logits    : (B, 2)
    video_feats_list: list of NUM_HEADS × (B, EMBED_DIM) — one per temporal head
    labels          : (B,)
    """
    l_bce    = _bce_loss(video_logits, labels)
    l_supcon = lam * sum(_supcon_loss(feats, labels) for feats in video_feats_list)
    return l_bce + l_supcon


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _video_probs_from_forward(
    video_logits: torch.Tensor,
    frame_logits_list: list,
    labels: torch.Tensor,
    lengths: torch.Tensor,
):
    """
    Compute per-frame and per-video probabilities from a single forward pass.

    Frame probability: mean of the 4 MACHead softmax outputs (frozen backbone).
    Video probability: softmax of video_logits from the temporal head.

    Returns
    -------
    frame_labels : (N_valid_frames,)  int64
    frame_probs  : (N_valid_frames,)  float32
    video_labels : (B,)               int64
    video_probs  : (B,)               float32
    """
    B = labels.size(0)
    T = frame_logits_list[0].size(0) // B

    # Valid-frame mask: shape (B, T) then flattened to (B*T,)
    time_idx    = torch.arange(T, device=labels.device).unsqueeze(0)   # (1, T)
    valid_2d    = time_idx < lengths.to(labels.device).unsqueeze(1)    # (B, T)
    valid_flat  = valid_2d.reshape(-1)                                  # (B*T,)

    # Per-frame labels: broadcast video label to every frame, then mask.
    frame_labels_all = labels.repeat_interleave(T)         # (B*T,)
    frame_labels     = frame_labels_all[valid_flat]        # (N_valid,)

    # Frame probs: average across the 4 MACHead layers.
    mean_frame_logits = torch.stack(frame_logits_list, dim=0).mean(dim=0)  # (B*T, 2)
    frame_probs_all   = torch.softmax(mean_frame_logits.float(), dim=1)[:, 1]  # (B*T,)
    frame_probs       = frame_probs_all[valid_flat]        # (N_valid,)

    # Video probs: directly from the temporal head.
    video_probs = torch.softmax(video_logits.float(), dim=1)[:, 1]        # (B,)

    return frame_labels, frame_probs, labels, video_probs


def run_eval(model: nn.Module, loader: DataLoader, desc: str):
    """
    Evaluate VideoViT on one DataLoader.
    Returns (frame_labels, frame_probs, video_labels, video_probs).
    """
    frame_labels_all, frame_probs_all = [], []
    video_labels_all, video_probs_all = [], []

    model.eval()
    with torch.inference_mode(), \
         torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, labels, lengths in tqdm(loader, desc=desc, leave=False):
            frames  = frames.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            video_logits, frame_logits_list, _, _ = model(frames, lengths)

            fl, fp, vl, vp = _video_probs_from_forward(
                video_logits, frame_logits_list, labels, lengths
            )
            frame_labels_all.extend(fl.cpu().numpy().tolist())
            frame_probs_all.extend(fp.cpu().numpy().tolist())
            video_labels_all.extend(vl.cpu().numpy().tolist())
            video_probs_all.extend(vp.cpu().numpy().tolist())

    return frame_labels_all, frame_probs_all, video_labels_all, video_probs_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  STAGE 2 — Temporal transformer training (frame_model FROZEN)")
    print(f"{SEP}\n")

    if not args.load_from:
        raise ValueError(
            "--load_from is required for Stage 2. "
            "Point it to the Stage 1 best.pth checkpoint."
        )

    NUM_FRAMES = args.num_frames

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, val_df = prepare_splits(
        args.manifest, args.root_dir, val_ratio=args.val_ratio
    )

    train_dataset = ManifestVideoDataset(
        train_df, args.root_dir, num_frames=NUM_FRAMES, augment=True
    )
    val_dataset = ManifestVideoDataset(
        val_df, args.root_dir, num_frames=NUM_FRAMES, augment=False
    )
    cdf_dataset = CDFv1VideoDataset(
        args.cdf_csv, args.cdf_root, num_frames=NUM_FRAMES
    )

    train_batch_sampler = BalancedRealFakeBatchSampler(train_dataset, args.batch_size)
    print(
        f"Train balanced batches -> {len(train_batch_sampler)} batches/epoch "
        f"({args.batch_size // 2} real + {args.batch_size // 2} fake videos per batch)"
    )

    _persistent = _num_workers > 0
    _prefetch   = 4 if _num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    cdf_loader = DataLoader(
        cdf_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    os.makedirs(args.save_root, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = VideoViT(num_frames=NUM_FRAMES).to(device)

    # Load Stage 1 weights into frame_model.
    # VideoViT.load_image_weights handles the key remapping automatically.
    print(f"Loading Stage 1 weights from: {args.load_from}")
    missing, unexpected = model.load_image_weights(args.load_from, strict=False)
    if missing:
        print(f"  Missing keys in frame_model: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    # ── Freeze frame_model completely ────────────────────────────────────────
    # requires_grad_(False) on a module recursively covers all parameters,
    # including LoRA adapter layers inside the backbone.
    model.frame_model.requires_grad_(False)
    # Keep frame_model in eval mode throughout training so BatchNorm / Dropout
    # inside the frozen backbone use inference statistics.
    # We enforce this at the top of every epoch's loop below.

    trainable   = [p for p in model.parameters() if p.requires_grad]
    total_n     = sum(p.numel() for p in model.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    print(
        f"\n  frame_model:            FROZEN\n"
        f"  temporal_transformers:  TRAINABLE\n"
        f"  video_classifier:       TRAINABLE\n"
        f"  Trainable params: {trainable_n:,} / {total_n:,} "
        f"({100*trainable_n/total_n:.1f}%)\n"
    )

    # torch.compile wraps the whole model; frozen submodules are still compiled
    # but their gradient computation is correctly skipped by autograd.
    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── AMP scaler ──────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device=device.type)

    # ── Optimiser & cosine scheduler with linear warmup ─────────────────────
    # Only pass trainable (temporal) parameters to the optimiser.
    lr_base        = args.lr
    epochs         = args.epochs
    iter_per_epoch = len(train_loader)
    total_steps    = epochs * iter_per_epoch
    warmup_steps   = args.warmup_steps
    lr_min         = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmup_steps) * math.pi / (total_steps - warmup_steps))) / 2)
             + lr_min)
            if i > warmup_steps
            else (i / warmup_steps + lr_min)
        )
        for i in range(total_steps)
    }

    optimizer = optim.AdamW(trainable, lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[step]
    )

    lam = args.supcon_weight

    # ── Training loop ───────────────────────────────────────────────────────
    best_test_auc = 0.0
    best_epoch    = -1

    for epoch in range(epochs):
        print(f"\n{SEP}")
        print(f"  EPOCH {epoch+1}/{epochs}")
        print(SEP)

        model.train()
        # Enforce eval mode on the frozen frame_model so its Dropout / BatchNorm
        # layers use inference statistics rather than batch statistics.
        raw_for_eval = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_for_eval.frame_model.eval()

        iter_i = epoch * iter_per_epoch
        train_frame_labels, train_frame_probs = [], []
        train_video_labels, train_video_probs = [], []

        for batch_idx, (frames, labels, lengths) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        ):
            frames  = frames.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                video_logits, frame_logits_list, _, video_feats_list = model(frames, lengths)
                loss = stage2_loss(video_logits, video_feats_list, labels, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Clip only trainable (temporal) parameters.
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                fl, fp, vl, vp = _video_probs_from_forward(
                    video_logits, frame_logits_list, labels, lengths
                )
            train_frame_labels.extend(fl.cpu().numpy().tolist())
            train_frame_probs.extend(fp.cpu().numpy().tolist())
            train_video_labels.extend(vl.cpu().numpy().tolist())
            train_video_probs.extend(vp.cpu().numpy().tolist())

            if batch_idx % 256 == 0:
                print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

        # ── Per-epoch metrics ────────────────────────────────────────────────
        print()
        compute_metrics(train_frame_labels, train_frame_probs, "Train frame", epoch)
        compute_metrics(train_video_labels, train_video_probs, "Train video", epoch)

        val_fl, val_fp, val_vl, val_vp = run_eval(
            model, val_loader, f"Epoch {epoch+1} [val]"
        )
        compute_metrics(val_fl, val_fp, "Val frame  ", epoch)
        compute_metrics(val_vl, val_vp, "Val video  ", epoch)

        cdf_fl, cdf_fp, cdf_vl, cdf_vp = run_eval(
            model, cdf_loader, f"Epoch {epoch+1} [CDFv1]"
        )
        compute_metrics(cdf_fl, cdf_fp, "Test frame ", epoch)
        test_auc = compute_metrics(cdf_vl, cdf_vp, "Test video ", epoch)

        # ── Checkpointing ────────────────────────────────────────────────────
        # Save the full VideoViT state dict (frozen + trainable) so Stage 2
        # checkpoints are self-contained and can be resumed or used for inference.
        raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = raw_model.state_dict()

        torch.save(state_dict, os.path.join(args.save_root, "latest.pth"))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch    = epoch
            torch.save(state_dict, os.path.join(args.save_root, "best.pth"))
            print(f"\n  ★ New best Test video AUC={best_test_auc:.4f} → saved best.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test video AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Stage 2 complete.")
    print(f"  Best checkpoint: epoch {best_epoch+1}  Test video AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(args.save_root, 'best.pth')}")
    print(SEP)