import os, math, torch, argparse, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from torch import nn
from augmentations import augment_batch, load_and_resize, normalize
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score,
)
from video_model_small import VideoViT


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',        default=50,   type=int)
parser.add_argument('--batch_size',    default=4,    type=int,
                    help='Videos per batch. Training batches are class-balanced, '
                         'so this must be an even number.')
parser.add_argument('--num_frames',    default=32,   type=int,
                    help='Frames to sample per video. Must match VideoViT.num_frames.')
parser.add_argument('--num_workers',   default=6,    type=int,
                    help='6 workers suits Ryzen 7000; tune down if RAM is tight')
parser.add_argument('--save_root',     default='checkpoints_vit_video', type=str)
parser.add_argument('--load_from',     default='',   type=str)
parser.add_argument('--image_ckpt',    default='',   type=str,
                    help='Optional: path to pretrained image model .pth to warm-start '
                         'the ViT backbone and MACHeads.')
parser.add_argument('--manifest',      default='E:/Work/sampled_30k/manifest_onct.csv', type=str)
parser.add_argument('--root_dir',      default='E:/Work/sampled_30k/', type=str)
parser.add_argument('--cdf_root',      default='E:/Work/cdfv1_onct_out', type=str)
parser.add_argument('--cdf_csv',       default='E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv', type=str)
parser.add_argument('--val_ratio',     default=0.2, type=float)
parser.add_argument('--frame_loss_weight', default=3.0, type=float,
                    help='Weight for frame-level CE+SupCon. Total loss is averaged per valid frame.')
parser.add_argument('--supcon_weight', default=1/16, type=float)
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

# cuDNN benchmark: profile conv algorithms once, then use the fastest.
# Fixed input size (B×3×266×266) means the profile stays valid all run.
torch.backends.cudnn.benchmark = True



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
# Data splits  (video-level — no frame leakage)
# ---------------------------------------------------------------------------

def _extract_video_id(sample_dir: str) -> str:
    """
    Parse the video ID from a sample_dir string, preserving the method
    subfolder so different manipulation methods on the same source video
    remain separate entries.

    Examples
    --------
    'real/000_frame_03'                  → 'real/000'
    'fake/FaceSwap/922_898_frame_31'     → 'fake/FaceSwap/922_898'
    'fake/Deepfakes/922_898_frame_31'    → 'fake/Deepfakes/922_898'
    """
    parts = Path(sample_dir).parts        # e.g. ('fake', 'FaceSwap', '922_898_frame_31')
    basename = parts[-1]                  # '922_898_frame_31' or 'frame_000000'
    marker = '_frame_'
    idx = basename.rfind(marker)
    if idx != -1:
        clip_id = basename[:idx]          # '922_898'
        prefix = "/".join(parts[:-1])     # 'fake/FaceSwap'
        return f"{prefix}/{clip_id}" if prefix else clip_id

    if basename.startswith("frame_") and len(parts) > 1:
        return "/".join(parts[:-1])       # CDF layout: 'real/00011'

    return sample_dir.replace("\\", "/")


def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    """
    Split at the VIDEO level so no video's frames appear in both train and val.

    Returns two DataFrames of individual frame rows, but the val/train boundary
    is determined by video ID, not by frame.
    """
    df = pd.read_csv(manifest_csv)
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {required}. Found: {list(df.columns)}")

    df["video_id"] = df["sample_dir"].apply(_extract_video_id)

    # Collect unique video IDs per class
    real_vids = df[df["label"] == 0]["video_id"].unique()
    fake_vids = df[df["label"] == 1]["video_id"].unique()

    rng = np.random.default_rng(42)
    real_vids = rng.permutation(real_vids)
    fake_vids = rng.permutation(fake_vids)

    print(f"Full dataset -> Real videos: {len(real_vids)} | Fake videos: {len(fake_vids)}")

    real_val_n = max(1, int(len(real_vids) * val_ratio))
    fake_val_n = max(1, int(len(fake_vids) * val_ratio))

    real_val_ids   = set(real_vids[:real_val_n])
    fake_val_ids   = set(fake_vids[:fake_val_n])
    val_ids        = real_val_ids | fake_val_ids

    train_df = df[~df["video_id"].isin(val_ids)].reset_index(drop=True)
    val_df   = df[ df["video_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train -> frames: {len(train_df)}  "
          f"(real vids: {len(real_vids)-real_val_n}  fake vids: {len(fake_vids)-fake_val_n})")
    print(f"Val   -> frames: {len(val_df)}  "
          f"(real vids: {real_val_n}  fake vids: {fake_val_n})")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _load_video_frames(frame_paths: list, img_size: int) -> torch.Tensor:
    """
    Load a list of image paths and return a stacked tensor (T, 3, H, W).
    Frames that fail to load are replaced by a zero tensor so the shape is
    always (T, 3, H, W) regardless of missing files.
    """
    frames = []
    for p in frame_paths:
        try:
            img = load_and_resize(p, img_size)
            img = normalize(img)
        except Exception:
            img = torch.zeros(3, img_size, img_size)
        frames.append(img)
    return torch.stack(frames, dim=0)   # (T, 3, H, W)


def video_collate_fn(batch):
    frames_list, labels = zip(*batch)
    lengths = torch.tensor([frames.size(0) for frames in frames_list], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_frames = []
    for frames in frames_list:
        pad_t = max_len - frames.size(0)
        if pad_t > 0:
            frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_t))
        padded_frames.append(frames)

    return torch.stack(padded_frames, dim=0), torch.tensor(labels, dtype=torch.long), lengths


class ManifestVideoDataset(Dataset):
    """
    Train/val dataset.  label: 0=Real, 1=Fake.

    Groups per-frame CSV rows by video ID, then for each video samples exactly
    num_frames frames (uniform stride if more are available, repeat if fewer).

    __getitem__ returns:
        frames : (T, 3, H, W)   — T = num_frames
        label  : int            — video-level label (same for all frames)
    """

    def __init__(self, df: pd.DataFrame, root_dir: str,
                 num_frames: int = 32, augment: bool = True):
        self.num_frames = num_frames
        self.augment    = augment
        self.root_dir   = root_dir

        # Build one entry per video: (sorted_frame_paths, label)
        self.videos: list = []

        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])

            # Each sample_dir is a FOLDER; load only image.png from it.
            paths = []
            for rel in group["sample_dir"].str.replace("\\", "/", regex=False):
                frame_dir = Path(root_dir) / rel
                if frame_dir.is_dir():
                    img = frame_dir / "image.png"
                    if img.is_file():
                        paths.append(img)
            paths = sorted(str(p) for p in paths)
            if len(paths) == 0:
                continue
            self.videos.append((paths, label))

        print(f"  [ManifestVideoDataset] {len(self.videos)} videos "
              f"({sum(1 for _,l in self.videos if l==0)} real, "
              f"{sum(1 for _,l in self.videos if l==1)} fake)")

    def _sample_frames(self, paths: list) -> list:
        """
        Return exactly self.num_frames paths.
        - If len(paths) >= num_frames: uniform stride (no repeats).
        - If len(paths) <  num_frames: tile and trim.
        """
        n = len(paths)
        T = self.num_frames
        if n >= T:
            indices = np.linspace(0, n - 1, T, dtype=int)
        else:
            indices = np.arange(n)
        return [paths[i] for i in indices]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        frame_paths  = self._sample_frames(paths)
        frames       = _load_video_frames(frame_paths, IMG_SIZE)   # (T, 3, H, W)
        if self.augment:
            # augment_batch expects (B, 3, H, W) — treat frames as a batch
            frames = augment_batch(frames)
        return frames, label


class BalancedRealFakeBatchSampler(Sampler):
    """Yield class-balanced batches, oversampling the minority class if needed."""

    def __init__(self, dataset: ManifestVideoDataset, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("BalancedRealFakeBatchSampler requires an even batch_size.")
        self.batch_size = batch_size
        self.per_class = batch_size // 2
        self.real_indices = [i for i, (_, label) in enumerate(dataset.videos) if label == 0]
        self.fake_indices = [i for i, (_, label) in enumerate(dataset.videos) if label == 1]
        if not self.real_indices or not self.fake_indices:
            raise ValueError("Balanced batches need at least one real and one fake video.")
        self.num_batches = math.ceil(
            max(len(self.real_indices), len(self.fake_indices)) / self.per_class
        )

    def __iter__(self):
        n_per_class = self.num_batches * self.per_class
        real_perm = self._sample_class(self.real_indices, n_per_class)
        fake_perm = self._sample_class(self.fake_indices, n_per_class)
        for i in range(self.num_batches):
            start = i * self.per_class
            end = start + self.per_class
            batch = real_perm[start:end] + fake_perm[start:end]
            order = torch.randperm(len(batch)).tolist()
            yield [batch[j] for j in order]

    @staticmethod
    def _sample_class(indices, n):
        if len(indices) >= n:
            return [indices[i] for i in torch.randperm(len(indices)).tolist()[:n]]
        base = [indices[i] for i in torch.randperm(len(indices)).tolist()]
        extra = [indices[i] for i in torch.randint(len(indices), (n - len(indices),)).tolist()]
        return base + extra

    def __len__(self):
        return self.num_batches


class CDFv1VideoDataset(Dataset):
    """
    CDFv1 test dataset (video-level).
    Manifest convention: 1=Real, 0=Fake — flipped on load to match 0=Real, 1=Fake.
    """

    def __init__(self, csv_path: str, data_root: str, num_frames: int = 32):
        self.num_frames = num_frames

        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"]    = 1 - df["label"].astype(int)
        df["video_id"] = df["sample_dir"].apply(_extract_video_id)

        print(f"CDFv1 -> Real: {(df['label']==0).sum()} frames | "
              f"Fake: {(df['label']==1).sum()} frames | Total: {len(df)}")

        root = Path(data_root)
        self.videos: list = []

        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
            # Each sample_dir is a FOLDER; load only image.png from it.
            paths = []
            for d in group["sample_dir"].str.replace("\\", "/", regex=False):
                frame_dir = root / d
                if frame_dir.is_dir():
                    img = frame_dir / "image.png"
                    if img.is_file():
                        paths.append(img)
            paths = sorted(str(p) for p in paths)
            if len(paths) == 0:
                continue
            self.videos.append((paths, label))

        skipped_vids = df["video_id"].nunique() - len(self.videos)
        if skipped_vids:
            print(f"  [CDFv1] Skipped {skipped_vids} videos with no frames on disk")
        print(f"  [CDFv1] {len(self.videos)} videos loaded.")

    def _sample_frames(self, paths: list) -> list:
        n = len(paths)
        T = self.num_frames
        if n >= T:
            indices = np.linspace(0, n - 1, T, dtype=int)
        else:
            indices = np.arange(n)
        return [paths[i] for i in indices]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        frame_paths  = self._sample_frames(paths)
        frames       = _load_video_frames(frame_paths, IMG_SIZE)   # (T, 3, H, W)
        return frames, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

bce_loss = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()

def _legacy_video_loss(video_logits, frame_logits_list, labels, lengths, frame_weight=1.0):
    """
    Primary loss  : CE + SupCon on video-level logits/features.
    Auxiliary loss: mean CE across all 4 MACHead layers' frame logits.
                    Frame labels = video label broadcast to all T frames.
                    SupCon is skipped for frame aux (frame features are not
                    video-aligned enough to make a good embedding space).

    Args:
        video_logits      : (B, 2)
        video_feats       : (B, 192)   — from TemporalTransformer layer 3
        frame_logits_list : list of 4 × (B*T, 2)
        labels            : (B,)       — video-level
        lam               : supcon weight
    """
    l_video = bce_loss(video_logits, labels)

    # Broadcast video label to all T frames: (B,) → (B*T,)
    T = frame_logits_list[0].size(0) // labels.size(0)
    frame_labels = labels.repeat_interleave(T)   # (B*T,)
    time_idx = torch.arange(T, device=labels.device).unsqueeze(0)
    valid_mask = time_idx < lengths.to(labels.device).unsqueeze(1)
    valid_mask = valid_mask.reshape(-1)

    l_frame = sum(
        bce_loss(fl[valid_mask], frame_labels[valid_mask]) for fl in frame_logits_list
    ) / len(frame_logits_list)

    return l_video + frame_weight * l_frame


def frame_loss(logits, features, labels, lam):
    return bce_loss(logits, labels) + lam * supcon_loss(features, labels)


def video_frame_loss(video_logits, frame_logits_list, frame_feats_list,
                     labels, lengths, frame_weight=1.0, lam=1/16):
    B = labels.size(0)
    T = frame_logits_list[0].size(0) // B
    frame_labels = labels.repeat_interleave(T)
    video_logits_per_frame = video_logits.repeat_interleave(T, dim=0)

    time_idx = torch.arange(T, device=labels.device).unsqueeze(0)
    valid_mask = time_idx < lengths.to(labels.device).unsqueeze(1)
    valid_mask = valid_mask.reshape(-1)

    l_video = bce_loss(video_logits_per_frame[valid_mask], frame_labels[valid_mask])
    l_frame = sum(
        frame_loss(fl[valid_mask], ff[valid_mask], frame_labels[valid_mask], lam)
        for fl, ff in zip(frame_logits_list, frame_feats_list)
    ) / len(frame_logits_list)

    return l_video + frame_weight * l_frame


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, loader, desc, device):
    """
    inference_mode() is faster than no_grad(): it also disables the version
    counter, saving a few microseconds per tensor op.
    autocast gives FP16 throughput during eval too.
    """
    all_labels, all_probs = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, labels, lengths in tqdm(loader, desc=desc, leave=False):
            frames = frames.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            # VideoViT returns (video_logits, frame_logits, frame_feats, video_feats)
            video_logits, _, _, _ = model(frames, lengths)
            probs = torch.softmax(video_logits.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    NUM_FRAMES = args.num_frames

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, val_df = prepare_splits(args.manifest, args.root_dir, val_ratio=args.val_ratio)

    train_dataset = ManifestVideoDataset(train_df, args.root_dir, num_frames=NUM_FRAMES, augment=True)
    val_dataset   = ManifestVideoDataset(val_df,   args.root_dir, num_frames=NUM_FRAMES, augment=False)
    cdf_dataset   = CDFv1VideoDataset(args.cdf_csv, args.cdf_root, num_frames=NUM_FRAMES)

    _persistent = _num_workers > 0
    _prefetch   = 4 if _num_workers > 0 else None
    train_batch_sampler = BalancedRealFakeBatchSampler(train_dataset, args.batch_size)
    print(f"Train balanced batches -> {len(train_batch_sampler)} batches/epoch "
          f"({args.batch_size // 2} real + {args.batch_size // 2} fake videos per batch; "
          f"minority class oversampled)")

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=_num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=_num_workers,
        pin_memory=True, shuffle=False,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )
    cdf_loader = DataLoader(
        cdf_dataset, batch_size=args.batch_size, num_workers=_num_workers,
        pin_memory=True, shuffle=False,
        collate_fn=video_collate_fn,
        persistent_workers=_persistent, prefetch_factor=_prefetch,
    )

    os.makedirs(save_root, exist_ok=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = VideoViT(num_frames=NUM_FRAMES).to(device)

    if args.image_ckpt:
        # Warm-start ViT backbone + MACHeads from pretrained image model.
        # TemporalTransformers and video_classifier initialise from scratch.
        missing, unexpected = model.load_image_weights(args.image_ckpt, strict=False)
        print(f"  Warm-started from image checkpoint: {args.image_ckpt}")

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location='cpu'))
        print(f"Loaded video checkpoint from {args.load_from}")

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

    frame_loss_weight = args.frame_loss_weight
    lam = args.supcon_weight

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

        for batch_idx, (frames, labels, lengths) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        ):
            # frames : (B, T, 3, H, W)
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            # Note: augmentation is already applied in the Dataset worker.

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                video_logits, frame_logits_list, frame_feats_list, _ = model(frames, lengths)
                loss = video_frame_loss(
                    video_logits,
                    frame_logits_list,
                    frame_feats_list,
                    labels,
                    lengths,
                    frame_loss_weight,
                    lam,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(iter_i + batch_idx)

            with torch.inference_mode():
                probs = torch.softmax(video_logits.float(), dim=1)[:, 1].cpu().numpy()
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
    print(SEP)
