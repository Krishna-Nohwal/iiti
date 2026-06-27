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
parser.add_argument('--stage',         default=1,    type=int, choices=[1, 2],
                    help='Stage 1: train backbone+MACHeads as pure image model (identical to '
                         'train.py). Stage 2: freeze backbone+MACHeads, train temporal only.')
parser.add_argument('--epochs',        default=30,   type=int)
parser.add_argument('--batch_size',    default=32,   type=int,
                    help='Stage 1: images per batch (plain shuffle). '
                         'Stage 2: videos per batch (must be even for balanced sampler).')
parser.add_argument('--num_frames',    default=12,   type=int,
                    help='Frames to sample per video (Stage 2 only).')
parser.add_argument('--num_workers',   default=10,   type=int)
parser.add_argument('--save_root',     default='checkpoints_vit_video', type=str)
parser.add_argument('--load_from',     default='',   type=str,
                    help='Resume from checkpoint. For stage 2, point to stage 1 best_s1.pth.')
parser.add_argument('--image_ckpt',    default='',   type=str,
                    help='Optional: path to pretrained image model .pth to warm-start '
                         'the ViT backbone and MACHeads (stage 1 only).')
parser.add_argument('--manifest',      default='E:/Work/sampled_30k/manifest_onct.csv', type=str)
parser.add_argument('--root_dir',      default='E:/Work/sampled_30k/', type=str)
parser.add_argument('--cdf_root',      default='E:/Work/cdfv1_onct_out', type=str)
parser.add_argument('--cdf_csv',       default='E:/Work/cdfv1_onct_out/manifest_cdfv1_onct.csv', type=str)
parser.add_argument('--val_ratio',     default=0.25, type=float)   # matched to train.py
parser.add_argument('--supcon_weight', default=1/16, type=float)
parser.add_argument('--lr_stage1',     default=1e-4, type=float,
                    help='Base LR for stage 1 (backbone + MACHeads).')
parser.add_argument('--lr_stage2',     default=1e-3, type=float,
                    help='Base LR for stage 2 (temporal transformers + video_classifier). '
                         'Higher than stage 1 since temporal starts from scratch.')
parser.add_argument('--warmup_steps',  default=512,  type=int,
                    help='LR warmup steps. Stage 2 uses 64 by default.')
parser.add_argument('--no_compile',    action='store_true',
                    help='Disable torch.compile (useful for debugging)')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

save_root    = args.save_root
IMG_SIZE     = 256   # matched to train.py
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Data splits  (always video-level to prevent leakage in both stages)
# ---------------------------------------------------------------------------

def _extract_video_id(sample_dir: str) -> str:
    """
    Parse the video ID from a sample_dir string, preserving the method
    subfolder so different manipulation methods on the same source video
    remain separate entries.

    Matched exactly to train.py's implementation.

    Examples
    --------
    'real/000_frame_03'                  -> 'real/000'
    'fake/FaceSwap/922_898_frame_31'     -> 'fake/FaceSwap/922_898'
    'fake/Deepfakes/922_898_frame_31'    -> 'fake/Deepfakes/922_898'
    """
    parts    = Path(sample_dir).parts        # e.g. ('fake', 'FaceSwap', '922_898_frame_31')
    basename = parts[-1]                     # last component: '922_898_frame_31'
    marker   = '_frame_'
    idx      = basename.rfind(marker)
    clip_id  = basename[:idx] if idx != -1 else basename   # '922_898'
    # Rejoin with parent path (everything except the last component)
    prefix   = "/".join(parts[:-1])          # 'fake/FaceSwap'
    return f"{prefix}/{clip_id}" if prefix else clip_id


def prepare_splits(manifest_csv: str, root_dir: str, val_ratio: float = 0.05):
    """
    Video-level split: assign whole videos to train/val so no frames from
    the same video appear in both sets. Works correctly for both stages:
      - Stage 1 flattens the returned DataFrames back to individual frame rows.
      - Stage 2 groups them by video_id.
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

    real_val_ids = set(real_vids[:real_val_n])
    fake_val_ids = set(fake_vids[:fake_val_n])
    val_ids      = real_val_ids | fake_val_ids

    train_df = df[~df["video_id"].isin(val_ids)].reset_index(drop=True)
    val_df   = df[ df["video_id"].isin(val_ids)].reset_index(drop=True)

    real_train_n = len(real_vids) - real_val_n
    fake_train_n = len(fake_vids) - fake_val_n
    print(f"Train -> frames: {len(train_df)}  "
          f"(real vids: {real_train_n}  fake vids: {fake_train_n})")
    print(f"Val   -> frames: {len(val_df)}  "
          f"(real vids: {real_val_n}  fake vids: {fake_val_n})")
    return train_df, val_df


# ---------------------------------------------------------------------------
# Stage 1 datasets  (flat image, identical to train.py)
# ---------------------------------------------------------------------------

class ManifestImageDataset(Dataset):
    """
    Stage 1 dataset - one image per sample, shuffled.
    Replicates train.py's ManifestVideoDataset frame-loading logic exactly:
      - Uses Path(root_dir) / rel / "image.png" for path construction.
      - Returns (img, label) with load_and_resize + normalize.
      - Augmentation is applied per-batch in the training loop (augment_batch),
        not here, matching train.py.
    label: 0=Real, 1=Fake.
    """

    def __init__(self, df: pd.DataFrame, root_dir: str):
        root = Path(root_dir)
        entries = []
        for _, row in df.iterrows():
            rel = row["sample_dir"].replace("\\", "/")
            img_path = root / rel / "image.png"
            if img_path.is_file():
                entries.append((str(img_path), int(row["label"])))

        skipped = len(df) - len(entries)
        if skipped:
            print(f"  [Dataset] Skipped {skipped} missing image.png ({len(entries)} remaining)")

        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


class CDFv1ImageDataset(Dataset):
    """
    Stage 1 CDFv1 test dataset (flat images, one per row).
    Manifest convention: 1=Real, 0=Fake - flipped on load to match 0=Real, 1=Fake.
    Replicates train.py's CDFv1VideoDataset frame-loading logic exactly.
    """

    def __init__(self, csv_path: str, data_root: str):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df["label"] = 1 - df["label"].astype(int)   # flip: 1=Real->0, 0=Fake->1

        print(f"CDFv1 -> Real: {(df['label']==0).sum()} | Fake: {(df['label']==1).sum()} | Total: {len(df)}")

        root = Path(data_root)
        entries = []
        for _, row in df.iterrows():
            rel = row["sample_dir"].replace("\\", "/")
            img_path = root / rel / "image.png"
            if img_path.is_file():
                entries.append((str(img_path), int(row["label"])))

        skipped = len(df) - len(entries)
        if skipped:
            print(f"  [CDFv1] Skipped {skipped} missing image.png ({len(entries)} remaining)")

        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        img = load_and_resize(img_path, IMG_SIZE)
        img = normalize(img)
        return img, label


# ---------------------------------------------------------------------------
# Stage 2 datasets  (video-level)
# ---------------------------------------------------------------------------

def _load_video_frames(frame_paths: list, img_size: int) -> torch.Tensor:
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
    Stage 2 dataset.  label: 0=Real, 1=Fake.

    Groups per-frame CSV rows by video ID, then for each video samples exactly
    num_frames frames (uniform stride if more are available, tile if fewer).
    """

    def __init__(self, df: pd.DataFrame, root_dir: str,
                 num_frames: int = 32, augment: bool = True):
        self.num_frames = num_frames
        self.augment    = augment
        self.root_dir   = root_dir
        self.videos: list = []

        for video_id, group in df.groupby("video_id"):
            label = int(group["label"].iloc[0])
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
        n = len(paths)
        T = self.num_frames
        if n >= T:
            indices = np.linspace(0, n - 1, T, dtype=int)
        else:
            indices = np.tile(np.arange(n), math.ceil(T / n))[:T]
        return [paths[i] for i in indices]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        frame_paths  = self._sample_frames(paths)
        frames       = _load_video_frames(frame_paths, IMG_SIZE)
        if self.augment:
            frames = augment_batch(frames)
        return frames, label


class BalancedRealFakeBatchSampler(Sampler):
    """Yield class-balanced batches, oversampling the minority class if needed."""

    def __init__(self, dataset: ManifestVideoDataset, batch_size: int):
        if batch_size % 2 != 0:
            raise ValueError("BalancedRealFakeBatchSampler requires an even batch_size.")
        self.batch_size   = batch_size
        self.per_class    = batch_size // 2
        self.real_indices = [i for i, (_, label) in enumerate(dataset.videos) if label == 0]
        self.fake_indices = [i for i, (_, label) in enumerate(dataset.videos) if label == 1]
        if not self.real_indices or not self.fake_indices:
            raise ValueError("Balanced batches need at least one real and one fake video.")
        self.num_batches = math.ceil(
            max(len(self.real_indices), len(self.fake_indices)) / self.per_class
        )

    def __iter__(self):
        n_per_class = self.num_batches * self.per_class
        real_perm   = self._sample_class(self.real_indices, n_per_class)
        fake_perm   = self._sample_class(self.fake_indices, n_per_class)
        for i in range(self.num_batches):
            start = i * self.per_class
            end   = start + self.per_class
            batch = real_perm[start:end] + fake_perm[start:end]
            order = torch.randperm(len(batch)).tolist()
            yield [batch[j] for j in order]

    @staticmethod
    def _sample_class(indices, n):
        if len(indices) >= n:
            return [indices[i] for i in torch.randperm(len(indices)).tolist()[:n]]
        base  = [indices[i] for i in torch.randperm(len(indices)).tolist()]
        extra = [indices[i] for i in torch.randint(len(indices), (n - len(indices),)).tolist()]
        return base + extra

    def __len__(self):
        return self.num_batches


class CDFv1VideoDataset(Dataset):
    """
    Stage 2 CDFv1 test dataset (video-level).
    Manifest convention: 1=Real, 0=Fake - flipped on load to match 0=Real, 1=Fake.
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
            indices = np.tile(np.arange(n), math.ceil(T / n))[:T]
        return [paths[i] for i in indices]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        paths, label = self.videos[idx]
        frame_paths  = self._sample_frames(paths)
        frames       = _load_video_frames(frame_paths, IMG_SIZE)
        return frames, label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# Stage 1: replicates train.py cls_loss exactly (mean CrossEntropy).
bce_loss    = nn.CrossEntropyLoss()
supcon_loss = SupConLoss()

def stage1_loss(logits_list, features_list, labels, lam):
    """
    Identical to train.py cls_loss structure:
      primary = CrossEntropy(logits[3], labels) + lam * SupCon(features[3], labels)
      aux     = (cls_loss[0] + cls_loss[1] + cls_loss[2]) / 4
      total   = primary + aux
    """
    def cls_loss(logits, features):
        return bce_loss(logits, labels) + lam * supcon_loss(features, labels)

    l_primary = cls_loss(logits_list[3], features_list[3])
    l_aux     = (
        cls_loss(logits_list[0], features_list[0]) +
        cls_loss(logits_list[1], features_list[1]) +
        cls_loss(logits_list[2], features_list[2])
    ) / 4.0
    return l_primary + l_aux


# Stage 2: video-level loss only (backbone is frozen, frame terms give zero grad).
bce_loss_sum = nn.CrossEntropyLoss(reduction='sum')

def stage2_loss(video_logits, video_feats_list, labels, lam=1/16):
    """
    Stage 2: temporal head loss only. Backbone + MACHeads are frozen so
    frame terms are omitted entirely.

    Video terms:
        BCE(video_logits)
      + lam * SupCon(video_feats[layer_i])  i in [0,1,2,3]
    """
    l_video_bce    = bce_loss_sum(video_logits, labels) / labels.size(0)
    l_supcon_video = lam * sum(
        supcon_loss(video_feats_list[i], labels)
        for i in range(4)
    )
    return l_video_bce + l_supcon_video


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval_stage1(model, loader, desc, device):
    """
    Stage 1 eval - identical to train.py run_eval.
    Calls only model.frame_model; temporal path stays idle.
    Uses logits[3] to match train.py exactly.
    """
    all_labels, all_probs = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            logits_list, _, _ = model.frame_model(imgs)
            probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_probs


def frame_and_video_predictions(video_logits, frame_logits_list, labels, lengths):
    B = labels.size(0)
    T = frame_logits_list[0].size(0) // B

    time_idx       = torch.arange(T, device=labels.device).unsqueeze(0)
    valid_by_video = time_idx < lengths.to(labels.device).unsqueeze(1)  # (B, T)
    valid_mask     = valid_by_video.reshape(-1)

    frame_labels = labels.repeat_interleave(T)

    # MACHead: average of 4 layers -> (B*T,)
    mean_frame_logits = torch.stack(frame_logits_list, dim=0).mean(dim=0)
    mac_probs = torch.softmax(mean_frame_logits.float(), dim=1)[:, 1]

    # Video logit broadcast to frames -> (B*T,)
    video_probs_per_frame = torch.softmax(video_logits.float(), dim=1)[:, 1]
    video_probs_per_frame = video_probs_per_frame.repeat_interleave(T)

    # Frame prob: MACHead weighted 5x over video logit
    frame_probs_all = (5 * mac_probs + video_probs_per_frame) / 6   # (B*T,)
    frame_probs     = frame_probs_all[valid_mask]

    # Video prob: mean of valid frame probs
    frame_probs_2d = frame_probs_all.reshape(B, T)
    video_probs = (
        (frame_probs_2d * valid_by_video.float()).sum(dim=1)
        / lengths.to(labels.device).clamp_min(1).float()
    )

    return frame_labels[valid_mask], frame_probs, labels, video_probs


def run_eval_stage2(model, loader, desc, device):
    frame_labels_all, frame_probs_all = [], []
    video_labels_all, video_probs_all = [], []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for frames, labels, lengths in tqdm(loader, desc=desc, leave=False):
            frames  = frames.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            video_logits, frame_logits_list, _, _ = model(frames, lengths)
            frame_labels, frame_probs, video_labels, video_probs = frame_and_video_predictions(
                video_logits, frame_logits_list, labels, lengths,
            )
            frame_probs_all.extend(frame_probs.cpu().numpy().tolist())
            frame_labels_all.extend(frame_labels.cpu().numpy().tolist())
            video_probs_all.extend(video_probs.cpu().numpy().tolist())
            video_labels_all.extend(video_labels.cpu().numpy().tolist())
    return frame_labels_all, frame_probs_all, video_labels_all, video_probs_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    NUM_FRAMES = args.num_frames
    STAGE      = args.stage
    print(f"\n{'='*80}")
    print(f"  STAGE {STAGE} TRAINING")
    if STAGE == 1:
        print("  Pure image training - backbone + MACHeads only (identical to train.py).")
        print("  Temporal transformers exist in model but are fully frozen and not called.")
    else:
        print("  Temporal transformers training | Backbone + MACHeads FROZEN")
    print(f"{'='*80}\n")

    # ---- Data ---------------------------------------------------------------
    # Always split video-level so val is leak-free for both stages.
    train_df, val_df = prepare_splits(args.manifest, args.root_dir, val_ratio=args.val_ratio)

    _persistent = _num_workers > 0
    _prefetch   = 4 if _num_workers > 0 else None

    if STAGE == 1:
        # Stage 1: flat image datasets, shuffle=True - identical to train.py
        train_dataset = ManifestImageDataset(train_df, args.root_dir)
        val_dataset   = ManifestImageDataset(val_df,   args.root_dir)
        cdf_dataset   = CDFv1ImageDataset(args.cdf_csv, args.cdf_root)

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
    else:
        # Stage 2: video datasets, balanced sampler
        train_dataset = ManifestVideoDataset(train_df, args.root_dir, num_frames=NUM_FRAMES, augment=True)
        val_dataset   = ManifestVideoDataset(val_df,   args.root_dir, num_frames=NUM_FRAMES, augment=False)
        cdf_dataset   = CDFv1VideoDataset(args.cdf_csv, args.cdf_root, num_frames=NUM_FRAMES)

        train_batch_sampler = BalancedRealFakeBatchSampler(train_dataset, args.batch_size)
        print(f"Train balanced batches -> {len(train_batch_sampler)} batches/epoch "
              f"({args.batch_size // 2} real + {args.batch_size // 2} fake videos per batch)")

        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=_num_workers,
            pin_memory=True, collate_fn=video_collate_fn,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=_num_workers,
            pin_memory=True, shuffle=False, collate_fn=video_collate_fn,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )
        cdf_loader = DataLoader(
            cdf_dataset, batch_size=args.batch_size, num_workers=_num_workers,
            pin_memory=True, shuffle=False, collate_fn=video_collate_fn,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )

    os.makedirs(save_root, exist_ok=True)

    # ---- Model --------------------------------------------------------------
    model = VideoViT(num_frames=NUM_FRAMES).to(device)

    if args.image_ckpt and STAGE == 1:
        missing, unexpected = model.load_image_weights(args.image_ckpt, strict=False)
        print(f"  Warm-started from image checkpoint: {args.image_ckpt}")

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location='cpu'))
        print(f"  Loaded checkpoint from {args.load_from}")

    # ---- Freeze / unfreeze based on stage -----------------------------------
    print("  Model children:", [name for name, _ in model.named_children()])

    if STAGE == 1:
        # Freeze temporal path entirely - gradients never flow into it.
        model.temporal_transformers.requires_grad_(False)
        if hasattr(model, 'video_classifier'):
            model.video_classifier.requires_grad_(False)
        print("  Stage 1: temporal_transformers + video_classifier FROZEN")
        print("           frame_model trains as a pure image model")
    else:
        # Freeze backbone; only temporal transformers + video_classifier train.
        model.frame_model.requires_grad_(False)
        print("  Stage 2: frame_model FROZEN | temporal_transformers + video_classifier TRAIN")

    trainable   = [p for p in model.parameters() if p.requires_grad]
    total       = sum(p.numel() for p in model.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {trainable_n:,} / {total:,} "
          f"({100*trainable_n/total:.1f}%)\n")

    if not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # ---- AMP scaler ---------------------------------------------------------
    scaler = torch.amp.GradScaler(device=device.type)

    # ---- Optimiser & scheduler ----------------------------------------------
    lr_base  = args.lr_stage1 if STAGE == 1 else args.lr_stage2
    warmstep = args.warmup_steps if STAGE == 1 else 64

    epochs         = args.epochs
    iter_per_epoch = len(train_loader)
    totalstep      = epochs * iter_per_epoch
    lr_min         = 1e-6 / lr_base

    lr_dict = {
        i: (
            (((1 + math.cos((i - warmstep) * math.pi / (totalstep - warmstep))) / 2) + lr_min)
            if i > warmstep
            else (i / warmstep + lr_min)
        )
        for i in range(totalstep)
    }

    optimizer = optim.AdamW(trainable, lr=lr_base, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_dict[step]
    )

    lam = args.supcon_weight

    # ---- Training loop ------------------------------------------------------
    best_test_auc = 0.0
    best_epoch    = -1
    SEP           = "=" * 80

    for epoch in range(epochs):
        print(f"\n{SEP}")
        print(f"  STAGE {STAGE} | EPOCH {epoch+1}/{epochs}")
        print(SEP)

        model.train()
        # Keep frozen submodules in eval mode: BatchNorm/Dropout use inference statistics.
        if STAGE == 1:
            model.temporal_transformers.eval()
            if hasattr(model, 'video_classifier'):
                model.video_classifier.eval()
        else:
            model.frame_model.eval()

        iter_i = epoch * iter_per_epoch

        # ==== Stage 1 training loop (identical behaviour to train.py) ========
        if STAGE == 1:
            train_labels, train_probs = [], []

            for batch_idx, (imgs, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
            ):
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                imgs   = augment_batch(imgs)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    # Call only the frame model. Temporal path is frozen and not invoked.
                    logits_list, features_list, _ = model.frame_model(imgs)
                    loss = stage1_loss(logits_list, features_list, labels, lam)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(iter_i + batch_idx)

                with torch.inference_mode():
                    probs = torch.softmax(logits_list[3].float(), dim=1)[:, 1].cpu().numpy()
                train_probs.extend(probs.tolist())
                train_labels.extend(labels.cpu().numpy().tolist())

                if batch_idx % 256 == 0:
                    print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

            # Stage 1 metrics
            print()
            compute_metrics(train_labels, train_probs, "Train", epoch)

            val_labels, val_probs = run_eval_stage1(
                model, val_loader, f"Epoch {epoch+1} [val]", device)
            compute_metrics(val_labels, val_probs, "Val  ", epoch)

            cdf_labels, cdf_probs = run_eval_stage1(
                model, cdf_loader, f"Epoch {epoch+1} [CDFv1]", device)
            test_auc = compute_metrics(cdf_labels, cdf_probs, "Test ", epoch)

        # ==== Stage 2 training loop (video, temporal transformer) ============
        else:
            train_frame_labels, train_frame_probs = [], []
            train_video_labels, train_video_probs = [], []

            for batch_idx, (frames, labels, lengths) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
            ):
                frames  = frames.to(device, non_blocking=True)
                labels  = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    video_logits, frame_logits_list, frame_feats_list, video_feats_list = model(frames, lengths)
                    loss = stage2_loss(video_logits, video_feats_list, labels, lam)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(iter_i + batch_idx)

                with torch.inference_mode():
                    frame_labels, frame_probs, video_labels, video_probs = frame_and_video_predictions(
                        video_logits, frame_logits_list, labels, lengths,
                    )
                train_frame_probs.extend(frame_probs.cpu().numpy().tolist())
                train_frame_labels.extend(frame_labels.cpu().numpy().tolist())
                train_video_probs.extend(video_probs.cpu().numpy().tolist())
                train_video_labels.extend(video_labels.cpu().numpy().tolist())

                if batch_idx % 256 == 0:
                    print(f"  batch={batch_idx:4d}/{iter_per_epoch}  loss={loss.item():.4f}")

            # Stage 2 metrics
            print()
            compute_metrics(train_frame_labels, train_frame_probs, "Train frame", epoch)
            compute_metrics(train_video_labels, train_video_probs, "Train video", epoch)

            val_frame_labels, val_frame_probs, val_video_labels, val_video_probs = run_eval_stage2(
                model, val_loader, f"Epoch {epoch+1} [val]", device
            )
            compute_metrics(val_frame_labels, val_frame_probs, "Val frame  ", epoch)
            compute_metrics(val_video_labels, val_video_probs, "Val video  ", epoch)

            cdf_frame_labels, cdf_frame_probs, cdf_video_labels, cdf_video_probs = run_eval_stage2(
                model, cdf_loader, f"Epoch {epoch+1} [CDFv1]", device
            )
            compute_metrics(cdf_frame_labels, cdf_frame_probs, "Test frame ", epoch)
            test_auc = compute_metrics(cdf_video_labels, cdf_video_probs, "Test video ", epoch)

        # ---- Checkpointing (both stages) ------------------------------------
        state_dict = (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict()
        torch.save(state_dict, os.path.join(save_root, f'latest_s{STAGE}.pth'))
        vit_module = (model._orig_mod if hasattr(model, '_orig_mod') else model).vit
        vit_module.save_pretrained(os.path.join(save_root, f'latest_s{STAGE}_lora'))

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch    = epoch
            torch.save(state_dict, os.path.join(save_root, f'best_s{STAGE}.pth'))
            vit_module.save_pretrained(os.path.join(save_root, f'best_s{STAGE}_lora'))
            print(f"\n  New best Test AUC={best_test_auc:.4f} -> saved best_s{STAGE}.pth")
        else:
            print(f"\n  Best so far: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")

    print(f"\n{SEP}")
    print(f"  Stage {STAGE} complete. Best: epoch {best_epoch+1}  Test AUC={best_test_auc:.4f}")
    print(f"  Saved to: {os.path.join(save_root, f'best_s{STAGE}.pth')}")
    if STAGE == 1:
        print(f"\n  To run stage 2:")
        print(f"    python alt_train.py --stage 2 "
              f"--load_from {os.path.join(save_root, 'best_s1.pth')} "
              f"--save_root {save_root} --epochs <N>")
    print(SEP)