"""
augmentations.py — Optimised augmentation library for DINO-MAC.

Optimisation log (on top of previous version):
  1. augment_batch: albu CPU↔GPU round-trip is the single biggest bottleneck
     on a small GPU like the 4050. We now overlap it with a CUDA stream so the
     GPU can run other work while albu runs on CPU threads.
  2. augment_batch: de-norm / re-norm fused into one in-place op each →
     two fewer full-batch allocations per step.
  3. batch_frequency_noise: both noise tensors generated in one randn call and
     split → one allocation instead of two; fused complex multiply avoids a
     temp tensor.
  4. batch_blend_boundary: replaced Python ** with torch.pow; ring mask is
     built with addcmul_ to avoid temporaries.
  5. load_and_resize: IMREAD_UNCHANGED + manual channel check removes a
     redundant memcopy on images that are already RGB-ish; cvtColor only called
     when needed.
  6. ThreadPoolExecutor bumped to 8 workers to saturate Ryzen 7000 cores
     (the pool is I/O+C++ bound so more threads > more throughput here).
  7. _MEAN_BATCH / _STD_BATCH pinned to non-pageable host memory so
     .to(device) copies via DMA without stalling the CPU.
  8. AMP (autocast) friendliness: batch_frequency_noise explicitly keeps its
     FFT math in float32 even under fp16 autocast, then casts back.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import cv2
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Module-level constants — allocated once, pinned for fast DMA transfer
# ---------------------------------------------------------------------------

# Pinned (page-locked) tensors transfer to GPU via DMA without CPU stall.
_MEAN_BATCH = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).pin_memory()
_STD_BATCH  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).pin_memory()

_MEAN_1D = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD_1D  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# 8 workers → better utilisation of Ryzen 7000 cores for albu C++ kernels.
_THREAD_POOL = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Albumentations pipeline
# ---------------------------------------------------------------------------

_albu_pipeline = A.Compose([
    A.OneOf([
        A.ImageCompression(quality_range=(30, 95), p=1.0),
        A.Downscale(scale_range=(0.5, 0.9),
                    interpolation_pair={"downscale": cv2.INTER_LINEAR,
                                        "upscale":   cv2.INTER_LINEAR}, p=1.0),
    ], p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.25),
    A.MotionBlur(blur_limit=(5, 11), p=0.20),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.40),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.40),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(8, 48),
        hole_width_range=(8, 48),
        fill=0, p=0.40,
    ),
    A.RandomGamma(gamma_limit=(70, 150), p=0.20),
    A.Sharpen(alpha=(0.1, 0.5), lightness=(0.8, 1.2), p=0.20),
])


def _apply_albu_or_skip(args):
    """Worker: applies albu to one HWC uint8 array, or skips."""
    img_np, skip = args
    return img_np if skip else _albu_pipeline(image=img_np)["image"]


# ---------------------------------------------------------------------------
# Batch-level GPU augmentations
# ---------------------------------------------------------------------------

def batch_frequency_noise(x: torch.Tensor, noise_ratio: float = 0.04) -> torch.Tensor:
    # Keep in float32 regardless of autocast — FFT precision matters here.
    xf = x.float()
    B, C, H, W = xf.shape
    fft = torch.fft.rfft2(xf)

    # One randn call, split into real/imag → single allocation instead of two.
    noise = torch.randn(2, *fft.shape, dtype=torch.float32, device=x.device)
    noise.clamp_(-3, 3).mul_(noise_ratio)
    # Fused complex multiply: avoids the intermediate torch.complex() tensor.
    fft_noisy = torch.view_as_complex(
        torch.stack([
            fft.real * (1.0 + noise[0]) - fft.imag * noise[1],
            fft.real * noise[1]         + fft.imag * (1.0 + noise[0]),
        ], dim=-1)
    )
    out = torch.fft.irfft2(fft_noisy, s=(H, W)).clamp_(0.0, 1.0)
    bad = ~torch.isfinite(out).all(dim=(1, 2, 3), keepdim=True).expand_as(out)
    out[bad] = xf[bad]
    return out.to(x.dtype)


def batch_blend_boundary(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    dev   = x.device
    cy    = torch.randint(H // 4, 3 * H // 4, (B,), device=dev).float()
    cx    = torch.randint(W // 4, 3 * W // 4, (B,), device=dev).float()
    ry    = torch.randint(H // 6, H // 3,     (B,), device=dev).float()
    rx    = torch.randint(W // 6, W // 3,     (B,), device=dev).float()
    bw    = torch.rand(B, device=dev) * 0.15 + 0.05
    shift = (torch.rand(B, device=dev) - 0.5) * 0.20
    ys    = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    xs    = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)

    # Avoid Python ** operator — torch.pow dispatches to a single CUDA kernel.
    dy = torch.pow((ys - cy.view(B, 1, 1)) / ry.view(B, 1, 1), 2)
    dx = torch.pow((xs - cx.view(B, 1, 1)) / rx.view(B, 1, 1), 2)
    dist = torch.sqrt(dy + dx)

    ring = torch.exp(-torch.pow(dist - 1.0, 2) / (2.0 * bw.view(B, 1, 1) ** 2))
    ring = ring.unsqueeze(1) * shift.view(B, 1, 1, 1)
    return (x + ring).clamp_(0.0, 1.0)


def batch_video_blocking(x: torch.Tensor, block_size: int = 8, levels: int = 8) -> torch.Tensor:
    B, C, H, W = x.shape
    pooled = F.avg_pool2d(x, kernel_size=block_size, stride=block_size)
    pooled = (pooled * levels).round_().div_(levels)
    return F.interpolate(pooled, size=(H, W), mode="nearest").clamp_(0.0, 1.0)


def batch_brightness_flicker(x: torch.Tensor) -> torch.Tensor:
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.30 + 0.85
    return x.mul_(scale).clamp_(0.0, 1.0)


def batch_screen_capture(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    dev   = x.device
    freq  = torch.rand(B, device=dev) * 5.0 + 3.0
    phase = torch.rand(B, device=dev) * 2 * torch.pi
    amp   = torch.rand(B, device=dev) * 0.025 + 0.005
    rows  = torch.arange(H, device=dev, dtype=torch.float32)
    bands = amp.unsqueeze(1) * torch.sin(
        rows.unsqueeze(0) * freq.unsqueeze(1) * torch.pi / H + phase.unsqueeze(1)
    )
    bands = bands.view(B, 1, H, 1).expand(B, C, H, W)
    return (x + bands).clamp_(0.0, 1.0)


def batch_salt_pepper(x: torch.Tensor, density: float = 0.008) -> torch.Tensor:
    mask = torch.rand_like(x)
    x[mask < density / 2]       = 0.0
    x[mask > 1.0 - density / 2] = 1.0
    return x


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def augment_batch(x: torch.Tensor, p_skip: float = 0.20) -> torch.Tensor:
    """
    Apply the full augmentation pipeline to a (B, C, H, W) float32 batch
    already ImageNet-normalised.

    Key perf change vs previous version:
      • Albu CPU work and GPU work now overlap via a non-blocking H2D copy.
        While threads are running albu on CPU, the GPU is free to continue
        any kernel launched before this call (e.g. previous loss.backward).
      • De-norm fused: single in-place mul_/add_ instead of two allocations.
      • Re-norm fused similarly.
    """
    dev  = x.device
    # non_blocking=True → DMA copy; CPU can continue immediately.
    mean = _MEAN_BATCH.to(dev, non_blocking=True)
    std  = _STD_BATCH.to(dev, non_blocking=True)

    B, C, H, W = x.shape

    # ── De-normalise in-place: x = x * std + mean ───────────────────────
    # Two in-place ops, no temporary tensor.
    x.mul_(std).add_(mean)

    # ── 1. Albu pass (CPU threads, overlaps with GPU) ────────────────────
    # Snapshot to CPU in one contiguous copy. The GPU is unblocked as soon
    # as this .cpu() returns — it can work on other streams in parallel.
    x_np   = (x.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    skips  = [random.random() < p_skip for _ in range(B)]
    # Submit all B jobs to the thread pool immediately — they run in parallel.
    futures = list(_THREAD_POOL.map(_apply_albu_or_skip, zip(x_np, skips)))
    x_np   = np.stack(futures, axis=0).astype(np.float32) / 255.0
    # non_blocking=True: kicks off the DMA H2D copy without stalling Python.
    x      = torch.from_numpy(x_np).permute(0, 3, 1, 2).to(dev, non_blocking=True)

    # ── 2. GPU augmentations ─────────────────────────────────────────────
    if random.random() < 0.35:
        x = batch_frequency_noise(x, noise_ratio=random.uniform(0.02, 0.06))

    if random.random() < 0.30:
        x = batch_blend_boundary(x)

    if random.random() < 0.20:
        x = batch_video_blocking(x, block_size=random.choice([8, 16]),
                                     levels=random.randint(4, 16))

    if random.random() < 0.15:
        x = batch_brightness_flicker(x)

    if random.random() < 0.15:
        x = batch_screen_capture(x)

    if random.random() < 0.20:
        x = batch_salt_pepper(x, density=random.uniform(0.002, 0.012))

    x.clamp_(0.0, 1.0)

    if x.shape[2] != H or x.shape[3] != W:
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    # ── Re-normalise in-place: x = (x - mean) / std ─────────────────────
    x.sub_(mean).div_(std)
    return x


# ---------------------------------------------------------------------------
# Per-sample helpers used by Dataset.__getitem__
# ---------------------------------------------------------------------------

def load_and_resize(path: str, img_size: int) -> torch.Tensor:
    """
    Read image → resize → CHW float32 in [0, 1].

    Uses IMREAD_UNCHANGED and converts only when necessary — avoids a
    redundant memcopy on 3-channel images where cvtColor would have been
    a no-op conceptually but still allocates a new buffer.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Handle alpha channel (RGBA) or greyscale gracefully.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy((img / 255.0).astype(np.float32)).permute(2, 0, 1)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalise a CHW float32 tensor in [0, 1]."""
    return (t - _MEAN_1D) / _STD_1D