import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, List

from frame_model import ViT


# ---------------------------------------------------------------------------
# Real-video memory bank
# ---------------------------------------------------------------------------

class RealVideoMemoryBank:
    """
    Per-head kNN memory bank of real video embeddings.

    Stores L2-normalised embeddings from real training videos, one bank per
    temporal transformer head. At query time, returns the mean cosine
    similarity between the query video embedding and its k nearest real
    neighbours — one scalar per head.

    Because the backbone is frozen, embeddings are stable across training:
    a video processed in epoch 1 and epoch 30 produces identical embeddings,
    so the bank never needs updating.

    Uses exact inner-product search over L2-normalised vectors, which is
    equivalent to cosine similarity. FAISS is used if available; falls back
    to a pure-PyTorch brute-force search for small banks.

    Args
    ----
    embed_dim : int   — embedding dimension per head (1024 for ViT-Large)
    num_heads : int   — number of temporal transformer heads
    k         : int   — number of nearest neighbours to retrieve
    device    : str   — 'cpu' (bank always lives on CPU; queries are moved)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 4,
        k:         int = 32,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k         = k
        self._banks: List[np.ndarray] = [np.zeros((0, embed_dim), dtype=np.float32)
                                          for _ in range(num_heads)]
        self._use_faiss = False
        self._faiss_indices = None

        try:
            import faiss
            self._use_faiss = True
            print("  [RealVideoMemoryBank] FAISS available — using exact IP search.")
        except ImportError:
            print("  [RealVideoMemoryBank] FAISS not found — using PyTorch brute-force search.")

    # ── Building ────────────────────────────────────────────────────────────

    def add(self, video_feats_list: List[Tensor]):
        """
        Add a batch of real video embeddings to the bank.

        video_feats_list : list of NUM_HEADS × (B, embed_dim) tensors
                           (output of temporal transformers for real videos only)
        """
        for h, feats in enumerate(video_feats_list):
            # L2-normalise before storing so inner product = cosine similarity.
            normed = F.normalize(feats.float(), dim=-1).detach().cpu().numpy()
            self._banks[h] = np.concatenate([self._banks[h], normed], axis=0)

    def build(self):
        """
        Finalise the bank after all real videos have been added.
        Builds FAISS indices if available.
        """
        n = self._banks[0].shape[0]
        print(f"  [RealVideoMemoryBank] Built with {n} real video embeddings per head.")

        if self._use_faiss:
            import faiss
            self._faiss_indices = []
            for h in range(self.num_heads):
                index = faiss.IndexFlatIP(self.embed_dim)   # exact inner product
                index.add(self._banks[h])
                self._faiss_indices.append(index)

    def __len__(self):
        return self._banks[0].shape[0]

    # ── Querying ────────────────────────────────────────────────────────────

    def query(self, video_feats_list: List[Tensor]) -> Tensor:
        """
        Query the bank for each video in the batch.

        video_feats_list : list of NUM_HEADS × (B, embed_dim) tensors

        Returns
        -------
        sim_scores : (B, NUM_HEADS)  mean cosine similarity to k nearest
                     real neighbours, one score per head per video.
                     Higher = more similar to real videos.
        """
        B      = video_feats_list[0].size(0)
        device = video_feats_list[0].device
        scores = []

        k_eff = min(self.k, len(self))   # can't retrieve more than bank size

        for h, feats in enumerate(video_feats_list):
            normed = F.normalize(feats.float(), dim=-1).detach().cpu()

            if self._use_faiss:
                sims, _ = self._faiss_indices[h].search(
                    normed.numpy().astype(np.float32), k_eff
                )                                              # (B, k_eff)
                mean_sim = torch.from_numpy(sims).mean(dim=1) # (B,)
            else:
                # Brute-force: cosine sim between queries and entire bank.
                bank   = torch.from_numpy(self._banks[h])     # (N, D)
                sim    = normed @ bank.T                       # (B, N)
                topk   = sim.topk(k_eff, dim=1).values        # (B, k_eff)
                mean_sim = topk.mean(dim=1)                    # (B,)

            scores.append(mean_sim)

        return torch.stack(scores, dim=1).to(device)          # (B, NUM_HEADS)


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def temporal_augment(
    frame_cls: Tensor,
    key_padding_mask: Optional[Tensor],
    blank_prob:   float = 0.15,
    repeat_prob:  float = 0.10,
    noise_std:    float = 0.02,
    mixup_prob:   float = 0.10,
    shuffle_prob: float = 0.05,
    reverse_prob: float = 0.05,
    speed_prob:   float = 0.10,
) -> Tensor:
    """
    Temporal augmentation applied to CLS token sequences in embedding space,
    after the frozen backbone and before the temporal transformers.

    All augmentations operate on valid frames only (those not masked by
    key_padding_mask). Padding slots are left as-is so the transformer's
    padding mask remains consistent.

    Args
    ----
    frame_cls        : (B, T, D)  CLS token sequence
    key_padding_mask : (B, T) bool — True = padding, False = valid frame.
                       None means all frames are valid.

    Augmentations (applied independently per sample in the batch)
    -------------------------------------------------------------
    blank_prob   — replace a random valid frame with a zero vector
    repeat_prob  — duplicate a random valid frame into a random adjacent slot
    noise_std    — add per-frame Gaussian noise scaled by the token's own norm
    mixup_prob   — interpolate two randomly chosen valid frames in-place
    shuffle_prob — randomly permute the order of valid frames
    reverse_prob — reverse the temporal order of valid frames
    speed_prob   — re-sample valid frames at a non-uniform stride

    Returns
    -------
    frame_cls : (B, T, D)  augmented in-place copy
    """
    B, T, D = frame_cls.shape
    out = frame_cls.clone()

    for b in range(B):
        if key_padding_mask is not None:
            valid = (~key_padding_mask[b]).nonzero(as_tuple=True)[0]
        else:
            valid = torch.arange(T, device=frame_cls.device)

        n_valid = valid.numel()
        if n_valid < 2:
            continue

        if torch.rand(1).item() < blank_prob:
            idx = valid[torch.randint(n_valid, (1,)).item()]
            out[b, idx] = 0.0

        if torch.rand(1).item() < repeat_prob and n_valid >= 2:
            src_pos = torch.randint(n_valid, (1,)).item()
            src_idx = valid[src_pos]
            offsets = [-1, 1]
            dst_pos = (src_pos + offsets[torch.randint(2, (1,)).item()]) % n_valid
            dst_idx = valid[dst_pos]
            out[b, dst_idx] = out[b, src_idx].clone()

        if noise_std > 0:
            tokens = out[b, valid]
            norms  = tokens.norm(dim=-1, keepdim=True)
            noise  = torch.randn_like(tokens) * noise_std * norms
            out[b, valid] = tokens + noise

        if torch.rand(1).item() < mixup_prob and n_valid >= 2:
            perm   = torch.randperm(n_valid, device=frame_cls.device)
            i1, i2 = valid[perm[0]], valid[perm[1]]
            lam    = torch.rand(1, device=frame_cls.device).item()
            mixed  = lam * out[b, i1] + (1 - lam) * out[b, i2]
            out[b, i1] = mixed

        if torch.rand(1).item() < shuffle_prob:
            perm          = torch.randperm(n_valid, device=frame_cls.device)
            out[b, valid] = out[b, valid[perm]]

        if torch.rand(1).item() < reverse_prob:
            out[b, valid] = out[b, valid.flip(0)]

        if torch.rand(1).item() < speed_prob and n_valid >= 4:
            rand_pos      = torch.randint(0, n_valid, (n_valid,), device=frame_cls.device)
            rand_pos, _   = rand_pos.sort()
            out[b, valid] = out[b, valid[rand_pos]]

    return out


# ---------------------------------------------------------------------------
# Temporal transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """Temporal encoder that uses only one CLS token per frame."""

    def __init__(
        self,
        embed_dim:  int = 1024,
        num_frames: int = 32,
        num_layers: int = 2,
        num_heads:  int = 8,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.num_frames  = num_frames
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.video_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed,   std=0.02)
        nn.init.trunc_normal_(self.video_token, std=0.02)

    def forward(self, frame_cls: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = frame_cls.shape
        if T > self.num_frames:
            raise ValueError(f"Expected at most {self.num_frames} frames, got {T}")

        x           = frame_cls + self.pos_embed[:, :T, :]
        video_token = self.video_token.expand(B, -1, -1)
        x           = torch.cat([video_token, x], dim=1)

        if key_padding_mask is not None:
            video_mask       = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([video_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0]


# ---------------------------------------------------------------------------
# VideoViT
# ---------------------------------------------------------------------------

class VideoViT(nn.Module):
    """
    Frame + video deepfake detection model.

    Fusion inputs to fusion_classifier:
      - temporal_vec      : concat of NUM_HEADS temporal transformer outputs
                            (NUM_HEADS * EMBED_DIM = 4096)
      - frame_mean_logits : mean of deepest MACHead logits over valid frames
                            (2)
      - real_sim_scores   : mean cosine similarity to k nearest real training
                            videos, one score per temporal head (NUM_HEADS = 4)
                            ONLY present when memory_bank is set; otherwise
                            this slot is absent and fusion_classifier has
                            input dim 4096 + 2 = 4098.

    When memory_bank is attached:
      fusion_classifier input dim = 4096 + 2 + 4 = 4102

    The memory_bank is NOT a nn.Module and is NOT saved in state_dict.
    It must be rebuilt and re-attached after loading a checkpoint.
    """

    EMBED_DIM = ViT.EMBED_DIM   # 1024
    NUM_HEADS = ViT.NUM_HEADS   # 4

    def __init__(
        self,
        num_frames:       int   = 32,
        temporal_layers:  int   = 2,
        temporal_heads:   int   = 8,
        temporal_dropout: float = 0.1,
        use_memory_bank:  bool  = False,
    ):
        super().__init__()
        self.num_frames      = num_frames
        self.use_memory_bank = use_memory_bank
        self.memory_bank: Optional[RealVideoMemoryBank] = None

        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim  = self.EMBED_DIM,
                num_frames = num_frames,
                num_layers = temporal_layers,
                num_heads  = temporal_heads,
                dropout    = temporal_dropout,
            )
            for _ in range(self.NUM_HEADS)
        ])

        # fusion_classifier input dim depends on whether memory bank is used.
        knn_dim = self.NUM_HEADS if use_memory_bank else 0
        self.fusion_classifier = nn.Linear(
            self.NUM_HEADS * self.EMBED_DIM + 2 + knn_dim, 2
        )

    def attach_memory_bank(self, bank: RealVideoMemoryBank):
        """
        Attach a pre-built RealVideoMemoryBank.
        Must be called after building the bank in train_stage2.py.
        """
        assert self.use_memory_bank, \
            "VideoViT was not constructed with use_memory_bank=True."
        assert bank.num_heads == self.NUM_HEADS, \
            f"Bank has {bank.num_heads} heads but model has {self.NUM_HEADS}."
        self.memory_bank = bank

    @property
    def vit(self):
        return self.frame_model.vit

    def forward(self, video: Tensor, lengths: Optional[Tensor] = None):
        B, T, C, H, W = video.shape
        if T > self.num_frames:
            raise ValueError(f"Expected at most {self.num_frames} frames, got {T}")
        frames = video.reshape(B * T, C, H, W)

        frame_logits_list, frame_feats_list, cls_list = self.frame_model(frames)

        if lengths is None:
            key_padding_mask = None
            valid_counts     = torch.full((B,), T, dtype=torch.float32, device=video.device)
        else:
            time_idx         = torch.arange(T, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)
            valid_counts     = lengths.to(video.device).float()

        # frame_mean_logits: mean of deepest MACHead logits over valid frames.
        frame_logits_bt  = frame_logits_list[3].reshape(B, T, 2).float()   # (B, T, 2)
        if key_padding_mask is not None:
            valid_mask       = (~key_padding_mask).float().unsqueeze(-1)    # (B, T, 1)
            frame_logits_bt  = frame_logits_bt * valid_mask
        frame_mean_logits = (
            frame_logits_bt.sum(dim=1) /
            valid_counts.unsqueeze(1).clamp(min=1)
        )                                                                    # (B, 2)

        # Temporal transformers.
        video_feats_list = []
        for temporal_tfm, cls_tokens in zip(self.temporal_transformers, cls_list):
            frame_cls = cls_tokens.reshape(B, T, self.EMBED_DIM)
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)                   # (B, 4096)

        # kNN real-video similarity scores (optional).
        if self.use_memory_bank:
            assert self.memory_bank is not None, \
                "use_memory_bank=True but no bank attached. Call attach_memory_bank() first."
            real_sim = self.memory_bank.query(video_feats_list)             # (B, 4)
            fused    = torch.cat([temporal_vec, frame_mean_logits, real_sim], dim=1)  # (B, 4102)
        else:
            fused    = torch.cat([temporal_vec, frame_mean_logits], dim=1)  # (B, 4098)

        video_logits = self.fusion_classifier(fused)                        # (B, 2)

        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        ckpt  = torch.load(image_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))

        frame_state = {}
        for key, value in state.items():
            if key.startswith("frame_model."):
                frame_state[key[len("frame_model."):]] = value
            elif not key.startswith(("temporal_transformers.", "fusion_classifier.")):
                frame_state[key] = value

        missing, unexpected = self.frame_model.load_state_dict(frame_state, strict=strict)
        print(f"Loaded image weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected