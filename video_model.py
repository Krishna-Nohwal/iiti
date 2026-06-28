import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional

from frame_model import ViT


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
                   (neutral in normalised embedding space)
    repeat_prob  — duplicate a random valid frame into a random adjacent slot
    noise_std    — add per-frame Gaussian noise scaled by the token's own norm,
                   so the perturbation is proportional to the feature magnitude
    mixup_prob   — interpolate two randomly chosen valid frames in-place
    shuffle_prob — randomly permute the order of valid frames (low probability
                   since real and fake videos are equally reversible; mainly
                   acts as a regulariser against strict temporal order)
    reverse_prob — reverse the temporal order of valid frames
    speed_prob   — re-sample valid frames at a non-uniform stride, simulating
                   variable frame rate (clusters some frames, spreads others)

    Returns
    -------
    frame_cls : (B, T, D)  augmented in-place copy
    """
    B, T, D = frame_cls.shape
    out = frame_cls.clone()

    for b in range(B):
        # Build list of valid frame indices for this sample.
        if key_padding_mask is not None:
            valid = (~key_padding_mask[b]).nonzero(as_tuple=True)[0]   # 1-D tensor
        else:
            valid = torch.arange(T, device=frame_cls.device)

        n_valid = valid.numel()
        if n_valid < 2:
            continue   # nothing meaningful to augment with < 2 frames

        # ── Blank frame ───────────────────────────────────────────────────
        # Replace one valid frame with a zero vector.  Zeros are more neutral
        # than black pixels would be after normalisation.
        if torch.rand(1).item() < blank_prob:
            idx = valid[torch.randint(n_valid, (1,)).item()]
            out[b, idx] = 0.0

        # ── Frame repetition ─────────────────────────────────────────────
        # Pick a source frame and copy it into a randomly chosen neighbour slot.
        if torch.rand(1).item() < repeat_prob and n_valid >= 2:
            src_pos  = torch.randint(n_valid, (1,)).item()
            src_idx  = valid[src_pos]
            # Choose a neighbour that is different from the source.
            offsets  = [-1, 1]
            dst_pos  = (src_pos + offsets[torch.randint(2, (1,)).item()]) % n_valid
            dst_idx  = valid[dst_pos]
            out[b, dst_idx] = out[b, src_idx].clone()

        # ── Embedding-space Gaussian noise ───────────────────────────────
        # Scale noise by each frame token's own L2 norm so that strongly
        # activating tokens receive proportionally larger perturbations.
        if noise_std > 0:
            tokens     = out[b, valid]                          # (n_valid, D)
            norms      = tokens.norm(dim=-1, keepdim=True)     # (n_valid, 1)
            noise      = torch.randn_like(tokens) * noise_std * norms
            out[b, valid] = tokens + noise

        # ── CLS token mixup ──────────────────────────────────────────────
        # Interpolate two randomly chosen valid frames in embedding space.
        # Destroys localised temporal cues without discarding information.
        if torch.rand(1).item() < mixup_prob and n_valid >= 2:
            perm   = torch.randperm(n_valid, device=frame_cls.device)
            i1, i2 = valid[perm[0]], valid[perm[1]]
            lam    = torch.rand(1, device=frame_cls.device).item()
            mixed  = lam * out[b, i1] + (1 - lam) * out[b, i2]
            out[b, i1] = mixed

        # ── Temporal shuffle ─────────────────────────────────────────────
        # Randomly permute valid frame order.  Low default probability because
        # temporal order is only a weak cue for real/fake distinction.
        if torch.rand(1).item() < shuffle_prob:
            perm           = torch.randperm(n_valid, device=frame_cls.device)
            out[b, valid]  = out[b, valid[perm]]

        # ── Temporal reverse ─────────────────────────────────────────────
        # Reverse the valid frame sequence.
        if torch.rand(1).item() < reverse_prob:
            out[b, valid] = out[b, valid.flip(0)]

        # ── Speed perturbation ───────────────────────────────────────────
        # Re-sample valid frames at a non-uniform stride by sorting a set of
        # random positions — this clusters some frames and spreads others,
        # simulating variable frame rate without dropping or adding frames.
        if torch.rand(1).item() < speed_prob and n_valid >= 4:
            # Sample n_valid positions in [0, n_valid) and sort them to get
            # a monotone but non-uniform index sequence.
            rand_pos = torch.randint(0, n_valid, (n_valid,), device=frame_cls.device)
            rand_pos, _ = rand_pos.sort()
            out[b, valid] = out[b, valid[rand_pos]]

    return out


class TemporalTransformer(nn.Module):
    """Temporal encoder that uses only one CLS token per frame."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_frames: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
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
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.video_token, std=0.02)

    def forward(self, frame_cls: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = frame_cls.shape
        if T > self.num_frames:
            raise ValueError(f"Expected at most {self.num_frames} frames, got {T}")

        x = frame_cls + self.pos_embed[:, :T, :]
        video_token = self.video_token.expand(B, -1, -1)
        x = torch.cat([video_token, x], dim=1)
        if key_padding_mask is not None:
            video_mask = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([video_mask, key_padding_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0]


class VideoViT(nn.Module):
    """
    Frame + video model.

    Frame logits come directly from frame_model.ViT. Video logits are produced
    by fusing two signals:
      - temporal_vec  : concat of NUM_HEADS temporal transformer outputs  (NUM_HEADS * EMBED_DIM)
      - frame_mean_logits : mean of deepest MACHead logits over valid frames  (2)

    These are concatenated and passed through fusion_classifier -> (B, 2).

    During training, temporal_augment() is applied to each head's CLS token
    sequence in embedding space, after the frozen backbone and before the
    temporal transformer. At eval time the augmentation is bypassed.
    """

    EMBED_DIM = ViT.EMBED_DIM
    NUM_HEADS = ViT.NUM_HEADS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList(
            [
                TemporalTransformer(
                    embed_dim=self.EMBED_DIM,
                    num_frames=num_frames,
                    num_layers=temporal_layers,
                    num_heads=temporal_heads,
                    dropout=temporal_dropout,
                )
                for _ in range(self.NUM_HEADS)
            ]
        )
        # Input: temporal_vec (NUM_HEADS * EMBED_DIM) + frame_mean_logits (2)
        self.fusion_classifier = nn.Linear(self.NUM_HEADS * self.EMBED_DIM + 2, 2)

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
            valid_counts = torch.full((B,), T, dtype=torch.float32, device=video.device)
        else:
            time_idx = torch.arange(T, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)
            valid_counts = lengths.to(video.device).float()

        # frame_mean_logits: mean of deepest MACHead logits over valid frames.
        # frame_logits_list[3] shape: (B*T, 2) — reshape to (B, T, 2),
        # zero out padding positions, sum, divide by valid count.
        frame_logits_bt = frame_logits_list[3].reshape(B, T, 2).float()  # (B, T, 2)
        if key_padding_mask is not None:
            # key_padding_mask: True = padding; zero those out before summing.
            valid_mask = (~key_padding_mask).float().unsqueeze(-1)        # (B, T, 1)
            frame_logits_bt = frame_logits_bt * valid_mask
        frame_mean_logits = frame_logits_bt.sum(dim=1) / valid_counts.unsqueeze(1).clamp(min=1)  # (B, 2)

        video_feats_list = []
        for temporal_tfm, cls_tokens in zip(self.temporal_transformers, cls_list):
            frame_cls = cls_tokens.reshape(B, T, self.EMBED_DIM)

            # Augment in embedding space during training only.
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)

            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)                # (B, NUM_HEADS * EMBED_DIM)
        fused        = torch.cat([temporal_vec, frame_mean_logits], dim=1)  # (B, NUM_HEADS * EMBED_DIM + 2)
        video_logits = self.fusion_classifier(fused)                      # (B, 2)

        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        ckpt = torch.load(image_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))

        frame_state = {}
        for key, value in state.items():
            if key.startswith("frame_model."):
                frame_state[key[len("frame_model."):]] = value
            elif not key.startswith(("temporal_transformers.", "fusion_classifier.")):
                frame_state[key] = value

        missing, unexpected = self.frame_model.load_state_dict(frame_state, strict=strict)
        print(f"Loaded image weights - missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected