from typing import Optional

import torch
from torch import Tensor, nn

from frame_model_small import ViT


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_frames: int = 32,
        num_layers: int = 2,
        num_heads: int = 6,
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
    EMBED_DIM = ViT.EMBED_DIM
    NUM_HEADS = ViT.NUM_HEADS

    def __init__(
        self,
        num_frames: int = 32,
        temporal_layers: int = 2,
        temporal_heads: int = 6,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_model = ViT()
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim=self.EMBED_DIM,
                num_frames=num_frames,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=temporal_dropout,
            )
            for _ in range(self.NUM_HEADS)
        ])
        self.video_classifier = nn.Linear(self.NUM_HEADS * self.EMBED_DIM, 2)

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
        else:
            time_idx = torch.arange(T, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)

        video_feats_list = []
        for temporal_tfm, cls_tokens in zip(self.temporal_transformers, cls_list):
            frame_cls = cls_tokens.reshape(B, T, self.EMBED_DIM)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))

        video_vec = torch.cat(video_feats_list, dim=1)
        video_logits = self.video_classifier(video_vec)
        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        ckpt = torch.load(image_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))

        frame_state = {}
        for key, value in state.items():
            if key.startswith("frame_model."):
                frame_state[key[len("frame_model."):]] = value
            elif not key.startswith(("temporal_transformers.", "video_classifier.")):
                frame_state[key] = value

        missing, unexpected = self.frame_model.load_state_dict(frame_state, strict=strict)
        print(f"Loaded image weights - missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected
