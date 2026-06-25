import timm
import torch
from torch import nn
from torch import Tensor
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from image model
# ─────────────────────────────────────────────────────────────────────────────

class MACHead(nn.Module):
    """
    Multi-Aspect Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 192-dim intermediate feature vector.

    Input dim:  (1 + num_reg + 1) * embed_dim  =  6 * 384  =  2304
    Hidden dim: embed_dim                        =  384
    Bottle dim: embed_dim // 2                   =  192      ← returned as `h`
    Output dim: 2  (real / fake logits)
    """
    def __init__(self, embed_dim: int = 384, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg = num_reg
        in_dim = (1 + num_reg + 1) * embed_dim   # 6C = 2304

        self.head = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Linear(embed_dim // 2, 2)

    def forward(self, cls_tok: Tensor, reg_tok: Tensor, patch_tok: Tensor):
        """
        Args:
            cls_tok   : (B, 1,       embed_dim)
            reg_tok   : (B, num_reg, embed_dim)
            patch_tok : (B, H*W,     embed_dim)
        Returns:
            logits : (B, 2)
            h      : (B, embed_dim // 2)  — 192-dim discriminative features
        """
        B     = cls_tok.size(0)
        f_avg = patch_tok.mean(dim=1)
        f_cls = cls_tok.squeeze(1)
        f_reg = reg_tok.reshape(B, -1)
        inp   = torch.cat([f_cls, f_reg, f_avg], dim=1).float()
        h     = self.head(inp)
        return self.classifier(h), h


# ─────────────────────────────────────────────────────────────────────────────
# New: temporal self-attention block
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """
    Standard multi-head self-attention over a sequence of T frame features.

    Every frame attends to every other frame simultaneously (parallel, not
    recurrent). Positional embeddings are added before attention so the model
    knows frame order without being constrained by it.

    Args:
        dim       : feature dimension of each frame token (192)
        num_heads : number of attention heads (dim must be divisible by this)
        dropout   : dropout on attention weights

    Input / output shape:  (B, T, dim)  — unchanged, just contextualised
    """
    def __init__(self, dim: int = 192, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Q, K, V projections — each maps (B, T, dim) → (B, T, dim)
        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, T, dim)   T = number of frames = 32
        Returns:
            out : (B, T, dim)  each frame is now a weighted mix of all frames
        """
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # Project and split into heads
        # (B, T, D) → (B, T, H, Dh) → (B, H, T, Dh)
        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product attention across the T=32 frame dimension
        # attn[b, h, t, t'] = "how much should frame t attend to frame t'?"
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values, then merge heads back
        # (B, H, T, Dh) → (B, T, H, Dh) → (B, T, D)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TemporalTransformerLayer(nn.Module):
    """
    One transformer encoder layer over the frame sequence:
        LayerNorm → TemporalSelfAttention → residual
        LayerNorm → FFN                  → residual

    Pre-norm (norm before attention) is more stable than post-norm,
    especially with a small number of layers.
    """
    def __init__(self, dim: int = 192, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = TemporalSelfAttention(dim, num_heads, dropout)

        mlp_dim    = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x : (B, T, dim)
        x = x + self.attn(self.norm1(x))   # attention  + residual
        x = x + self.ffn(self.norm2(x))    # feed-forward + residual
        return x


class TemporalTransformer(nn.Module):
    """
    Stacks N transformer layers over a sequence of T frame features.
    Adds a learnable positional embedding per frame slot before the first layer.
    Appends a learnable [VID] token whose output is used for final classification —
    this is the same pattern as the [CLS] token in BERT.

    Args:
        feat_dim   : dimension of per-frame features coming from MACHead (192)
        num_frames : fixed number of frames per video (32)
        num_layers : how many transformer layers to stack (2 is enough to start)
        num_heads  : attention heads inside each layer
        dropout    : dropout throughout

    Input:
        frame_feats : (B, T, feat_dim)   — one 192-dim vector per frame

    Output:
        vid_feat : (B, feat_dim)   — single video-level representation
    """
    def __init__(self, feat_dim: int = 192, num_frames: int = 32,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_dim   = feat_dim
        self.num_frames = num_frames

        # Learnable positional embedding: one vector per frame slot.
        # Tells the model "this is frame 0", "this is frame 1", etc.
        self.pos_embed = nn.Embedding(num_frames, feat_dim)
        self.register_buffer('pos_idx', torch.arange(num_frames))  # (T,) — fixed index tensor

        # Learnable [VID] token, prepended to the sequence.
        # After all layers, this token summarises the entire video.
        self.vid_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.vid_token, std=0.02)

        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(feat_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, frame_feats: Tensor) -> Tensor:
        """
        Args:
            frame_feats : (B, T, 192)   one feature vector per frame
        Returns:
            vid_feat    : (B, 192)      video-level feature from [VID] token
        """
        B, T, D = frame_feats.shape

        # 1. Add positional embeddings so the model knows frame order
        x = frame_feats + self.pos_embed(self.pos_idx)   # (B, T, D)

        # 2. Prepend the [VID] token — it will attend to all frames
        vid_tok = self.vid_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([vid_tok, x], dim=1)               # (B, T+1, D)

        # 3. Run all transformer layers — all frames attend to all others
        for layer in self.layers:
            x = layer(x)                                 # (B, T+1, D)

        x = self.norm(x)

        # 4. Extract only the [VID] token output as the video summary
        vid_feat = x[:, 0]                               # (B, D)
        return vid_feat


# ─────────────────────────────────────────────────────────────────────────────
# Video-level model
# ─────────────────────────────────────────────────────────────────────────────

class VideoViT(nn.Module):
    """
    Extends ViT (image model) to video by adding a TemporalTransformer on top
    of the per-frame MACHead features.

    Pipeline:
        video (B, T, 3, H, W)
            ↓  reshape to (B*T, 3, H, W) — process all frames at once
        ViT backbone + 4 × MACHead
            ↓  per-frame features: 4 × (B*T, 192)
            ↓  reshape to 4 × (B, T, 192)
        4 × TemporalTransformer (one per MACHead layer)
            ↓  per-layer video feature: 4 × (B, 192)
        Concat → (B, 4*192 = 768)
        Linear classifier → (B, 2)

    The ViT backbone + MACHeads are identical to the image model and can be
    initialised from a pretrained image checkpoint. Only the TemporalTransformers
    and the final video classifier are new parameters.

    Args:
        num_frames       : frames per video (default 32)
        temporal_layers  : transformer layers in each TemporalTransformer (default 2)
        temporal_heads   : attention heads in each TemporalTransformer (default 4)
        temporal_dropout : dropout in temporal transformer (default 0.1)
    """

    EMBED_DIM = 384
    NUM_REG   = 4
    NUM_HEADS = 4          # number of MACHead layers tapped
    LAYERS    = [8, 9, 10, 11]
    DROP_PATH = 0.15
    MAC_DROP  = 0.4
    FEAT_DIM  = 192        # MACHead output dimension

    def __init__(
        self,
        num_frames:       int   = 32,
        temporal_layers:  int   = 2,
        temporal_heads:   int   = 4,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames

        # ── Backbone (identical to image ViT) ───────────────────────────
        self.vit = timm.create_model(
            'vit_small_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["attn.qkv"],
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)

        # ── Per-layer MACHeads (identical to image ViT) ─────────────────
        self.mac_heads = nn.ModuleList([
            MACHead(self.EMBED_DIM, self.NUM_REG, self.MAC_DROP)
            for _ in range(self.NUM_HEADS)
        ])

        # ── One TemporalTransformer per MACHead layer (NEW) ──────────────
        # Each operates on (B, T, 192) independently so they can specialise
        # on the features at their respective ViT depth.
        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                feat_dim   = self.FEAT_DIM,
                num_frames = num_frames,
                num_layers = temporal_layers,
                num_heads  = temporal_heads,
                dropout    = temporal_dropout,
            )
            for _ in range(self.NUM_HEADS)
        ])

        # ── Final video-level classifier (NEW) ──────────────────────────
        # Concatenates the [VID] token from all 4 temporal transformers.
        self.video_classifier = nn.Linear(self.NUM_HEADS * self.FEAT_DIM, 2)

    def forward(self, video: Tensor):
        """
        Args:
            video : (B, T, 3, H, W)   B=batch, T=frames, spatial H×W

        Returns:
            video_logits  : (B, 2)                  — video-level real/fake
            frame_logits  : list of 4 × (B*T, 2)   — per-frame logits (for aux loss)
            frame_feats   : list of 4 × (B, T, 192) — per-frame features
            video_feats   : list of 4 × (B, 192)    — per-layer video features
        """
        B, T, C, H, W = video.shape

        # ── 1. Run ViT on all frames at once ────────────────────────────
        # Merge batch and time: (B, T, C, H, W) → (B*T, C, H, W)
        frames = video.reshape(B * T, C, H, W)

        _, intermediates = self.vit.forward_intermediates(
            frames,
            indices=self.LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        # ── 2. MACHead: produce per-frame features ───────────────────────
        frame_logits_list: list = []
        frame_feats_list:  list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            BT, Cemb, Hg, Wg = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(BT, Hg * Wg, Cemb)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            # logits: (B*T, 2)   feats: (B*T, 192)
            logits, feats = self.mac_heads[i](cls_tok, reg_tok, patch_tok)
            frame_logits_list.append(logits)

            # Restore time dimension: (B*T, 192) → (B, T, 192)
            frame_feats_list.append(feats.reshape(B, T, self.FEAT_DIM))

        # ── 3. TemporalTransformer: attend across frames ─────────────────
        # Each transformer receives (B, T, 192) and returns (B, 192).
        video_feats_list: list = []
        for i, temporal_tfm in enumerate(self.temporal_transformers):
            vid_feat = temporal_tfm(frame_feats_list[i])   # (B, 192)
            video_feats_list.append(vid_feat)

        # ── 4. Final video-level classification ──────────────────────────
        # Cat all 4 video features: (B, 4*192 = 768) → (B, 2)
        video_vec    = torch.cat(video_feats_list, dim=1)  # (B, 768)
        video_logits = self.video_classifier(video_vec)    # (B, 2)

        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        """
        Load ViT backbone + MACHead weights from a pretrained image checkpoint.
        TemporalTransformers and video_classifier start from random init.

        Args:
            image_ckpt_path : path to the image model .pt / .pth checkpoint
            strict          : whether to require all keys to match (default False,
                              since the checkpoint won't have temporal layers)
        """
        ckpt = torch.load(image_ckpt_path, map_location='cpu')
        # Checkpoints are sometimes wrapped in a 'state_dict' or 'model' key
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f"Loaded image weights — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected