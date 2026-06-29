import timm
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class AttentionPool(nn.Module):
    """
    Lightweight learnable attention pooling over a token sequence.

    Scores each token via a single learned query vector (no key projection),
    then returns a softmax-weighted sum. Works for any sequence length N.

    Params: embed_dim  (just the query vector — 1,024 for ViT-Large)

    Input : (B, N, C)
    Output: (B, C)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.scale  = embed_dim ** -0.5
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, C)
        Returns:
            pooled : (B, C)
        """
        q    = self.query.expand(x.size(0), -1, -1)   # (B, 1, C)
        attn = (q @ x.transpose(-2, -1)) * self.scale  # (B, 1, N)
        attn = attn.softmax(dim=-1)                     # (B, 1, N)
        return (attn @ x).squeeze(1)                    # (B, C)


class SpatialHead(nn.Module):
    """
    Spatial Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 512-dim intermediate feature vector.

    Both REG and patch tokens are attention-pooled to (B, C) before
    concatenation, so all three spatial granularities contribute equally:

        [f_cls | f_reg | f_patch]  →  (B, 3C) = (B, 3072)

        f_cls   — global spatial summary
        f_reg   — artifact-localized spatial outliers
        f_patch — local spatial features

    Previously REG was flattened to (B, 4C), dominating 4/6 of the input.
    Now each component is (B, C), contributing 1/3 each.

    Params added per head:
        patch_pool.query : C     =  1,024
        reg_pool.query   : C     =  1,024
        Total            :       ~  2,048  (essentially free)

    Input dim:  3 * embed_dim  =  3 * 1024  =  3072
    Hidden dim: embed_dim      =  1024
    Bottle dim: embed_dim // 2 =  512        ← returned as `features`
    Output dim: 2  (real / fake logits)
    """
    def __init__(self, embed_dim: int = 1024, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg    = num_reg
        in_dim          = 3 * embed_dim   # 3C = 3072  (cls + reg_pooled + patch_pooled)

        self.patch_pool = AttentionPool(embed_dim)   # 256 patch tokens → (B, C)
        self.reg_pool   = AttentionPool(embed_dim)   # 4   reg   tokens → (B, C)

        self.head = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Linear(embed_dim // 2, 2)

    def forward(self, cls_tok, reg_tok, patch_tok):
        """
        Args:
            cls_tok   : (B, 1,       embed_dim)
            reg_tok   : (B, num_reg, embed_dim)
            patch_tok : (B, H*W,     embed_dim)
        Returns:
            dict with keys:
                f_cls    : (B, C)
                f_reg    : (B, C)   attention-pooled (was flattened 4C before)
                f_patch  : (B, C)   attention-pooled
                logits   : (B, 2)
                features : (B, C//2 = 512)
        """
        f_cls   = cls_tok.squeeze(1)          # (B, C)
        f_reg   = self.reg_pool(reg_tok)       # (B, C)  — 4 tokens → 1
        f_patch = self.patch_pool(patch_tok)   # (B, C)  — 256 tokens → 1

        inp = torch.cat([f_cls, f_reg, f_patch], dim=1).float()  # (B, 3C)
        h   = self.head(inp)                                       # (B, C/2)

        return {
            "f_cls":    f_cls,
            "f_reg":    f_reg,
            "f_patch":  f_patch,
            "logits":   self.classifier(h),
            "features": h,
        }


class ViT(nn.Module):
    """
    DINOv3 ViT-Large/16 with 4 register tokens, finetuned with LoRA.

    Forward pass taps intermediate outputs from layers [19, 20, 21, 22, 23].
    Each layer feeds its own SpatialHead → 5 sets of (logits, 512-dim features).

    Architecture differences vs ViT-Base:
        - embed_dim 1024 (vs 768)
        - 24 transformer blocks (vs 12)   →  tapped layers: [20, 21, 22, 23]
        - patch size 16 (unchanged)       →  patch grid for 256×256: 16×16 = 256 patches
        - SwiGLU FFN + RoPE positional encoding (same as Base variant)
        - Distilled from 7B teacher on LVD-1689M dataset

    Shapes per forward call (batch size B, image size 256×256):
        patch grid    : 16×16 = 256 patches
        prefix_tokens : [CLS, REG_1, REG_2, REG_3, REG_4]  → 5 tokens
        spatial_map   : (B, 1024, 16, 16)
        patch_tok     : (B, 256, 1024)
        cls_tok       : (B, 1,   1024)
        reg_tok       : (B, 4,   1024)

    Returns:
        logits_list   : list of 5 × (B, 2)    — one per tapped layer
        features_list : list of 5 × (B, 512)  — one per tapped layer
    """
    EMBED_DIM  = 1024   # ViT-Large hidden size
    NUM_REG    = 4
    NUM_LAYERS = 5      # number of tapped layers → one SpatialHead each
    LAYERS     = [19, 20, 21, 22, 23]   # ViT-Large has 24 blocks (0-indexed)
    DROP_PATH  = 0.10
    HEAD_DROP  = 0.4

    def __init__(self):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────
        self.vit = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=32,
            lora_alpha=64,        # 2× r
            target_modules=["attn.qkv"],  # verify attr name: may be "qkv" depending on timm version
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)

        # ── One SpatialHead per tapped layer ────────────────────────────
        self.spatial_heads = nn.ModuleList([
            SpatialHead(self.EMBED_DIM, self.NUM_REG, self.HEAD_DROP)
            for _ in range(self.NUM_LAYERS)
        ])

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=self.LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        logits_list:   list = []
        features_list: list = []
        cls_list:      list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            result = self.spatial_heads[i](cls_tok, reg_tok, patch_tok)
            logits_list.append(result["logits"])
            features_list.append(result["features"])
            cls_list.append(result["f_cls"])


        return logits_list, features_list, cls_list