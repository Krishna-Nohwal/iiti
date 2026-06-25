import timm
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class MACHead(nn.Module):
    """
    Multi-Aspect Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 512-dim intermediate feature vector.

    Input dim:  (1 + num_reg + 1) * embed_dim  =  6 * 1024  =  6144
    Hidden dim: embed_dim                        =  1024
    Bottle dim: embed_dim // 2                   =  512       ← returned as `h`
    Output dim: 2  (real / fake logits)
    """
    def __init__(self, embed_dim: int = 1024, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg = num_reg
        in_dim = (1 + num_reg + 1) * embed_dim   # 6C = 6144

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
            logits : (B, 2)
            h      : (B, embed_dim // 2)  — 512-dim discriminative features
        """
        B     = cls_tok.size(0)
        f_avg = patch_tok.mean(dim=1)        # (B, C)    spatial average
        f_cls = cls_tok.squeeze(1)           # (B, C)    CLS token
        f_reg = reg_tok.reshape(B, -1)       # (B, 4*C)  REG tokens flattened
        inp   = torch.cat([f_cls, f_reg, f_avg], dim=1).float()  # (B, 6C)
        h     = self.head(inp)                                     # (B, C/2 = 512)
        result = {"f_avg":f_avg,
                  "f_cls":f_cls,
                  "f_reg":f_reg,
                  "logits":self.classifier(h), #logits
                  "features":h} #features
        return  result


class ViT(nn.Module):
    """
    DINOv3 ViT-Large/16 with 4 register tokens, finetuned with LoRA.

    Forward pass taps intermediate outputs from layers [20, 21, 22, 23].
    Each layer feeds its own MACHead → 4 sets of (logits, 512-dim features).

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
        logits_list   : list of 4 × (B, 2)    — one per tapped layer
        features_list : list of 4 × (B, 512)  — one per tapped layer
    """
    EMBED_DIM = 1024   # ViT-Large hidden size
    NUM_REG   = 4
    NUM_HEADS = 4
    LAYERS    = [20, 21, 22, 23]   # ViT-Large has 24 blocks (0-indexed)
    DROP_PATH = 0.10
    MAC_DROP  = 0.4

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
            r=64,
            lora_alpha=128,        # 2× r
            target_modules=["attn.qkv"],  # verify attr name: may be "qkv" depending on timm version
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)

        #modify here to use the new "results" dict

        # ── One MACHead per tapped layer ────────────────────────────────
        self.mac_heads = nn.ModuleList([
            MACHead(self.EMBED_DIM, self.NUM_REG, self.MAC_DROP)
            for _ in range(self.NUM_HEADS)
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
        cls_list : list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            result = self.mac_heads[i](cls_tok, reg_tok, patch_tok)
            logits_list.append(result["logits"])
            features_list.append(result["feats"])
            cls_list : list = [] 

        return logits_list, features_list, cls_list