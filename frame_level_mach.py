import timm
import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model


class AttentionPooler(nn.Module):
    """
    Single-query attention pooler over patch tokens.

    Reduces (B, N, embed_dim) patch tokens to a (B, head_dim) vector using
    a learned query. Uses a smaller head_dim (default 256) for k/v projections
    to keep parameter count low.

    Params per instance (embed_dim=1024, head_dim=256):
        query   : head_dim                     =   256
        k_proj  : embed_dim x head_dim (no b.) = 262,144
        v_proj  : embed_dim x head_dim (no b.) = 262,144
        Total                                  = 524,544
    """
    def __init__(self, embed_dim: int = 1024, head_dim: int = 256):
        super().__init__()
        self.scale  = head_dim ** -0.5
        self.query  = nn.Parameter(torch.zeros(1, 1, head_dim))
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, patch_tok: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tok : (B, N, embed_dim)
        Returns:
            pooled    : (B, head_dim)
        """
        B = patch_tok.size(0)
        q = self.query.expand(B, -1, -1)                        # (B, 1, head_dim)
        k = self.k_proj(patch_tok)                               # (B, N, head_dim)
        v = self.v_proj(patch_tok)                               # (B, N, head_dim)
        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) * self.scale, dim=-1      # (B, 1, N)
        )
        return (attn @ v).squeeze(1)                             # (B, head_dim)


class JointMACHead(nn.Module):
    """
    Joint Multi-Aspect Classification head consuming all 4 tapped layers
    simultaneously, enabling cross-layer interaction.

    Token streams per layer
    -----------------------
      cls_tok   : (B, 1,       embed_dim)
      reg_tok   : (B, num_reg, embed_dim)
      patch_tok : (B, H*W,     embed_dim)  -> attention-pooled to (B, head_dim)

    Grouping across NUM_LAYERS=4 layers
    ------------------------------------
      CLS stack : (B, 4 x 1024) = (B, 4096)
      REG stack : (B, 4 x 4 x 1024) = (B, 16384)
      AVG stack : (B, 4 x 256) = (B, 1024)   <- attn-pooled patches

    Each group is projected independently to proj_dim=256, then:
      - Three auxiliary Linear(256, 2) heads supervise each projector.
      - The three 256-dim vectors concatenate -> (B, 768) -> fusion MLP -> (B, 2).

    Parameter count (embed_dim=1024, num_reg=4, head_dim=256, proj_dim=256):
        4 x AttentionPooler                        2,098,176
        cls_proj  Linear(4096  -> 256)             1,048,832
        reg_proj  Linear(16384 -> 256)             4,194,560
        avg_proj  Linear(1024  -> 256)               262,400
        3 x aux   Linear(256   -> 2)                   1,542
        fusion    Linear(768->512) + Linear(512->256)
                  + Linear(256->2)                   525,570
        -------------------------------------------------
        Total                                      8,131,080

        vs old 4 x MACHead                       27,273,224
        Reduction                               -19,142,144  (-70.2%)

    Returns
    -------
    logits          : (B, 2)
    features        : (B, proj_dim)        fused feature vector for SupCon
    aux_logits_list : list of 3 x (B, 2)  [cls_aux, reg_aux, avg_aux]
    aux_feats_list  : list of 3 x (B, proj_dim)
    """
    NUM_LAYERS = 4

    def __init__(
        self,
        embed_dim: int = 1024,
        num_reg:   int = 4,
        head_dim:  int = 256,
        proj_dim:  int = 256,
        dropout_p: float = 0.4,
    ):
        super().__init__()
        self.num_reg  = num_reg
        self.head_dim = head_dim
        self.proj_dim = proj_dim

        # One attention pooler per tapped layer
        self.attn_poolers = nn.ModuleList([
            AttentionPooler(embed_dim, head_dim)
            for _ in range(self.NUM_LAYERS)
        ])

        # Grouped projectors
        cls_in = self.NUM_LAYERS * embed_dim            # 4096
        reg_in = self.NUM_LAYERS * num_reg * embed_dim  # 16384
        avg_in = self.NUM_LAYERS * head_dim             # 1024

        self.cls_proj = nn.Sequential(
            nn.Linear(cls_in, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.reg_proj = nn.Sequential(
            nn.Linear(reg_in, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.avg_proj = nn.Sequential(
            nn.Linear(avg_in, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # Per-projector auxiliary classifiers (deep supervision)
        self.cls_aux = nn.Linear(proj_dim, 2)
        self.reg_aux = nn.Linear(proj_dim, 2)
        self.avg_aux = nn.Linear(proj_dim, 2)

        # Fusion MLP
        fused_in = proj_dim * 3   # 768
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Linear(proj_dim, 2)

    def forward(
        self,
        cls_list:   list,   # NUM_LAYERS x (B, 1,       embed_dim)
        reg_list:   list,   # NUM_LAYERS x (B, num_reg, embed_dim)
        patch_list: list,   # NUM_LAYERS x (B, H*W,     embed_dim)
    ):
        B = cls_list[0].size(0)

        # Attention-pool patch tokens per layer -> (B, head_dim) each
        avg_tokens = [
            pooler(patch_tok)
            for pooler, patch_tok in zip(self.attn_poolers, patch_list)
        ]

        # Stack and flatten across layers
        cls_stack = torch.cat([t.squeeze(1)   for t in cls_list],  dim=1).float()  # (B, 4096)
        reg_stack = torch.cat([t.reshape(B,-1) for t in reg_list], dim=1).float()  # (B, 16384)
        avg_stack = torch.cat(avg_tokens,                           dim=1).float()  # (B, 1024)

        # Grouped projections
        f_cls = self.cls_proj(cls_stack)   # (B, proj_dim)
        f_reg = self.reg_proj(reg_stack)   # (B, proj_dim)
        f_avg = self.avg_proj(avg_stack)   # (B, proj_dim)

        # Auxiliary logits for deep supervision
        aux_logits_list = [self.cls_aux(f_cls), self.reg_aux(f_reg), self.avg_aux(f_avg)]
        aux_feats_list  = [f_cls, f_reg, f_avg]

        # Fusion
        fused    = torch.cat([f_cls, f_reg, f_avg], dim=1)  # (B, 768)
        features = self.fusion(fused)                         # (B, proj_dim)
        logits   = self.classifier(features)                  # (B, 2)

        return logits, features, aux_logits_list, aux_feats_list


class ViT(nn.Module):
    """
    DINOv3 ViT-Large/16 with 4 register tokens, finetuned with LoRA.

    Forward pass taps intermediate outputs from layers [20, 21, 22, 23].
    All 4 layers feed a single JointMACHead for cross-layer interaction,
    with attention pooling over patch tokens instead of simple averaging.

    Return signature is kept compatible with video_model.VideoViT:
        logits_list   : list of 4 x (B, 2)   — index [3] is the primary fused
                        logit; indices [0..2] are the three aux logits reordered
                        so that video_model's frame_logits_list[3] and [0] both
                        give valid (B, 2) tensors of the right shape.
        features_list : list of 4 x (B, 256) — mirrors logits_list layout.
        cls_list      : list of 4 x (B, 1024) — raw per-layer CLS tokens,
                        consumed by VideoViT's temporal transformers unchanged.

    Shapes per forward call (batch size B, image size 256x256):
        patch grid    : 16x16 = 256 patches
        prefix_tokens : [CLS, REG_1, REG_2, REG_3, REG_4] -> 5 tokens
        spatial_map   : (B, 1024, 16, 16)
        patch_tok     : (B, 256,  1024)
        cls_tok       : (B, 1,    1024)
        reg_tok       : (B, 4,    1024)
    """
    EMBED_DIM = 1024
    NUM_REG   = 4
    NUM_HEADS = 4          # kept for VideoViT compatibility (num temporal transformers)
    HEAD_DIM  = 256        # attention pooler k/v dim
    PROJ_DIM  = 256        # grouped projector output dim
    LAYERS    = [20, 21, 22, 23]
    DROP_PATH = 0.10
    MAC_DROP  = 0.4

    def __init__(self):
        super().__init__()

        # Backbone
        self.vit = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m',
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

        # Single joint head consuming all 4 tapped layers
        self.mac_head = JointMACHead(
            embed_dim=self.EMBED_DIM,
            num_reg=self.NUM_REG,
            head_dim=self.HEAD_DIM,
            proj_dim=self.PROJ_DIM,
            dropout_p=self.MAC_DROP,
        )

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=self.LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        cls_list   = []
        reg_list   = []
        patch_list = []

        for spatial_map, prefix_tokens in intermediates:
            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            cls_list.append(cls_tok)
            reg_list.append(reg_tok)
            patch_list.append(patch_tok)

        logits, features, aux_logits_list, aux_feats_list = self.mac_head(
            cls_list, reg_list, patch_list
        )

        # ── Compatibility shim for video_model.VideoViT ──────────────────────
        # VideoViT expects:
        #   frame_logits_list[0]  — any (B, 2) tensor  (used only for shape: B*T // B = T)
        #   frame_logits_list[3]  — primary logits      (used for frame_mean_logits)
        #   features_list         — passed through, not used in stage2 loss
        #   cls_list              — raw CLS tokens per layer for temporal transformers
        #
        # We pack aux logits at [0..2] and the primary fused logit at [3],
        # matching the old 4-head layout positionally.
        logits_list   = aux_logits_list + [logits]          # [aux_cls, aux_reg, aux_avg, fused]
        features_list = aux_feats_list  + [features]        # mirrors logits_list

        # cls_list entries are (B, 1, embed_dim); squeeze to (B, embed_dim) to match
        # the old layout that VideoViT's temporal transformers reshape from (B*T, C)
        # back to (B, T, C). We keep the raw 1-token dim so the reshape in
        # video_model (cls_tokens.reshape(B, T, EMBED_DIM)) still works correctly
        # — the squeeze happens there via .squeeze(1) already done in the old code.
        # So cls_list shape stays: 4 x (B, 1, embed_dim), same as before.

        return logits_list, features_list, cls_list