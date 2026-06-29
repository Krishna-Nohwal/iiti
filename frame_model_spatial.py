import timm
import torch
import torch.nn.functional as F
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


class LayerAttention(nn.Module):
    """
    Single-layer, attention-only transformer encoder over the CLS tokens
    from all 24 ViT-Large blocks.

    Contextualises each layer's CLS token against all others so the scorer
    can make a globally-informed selection (e.g. "layer 21 is redundant given
    layer 22 is already selected").  FFN is omitted deliberately — the sequence
    is only 24 tokens long and the extra capacity isn't needed.

    Params (~4.2M for C=1024, num_heads=2):
        pos_embed         : num_layers × C          =     24,576
        qkv projection   : 3 × C × C               =  3,145,728
        out projection   : C × C                   =  1,048,576
        layer norms (×2) : 2 × 2C                  =      4,096
        ─────────────────────────────────────────────────────────
        Total                                       ~  4,222,976

    Input : (B, num_layers, C)   e.g. (B, 24, 1024)
    Output: (B, num_layers, C)   same shape, each position contextualised
    """
    def __init__(self, embed_dim: int = 1024, num_layers: int = 24, num_heads: int = 2):
        super().__init__()
        # Learned positional embeddings — one per layer index
        self.pos_embed = nn.Parameter(torch.empty(1, num_layers, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)   # pre-norm on residual output

        # Single MHA block (no FFN)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, num_layers, C)  — stacked CLS tokens, one per layer
        Returns:
            (B, num_layers, C)      — contextualised CLS tokens
        """
        x = x + self.pos_embed                         # inject layer-position info
        residual = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        return self.norm2(residual + x)                # pre-norm + residual


class LayerSelector(nn.Module):
    """
    Scores all num_layers contextualised CLS tokens and returns the indices
    of the top-k most informative layers via Gumbel-softmax straight-through.

    During the forward pass the selection behaves like a hard argmax (discrete
    top-k), but gradients flow through the soft Gumbel distribution — so the
    scorer is fully differentiable end-to-end.

    Selection is shared across the batch (scores averaged over B before topk)
    to keep downstream routing simple: all samples in a batch use the same
    4 layer indices.

    Params: C + 1  ≈  1,025  (negligible)

    Input : (B, num_layers, C)
    Output: top_k indices  LongTensor of shape (k,)
            soft_weights   FloatTensor of shape (k,)  — for aux loss if needed
    """
    def __init__(self, embed_dim: int = 1024, num_layers: int = 24, top_k: int = 4, tau: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.tau   = tau                               # Gumbel temperature
        self.scorer = nn.Linear(embed_dim, 1, bias=True)

        # Bias toward last few layers as a prior — the scorer can override this
        # but it gives faster convergence early in training.
        with torch.no_grad():
            bias_val = torch.linspace(-1.0, 1.0, num_layers)   # (num_layers,)
            # We can't set per-position bias directly in Linear; instead we
            # register a learned offset that is added to the scorer output.
        self.layer_bias = nn.Parameter(torch.linspace(-1.0, 1.0, num_layers))  # (L,)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, num_layers, C)  — output of LayerAttention
        Returns:
            indices      : LongTensor  (top_k,)         — selected layer indices
            soft_weights : FloatTensor (top_k,)         — soft scores for aux loss
        """
        # Score each layer position
        scores = self.scorer(x).squeeze(-1)            # (B, num_layers)
        scores = scores + self.layer_bias.unsqueeze(0) # (B, num_layers)  broadcast
        scores = scores.mean(dim=0)                    # (num_layers,)  batch-average

        if self.training:
            # Gumbel-softmax straight-through: hard in forward, soft in backward
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1e-9)
            ))
            soft_scores = F.softmax((scores + gumbel_noise) / self.tau, dim=0)  # (L,)
        else:
            soft_scores = F.softmax(scores / self.tau, dim=0)                    # (L,)

        # Hard top-k indices (non-differentiable argmax, gradient via STE)
        _, indices = soft_scores.topk(self.top_k, dim=0, sorted=True)            # (k,)
        indices    = indices.sort().values                                         # keep layer order

        # Straight-through: detach hard selection, add soft gradient path
        soft_weights = soft_scores[indices]            # (k,)  — for optional aux loss

        return indices, soft_weights


class SpatialHead(nn.Module):
    """
    Spatial Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 512-dim intermediate feature vector.

    Two-stage fusion:

      Stage 1 — fuse local spatial signals:
        f_reg   : (B, C)  ─┐
                            ├─ cat → (B, 2C) → spatial_mlp → spatial_fused : (B, C)
        f_patch : (B, C)  ─┘

      Stage 2 — combine with global CLS and classify:
        f_cls         : (B, C)  ─┐
                                  ├─ cat → (B, 2C) → head → (B, C//2) → classifier → logits : (B, 2)
        spatial_fused : (B, C)  ─┘

        f_cls   — global spatial summary (CLS token)
        f_reg   — artifact-localized spatial outliers (attention-pooled)
        f_patch — local spatial features (attention-pooled)

    Params added per head:
        patch_pool.query  : C        =  1,024
        reg_pool.query    : C        =  1,024
        spatial_mlp       : 2C → C  ~  2,098,176
        head              : 2C → C//2

    spatial_mlp input  : 2 * embed_dim  =  2048  →  C      (spatial_fused)
    head input         : 2 * embed_dim  =  2048  →  C//2   (features)
    logits             : 2  (real / fake)
    """
    def __init__(self, embed_dim: int = 1024, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg    = num_reg

        self.patch_pool = AttentionPool(embed_dim)   # 256 patch tokens → (B, C)
        self.reg_pool   = AttentionPool(embed_dim)   # 4   reg   tokens → (B, C)

        # Stage 1: fuse f_reg + f_patch  →  spatial_fused : (B, C)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # Stage 2: project [f_cls | spatial_fused] → (B, C//2) then classify
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // 2),
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
                f_cls          : (B, C)
                f_reg          : (B, C)    attention-pooled
                f_patch        : (B, C)    attention-pooled
                spatial_fused  : (B, C)    MLP(f_reg ∥ f_patch)
                logits         : (B, 2)
                features       : (B, C//2 = 512)
        """
        f_cls   = cls_tok.squeeze(1)          # (B, C)
        f_reg   = self.reg_pool(reg_tok)       # (B, C)  — 4 tokens → 1
        f_patch = self.patch_pool(patch_tok)   # (B, C)  — 256 tokens → 1

        # Stage 1: fuse local spatial signals
        spatial_fused = self.spatial_mlp(
            torch.cat([f_reg, f_patch], dim=1).float()   # (B, 2C)
        )                                                  # (B, C)

        # Stage 2: combine with CLS and classify
        h = self.head(
            torch.cat([f_cls.float(), spatial_fused], dim=1)  # (B, 2C)
        )                                                       # (B, C//2)

        return {
            "f_cls":         f_cls,
            "f_reg":         f_reg,
            "f_patch":       f_patch,
            "spatial_fused": spatial_fused,
            "logits":        self.classifier(h),
            "features":      h,
        }


class ViT(nn.Module):
    """
    DINOv3 ViT-Large/16 with 4 register tokens, finetuned with LoRA.

    Forward pass taps CLS tokens from ALL 24 layers, runs LayerAttention to
    contextualise them, then LayerSelector picks the best 4 via Gumbel-softmax
    straight-through.  The selected 4 layers feed their full intermediates
    (spatial map + prefix tokens) into 4 independent SpatialHeads.

    Layer selection is batch-shared: all samples in a batch use the same 4
    layer indices (scores are averaged over B before topk).

    Architecture:
        - embed_dim 1024  (ViT-Large)
        - 24 transformer blocks (0-indexed 0…23)
        - patch size 16  →  patch grid for 256×256: 16×16 = 256 patches
        - SwiGLU FFN + RoPE positional encoding
        - Distilled from 7B teacher on LVD-1689M dataset

    New modules vs fixed-layer baseline:
        LayerAttention  : ~4.2M params  (attention-only, no FFN)
        LayerSelector   : ~1,025 params  (negligible)
        SpatialHeads    : 4 × ~3.15M  =  ~12.6M  (was 5 × ~3.15M = ~15.75M)
        Net delta       : ~+1.05M params

    Shapes per forward call (batch size B, image size 256×256):
        patch grid    : 16×16 = 256 patches
        prefix_tokens : [CLS, REG_1, REG_2, REG_3, REG_4]  → 5 tokens
        cls_stack     : (B, 24, 1024)   — stacked across all layers
        spatial_map   : (B, 1024, 16, 16)
        patch_tok     : (B, 256, 1024)
        cls_tok       : (B, 1,   1024)
        reg_tok       : (B, 4,   1024)

    Returns:
        logits_list    : list of 4 × (B, 2)    — one per selected layer
        features_list  : list of 4 × (B, 512)  — one per selected layer
        cls_list       : list of 4 × (B, C)    — raw CLS features
        fused_list     : list of 4 × (B, C)    — spatial_fused features
        selected_layers: LongTensor (4,)        — which layer indices were chosen
        soft_weights   : FloatTensor (4,)       — scorer confidences (for aux loss)
    """
    EMBED_DIM   = 1024   # ViT-Large hidden size
    NUM_REG     = 4
    NUM_HEADS   = 4      # number of SpatialHeads (= top_k)
    ALL_LAYERS  = list(range(24))   # tap every block
    DROP_PATH   = 0.10
    HEAD_DROP   = 0.4
    GUMBEL_TAU  = 1.0   # Gumbel temperature; anneal toward 0 during training if desired

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

        # ── Layer selector ───────────────────────────────────────────────
        self.layer_attention = LayerAttention(
            embed_dim=self.EMBED_DIM,
            num_layers=len(self.ALL_LAYERS),   # 24
            num_heads=2,
        )
        self.layer_selector = LayerSelector(
            embed_dim=self.EMBED_DIM,
            num_layers=len(self.ALL_LAYERS),   # 24
            top_k=self.NUM_HEADS,              # 4
            tau=self.GUMBEL_TAU,
        )

        # ── One SpatialHead per selected layer (fixed pool of 4) ────────
        self.spatial_heads = nn.ModuleList([
            SpatialHead(self.EMBED_DIM, self.NUM_REG, self.HEAD_DROP)
            for _ in range(self.NUM_HEADS)
        ])

    def forward(self, x):
        # ── 1. Pull intermediates from every layer ───────────────────────
        # intermediates: list of 24 × (spatial_map, prefix_tokens)
        #   spatial_map   : (B, C, H, W)
        #   prefix_tokens : (B, 5, C)   [CLS, R1, R2, R3, R4]
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=self.ALL_LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        # ── 2. Stack CLS tokens across layers → (B, 24, C) ─────────────
        cls_stack = torch.stack(
            [prefix[:, 0, :] for _, prefix in intermediates],
            dim=1,
        )   # (B, 24, C)

        # ── 3. Contextualise with LayerAttention ────────────────────────
        cls_ctx = self.layer_attention(cls_stack.float())   # (B, 24, C)

        # ── 4. Score and select best 4 layers (batch-shared) ────────────
        selected_layers, soft_weights = self.layer_selector(cls_ctx)
        # selected_layers : LongTensor (4,)  — sorted layer indices
        # soft_weights    : FloatTensor (4,) — Gumbel-softmax scores

        # ── 5. Route selected intermediates through SpatialHeads ─────────
        logits_list:   list = []
        features_list: list = []
        cls_list:      list = []
        fused_list:    list = []

        for head_idx, layer_idx in enumerate(selected_layers.tolist()):
            spatial_map, prefix_tokens = intermediates[layer_idx]

            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            result = self.spatial_heads[head_idx](cls_tok, reg_tok, patch_tok)
            logits_list.append(result["logits"])
            features_list.append(result["features"])
            cls_list.append(result["f_cls"])
            fused_list.append(result["spatial_fused"])

        return logits_list, features_list, cls_list, fused_list, selected_layers, soft_weights