import timm
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class MACHead(nn.Module):
    def __init__(self, embed_dim: int = 384, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg = num_reg
        in_dim = (1 + num_reg + 1) * embed_dim

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
        B = cls_tok.size(0)
        f_avg = patch_tok.mean(dim=1)
        f_cls = cls_tok.squeeze(1)
        f_reg = reg_tok.reshape(B, -1)
        inp = torch.cat([f_cls, f_reg, f_avg], dim=1).float()
        h = self.head(inp)
        return {
            "f_avg": f_avg,
            "f_cls": f_cls,
            "f_reg": f_reg,
            "logits": self.classifier(h),
            "features": h,
        }


class ViT(nn.Module):
    EMBED_DIM = 384
    NUM_REG = 4
    NUM_HEADS = 4
    LAYERS = [8, 9, 10, 11]
    DROP_PATH = 0.10
    MAC_DROP = 0.4

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
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

        logits_list = []
        features_list = []
        cls_list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok = prefix_tokens[:, :1, :]
            reg_tok = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            result = self.mac_heads[i](cls_tok, reg_tok, patch_tok)
            logits_list.append(result["logits"])
            features_list.append(result["features"])
            cls_list.append(result["f_cls"])

        return logits_list, features_list, cls_list
