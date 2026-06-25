import torch
from torch import nn
import torch.nn.functional as F

"""
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
    def forward(self,q,k,v):
        outputs, weights = self.mha(query=q, key=k, value=v)
        return outputs"""
    
class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feat_dim, num_layers, num_frames):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embed = nn.Embedding(num_frames, feat_dim)
        self.layers = nn.TransformerDecoderLayer(d_model, nhead=num_heads, dim_feedforward=2048, activation="relu",batch_first=True, device=None)

    def forward(self, x):
        pos = self.pos_embed(x)
        return nn.Linear(nn.TransformerDecoder(self.layers(x),nlayers = self.num_layers)) #logits from temporal data???
    
    

