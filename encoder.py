import math
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbeddings(nn.Module):
    def __init__(self, C, image_size, embedding_dim, patch_size):
        super().__init__()
        H, W = image_size
        self.embedding_dim = embedding_dim
        self.num_patches = (H // patch_size) * (W // patch_size)

        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

        self.conv1 = nn.Conv2d(in_channels = C,
                                out_channels = embedding_dim,
                                kernel_size = patch_size,
                                stride = patch_size,
                                padding = 0)
        
        self.cls_token = nn.Parameter(torch.randn((1, 1, self.embedding_dim)))
        self.pos_embeddings = nn.Parameter(torch.randn((1, self.num_patches + 1, self.embedding_dim)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x += self.pos_embeddings
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, scale_factor, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * scale_factor)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * scale_factor, input_dim)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.drop1(self.gelu(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x
    

class LayerNorm(nn.Module):
    def __init__(self, feature_dim, eps = 1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        var, mu = torch.var_mean(x, dim=-1, keepdim=True) 
        x_hat = (x - mu)/torch.sqrt(var + self.eps)
        y = x_hat * self.gamma + self.beta
        return y 


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.d_k = self.embedding_dim // self.num_heads
        self.w_qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

    def _to_heads(self, q, k, v):
        B, T, D = q.shape
        q, k, v = [rearrange(item.view(B, T, self.num_heads, self.d_k), 'b t h d -> b h t d') for item in (q, k, v)]
        return q, k, v

    def sdpa(self, q, k, v):
        scale = 1 / math.sqrt(self.d_k)
        scaled_score = (q @ rearrange(k, 'b h t d -> b h d t')) * scale
        attn_weights = torch.softmax(scaled_score, dim = -1)
        return self.attn_dropout(attn_weights) @ v # output shape is b h t d

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._to_heads(q, k, v)
        attn_out = self.sdpa(q, k, v)
        attn_out = rearrange(attn_out, 'b h t d -> b t (h d)')
        attn_out = self.proj_dropout(self.proj(attn_out))
        return attn_out


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.embedding_dim, config.eps)
        self.MHSA = MultiHeadSelfAttention(config.num_heads, config.embedding_dim, config.dropout_rate)
        self.layer_norm2 = LayerNorm(config.embedding_dim, config.eps)
        self.ffn = MLP(config.embedding_dim, config.scale_factor, config.dropout_rate)

    def forward(self, x):
        x = x + self.MHSA(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_blocks)])
        self.final_norm = LayerNorm(config.embedding_dim, config.eps)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        x = self.final_norm(x)
        return x
        

