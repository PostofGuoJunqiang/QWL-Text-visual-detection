import torch
import torch.nn as nn
import torch.nn.functional as F

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = dilation * (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size, dilation, padding, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape(B, d, 1, H * W).permute(0, 3, 2, 1)  # B, N, 1, d
        k = self.unfold(k).reshape(B, d, self.kernel_size * self.kernel_size, H * W).permute(0, 3, 1, 2)  # B, N, d, k*k
        attn = (q @ k) * self.scale  # B, N, 1, k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(B, d, self.kernel_size * self.kernel_size, H * W).permute(0, 3, 2, 1)  # B, N, k*k, d
        x = (attn @ v).squeeze(2).reshape(B, H, W, d).permute(0, 3, 1, 2)  # B, d, H, W
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Multi-Dilate Local Attention"

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads {num_heads} must be a multiple of num_dilation {self.num_dilation}"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(self.head_dim, qk_scale, attn_drop, kernel_size, dilation[i % self.num_dilation])
             for i in range(num_heads)])
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W).permute(2, 1, 0, 3, 4, 5)
        q, k, v = qkv.unbind(dim=1)

        attended_features = []
        for i in range(self.num_heads):
            attended_features.append(self.dilate_attention[i](q[i], k[i], v[i]))

        out = torch.stack(attended_features, dim=1).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out