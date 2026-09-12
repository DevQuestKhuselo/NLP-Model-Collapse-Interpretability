import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rope(x, base=10000.0):
    """
    x: [B,H,T,Dh], Dh must be even.
    Rotary positional embedding applied to Q or K.
    """
    B,H,T,D = x.shape
    assert D % 2 == 0
    pos = torch.arange(T, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D))
    angles = torch.einsum("t,d->td", pos, inv_freq)
    sin = angles.sin()[None,None,:,:]
    cos = angles.cos()[None,None,:,:]

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=8, rope=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_attn=False):
        B,T,C = x.shape
        qkv = self.qkv(x).view(B,T,3,self.num_heads,self.head_dim)
        q,k,v = qkv.unbind(dim=2)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.rope:
            q = apply_rope(q)
            k = apply_rope(k)

        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        return (y, attn) if return_attn else (y, None)

class AttentionOnlyBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, rope=True)

    def forward(self, x, return_attn=False):
        y, a = self.attn(self.norm(x), return_attn=return_attn)
        return x + y, a

class ICLTransformer(nn.Module):
    """
    Attention-only causal Transformer:
      input projection -> 2 transformer blocks -> layer norm -> label head

    We classify from the final query position.
    """
    def __init__(self, input_dim, num_labels=5, d_model=64, num_heads=8, depth=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            AttentionOnlyBlock(d_model, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, x, return_attn=False):
        h = self.input_proj(x)
        attn_maps = []
        for block in self.blocks:
            h, a = block(h, return_attn=return_attn)
            if return_attn:
                attn_maps.append(a)
        h = self.norm(h)
        logits = self.head(h[:, -1])
        return (logits, attn_maps) if return_attn else logits
