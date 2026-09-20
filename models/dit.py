"""DiT (Diffusion Transformer) with adaLN-Zero conditioning.

Unconditional variant: the conditioning vector comes from the timestep
embedding alone (no class embedding, no CFG), matching the DDPM baseline.
Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t may be long or float, shape [B]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _sincos_1d(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = pos.reshape(-1)[:, None] * omega[None]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed 2D sincos positional embedding, [grid_size**2, embed_dim].

    Row-major (h, w) ordering, matching PatchEmbed's flatten order.
    """
    grid = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    return torch.cat([_sincos_1d(embed_dim // 2, grid[0]),
                      _sincos_1d(embed_dim // 2, grid[1])], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # [B, hidden, h, w]
        return x.flatten(2).transpose(1, 2)  # [B, N, hidden]


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero: the modulation MLP is zero-initialized,
    so every block starts as the identity function."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, img_size=64, patch_size=4, img_channels=3, hidden_size=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, freq_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim

        self.x_embedder = PatchEmbed(img_size, patch_size, img_channels, hidden_size)
        self.t_embedder = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        grid_size = img_size // patch_size
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(hidden_size, grid_size).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, img_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.x_embedder.proj.weight, std=0.02)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)

        # Zero-init the adaLN modulations (and the output projection): gates
        # start at 0 so the network begins as the identity, which is what makes
        # deep diffusion transformers trainable.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.img_channels
        h = w = self.img_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(timestep_embedding(t, self.freq_dim))
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))
