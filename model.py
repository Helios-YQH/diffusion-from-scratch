import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.net(t)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-2, -1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.to_out(attn)


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim)

        ch = base_channels
        self.conv_in = nn.Conv2d(img_channels, ch, 3, padding=1)

        # Down stages: 28 -> 14 -> 7
        self.down1_block1 = ResidualBlock(ch, ch, time_dim)
        self.down1_block2 = ResidualBlock(ch, ch, time_dim)
        self.down1_sample = DownSample(ch)

        self.down2_block1 = ResidualBlock(ch, ch * 2, time_dim)
        self.down2_block2 = ResidualBlock(ch * 2, ch * 2, time_dim)
        self.down2_sample = DownSample(ch * 2)

        self.down3_block1 = ResidualBlock(ch * 2, ch * 4, time_dim)
        self.down3_block2 = ResidualBlock(ch * 4, ch * 4, time_dim)

        # Middle
        self.mid_block1 = ResidualBlock(ch * 4, ch * 4, time_dim)
        self.mid_attn = AttentionBlock(ch * 4)
        self.mid_block2 = ResidualBlock(ch * 4, ch * 4, time_dim)

        # Up stages: 7 -> 14 -> 28
        self.up1_block1 = ResidualBlock(ch * 8, ch * 4, time_dim)
        self.up1_block2 = ResidualBlock(ch * 4, ch * 4, time_dim)

        self.up2_sample = UpSample(ch * 4)
        self.up2_block1 = ResidualBlock(ch * 6, ch * 2, time_dim)
        self.up2_block2 = ResidualBlock(ch * 2, ch * 2, time_dim)

        self.up3_sample = UpSample(ch * 2)
        self.up3_block1 = ResidualBlock(ch * 3, ch, time_dim)
        self.up3_block2 = ResidualBlock(ch, ch, time_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_embed(t)
        t = self.time_mlp(t)

        x = self.conv_in(x)                       # (B, 64, 28, 28)

        # Down 1: 28x28
        x = self.down1_block1(x, t)
        x = self.down1_block2(x, t)
        skip1 = x                                 # (B, 64, 28, 28)
        x = self.down1_sample(x)                  # (B, 64, 14, 14)

        # Down 2: 14x14
        x = self.down2_block1(x, t)
        x = self.down2_block2(x, t)
        skip2 = x                                 # (B, 128, 14, 14)
        x = self.down2_sample(x)                  # (B, 128, 7, 7)

        # Down 3: 7x7
        x = self.down3_block1(x, t)
        x = self.down3_block2(x, t)
        skip3 = x                                 # (B, 256, 7, 7)

        # Middle: 7x7
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)                 # (B, 256, 7, 7)

        # Up 1: 7x7
        x = torch.cat([x, skip3], dim=1)          # (B, 512, 7, 7)
        x = self.up1_block1(x, t)
        x = self.up1_block2(x, t)                 # (B, 256, 7, 7)

        # Up 2: 14x14
        x = self.up2_sample(x)                    # (B, 256, 14, 14)
        x = torch.cat([x, skip2], dim=1)          # (B, 384, 14, 14)
        x = self.up2_block1(x, t)
        x = self.up2_block2(x, t)                 # (B, 128, 14, 14)

        # Up 3: 28x28
        x = self.up3_sample(x)                    # (B, 128, 28, 28)
        x = torch.cat([x, skip1], dim=1)          # (B, 192, 28, 28)
        x = self.up3_block1(x, t)
        x = self.up3_block2(x, t)                 # (B, 64, 28, 28)

        return self.conv_out(x)
