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
    def __init__(self, img_channels=1, base_channels=64, time_dim=256,
                 num_downs=3, blocks_per_stage=2):
        super().__init__()
        self.img_channels = img_channels
        self.base_channels = base_channels
        self.num_downs = num_downs
        self.blocks_per_stage = blocks_per_stage

        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim)

        ch = base_channels
        self.conv_in = nn.Conv2d(img_channels, ch, 3, padding=1)

        # Down stages
        self.downs = nn.ModuleList()
        in_ch = ch
        for i in range(num_downs):
            stage_ch = ch * (2 ** i)
            stage = nn.ModuleList()
            stage.append(ResidualBlock(in_ch, stage_ch, time_dim))
            for _ in range(blocks_per_stage - 1):
                stage.append(ResidualBlock(stage_ch, stage_ch, time_dim))
            if i < num_downs - 1:
                stage.append(DownSample(stage_ch))
            self.downs.append(stage)
            in_ch = stage_ch

        # Middle
        mid_ch = in_ch
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_dim)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_dim)

        # Up stages
        self.ups = nn.ModuleList()
        for i in reversed(range(num_downs)):
            stage_ch = ch * (2 ** i)
            stage = nn.ModuleList()
            if i < num_downs - 1:
                # upsample doubles spatial size, keeps channels from previous
                prev_ch = ch * (2 ** (i + 1))
                stage.append(UpSample(prev_ch))
            # First residual block: concat(skip=stage_ch, prev=prev_after_up)
            block_in = stage_ch + (ch * (2 ** (i + 1)) if i < num_downs - 1 else mid_ch)
            stage.append(ResidualBlock(block_in, stage_ch, time_dim))
            for _ in range(blocks_per_stage - 1):
                stage.append(ResidualBlock(stage_ch, stage_ch, time_dim))
            self.ups.append(stage)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_embed(t)
        t = self.time_mlp(t)

        x = self.conv_in(x)
        skips = []

        # Down
        for stage in self.downs:
            blocks = list(stage)
            # Process all blocks; capture skip before the downsample (if present)
            has_down = isinstance(blocks[-1], DownSample)
            res_blocks = blocks[:-1] if has_down else blocks
            for block in res_blocks:
                x = block(x, t)
            skips.append(x)
            if has_down:
                x = blocks[-1](x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Up — each stage consumes one skip on its FIRST ResidualBlock
        for stage in self.ups:
            blocks = list(stage)
            first_residual = True
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    if first_residual:
                        skip = skips.pop()
                        x = block(torch.cat([x, skip], dim=1), t)
                        first_residual = False
                    else:
                        x = block(x, t)
                else:
                    x = block(x)

        return self.conv_out(x)
