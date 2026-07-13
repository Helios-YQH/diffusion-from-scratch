"""
Debug v2: isolate what inside the model hangs under DataParallel.

Tests:
  python debug_v2.py 1   # Forward-only (no backward), 2 GPUs
  python debug_v2.py 2   # Forward+backward, SDP backend='math', 2 GPUs
  python debug_v2.py 3   # Forward+backward, manual attention (no F.scaled_dot_product_attention), 2GPUs
  python debug_v2.py 4   # Forward+backward, NO attention block at all, 2 GPUs
  python debug_v2.py 0   # run all

If test 1 hangs → forward itself is dead (scatter/gather)
If test 2 passes → F.scaled_dot_product_attention is the culprit
If test 3 passes → confirmed: SDPA backend + DataParallel = deadlock
If test 4 passes → AttentionBlock specifically is the problem
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Custom AttentionBlocks for comparison ─────────────────────────

class AttentionBlockManual(nn.Module):
    """Same as original but with manual softmax attention instead of F.scaled_dot_product_attention."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, -1).transpose(-2, -1)
        # Manual attention (no SDPA backend)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = attn @ v
        attn = attn.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.to_out(attn)


class UNetNoAttention(nn.Module):
    """Minimal UNet: no AttentionBlock at all. Tests if Attention is the problem."""
    def __init__(self, img_channels=3, base_channels=64, num_downs=4):
        super().__init__()
        self.time_dim = 256
        # Minimal time emb
        self.time_embed = nn.Linear(1, 256)
        self.time_mlp = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))

        ch = base_channels
        self.conv_in = nn.Conv2d(img_channels, ch, 3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = ch
        for i in range(num_downs):
            stage_ch = ch * (2 ** i)
            stage = nn.ModuleList()
            stage.append(self._resblock(in_ch, stage_ch))
            for _ in range(1):
                stage.append(self._resblock(stage_ch, stage_ch))
            if i < num_downs - 1:
                stage.append(nn.Conv2d(stage_ch, stage_ch, 3, stride=2, padding=1))
            self.downs.append(stage)
            in_ch = stage_ch

        mid_ch = in_ch
        self.mid_block1 = self._resblock(mid_ch, mid_ch)
        # NO attention — just another resblock
        self.mid_block2 = self._resblock(mid_ch, mid_ch)

        self.ups = nn.ModuleList()
        for i in reversed(range(num_downs)):
            stage_ch = ch * (2 ** i)
            stage = nn.ModuleList()
            if i < num_downs - 1:
                prev_ch = ch * (2 ** (i + 1))
                stage.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(prev_ch, prev_ch, 3, padding=1),
                ))
            block_in = stage_ch + (ch * (2 ** (i + 1)) if i < num_downs - 1 else mid_ch)
            stage.append(self._resblock(block_in, stage_ch))
            for _ in range(1):
                stage.append(self._resblock(stage_ch, stage_ch))
            self.ups.append(stage)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, img_channels, 3, padding=1),
        )

    def _resblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t = self.time_embed(t.float().unsqueeze(-1))
        t = self.time_mlp(t)
        x = self.conv_in(x)
        skips = []
        for stage in self.downs:
            blocks = list(stage)
            has_down = isinstance(blocks[-1], nn.Conv2d) and blocks[-1].stride == (2, 2)
            res_blocks = blocks[:-1] if has_down else blocks
            for block in res_blocks:
                x = block(x)
            skips.append(x)
            if has_down:
                x = blocks[-1](x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        for stage in self.ups:
            blocks = list(stage)
            first_residual = True
            for block in blocks:
                if first_residual and hasattr(block, '__iter__'):
                    continue  # upsample
                if first_residual and not hasattr(block, '__iter__'):
                    skip = skips.pop()
                    x = block(torch.cat([x, skip], dim=1))
                    first_residual = False
                elif not hasattr(block, '__iter__'):
                    x = block(x)
        return self.conv_out(x)


def _run_test(name, model, n_batches=3):
    """Returns True if completes, False if hangs (won't return)."""
    gpu_ids = [0, 1]
    device = f"cuda:{gpu_ids[0]}"
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    print(f"  Model on GPU, starting...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for i in range(n_batches):
        x0 = torch.randn(128, 3, 64, 64, device=device)
        t = torch.randint(0, 1000, (128,), device=device)
        noise = torch.randn_like(x0)
        pred = model(x0, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"    batch {i}: loss={loss.item():.4f}")
    print(f"  {name}: SUCCESS ({time.time()-t0:.1f}s)")
    return True


# ── Test 1: forward only, no backward ──────────────────────────────
def test1_forward_only():
    print("=== Test 1: Forward-only (no backward), 2 GPUs ===")

    from model import UNet
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    gpu_ids = [0, 1]
    device = f"cuda:{gpu_ids[0]}"
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    print("  Model on GPU, running forward only...")

    t0 = time.time()
    with torch.no_grad():
        for i in range(5):
            x0 = torch.randn(128, 3, 64, 64, device=device)
            t = torch.randint(0, 1000, (128,), device=device)
            pred = model(x0, t)
            print(f"    forward {i}: shape={pred.shape}")
    print(f"  Forward-only: SUCCESS ({time.time()-t0:.1f}s)")


# ── Test 2: SDP backend = 'math' ───────────────────────────────────
def test2_math_backend():
    print("=== Test 2: Forward+backward with SDP backend='math', 2 GPUs ===")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP: {torch.backends.cuda.math_sdp_enabled()}")

    from model import UNet
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    _run_test("Math backend", model)


# ── Test 3: manual attention (no F.scaled_dot_product_attention) ───
def test3_manual_attention():
    print("=== Test 3: Manual attention (no SDPA), 2 GPUs ===")

    # Patch the model to use manual attention
    import model as m
    m.AttentionBlock = AttentionBlockManual

    unet = m.UNet(img_channels=3, base_channels=64, num_downs=4)
    _run_test("Manual attention", unet)


# ── Test 4: No attention at all ────────────────────────────────────
def test4_no_attention():
    print("=== Test 4: No attention block, 2 GPUs ===")
    model = UNetNoAttention(img_channels=3, base_channels=64, num_downs=4)
    _run_test("No attention", model)


if __name__ == "__main__":
    test = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if test == 0:
        for fn in [test1_forward_only, test4_no_attention, test2_math_backend, test3_manual_attention]:
            fn()
            print()
            torch.cuda.empty_cache()
    elif test == 1:
        test1_forward_only()
    elif test == 2:
        test2_math_backend()
    elif test == 3:
        test3_manual_attention()
    elif test == 4:
        test4_no_attention()
