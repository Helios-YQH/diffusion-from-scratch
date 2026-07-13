"""
Debug script: isolate DataLoader + CUDA + DataParallel interaction.

Usage on server:
  python debug_dataloader.py 1    # DL before CUDA (fork), 2 GPUs — should work
  python debug_dataloader.py 2    # DL after CUDA (fork), 2 GPUs  — current code, expect hang
  python debug_dataloader.py 3    # DL after CUDA (spawn), 2 GPUs — spawn fix, expect slow but ok
  python debug_dataloader.py 4    # fork + num_workers=0, 2 GPUs — should work
  python debug_dataloader.py 0    # run all (1→4→3→2)

Each test fetches 3 batches and runs forward+backward, then prints SUCCESS or hangs.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import UNet


def _make_data(n=10000):
    """Small synthetic data to keep tests fast."""
    path = "data/debug_celeba_test.pt"
    if not os.path.exists(path):
        x = torch.randn(n, 3, 64, 64)
        torch.save(x, path)
    return torch.load(path, weights_only=True)


def _train_loop(model, loader, device, n_batches=3):
    """Run n_batches of forward+backward. Returns True if it completes."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x0 = batch[0].to(device)
        t = torch.randint(0, 1000, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        pred = model(x0, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"    batch {i}: loss={loss.item():.4f}")
    return True


# ── Test 1: DataLoader BEFORE CUDA (fork) ──────────────────────────
def test1_dl_before_cuda():
    """Create DataLoader first, then init CUDA + DataParallel."""
    print("=== Test 1: DataLoader BEFORE CUDA (fork, 2 GPUs) === expect OK")

    data = _make_data()
    ds = TensorDataset(data)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    print("  DataLoader created (CUDA not yet init)")

    gpu_ids = [0, 1]
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(f"cuda:{gpu_ids[0]}")
    print("  Model on GPU, starting train loop...")

    t0 = time.time()
    _train_loop(model, loader, f"cuda:{gpu_ids[0]}")
    print(f"  SUCCESS  ({time.time()-t0:.1f}s)")


# ── Test 2: DataLoader AFTER CUDA (fork) — current broken code path ─
def test2_dl_after_cuda():
    """Init CUDA + DataParallel first, then create DataLoader. Expect HANG."""
    print("=== Test 2: DataLoader AFTER CUDA (fork, 2 GPUs) === expect HANG")

    gpu_ids = [0, 1]
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(f"cuda:{gpu_ids[0]}")
    print("  Model on GPU, CUDA inited")

    data = _make_data()
    ds = TensorDataset(data)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    print("  DataLoader created, starting train loop...")

    t0 = time.time()
    _train_loop(model, loader, f"cuda:{gpu_ids[0]}")
    print(f"  SUCCESS  ({time.time()-t0:.1f}s)")


# ── Test 3: DataLoader AFTER CUDA (spawn) — current fix attempt ────
def test3_spawn_after_cuda():
    """Spawn + DataLoader after CUDA. Should work but may be slow."""
    print("=== Test 3: DataLoader AFTER CUDA (spawn, 2 GPUs) === expect OK (slow)")

    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    print("  Start method: spawn")

    gpu_ids = [0, 1]
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(f"cuda:{gpu_ids[0]}")
    print("  Model on GPU, CUDA inited")

    data = _make_data()
    ds = TensorDataset(data)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    print("  DataLoader created, starting train loop...")

    t0 = time.time()
    _train_loop(model, loader, f"cuda:{gpu_ids[0]}")
    print(f"  SUCCESS  ({time.time()-t0:.1f}s)")


# ── Test 4: num_workers=0, fork, DataParallel ──────────────────────
def test4_noworker():
    """num_workers=0 with DataParallel. No fork issue at all."""
    print("=== Test 4: num_workers=0 (fork, 2 GPUs) === expect OK")

    gpu_ids = [0, 1]
    model = UNet(img_channels=3, base_channels=64, num_downs=4)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(f"cuda:{gpu_ids[0]}")
    print("  Model on GPU, CUDA inited")

    data = _make_data()
    ds = TensorDataset(data)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    print("  DataLoader created (num_workers=0), starting train loop...")

    t0 = time.time()
    _train_loop(model, loader, f"cuda:{gpu_ids[0]}")
    print(f"  SUCCESS  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    test = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if test == 0:
        # Run in order: best → worst, each isolated
        for t in [test1_dl_before_cuda, test4_noworker, test3_spawn_after_cuda, test2_dl_after_cuda]:
            t()
            print()
            torch.cuda.empty_cache()
    elif test == 1:
        test1_dl_before_cuda()
    elif test == 2:
        test2_dl_after_cuda()
    elif test == 3:
        test3_spawn_after_cuda()
    elif test == 4:
        test4_noworker()
