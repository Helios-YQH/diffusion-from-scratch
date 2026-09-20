"""Where does each backbone's eps error live? MSE(t) for a binned t sweep.

DDIM's first steps sit at t ~ 1000, so an error that is concentrated at high
noise levels is amplified along the deterministic trajectory while ancestral
sampling re-injects noise and masks it. This script measures that directly.

Usage (one GPU):
    CKPTS="checkpoints/mnist_dit_eps_best.pt checkpoints/mnist_unet_eps_best.pt" \
        python diag_eps_error.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from diffusion.ddpm import DDPM
from models.dit import DiT
from models.unet import UNet
from train import load_mnist

BINS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def build(arch):
    if arch.get("backbone") == "dit":
        return DiT(img_size=arch["img_size"], patch_size=arch["patch_size"],
                   img_channels=arch["img_channels"], hidden_size=arch["hidden_size"],
                   depth=arch["depth"], num_heads=arch["num_heads"])
    return UNet(img_channels=arch["img_channels"],
                base_channels=arch["base_channels"], time_dim=256,
                num_downs=arch["num_downs"])


def main():
    device = "cuda"
    x_all = load_mnist()[:4096].to(device)
    ckpts = os.environ.get("CKPTS", "checkpoints/mnist_dit_eps_best.pt "
                                     "checkpoints/mnist_unet_eps_best.pt").split()

    print(f"{'checkpoint':38s} {'weights':6s} " +
          " ".join(f"t<{b:<4d}" for b in BINS[1:9]))
    for path in ckpts:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        arch = raw["arch"]
        for weights in ("ema", "raw"):
            state = raw["ema"] if weights == "ema" else raw["model"]
            if state is None:
                continue
            model = build(arch).to(device)
            model.load_state_dict(state)
            model.eval()

            diff = DDPM(model, T=arch["timesteps"], device=device,
                        img_channels=arch["img_channels"], img_size=arch["img_size"])
            row = []
            with torch.no_grad():
                for lo, hi in zip(BINS[:-1], BINS[1:]):
                    losses = []
                    for _ in range(4):                      # 4 batches per bucket
                        x0 = x_all[torch.randint(0, x_all.shape[0], (64,), device=device)]
                        t = torch.randint(lo, hi, (64,), device=device)
                        xt, noise = diff.forward_diffusion(x0, t)
                        pred = model(xt, t)
                        losses.append(F.mse_loss(pred, noise).item())
                    row.append(sum(losses) / len(losses))
            print(f"{os.path.basename(path):38s} {weights:6s} " +
                  " ".join(f"{v:6.3f}" for v in row))


if __name__ == "__main__":
    main()
