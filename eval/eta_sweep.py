"""Is the DiT's DDIM failure about determinism? Sweep eta at fixed NFE.

eta=0 is deterministic DDIM; eta=1 makes each step's noise injection match
the DDPM ancestral variance. If the failure disappears as eta grows, the
deterministic trajectory is the culprit and the sampler's noise injection is
what keeps a slightly-off score model on the data manifold.

Usage:
    CKPT=checkpoints/mnist_dit_eps_best.pt NFE=50 python diag_ddim_eta.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.ddpm import DDPM
from models.dit import DiT
from models.unet import UNet

CKPT = os.environ.get("CKPT", "checkpoints/mnist_dit_eps_best.pt")
NFE = int(os.environ.get("NFE", "50"))
OUT = os.environ.get("OUT", "samples")


def build(arch):
    if arch.get("backbone") == "dit":
        return DiT(img_size=arch["img_size"], patch_size=arch["patch_size"],
                   img_channels=arch["img_channels"], hidden_size=arch["hidden_size"],
                   depth=arch["depth"], num_heads=arch["num_heads"])
    return UNet(img_channels=arch["img_channels"], base_channels=arch["base_channels"],
                time_dim=256, num_downs=arch["num_downs"])


raw = torch.load(CKPT, map_location="cpu", weights_only=False)
arch = raw["arch"]
model = build(arch)
model.load_state_dict(raw["ema"])
model.eval().to("cuda")
d = DDPM(model, T=arch["timesteps"], device="cuda",
         img_channels=arch["img_channels"], img_size=arch["img_size"])

print(f"{os.path.basename(CKPT)}  NFE={NFE}")
print(f"{'eta':>6} {'std':>7} {'frac|x|>0.99':>13} {'mean|x|':>8}")
for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
    x = d.sample_ddim(16, num_steps=NFE, eta=eta)
    sat = float((x.abs() > 0.99).float().mean())
    print(f"{eta:6.2f} {float(x.std()):7.3f} {sat:13.3f} {float(x.abs().mean()):8.3f}")

    if eta in (0.0, 0.5, 1.0):
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            img = (x[i].cpu() * 0.5 + 0.5).clamp(0, 1)
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(f"{os.path.basename(CKPT)}  DDIM NFE={NFE}  eta={eta}")
        plt.tight_layout()
        path = os.path.join(OUT, f"diag_eta{eta}.png")
        plt.savefig(path, dpi=80)
        plt.close()
        print("   wrote", path)
