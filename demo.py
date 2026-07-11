"""Demonstrate the diffusion denoising process using a trained checkpoint.

Usage:
  python demo.py --checkpoint checkpoints/ddpm_epoch170.pt
  python demo.py --checkpoint checkpoints/ddpm_epoch170.pt --n 64  # 64-digit gallery
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from diffusion import DDPM


def to_image(tensor):
    """Convert [-1,1] tensor to [0,1] numpy array for display."""
    img = tensor.detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img


def demo_diffusion_steps(diffusion, n_digits=4, saved_every=50):
    """
    Generate digits and capture the full denoising process.
    Displays a figure showing pure noise → intermediate steps → final digits.
    """
    print(f"Running full {diffusion.T}-step reverse diffusion...")
    model = diffusion.model
    model.eval()

    x = torch.randn(n_digits, 1, 28, 28, device=diffusion.device)

    # Record every `saved_every` steps plus the final
    save_at = set(range(0, diffusion.T, saved_every)) | {diffusion.T - 1}
    frames = [(0, x.clone())]

    with torch.no_grad():
        for t in reversed(range(diffusion.T)):
            t_batch = torch.full((n_digits,), t, device=diffusion.device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha = diffusion.alphas[t]
            alpha_bar = diffusion.alpha_bars[t]
            beta = diffusion.betas[t]

            coeff1 = 1.0 / alpha.sqrt()
            coeff2 = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
            x = coeff1 * (x - coeff2 * pred_noise)

            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

            if t in save_at:
                frames.append((t, x.clone()))

    x = torch.clamp(x, -1.0, 1.0)

    # Plot: rows=digits, cols=frames
    fig, axes = plt.subplots(n_digits, len(frames), figsize=(len(frames) * 1.2, n_digits * 1.2))
    if n_digits == 1:
        axes = axes[None, :]
    for row in range(n_digits):
        for col, (t, img) in enumerate(frames):
            axes[row, col].imshow(to_image(img[row])[0], cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                if col == 0:
                    axes[row, col].set_title("t=1000\n(noise)", fontsize=8)
                elif col == len(frames) - 1:
                    axes[row, col].set_title("t=0\n(done)", fontsize=8)
                else:
                    axes[row, col].set_title(f"t={t}", fontsize=8)

    fig.suptitle("DDPM Denoising Process: Pure Noise → Handwritten Digit", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join("samples", "demo_denoising_process.png")
    os.makedirs("samples", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def demo_gallery(diffusion, n_digits=64, nrow=8):
    """Generate a gallery of digits in one shot."""
    print(f"Generating {n_digits} digits...")
    samples = diffusion.sample(n_digits)

    ncol = max(1, n_digits // nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5))
    if ncol == 1 and nrow == 1:
        axes = np.array([[axes]])
    elif ncol == 1:
        axes = axes[:, None]
    elif nrow == 1:
        axes = axes[None, :]
    for i, ax in enumerate(axes.flat):
        if i < n_digits:
            ax.imshow(to_image(samples[i])[0], cmap="gray")
        ax.axis("off")

    fig.suptitle(f"DDPM Generated MNIST Digits ({n_digits} samples)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join("samples", "demo_gallery.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def demo_single_digit_zoom(diffusion):
    """
    Show a single digit's denoising in detail — one row with many snapshots,
    from t=1000 down to t=0.
    """
    print("Generating single digit with detailed step-by-step...")
    model = diffusion.model
    model.eval()

    x = torch.randn(1, 1, 28, 28, device=diffusion.device)

    # Capture ~12 frames — denser at low t where details emerge
    milestones = (
        list(range(900, 0, -100)) +
        [50, 30, 20, 10, 5, 0]
    )
    milestone_set = set(milestones)
    frames_by_t = {}

    with torch.no_grad():
        for t in reversed(range(diffusion.T)):
            t_batch = torch.full((1,), t, device=diffusion.device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha = diffusion.alphas[t]
            alpha_bar = diffusion.alpha_bars[t]
            beta = diffusion.betas[t]

            coeff1 = 1.0 / alpha.sqrt()
            coeff2 = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
            x = coeff1 * (x - coeff2 * pred_noise)

            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

            if t in milestone_set:
                frames_by_t[t] = x.clone()

    frames = [(t, frames_by_t[t]) for t in milestones if t in frames_by_t]

    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 1.5, 2))
    for i, (t, img) in enumerate(frames):
        axes[i].imshow(to_image(img[0])[0], cmap="gray")
        axes[i].axis("off")
        label = "noise" if t >= 900 else f"t={t}"
        axes[i].set_title(label, fontsize=9)

    fig.suptitle("Single Digit: Noise → Structure → Detail", fontsize=14, y=1.1)
    plt.tight_layout()
    path = os.path.join("samples", "demo_single_zoom.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser("DDPM Demo — visualize the diffusion process")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: latest in checkpoints/)")
    parser.add_argument("--n", type=int, default=64,
                        help="Number of digits for gallery mode")
    parser.add_argument("--mode", choices=["all", "process", "gallery", "zoom"], default="all",
                        help="Demo mode: all=everything, process=denoising steps, "
                             "gallery=digit gallery, zoom=single digit detail")
    parser.add_argument("--steps-per-frame", type=int, default=50,
                        help="How many diffusion steps between snapshots in process mode")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt_dir = "checkpoints"
        ckpt = os.path.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])
    print(f"Checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = UNet()
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    diffusion = DDPM(model, T=1000, device=device)

    if args.mode in ("all", "process"):
        demo_diffusion_steps(diffusion, n_digits=4, saved_every=args.steps_per_frame)
    if args.mode in ("all", "zoom"):
        demo_single_digit_zoom(diffusion)
    if args.mode in ("all", "gallery"):
        demo_gallery(diffusion, n_digits=args.n)

    print("Demo complete. All outputs saved to samples/.")


if __name__ == "__main__":
    main()
