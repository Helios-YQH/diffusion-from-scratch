"""Demonstrate the diffusion denoising process using a trained checkpoint.

Shows 4 digits (2x2 grid) evolving from pure noise to final digits in real time,
then saves the animation as a GIF.

Usage:
  python demo.py --checkpoint checkpoints/ddpm_epoch170.pt            # animate + gallery
  python demo.py --checkpoint checkpoints/ddpm_epoch170.pt --mode animate  # animate only
  python demo.py --checkpoint checkpoints/ddpm_epoch170.pt --mode gallery --n 64
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from model import UNet
from diffusion import DDPM


def to_image(tensor):
    img = tensor.detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img


def capture_reverse(diffusion, n_digits=4, capture_interval=10):
    """Run reverse diffusion, yielding (t, x_t) frames at each capture_interval."""
    model = diffusion.model
    model.eval()

    x = torch.randn(n_digits, 1, 28, 28, device=diffusion.device)
    yield (diffusion.T, x.clone())

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

            if t % capture_interval == 0:
                yield (t, x.clone())

    x = torch.clamp(x, -1.0, 1.0)
    yield (0, x)


def demo_animate(diffusion, capture_interval=10):
    """Live 2x2 animated denoising — popup window + saved GIF."""

    print(f"Running {diffusion.T}-step reverse diffusion "
          f"(capturing every {capture_interval} steps)...")
    frames = list(capture_reverse(diffusion, n_digits=4, capture_interval=capture_interval))
    print(f"  Captured {len(frames)} frames")

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    ims = [ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1, animated=True)
           for ax in axes]
    title = fig.suptitle("", fontsize=14)

    # Store current frame index for interactive control
    state = {"idx": 0, "playing": True, "frames": frames}

    def show_frame(idx):
        t, x = frames[idx]
        for i in range(4):
            ims[i].set_array(to_image(x[i])[0].numpy())
        progress = (diffusion.T - t) / diffusion.T
        title.set_text(f"DDPM Denoising Process — t={t:4d}  progress={progress:.0%}")
        state["idx"] = idx

    def on_key(event):
        if event.key == "right":
            state["idx"] = min(state["idx"] + 1, len(frames) - 1)
            show_frame(state["idx"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["idx"] = max(state["idx"] - 1, 0)
            show_frame(state["idx"])
            fig.canvas.draw_idle()
        elif event.key == " ":
            state["playing"] = not state["playing"]
            if state["playing"]:
                run_animation()
        elif event.key == "home":
            state["idx"] = 0
            show_frame(0)
            fig.canvas.draw_idle()
        elif event.key == "end":
            state["idx"] = len(frames) - 1
            show_frame(len(frames) - 1)
            fig.canvas.draw_idle()

    def run_animation():
        for idx in range(state["idx"], len(frames)):
            if not state["playing"]:
                return
            show_frame(idx)
            fig.canvas.draw_idle()
            plt.pause(0.03)
        state["playing"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Show first frame then auto-play
    show_frame(0)
    print("\n  Controls: [Space] pause/resume  [←→] step frame  [Home/End] jump ends")
    print("  Close the window when done.\n")
    plt.ion()
    plt.show()
    run_animation()
    plt.ioff()
    plt.show()

    # Save GIF (close and reopen to avoid animation state issues)
    plt.close("all")
    fig2, axes2 = plt.subplots(2, 2, figsize=(6, 6))
    axes2 = axes2.flatten()
    for ax in axes2:
        ax.axis("off")
    ims2 = [ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1, animated=True)
            for ax in axes2]
    title2 = fig2.suptitle("", fontsize=14)

    def update(frame_data):
        t, x = frame_data
        for i in range(4):
            ims2[i].set_array(to_image(x[i])[0].numpy())
        progress = (diffusion.T - t) / diffusion.T
        title2.set_text(f"DDPM Denoising — t={t:4d}  progress={progress:.0%}")
        return [*ims2, title2]

    print("  Rendering GIF...")
    ani = animation.FuncAnimation(
        fig2, update, frames=frames, interval=50, blit=True
    )
    os.makedirs("samples", exist_ok=True)
    gif_path = os.path.join("samples", "demo_denoising.gif")
    ani.save(gif_path, writer="pillow", fps=20, dpi=100)
    plt.close(fig2)
    print(f"  Saved {gif_path}")

    # Save final frame as PNG
    t_final, x_final = frames[-1]
    fig3, axes3 = plt.subplots(2, 2, figsize=(4, 4))
    for i, ax in enumerate(axes3.flat):
        ax.imshow(to_image(x_final[i])[0], cmap="gray")
        ax.axis("off")
    fig3.suptitle("Generated Digits", fontsize=14)
    plt.tight_layout()
    png_path = os.path.join("samples", "demo_final_2x2.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {png_path}")


def demo_gallery(diffusion, n_digits=64, nrow=8):
    """Generate a gallery of digits."""
    print(f"Generating gallery of {n_digits} digits...")
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
    os.makedirs("samples", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser("DDPM Demo — visualize the diffusion process")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["all", "animate", "gallery"], default="all")
    parser.add_argument("--n", type=int, default=64,
                        help="Number of digits for gallery")
    parser.add_argument("--capture-interval", type=int, default=10,
                        help="Diffusion steps between animation frames (default: 10)")
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

    if args.mode in ("all", "animate"):
        demo_animate(diffusion, capture_interval=args.capture_interval)
    if args.mode in ("all", "gallery"):
        demo_gallery(diffusion, n_digits=args.n)

    print("Demo complete.")


if __name__ == "__main__":
    main()
