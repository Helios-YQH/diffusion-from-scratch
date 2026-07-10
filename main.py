"""
DDPM Diffusion Model for MNIST Handwritten Digit Generation.

Usage:
  # Train on CPU (auto-detect)
  python main.py train --epochs 50 --batch-size 128

  # Train on 6 GPUs with DataParallel
  python main.py train --epochs 50 --batch-size 128 --gpus 0,1,2,3,4,5

  # Train on all available GPUs (default)
  python main.py train --epochs 50 --batch-size 128

  # Sample from a trained checkpoint (latest in checkpoints/ by default)
  python main.py sample --checkpoint checkpoints/ddpm_epoch50.pt --n 64

  # Sample with auto-detect latest checkpoint
  python main.py sample --n 16
"""

import argparse
import os
import torch
import torch.nn as nn

from model import UNet
from diffusion import DDPM
from train import train, load_mnist, save_sample_grid


def main():
    parser = argparse.ArgumentParser("DDPM MNIST")
    parser.add_argument("mode", choices=["train", "sample"], help="train or sample")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=16, help="number of images to sample")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU ids to use, e.g. '0,1,2,3,4,5'. Default: all available")
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpus is not None:
            gpu_ids = [int(x) for x in args.gpus.split(",")]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        device = f"cuda:{gpu_ids[0]}"
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    else:
        gpu_ids = []
        device = "cpu"
        print("Device: cpu")

    model = UNet(img_channels=1, base_channels=64, time_dim=256)

    use_data_parallel = len(gpu_ids) > 1
    if use_data_parallel:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        per_gpu_batch = args.batch_size
        batch_size = args.batch_size * len(gpu_ids)
        print(f"DataParallel mode: per-GPU batch={per_gpu_batch}, total batch={batch_size}")
    else:
        batch_size = args.batch_size

    diffusion = DDPM(model, T=args.timesteps, device=device, use_data_parallel=use_data_parallel)

    if args.mode == "train":
        train(diffusion, epochs=args.epochs, batch_size=batch_size,
              lr=args.lr, save_interval=args.save_interval,
              use_data_parallel=use_data_parallel)

    elif args.mode == "sample":
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = os.path.join("checkpoints", sorted(os.listdir("checkpoints"))[-1])
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        base_model = model.module if use_data_parallel else model
        base_model.load_state_dict(state)
        base_model.to(device)

        samples, steps = diffusion.sample(args.n, return_all=True)

        os.makedirs("samples", exist_ok=True)
        save_sample_grid(samples, os.path.join("samples", "generated.png"))
        print(f"Saved generated.png")

        # Also save intermediate steps
        if steps and len(steps) > 1:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            idxs = [0, len(steps) // 8, len(steps) // 4, len(steps) // 2,
                    len(steps) - 1] if len(steps) > 5 else range(len(steps))
            for i, idx in enumerate(idxs):
                if idx < len(steps):
                    r, c = i // 5, i % 5
                    img = (steps[idx][0].cpu() * 0.5 + 0.5).clamp(0, 1)
                    axes[r, c].imshow(img[0], cmap="gray")
                    t = args.timesteps - 1 - (idx * (args.timesteps // max(1, len(steps) - 1)))
                    axes[r, c].set_title(f"t={t}")
                    axes[r, c].axis("off")
            for i in range(len(idxs), 10):
                r, c = i // 5, i % 5
                axes[r, c].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join("samples", "diffusion_steps.png"), dpi=100)
            plt.close()
            print(f"Saved diffusion_steps.png")


if __name__ == "__main__":
    main()
