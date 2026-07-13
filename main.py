"""
DDPM Diffusion Model — MNIST (28x28) and CelebA (64x64).

注意设置NCCL_P2P_DISABLE=1

Training:
  # Single GPU
  python main.py train --dataset mnist --epochs 50 --batch-size 256

  # 5 GPU DDP (torchrun)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 \
  main.py train --dataset celeba --epochs 200 --batch-size 64

Sampling:
  python main.py sample --dataset celebA --checkpoint checkpoints/ddpm_celeba_best.pt --n 64
"""

import argparse
import os
import torch

from model import UNet
from diffusion import DDPM
from train import train, save_sample_grid


def main():
    parser = argparse.ArgumentParser("DDPM")
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--dataset", choices=["mnist", "celeba"], default="mnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear LR warmup steps")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=16, help="number of images to sample")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 28 for mnist, 64 for celeba)")
    parser.add_argument("--base-channels", type=int, default=None,
                        help="Base channels (default: 64)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "sample":
        if args.dataset == "celeba":
            img_channels, img_size = 3, args.image_size or 64
            base_channels = args.base_channels or 64
        else:
            img_channels, img_size = 1, args.image_size or 28
            base_channels = args.base_channels or 64

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = os.path.join("checkpoints", sorted(os.listdir("checkpoints"))[-1])
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)

        model = UNet(img_channels=img_channels, base_channels=base_channels,
                     time_dim=256, num_downs=4 if args.dataset == "celeba" else 3)
        model.load_state_dict(state)
        model.to(device)

        diffusion = DDPM(model, T=args.timesteps, device=device,
                         img_channels=img_channels, img_size=img_size)

        samples, steps = diffusion.sample(args.n, return_all=True)

        os.makedirs("samples", exist_ok=True)
        save_sample_grid(samples, os.path.join("samples", f"generated_{args.dataset}.png"))
        print(f"Saved generated_{args.dataset}.png")

        if steps and len(steps) > 1:
            import matplotlib.pyplot as plt
            is_rgb = img_channels == 3
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            idxs = [0, len(steps) // 8, len(steps) // 4, len(steps) // 2,
                    len(steps) - 1] if len(steps) > 5 else range(len(steps))
            for i, idx in enumerate(idxs):
                if idx < len(steps):
                    r, c = i // 5, i % 5
                    img = (steps[idx][0].cpu() * 0.5 + 0.5).clamp(0, 1)
                    if is_rgb:
                        axes[r, c].imshow(img.permute(1, 2, 0))
                    else:
                        axes[r, c].imshow(img[0], cmap="gray")
                    t_val = args.timesteps - 1 - (idx * (args.timesteps // max(1, len(steps) - 1)))
                    axes[r, c].set_title(f"t={t_val}")
                    axes[r, c].axis("off")
            for i in range(len(idxs), 10):
                r, c = i // 5, i % 5
                axes[r, c].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join("samples", f"diffusion_steps_{args.dataset}.png"), dpi=100)
            plt.close()
            print(f"Saved diffusion_steps_{args.dataset}.png")


if __name__ == "__main__":
    main()
