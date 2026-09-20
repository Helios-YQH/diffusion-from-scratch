"""
DDPM Diffusion Model — MNIST (28x28) and CelebA (64x64).

注意设置 NCCL_P2P_DISABLE=1

Training:
  # Single GPU with YAML config
  python main.py train --config configs/celeba.yml

  # 5 GPU DDP (torchrun), CLI overrides YAML
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 \
  main.py train --config configs/celeba.yml --batch-size 160

  # Without config (uses defaults)
  python main.py train --dataset mnist --epochs 50

  # Resume from latest checkpoint
  python main.py train --config configs/celeba.yml --resume

  # Force fresh start (ignore saved checkpoint)
  python main.py train --config configs/celeba.yml --no-resume

Sampling:
  python main.py sample --dataset celebA --checkpoint checkpoints/ddpm_celeba_best.pt --n 64
"""

import argparse
import os
import sys
import torch

from models.unet import UNet
from diffusion.ddpm import DDPM
from train import train, save_sample_grid


def _load_yaml_config(path):
    """Load a YAML config file. Gives a clear error if PyYAML is missing."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "YAML config support requires PyYAML. Install it first:\n"
            "  pip install pyyaml"
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(cfg)}")
    return cfg


def _scan_config_arg():
    """Quick first pass over sys.argv to find --config <path>."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main():
    # ── Phase 1: load YAML config (if any) ──────────────────────────
    config_path = _scan_config_arg()
    yaml_defaults = {}
    if config_path:
        yaml_defaults = _load_yaml_config(config_path)

    # ── Phase 2: build parser, YAML values as defaults ─────────────
    parser = argparse.ArgumentParser("DDPM")

    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--config", type=str, default=config_path,
                        help="Path to YAML config file")
    parser.add_argument("--dataset", choices=["mnist", "celeba"], default="mnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear LR warmup steps (before cosine decay)")
    parser.add_argument("--resume", action="store_true", default=None,
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Force fresh start, ignore saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=16, help="Number of images to sample")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save full checkpoint every N epochs")
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="Generate sample grid every N epochs")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 28 for mnist, 64 for celeba)")
    parser.add_argument("--base-channels", type=int, default=None,
                        help="UNet base channels (default: 128 for celeba, 64 for mnist)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")

    # set_defaults AFTER add_argument so YAML values override the
    # hardcoded defaults (action-level defaults win over parser-level)
    parser.set_defaults(**{k: v for k, v in yaml_defaults.items()
                           if k not in ("mode", "checkpoint")})

    args = parser.parse_args()

    # Resolve --resume / --no-resume priority:
    #   CLI --no-resume → False
    #   CLI --resume    → True
    #   neither + YAML resume: true → True
    #   neither + no YAML          → False
    if args.no_resume:
        args.resume = False
    elif args.resume is None:
        args.resume = yaml_defaults.get("resume", False)

    if args.mode == "train":
        train(args)

    elif args.mode == "sample":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # ── Find checkpoint ───────────────────────────────────────
        ckpt = args.checkpoint
        if ckpt is None:
            run_name = f"ddpm_{args.dataset}"
            # Prefer best > latest > any .pt file
            candidates = [
                os.path.join("checkpoints", f"{run_name}_best.pt"),
                os.path.join("checkpoints", f"{run_name}_latest.pt"),
            ]
            ckpt = next((c for c in candidates if os.path.exists(c)), None)
            if ckpt is None:
                pts = sorted([f for f in os.listdir("checkpoints")
                              if f.endswith(".pt")])
                if pts:
                    ckpt = os.path.join("checkpoints", pts[-1])
                else:
                    raise FileNotFoundError(
                        "No .pt checkpoint found in checkpoints/")

        print(f"Loading checkpoint: {ckpt}")
        raw = torch.load(ckpt, map_location=device, weights_only=False)

        # Support both raw state_dict and full training-state dict
        if isinstance(raw, dict) and "model" in raw:
            state = raw["model"]
            arch_cfg = raw.get("arch", {})
        else:
            state = raw
            arch_cfg = {}

        # ── Architecture (from checkpoint if available, else CLI/YAML) ──
        if args.dataset == "celeba":
            img_channels = arch_cfg.get("img_channels", 3)
            img_size = arch_cfg.get("img_size", args.image_size or 64)
            base_channels = arch_cfg.get("base_channels",
                                          args.base_channels or 128)
            num_downs = arch_cfg.get("num_downs", 4)
        else:
            img_channels = arch_cfg.get("img_channels", 1)
            img_size = arch_cfg.get("img_size", args.image_size or 28)
            base_channels = arch_cfg.get("base_channels",
                                          args.base_channels or 64)
            num_downs = arch_cfg.get("num_downs", 3)

        model = UNet(img_channels=img_channels, base_channels=base_channels,
                     time_dim=256, num_downs=num_downs)
        model.load_state_dict(state)
        model.to(device)

        diffusion = DDPM(model, T=args.timesteps, device=device,
                         img_channels=img_channels, img_size=img_size)

        samples, steps = diffusion.sample(args.n, return_all=True)

        os.makedirs("samples", exist_ok=True)
        save_sample_grid(samples, os.path.join("samples",
                         f"generated_{args.dataset}.png"))
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
