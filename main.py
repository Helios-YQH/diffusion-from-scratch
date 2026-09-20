"""
Diffusion Models from Scratch — DDPM / DiT / Rectified Flow.

Datasets: MNIST (28x28) and CelebA (64x64).

Note: on multi-GPU servers set NCCL_P2P_DISABLE=1 if training hangs.

Training:
  # Single GPU with YAML config
  uv run python main.py train --config configs/celeba_unet_eps.yml

  # 5-GPU DDP (torchrun); CLI args override YAML
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 \
    main.py train --config configs/celeba_dit_rf.yml

  # Backbone / objective can also be set on the CLI
  python main.py train --dataset celeba --backbone dit --objective rf

  # Resume from latest checkpoint / force a fresh start
  python main.py train --config configs/celeba_dit_rf.yml --resume
  python main.py train --config configs/celeba_dit_rf.yml --no-resume

  # Smoke test: stop after 20 optimizer steps
  python main.py train --config configs/celeba_dit_rf.yml --max-steps 20

Sampling:
  python main.py sample --run-name celeba_dit_rf --n 64
  python main.py sample --checkpoint checkpoints/celeba_unet_eps_best.pt --n 64
"""

import argparse
import os
import sys
import torch

from models.unet import UNet
from models.dit import DiT
from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from train import train, save_sample_grid


def _load_yaml_config(path):
    """Load a YAML config file. Gives a clear error if PyYAML is missing."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "YAML config support requires PyYAML. Install it first:\n"
            "  pip install pyyaml   (or: uv sync)"
        )
    with open(path, "r", encoding="utf-8") as f:
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


def build_parser(config_path):
    parser = argparse.ArgumentParser("Diffusion")
    parser.add_argument("mode", choices=["train", "sample"])
    parser.add_argument("--config", type=str, default=config_path,
                        help="Path to YAML config file")
    parser.add_argument("--dataset", choices=["mnist", "celeba"], default="mnist")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Checkpoint name (default: {dataset}_{backbone}_{objective})")

    # Backbone / objective
    parser.add_argument("--backbone", choices=["unet", "dit"], default="unet")
    parser.add_argument("--objective", choices=["eps", "rf"], default="eps")
    parser.add_argument("--patch-size", type=int, default=4, help="DiT patch size")
    parser.add_argument("--hidden-size", type=int, default=768, help="DiT width")
    parser.add_argument("--depth", type=int, default=12, help="DiT blocks")
    parser.add_argument("--num-heads", type=int, default=12, help="DiT attention heads")
    parser.add_argument("--t-sample", choices=["uniform", "logit_normal"],
                        default="uniform", help="Rectified-flow timestep sampling")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear LR warmup steps (before cosine decay)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N optimizer steps per epoch (0 = off; for smoke tests)")

    # Checkpointing / sampling
    parser.add_argument("--resume", action="store_true", default=None,
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="Force fresh start, ignore saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=16, help="Number of images to sample")
    parser.add_argument("--sample-steps", type=int, default=32,
                        help="ODE steps for rectified-flow sampling")
    parser.add_argument("--use-raw", action="store_true", default=False,
                        help="Sample with raw weights instead of EMA")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save full training state every N epochs (0 = final epoch only)")
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="Generate a sample grid every N epochs (0 = off)")
    parser.add_argument("--snapshot-interval", type=int, default=0,
                        help="Save EMA-only weight snapshots every N epochs "
                             "(0 = off; for FID-vs-steps curves)")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log to Weights & Biases (needs `uv sync --extra log`)")
    parser.add_argument("--wandb-project", type=str, default="diffusion-from-scratch")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 28 for mnist, 64 for celeba)")
    parser.add_argument("--base-channels", type=int, default=None,
                        help="UNet base channels (default: 128 for celeba, 64 for mnist)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")
    return parser


def _resolve_run_name(args):
    if args.run_name:
        return args.run_name
    return f"{args.dataset}_{args.backbone}_{args.objective}"


def sample_mode(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Find checkpoint ───────────────────────────────────────────
    ckpt = args.checkpoint
    if ckpt is None:
        run_name = _resolve_run_name(args)
        # Prefer best > latest > any .pt file
        candidates = [
            os.path.join("checkpoints", f"{run_name}_best.pt"),
            os.path.join("checkpoints", f"{run_name}_latest.pt"),
        ]
        ckpt = next((c for c in candidates if os.path.exists(c)), None)
        if ckpt is None:
            pts = sorted([f for f in os.listdir("checkpoints")
                          if f.endswith(".pt")]) if os.path.isdir("checkpoints") else []
            if pts:
                ckpt = os.path.join("checkpoints", pts[-1])
            else:
                raise FileNotFoundError("No .pt checkpoint found in checkpoints/")

    print(f"Loading checkpoint: {ckpt}")
    raw = torch.load(ckpt, map_location=device, weights_only=False)

    # Support raw state_dict, weights-only dict, and full training state
    if isinstance(raw, dict) and ("model" in raw or "ema" in raw):
        arch_cfg = raw.get("arch", {})
        if raw.get("ema") is not None and not args.use_raw:
            state = raw["ema"]
            print("Using EMA weights.")
        else:
            state = raw["model"] if raw.get("model") is not None else raw["ema"]
    else:
        state = raw
        arch_cfg = {}

    # ── Architecture (from checkpoint if available, else CLI/YAML) ──
    backbone = arch_cfg.get("backbone", args.backbone)
    objective = arch_cfg.get("objective", args.objective)
    default_size = 64 if args.dataset == "celeba" else 28
    img_channels = arch_cfg.get("img_channels", 3 if args.dataset == "celeba" else 1)
    img_size = arch_cfg.get("img_size", args.image_size or default_size)
    timesteps = arch_cfg.get("timesteps", args.timesteps)

    if backbone == "dit":
        model = DiT(img_size=img_size,
                    patch_size=arch_cfg.get("patch_size") or args.patch_size,
                    img_channels=img_channels,
                    hidden_size=arch_cfg.get("hidden_size") or args.hidden_size,
                    depth=arch_cfg.get("depth") or args.depth,
                    num_heads=arch_cfg.get("num_heads") or args.num_heads)
    else:
        model = UNet(img_channels=img_channels,
                     base_channels=arch_cfg.get("base_channels")
                     or args.base_channels
                     or (128 if args.dataset == "celeba" else 64),
                     time_dim=256,
                     num_downs=arch_cfg.get("num_downs",
                                            4 if args.dataset == "celeba" else 3))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Backbone: {backbone}  Objective: {objective}  "
          f"({img_channels}ch, {img_size}x{img_size})")

    if objective == "rf":
        diffusion = RectifiedFlow(model, device=device, img_channels=img_channels,
                                  img_size=img_size)
        samples, steps = diffusion.sample(args.n, num_steps=args.sample_steps,
                                          return_all=True)
    else:
        diffusion = DDPM(model, T=timesteps, device=device,
                         img_channels=img_channels, img_size=img_size)
        samples, steps = diffusion.sample(args.n, return_all=True)

    os.makedirs("samples", exist_ok=True)
    out_png = os.path.join("samples", f"generated_{_resolve_run_name(args)}.png")
    save_sample_grid(samples, out_png)
    print(f"Saved {out_png}")

    if steps and len(steps) > 1:
        import matplotlib.pyplot as plt
        is_rgb = img_channels == 3
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        n_show = min(10, len(steps))
        idxs = [round(i * (len(steps) - 1) / max(n_show - 1, 1))
                for i in range(n_show)]
        for i, idx in enumerate(idxs):
            r, c = i // 5, i % 5
            img = (steps[idx][0].cpu() * 0.5 + 0.5).clamp(0, 1)
            if is_rgb:
                axes[r, c].imshow(img.permute(1, 2, 0))
            else:
                axes[r, c].imshow(img[0], cmap="gray")
            if objective == "rf":
                label = f"step {idx}/{len(steps) - 1}"
            else:
                label = f"frame {idx}"
            axes[r, c].set_title(label)
            axes[r, c].axis("off")
        plt.tight_layout()
        step_png = os.path.join("samples", f"denoising_{_resolve_run_name(args)}.png")
        plt.savefig(step_png, dpi=100)
        plt.close()
        print(f"Saved {step_png}")


def main():
    # ── Phase 1: load YAML config (if any) ──────────────────────────
    config_path = _scan_config_arg()
    yaml_defaults = {}
    if config_path:
        yaml_defaults = _load_yaml_config(config_path)

    # ── Phase 2: build parser, YAML values as defaults ─────────────
    parser = build_parser(config_path)

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
        sample_mode(args)


if __name__ == "__main__":
    main()
