"""Cost accounting: parameter count, FLOPs per sample, and MFU.

FLOPs are measured with torch.utils.flop_counter (forward pass, multiply-adds
counted as 2) rather than closed-form formulas, so the UNet and the DiT are
measured the same way. Training FLOPs are approximated as 3x forward.

Usage:
  python eval/cost.py --config configs/celeba_dit_rf.yml
  python eval/cost.py --config configs/celeba_dit_rf.yml \
      --step-time 0.42 --batch-size 800 --gpu A6000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.dit import DiT
from models.unet import UNet

# Dense (non-sparse) peaks in FLOP/s. Override with --peak-tflops if your card
# is missing or the clock is different.
GPU_PEAKS = {
    "A6000": {"fp32": 38.7e12, "tf32": 77.4e12, "bf16": 154.8e12},
    "A100": {"fp32": 19.5e12, "tf32": 156e12, "bf16": 312e12},
    "H100": {"fp32": 67e12, "tf32": 495e12, "bf16": 989e12},
    "4090": {"fp32": 82.6e12, "tf32": 82.6e12, "bf16": 165.2e12},
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, img_size, img_channels, batch_size=1, device="cpu"):
    """Forward FLOPs for one batch, measured with FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model = model.to(device).eval()
    x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            model(x, t)
    model.train(was_training)
    return counter.get_total_flops()


def detect_peak_flops(precision="bf16"):
    """(peak FLOP/s, device name) for the current GPU, or (None, name)."""
    if not torch.cuda.is_available():
        return None, None
    name = torch.cuda.get_device_name(0)
    for key, table in GPU_PEAKS.items():
        if key.lower() in name.lower():
            return table.get(precision), name
    return None, name


def build_model_from_config(cfg_path):
    """Minimal YAML -> model, mirroring train.py's construction."""
    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img_size = cfg.get("image_size") or (64 if cfg.get("dataset") == "celeba" else 28)
    img_channels = 3 if cfg.get("dataset") == "celeba" else 1

    if cfg.get("backbone", "unet") == "dit":
        model = DiT(img_size=img_size,
                    patch_size=cfg.get("patch_size", 4),
                    img_channels=img_channels,
                    hidden_size=cfg.get("hidden_size", 768),
                    depth=cfg.get("depth", 12),
                    num_heads=cfg.get("num_heads", 12))
    else:
        model = UNet(img_channels=img_channels,
                     base_channels=cfg.get("base_channels")
                     or (128 if img_channels == 3 else 64),
                     time_dim=256,
                     num_downs=4 if img_channels == 3 else 3)
    return model, img_size, img_channels


def mfu(flops_per_sample_forward, global_batch, step_time_s, peak_flops,
        backward_factor=3.0):
    """Model FLOPs utilisation for a training step.

    Training FLOPs per sample are approximated as `backward_factor` times the
    measured forward FLOPs (fwd + bwd ~= 3x fwd).
    """
    return (backward_factor * flops_per_sample_forward * global_batch
            / step_time_s / peak_flops)


def summarize(model, img_size, img_channels, device="cpu"):
    params = count_params(model)
    flops = count_flops(model, img_size, img_channels, device=device)
    return {"params": params, "forward_flops_per_sample": flops}


def main():
    p = argparse.ArgumentParser("Cost accounting")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--step-time", type=float, default=None,
                   help="Measured seconds per optimizer step (from a training log)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Global batch size for that step")
    p.add_argument("--gpu", default=None, choices=sorted(GPU_PEAKS))
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="Override the dense peak (TFLOP/s)")
    p.add_argument("--precision", default="bf16",
                   choices=["fp32", "tf32", "bf16"])
    args = p.parse_args()

    model, img_size, img_channels = build_model_from_config(args.config)
    info = summarize(model, img_size, img_channels, args.device)

    print(f"Config: {args.config}")
    print(f"Model:  {type(model).__name__}  "
          f"({img_channels}ch, {img_size}x{img_size})")
    print(f"Params: {info['params'] / 1e6:.1f}M")
    print(f"FLOPs:  {info['forward_flops_per_sample'] / 1e9:.2f} GFLOPs/sample (forward)")

    if args.step_time and args.batch_size and (args.gpu or args.peak_tflops):
        if args.peak_tflops:
            peak = args.peak_tflops * 1e12
        else:
            peak = GPU_PEAKS[args.gpu][args.precision]
        val = mfu(info["forward_flops_per_sample"], args.batch_size,
                  args.step_time, peak)
        print(f"MFU:    {val * 100:.1f}%  "
              f"(global batch {args.batch_size}, {args.step_time:.3f} s/step, "
              f"{peak / 1e12:.1f} TFLOP/s {args.precision})")
        if val > 1.0:
            min_step = 3.0 * info["forward_flops_per_sample"] * args.batch_size / peak
            print(f"  (!) MFU > 100% is impossible — check step time / batch size. "
                  f"At 100% MFU one step would take {min_step:.2f} s.")


if __name__ == "__main__":
    main()
