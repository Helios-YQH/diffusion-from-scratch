"""Systems measurements for the diffusion stack — no convergence training needed.

Runs a short training benchmark (30 steps + 5 discarded warmups) under several
execution settings and reports step time, throughput, peak memory and MFU;
optionally profiles where the GPU time actually goes, and (multi-GPU) measures
the NCCL share of a DDP step. This mirrors the measurement loop of the CS336
assignment-2 systems report, applied to the diffusion models.

Usage:
  python eval/systems.py --config configs/celeba_dit_rf.yml
  python eval/systems.py --config configs/celeba_dit_rf.yml \
      --settings fp32 bf16 compile bf16+compile
  python eval/systems.py --config configs/celeba_dit_rf.yml --profile
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      eval/systems.py --config configs/celeba_dit_rf.yml --ddp
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from eval.cost import build_model_from_config, count_flops, detect_peak_flops, mfu

RESULTS_DIR = "results"

# Kernel-name -> category, for the profiler breakdown. Matched case-insensitively
# against the CUDA kernel names reported by torch.profiler.
CATEGORIES = [
    ("attention", ("flash", "attention", "sdpa", "mem_efficient", "efficient_attention")),
    ("matmul", ("gemm", "cutlass", "matmul", "s16816", "sgemm", "bgemm", "nvjet")),
    ("conv", ("conv", "implicit_gemm", "cudnn")),
    ("norm", ("layer_norm", "group_norm", "native_")),
    ("elementwise", ("elementwise", "mul", "add", "silu", "gelu", "sigmoid",
                     "vectorized", "copy", "cat", "clamp")),
    ("communication", ("nccl", "allreduce", "allgather", "reduce_scatter")),
]


def categorize(kernel_name):
    name = kernel_name.lower()
    for label, patterns in CATEGORIES:
        if any(p in name for p in patterns):
            return label
    return "other"


def _cuda():
    return torch.cuda.is_available()


def make_diffusion(model, cfg, device):
    objective = cfg.get("objective", "eps")
    img_size = cfg.get("image_size") or (64 if cfg.get("dataset") == "celeba" else 28)
    img_channels = 3 if cfg.get("dataset") == "celeba" else 1
    if objective == "rf":
        return RectifiedFlow(model, device=device, img_channels=img_channels,
                             img_size=img_size)
    return DDPM(model, T=cfg.get("timesteps", 1000), device=device,
                img_channels=img_channels, img_size=img_size)


def make_batch(cfg, batch_size, device):
    img_size = cfg.get("image_size") or (64 if cfg.get("dataset") == "celeba" else 28)
    img_channels = 3 if cfg.get("dataset") == "celeba" else 1
    return torch.randn(batch_size, img_channels, img_size, img_size, device=device)


def _one_step(model, diffusion, opt, x0, bf16):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and _cuda()):
        loss = diffusion.training_loss(x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def benchmark(model, diffusion, cfg, device, batch_size, steps, warmup,
              bf16, compile_model, tag):
    if compile_model:
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    for i in range(steps + warmup):
        x0 = make_batch(cfg, batch_size, device)
        if _cuda():
            torch.cuda.synchronize()
        t0 = time.time()
        _one_step(model, diffusion, opt, x0, bf16)
        if _cuda():
            torch.cuda.synchronize()
        dt = time.time() - t0
        if i >= warmup:
            times.append(dt)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if _cuda() else 0.0
    return {"setting": tag,
            "step_time_s": mean,
            "step_time_std_s": std,
            "step_time_cv": std / mean if mean else None,
            "throughput_img_s": batch_size / mean,
            "peak_memory_gb": peak_gb}


def profile_step(model, diffusion, cfg, device, batch_size, steps=6):
    from torch.profiler import ProfilerActivity, profile

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x0 = make_batch(cfg, batch_size, device)
    for _ in range(2):                      # warm up outside the trace
        _one_step(model, diffusion, opt, x0, bf16=False)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            _one_step(model, diffusion, opt, x0, bf16=False)

    totals = {}
    for evt in prof.key_averages():
        if evt.device_type != torch.autograd.DeviceType.CUDA or evt.self_device_time_total <= 0:
            continue
        totals[categorize(evt.key)] = totals.get(categorize(evt.key), 0.0) \
            + evt.self_device_time_total

    total = sum(totals.values())
    return {k: v / total for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def main():
    p = argparse.ArgumentParser("systems benchmark")
    p.add_argument("--config", required=True)
    p.add_argument("--settings", nargs="+", default=["fp32"],
                   choices=["fp32", "bf16", "compile", "bf16+compile"])
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-GPU batch size (default: the config's batch_size)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile", action="store_true",
                   help="Also report the per-category GPU-time breakdown")
    p.add_argument("--ddp", action="store_true",
                   help="Wrap in DDP (use under torchrun for the communication share)")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = args.ddp and world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        device = "cuda:0" if _cuda() else "cpu"
        rank = 0
    is_main = rank == 0

    batch_size = args.batch_size or cfg.get("batch_size", 128)
    model, img_size, img_channels = build_model_from_config(args.config)
    model = model.to(device)
    forward_flops = count_flops(model, img_size, img_channels, device=device)
    peak_flops, gpu_name = detect_peak_flops("bf16")

    if is_main:
        print(f"Config: {args.config}  ({cfg.get('backbone', 'unet')}, "
              f"{cfg.get('objective', 'eps')})  batch/GPU={batch_size}  "
              f"world={world_size}")
        print(f"Model:  {type(model).__name__}  "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
              f"{forward_flops / 1e9:.2f} GFLOPs/sample (forward)")
        if gpu_name:
            print(f"GPU:    {gpu_name}")

    results = []
    for setting in args.settings:
        # fresh module per setting so compile/autocast state cannot leak
        model, _, _ = build_model_from_config(args.config)
        model = model.to(device)
        diffusion = make_diffusion(model, cfg, device)
        bench_model = DDP(model, device_ids=[local_rank]) if is_distributed else model
        bf16 = "bf16" in setting
        compile_model = "compile" in setting
        res = benchmark(bench_model, diffusion, cfg, device, batch_size,
                        args.steps, args.warmup, bf16, compile_model,
                        setting if not is_distributed else f"{setting}+ddp")
        res["world_size"] = world_size
        res["global_throughput_img_s"] = res["throughput_img_s"] * world_size
        if peak_flops:
            res["mfu"] = mfu(forward_flops, batch_size * world_size,
                             res["step_time_s"], peak_flops)
        results.append(res)
        if is_main:
            line = (f"\n[{res['setting']}] {res['step_time_s'] * 1e3:.1f} ms/step "
                    f"(cv {res['step_time_cv'] * 100:.1f}%)  "
                    f"{res['global_throughput_img_s']:.0f} img/s  "
                    f"peak {res['peak_memory_gb']:.1f} GB")
            if "mfu" in res:
                line += f"  MFU {res['mfu'] * 100:.1f}%"
            print(line)

    baseline = next((r for r in results if r["setting"].startswith("fp32")), None)
    if is_main and baseline and len(results) > 1:
        print("\nrelative to fp32 baseline:")
        for r in results:
            sp = baseline["step_time_s"] / r["step_time_s"]
            print(f"  {r['setting']:<18} {sp:.2f}x")

    if args.profile:
        model, _, _ = build_model_from_config(args.config)
        model = model.to(device)
        diffusion = make_diffusion(model, cfg, device)
        fracs = profile_step(model, diffusion, cfg, device, batch_size)
        if is_main:
            print("\nGPU time by kernel category (fp32):")
            for label, frac in fracs.items():
                print(f"  {label:<16} {frac * 100:5.1f}%")

    if is_main:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.config))[0]
        out = os.path.join(RESULTS_DIR, f"systems_{stem}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "batch_size_per_gpu": batch_size,
                       "world_size": world_size, "steps": args.steps,
                       "params": sum(p.numel() for p in model.parameters()),
                       "forward_flops_per_sample": forward_flops,
                       "gpu_name": gpu_name, "results": results,
                       "profile_fractions": args.profile and fracs or None},
                      f, indent=2)
        print(f"\nWritten: {out}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
