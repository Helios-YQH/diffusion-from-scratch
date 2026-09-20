"""FID evaluation and NFE sweeps for the 2x2 study.

Reference statistics are always computed from the *training tensor*
(`data/celeba_64.pt`) so the reference distribution matches exactly what the
models were trained on. Never use third-party stats: any difference in
resizing or preprocessing makes the numbers incomparable.

Usage (single GPU or torchrun):
  # 1) build the reference stats once -> eval/stats/celeba64_ref_pool3.npz
  python eval/fid.py --build-ref

  # 2) evaluate one configuration
  python eval/fid.py --run-name celeba_dit_rf --n 10000 --sampler euler --nfe 16

  # 3) NFE sweep (Tier 1 = 10k samples)
  python eval/fid.py --run-name celeba_dit_rf --n 10000 --sampler euler \
      --nfe 4 8 16 32 64 128

  # Tier 2 = 50k samples, 2 GPUs
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      eval/fid.py --run-name celeba_dit_eps --n 50000 --sampler ddim --nfe 50
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from models.dit import DiT
from models.unet import UNet

REF_STATS_PATH = os.path.join("eval", "stats", "celeba64_ref_pool3.npz")
RESULTS_DIR = "results"
FEAT_DIM = 2048


# ── Inception features ────────────────────────────────────────────────

def get_inception(device):
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()  # keep pool3 (2048-d)
    return model.eval().to(device)


def _normalize_for_inception(x):
    """[-1, 1] images at 64x64 -> 299x299 ImageNet-normalized input."""
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    x = F.interpolate(x, size=(299, 299), mode="bilinear",
                      align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def inception_features(inception, images, device, batch_size=64):
    """images: float tensor [N, 3, H, W] in [-1, 1] -> [N, 2048] on CPU."""
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = _normalize_for_inception(images[i:i + batch_size].to(device))
        feats.append(inception(batch).float().cpu())
    return torch.cat(feats)


# ── FID ───────────────────────────────────────────────────────────────

def compute_stats(feats):
    arr = feats.double().numpy()
    return arr.mean(axis=0), np.cov(arr, rowvar=False)


def _sqrtm_psd(mat):
    """Symmetric PSD matrix square root (scipy-free, exact for PSD input)."""
    mat = (mat + mat.T) / 2.0
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def fid_from_stats(mu1, sigma1, mu2, sigma2):
    """Frechet distance: ||mu1-mu2||^2 + tr(S1+S2-2(S1^1/2 S2 S1^1/2)^1/2)."""
    diff = mu1 - mu2
    sqrt_s1 = _sqrtm_psd(sigma1)
    mid = sqrt_s1 @ sigma2 @ sqrt_s1
    eig = np.linalg.eigvalsh((mid + mid.T) / 2.0)
    trace_term = np.sqrt(np.clip(eig, 0.0, None)).sum()
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * trace_term)


# ── Reference statistics ──────────────────────────────────────────────

def build_ref_stats(data_path, inception, device, batch_size=64, chunk=4096):
    """Features over the full training tensor (memory-mapped, chunked)."""
    data = torch.load(data_path, weights_only=True, mmap=True)
    n = data.shape[0]
    feats = []
    for i in range(0, n, chunk):
        piece = data[i:i + chunk]
        piece = piece.float().div_(127.5).sub_(1.0) \
            if piece.dtype == torch.uint8 else piece.float()
        feats.append(inception_features(inception, piece, device, batch_size))
        print(f"  ref features: {min(i + chunk, n)}/{n}", flush=True)
    return torch.cat(feats)


def ensure_ref_stats(args, inception, device, rank, world_size):
    """Returns (mu, sigma); computes them on rank 0 if the cache is missing,
    then broadcasts to the other ranks."""
    if os.path.exists(REF_STATS_PATH):
        npz = np.load(REF_STATS_PATH)
        return npz["mu"], npz["sigma"]

    if rank == 0:
        print(f"Reference stats not found — building from {args.data} ...")
        feats = build_ref_stats(args.data, inception, device,
                                args.feat_batch)
        mu, sigma = compute_stats(feats)
        os.makedirs(os.path.dirname(REF_STATS_PATH), exist_ok=True)
        np.savez(REF_STATS_PATH, mu=mu, sigma=sigma, n=feats.shape[0])
        print(f"Saved {REF_STATS_PATH}  ({feats.shape[0]} reference images)")

    if world_size > 1:
        dist.barrier()
    npz = np.load(REF_STATS_PATH)
    return npz["mu"], npz["sigma"]


# ── Sampling ──────────────────────────────────────────────────────────

def load_model(ckpt_path, device, weights="ema"):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and ("model" in raw or "ema" in raw):
        arch = raw.get("arch") or {}
        if weights == "ema" and raw.get("ema") is not None:
            state = raw["ema"]
        else:
            state = raw["model"] if raw.get("model") is not None else raw["ema"]
    else:
        state, arch = raw, {}

    backbone = arch.get("backbone", "unet")
    img_channels = arch.get("img_channels", 3)
    img_size = arch.get("img_size", 64)
    if backbone == "dit":
        model = DiT(img_size=img_size,
                    patch_size=arch.get("patch_size") or 4,
                    img_channels=img_channels,
                    hidden_size=arch.get("hidden_size") or 768,
                    depth=arch.get("depth") or 12,
                    num_heads=arch.get("num_heads") or 12)
    else:
        model = UNet(img_channels=img_channels,
                     base_channels=arch.get("base_channels") or 128,
                     time_dim=256,
                     num_downs=arch.get("num_downs") or 4)
    model.load_state_dict(state)
    model.eval().to(device)

    meta = {
        "backbone": backbone,
        "objective": arch.get("objective", "eps"),
        "img_channels": img_channels,
        "img_size": img_size,
        "timesteps": arch.get("timesteps", 1000),
        "arch": arch,
    }
    return model, meta


def generate_features(model, meta, total, args, device, rank, seed):
    """Generate `total` samples on this rank and return their Inception features."""
    obj_kwargs = dict(device=device, img_channels=meta["img_channels"],
                      img_size=meta["img_size"])
    if meta["objective"] == "rf":
        obj = RectifiedFlow(model, **obj_kwargs)
    else:
        obj = DDPM(model, T=meta["timesteps"], **obj_kwargs)

    feats = []
    done = 0
    while done < total:
        b = min(args.batch_size, total - done)
        torch.manual_seed(seed + done)
        if meta["objective"] == "rf":
            x = obj.sample(b, num_steps=args.nfe)
        elif args.sampler == "ddim":
            x = obj.sample_ddim(b, num_steps=args.nfe)
        else:
            x = obj.sample(b)
        model.eval()
        feats.append(inception_features(args.inception, x, device,
                                        args.feat_batch))
        done += b
        if rank == 0:
            print(f"    generated {done}/{total}", flush=True)
    return torch.cat(feats)


def check_sampler(meta, sampler):
    objective = meta["objective"]
    if objective == "rf" and sampler != "euler":
        raise SystemExit(f"This checkpoint is rectified flow — use --sampler euler "
                         f"(got '{sampler}')")
    if objective == "eps" and sampler not in ("ancestral", "ddim"):
        raise SystemExit(f"This checkpoint is eps/DDPM — use --sampler ancestral|ddim "
                         f"(got '{sampler}')")


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("FID evaluation")
    p.add_argument("--build-ref", action="store_true",
                   help="Build reference stats and exit")
    p.add_argument("--data", type=str, default=os.path.join("data", "celeba_64.pt"),
                   help="Training tensor used for the reference stats")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--sampler", choices=["ancestral", "ddim", "euler"],
                   default="euler")
    p.add_argument("--nfe", type=int, nargs="+", default=[32],
                   help="ODE/DDIM steps to sweep (ancestral ignores this)")
    p.add_argument("--n", type=int, default=10000, help="Total samples (Tier 1 = 10k)")
    p.add_argument("--batch-size", type=int, default=64, help="Sampling batch size")
    p.add_argument("--feat-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None, help="JSON name suffix")
    return p.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = "LOCAL_RANK" in os.environ and world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rank = 0
    is_main = rank == 0

    args.inception = get_inception(device)

    if args.build_ref:
        ensure_ref_stats(args, args.inception, device, rank, world_size)
        if is_main:
            print("Reference stats ready.")
        if distributed:
            dist.destroy_process_group()
        return

    # ── Resolve checkpoint ────────────────────────────────────────
    ckpt = args.checkpoint
    if ckpt is None:
        run_name = args.run_name or "celeba_dit_rf"
        for kind in ("best", "latest"):
            cand = os.path.join("checkpoints", f"{run_name}_{kind}.pt")
            if os.path.exists(cand):
                ckpt = cand
                break
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint for run '{run_name}'")
    run_name = args.run_name or os.path.basename(ckpt).rsplit("_", 1)[0]

    if is_main:
        print(f"Checkpoint: {ckpt}  weights={args.weights}")

    model, meta = load_model(ckpt, device, args.weights)
    if is_main:
        print(f"Backbone: {meta['backbone']}  objective: {meta['objective']}  "
              f"({meta['img_channels']}ch, {meta['img_size']}x{meta['img_size']})")

    mu_ref, sigma_ref = ensure_ref_stats(args, args.inception, device, rank,
                                        world_size)

    # ── Sweep ─────────────────────────────────────────────────────
    per_rank = args.n // world_size
    records = []
    tag = args.tag or f"{args.weights}-{args.sampler}"
    out_path = os.path.join(RESULTS_DIR, f"fid_{run_name}.json")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            records = json.load(f).get("runs", [])

    for nfe in args.nfe:
        check_sampler(meta, args.sampler)
        args.nfe = nfe
        nfe_used = meta["timesteps"] if args.sampler == "ancestral" else nfe
        if is_main:
            print(f"\n[{tag}] NFE={nfe_used}  n={per_rank * world_size} "
                  f"({per_rank}/rank)")

        feats = generate_features(model, meta, per_rank, args, device, rank,
                                  seed=args.seed + 1000 * rank)

        if world_size > 1:
            gathered = [torch.zeros_like(feats) for _ in range(world_size)]
            dist.all_gather(gathered, feats)
            feats = torch.cat(gathered)

        if is_main:
            mu, sigma = compute_stats(feats)
            fid = fid_from_stats(mu_ref, sigma_ref, mu, sigma)
            print(f"  FID = {fid:.3f}   (n={feats.shape[0]}, weights={args.weights})")
            records = [r for r in records
                       if not (r["sampler"] == args.sampler
                               and r["nfe"] == nfe_used
                               and r["weights"] == args.weights)]
            records.append({"run_name": run_name, "sampler": args.sampler,
                            "nfe": nfe_used, "n": int(feats.shape[0]),
                            "weights": args.weights, "fid": fid,
                            "git_commit": _git_commit()})
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"run_name": run_name, "runs": sorted(
                    records, key=lambda r: r["nfe"])}, f, indent=2)
            print(f"  written: {out_path}")

    if is_main:
        print("\nsampler    NFE    FID")
        for r in sorted(records, key=lambda r: r["nfe"]):
            print(f"{r['sampler']:<10} {r['nfe']:>4}   {r['fid']:.3f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
