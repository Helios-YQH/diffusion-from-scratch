"""FID evaluation and NFE sweeps.

Reference statistics are always computed from the *training tensor* so the
reference distribution matches exactly what the models were trained on. Never
use third-party stats: any difference in resizing or preprocessing makes the
numbers incomparable.

Two feature extractors are supported:
  --feature-net inception   standard FID features (2048-d pool3), for CelebA
  --feature-net mnist       a small CNN trained on MNIST in this repo, for the
                            MNIST cells (non-standard but consistent; see the
                            report's limitations section)

Usage (single GPU or torchrun):
  # 1) build the reference stats once
  python eval/fid.py --build-ref --feature-net inception

  # 2) evaluate one configuration
  python eval/fid.py --run-name celeba_dit_rf --num-samples 10000 --sampler euler --nfe 16
  python eval/fid.py --run-name mnist_unet_rf   --num-samples 10000 --sampler euler --nfe 16 \
      --feature-net mnist

  # 3) NFE sweep
  python eval/fid.py --run-name celeba_unet_eps --num-samples 10000 --sampler ddim \
      --nfe 10 20 50
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from models.dit import DiT
from models.unet import UNet

STATS_DIR = os.path.join("eval", "stats")
RESULTS_DIR = "results"
MNIST_FEATNET = os.path.join(STATS_DIR, "mnist_featnet.pt")

DEFAULT_DATA = {
    "inception": os.path.join("data", "celeba_64_uint8.pt"),
    "mnist": os.path.join("data", "mnist", "mnist.pkl.gz"),
}


# ── Feature extractors ────────────────────────────────────────────────

def get_inception(device):
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                         transform_input=False)
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
def inception_features(net, images, device, batch_size=64):
    """images: float tensor [N, 3, H, W] in [-1, 1] -> [N, 2048] on CPU."""
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = _normalize_for_inception(images[i:i + batch_size].to(device))
        feats.append(net(batch).float().cpu())
    return torch.cat(feats)


class MnistFeatNet(nn.Module):
    """Small convolutional trunk used as an MNIST feature extractor.

    FID against Inception features is not meaningful for grayscale digits, so
    the MNIST cells are evaluated in the feature space of a classifier trained
    on the dataset itself. The metric is therefore non-standard and only
    comparable within this report.
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.trunk(x).flatten(1)


def train_mnist_featnet(device, epochs=3, cache=MNIST_FEATNET):
    """Train (or load) the MNIST feature trunk. Cached under eval/stats/."""
    net = MnistFeatNet()
    if os.path.exists(cache):
        net.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
        return net.eval().to(device)

    from train import load_mnist

    x, y = load_mnist(return_labels=True)
    head = nn.Linear(128, 10)
    net, head = net.to(device), head.to(device)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
    loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True)

    print("  training the MNIST feature net (one-time, ~1 min)...")
    for ep in range(epochs):
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = head(net(xb.repeat(1, 3, 1, 1)))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(-1) == yb).sum().item()
            total += yb.numel()
        print(f"    epoch {ep + 1}: classifier acc {correct / total:.3%}")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(net.state_dict(), cache)
    return net.eval().to(device)


@torch.no_grad()
def mnist_features(net, images, device, batch_size=64):
    """images [N, 1-or-3, H, W] in [-1,1] -> [N, 128] features on CPU."""
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = images[i:i + batch_size].to(device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != 28:
            x = F.interpolate(x, size=(28, 28), mode="bilinear",
                              align_corners=False, antialias=True)
        feats.append(net(x).float().cpu())
    return torch.cat(feats)


def build_feature_extractor(name, device):
    """Returns feat_fn(images, batch_size) -> CPU feature tensor."""
    if name == "mnist":
        net = train_mnist_featnet(device)
        return lambda images, bs=64: mnist_features(net, images, device, bs)

    net = get_inception(device)
    return lambda images, bs=64: inception_features(net, images, device, bs)


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

def load_reference_images(path):
    if path.endswith(".pkl.gz"):
        from train import load_mnist
        return load_mnist()          # [N, 1, 28, 28] float in [-1, 1]
    if not os.path.exists(path):
        # The CelebA training tensor is normally produced by a training run.
        # Build it on demand so evaluation does not depend on one having run.
        import re

        from train import extract_celeba, preprocess_celeba
        m = re.search(r"celeba_(\d+)_uint8\.pt$", path)
        size = int(m.group(1)) if m else 64
        print(f"{path} not found — building it (extract + resize, "
              f"one-time, ~20 min) ...", flush=True)
        extract_celeba()
        preprocess_celeba(image_size=size)
    return torch.load(path, weights_only=True, mmap=True)


def build_ref_stats(data_path, feat_fn, batch_size=64, chunk=4096):
    """Features over the full training set (memory-mapped, chunked)."""
    data = load_reference_images(data_path)
    n = data.shape[0]
    feats = []
    for i in range(0, n, chunk):
        piece = data[i:i + chunk]
        piece = piece.float().div_(127.5).sub_(1.0) \
            if piece.dtype == torch.uint8 else piece.float()
        feats.append(feat_fn(piece, batch_size))
        print(f"  ref features: {min(i + chunk, n)}/{n}", flush=True)
    return torch.cat(feats)


def ensure_ref_stats(args, feat_fn, rank, world_size):
    path = os.path.join(STATS_DIR, f"{args.feature_net}_ref.npz")
    if not os.path.exists(path) and rank == 0:
        print(f"Reference stats not found — building from {args.data} ...")
        feats = build_ref_stats(args.data, feat_fn, args.feat_batch)
        mu, sigma = compute_stats(feats)
        os.makedirs(STATS_DIR, exist_ok=True)
        np.savez(path, mu=mu, sigma=sigma, n=feats.shape[0])
        print(f"Saved {path}  ({feats.shape[0]} reference images)")
    if world_size > 1:
        dist.barrier()
    npz = np.load(path)
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


def generate_features(model, meta, total, args, device, rank, seed, feat_fn):
    """Generate `total` samples on this rank; returns (features, sample seconds).

    The returned time covers sampling only — feature extraction is excluded so
    that `sampling_img_s` reflects the generative cost, not the evaluation cost.
    """
    obj_kwargs = dict(device=device, img_channels=meta["img_channels"],
                      img_size=meta["img_size"])
    if meta["objective"] == "rf":
        obj = RectifiedFlow(model, **obj_kwargs)
    else:
        obj = DDPM(model, T=meta["timesteps"], **obj_kwargs)

    feats = []
    sample_seconds = 0.0
    done = 0
    while done < total:
        b = min(args.batch_size, total - done)
        torch.manual_seed(seed + done)
        t0 = time.time()
        if meta["objective"] == "rf":
            x = obj.sample(b, num_steps=args.nfe)
        elif args.sampler == "ddim":
            x = obj.sample_ddim(b, num_steps=args.nfe)
        else:
            x = obj.sample(b)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        sample_seconds += time.time() - t0
        model.eval()
        feats.append(feat_fn(x, args.feat_batch))
        done += b
        if rank == 0:
            print(f"    generated {done}/{total}", flush=True)
    return torch.cat(feats), sample_seconds


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
    p.add_argument("--data", type=str, default=None,
                   help="Training data used for the reference stats "
                        "(default depends on --feature-net)")
    p.add_argument("--feature-net", choices=["inception", "mnist"], default="inception")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--sampler", choices=["ancestral", "ddim", "euler"],
                   default="euler")
    p.add_argument("--nfe", type=int, nargs="+", default=[32],
                   help="ODE/DDIM steps to sweep (ancestral ignores this)")
    p.add_argument("--num-samples", type=int, default=10000, dest="n",
                   help="Total samples (Tier 1 = 10k). Deliberately not called "
                        "--n: torchrun's own parser rejects it as an ambiguous "
                        "prefix of --nnodes/--nproc-per-node.")
    p.add_argument("--batch-size", type=int, default=64, help="Sampling batch size")
    p.add_argument("--feat-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None, help="JSON name suffix")
    args = p.parse_args()
    if args.data is None:
        args.data = DEFAULT_DATA[args.feature_net]
    return args


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

    feat_fn = build_feature_extractor(args.feature_net, device)

    if args.build_ref:
        ensure_ref_stats(args, feat_fn, rank, world_size)
        if is_main:
            print("Reference stats ready.")
        if distributed:
            dist.destroy_process_group()
        return

    # ── Resolve checkpoint ────────────────────────────────────────
    ckpt = args.checkpoint
    run_name = args.run_name
    if ckpt is None:
        run_name = run_name or "celeba_dit_rf"
        for kind in ("best", "latest"):
            cand = os.path.join("checkpoints", f"{run_name}_{kind}.pt")
            if os.path.exists(cand):
                ckpt = cand
                break
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint for run '{run_name}'")
    run_name = run_name or os.path.basename(ckpt).rsplit("_", 1)[0]

    if is_main:
        print(f"Checkpoint: {ckpt}  weights={args.weights}  "
              f"features={args.feature_net}")

    model, meta = load_model(ckpt, device, args.weights)
    if is_main:
        print(f"Backbone: {meta['backbone']}  objective: {meta['objective']}  "
              f"({meta['img_channels']}ch, {meta['img_size']}x{meta['img_size']})")

    mu_ref, sigma_ref = ensure_ref_stats(args, feat_fn, rank, world_size)

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

        feats, sample_seconds = generate_features(
            model, meta, per_rank, args, device, rank,
            seed=args.seed + 1000 * rank, feat_fn=feat_fn)

        if world_size > 1:
            # NCCL only talks CUDA tensors, and the features are computed on CPU
            # (deliberately, so a 50k-sample sweep never has to hold activations
            # on the GPU). Move them over for the collective, then back.
            feats_dev = feats.to(device, non_blocking=True)
            gathered = [torch.empty_like(feats_dev) for _ in range(world_size)]
            dist.all_gather(gathered, feats_dev)
            feats = torch.cat([g.cpu() for g in gathered])
            t = torch.tensor([sample_seconds], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            sample_seconds = float(t.item())

        if is_main:
            mu, sigma = compute_stats(feats)
            fid = fid_from_stats(mu_ref, sigma_ref, mu, sigma)
            n_total = int(feats.shape[0])
            img_s = n_total / sample_seconds if sample_seconds > 0 else None
            print(f"  FID = {fid:.3f}   (n={n_total}, weights={args.weights}, "
                  f"{img_s:.1f} img/s)" if img_s else
                  f"  FID = {fid:.3f}   (n={n_total}, weights={args.weights})")
            records = [r for r in records
                       if not (r["sampler"] == args.sampler
                               and r["nfe"] == nfe_used
                               and r["weights"] == args.weights)]
            records.append({"run_name": run_name, "sampler": args.sampler,
                            "nfe": nfe_used, "n": n_total,
                            "weights": args.weights, "fid": fid,
                            "feature_net": args.feature_net,
                            "sampling_img_s": img_s,
                            "git_commit": _git_commit()})
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"run_name": run_name, "runs": sorted(
                    records, key=lambda r: r["nfe"])}, f, indent=2)
            print(f"  written: {out_path}")

    if is_main:
        print("\nsampler    NFE    FID      img/s")
        for r in sorted(records, key=lambda r: r["nfe"]):
            ips = r.get("sampling_img_s")
            print(f"{r['sampler']:<10} {r['nfe']:>4}   {r['fid']:>6.3f}   "
                  f"{ips:>7.1f}" if ips else
                  f"{r['sampler']:<10} {r['nfe']:>4}   {r['fid']:>6.3f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
