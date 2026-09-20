import contextlib
import glob
import gzip
import json
import os
import pickle
import random
import subprocess
import time
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm

from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from eval.cost import count_flops, detect_peak_flops, mfu
from models.dit import DiT
from models.unet import UNet

DATA_PATH = os.path.join("data", "mnist", "mnist.pkl.gz")
CELEBA_PATH = os.path.join("data", "celeba.zip")
CELEBA_DIR = os.path.join("data", "celeba")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"


# ── Data ──────────────────────────────────────────────────────────────

def extract_celeba():
    """Extract CelebA zip to data/celeba/ once. Skips if already done."""
    done_marker = os.path.join(CELEBA_DIR, ".extracted")
    if os.path.exists(done_marker):
        return

    os.makedirs(CELEBA_DIR, exist_ok=True)
    with zipfile.ZipFile(CELEBA_PATH) as zf:
        members = [m for m in zf.infolist()
                   if m.filename.endswith((".jpg", ".jpeg", ".png"))]
        print(f"Extracting {len(members)} images from celeba.zip "
              f"to {CELEBA_DIR}/ ...")
        for m in tqdm(members, desc="Extracting", ncols=80):
            zf.extract(m, CELEBA_DIR)

    with open(done_marker, "w") as f:
        f.write("done")
    print("Extraction complete.")


def preprocess_celeba(image_size=32):
    """Resize all CelebA images into a uint8 .pt cache.

    Images are stored as uint8 [0, 255] and normalized to [-1, 1] on the fly
    in the training loop: the cache is ~2.5GB instead of ~10GB in float32.
    """
    cache_path = os.path.join("data", f"celeba_{image_size}_uint8.pt")
    if os.path.exists(cache_path):
        print(f"Loading preprocessed cache: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    files = sorted(glob.glob(os.path.join(CELEBA_DIR, "**", "*.jpg"),
                             recursive=True))
    if not files:
        raise RuntimeError(
            f"No .jpg files found in {CELEBA_DIR}. "
            f"Did you run extract_celeba() first?")

    print(f"Preprocessing {len(files)} images to {image_size}x{image_size}...")
    tensors = []
    for f in tqdm(files, desc="Preprocessing", ncols=80):
        img = Image.open(f).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # HWC, [0, 255]
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))

    data = torch.stack(tensors)
    torch.save(data, cache_path)
    print(f"Saved {cache_path}  ({data.shape}, uint8)")
    return data


def to_model_input(batch, device):
    """uint8 [0,255] -> float [-1,1] on the training device."""
    x = batch.to(device, non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.float().div_(127.5).sub_(1.0)
    return x


def load_mnist(normalize=True):
    with gzip.open(DATA_PATH, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*align.*")
            train, val, test = pickle.load(f, encoding="latin1")
    x_train, y_train = train
    x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32)
    if normalize:
        x_train = x_train * 2.0 - 1.0  # [0,1] -> [-1,1]
    return torch.from_numpy(x_train)


def save_sample_grid(images, path, nrow=4):
    images = images.detach().cpu()
    images = (images * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    n = min(images.shape[0], nrow * nrow)
    is_rgb = images.shape[1] == 3
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 1.5, nrow * 1.5))
    for i in range(n):
        r, c = i // nrow, i % nrow
        img = images[i]
        if is_rgb:
            axes[r, c].imshow(img.permute(1, 2, 0))
        else:
            axes[r, c].imshow(img[0], cmap="gray")
        axes[r, c].axis("off")
    for i in range(n, nrow * nrow):
        r, c = i // nrow, i % nrow
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ── Model / objective construction ────────────────────────────────────

def build_model(args, img_channels, img_size):
    if args.backbone == "dit":
        return DiT(img_size=img_size, patch_size=args.patch_size,
                   img_channels=img_channels, hidden_size=args.hidden_size,
                   depth=args.depth, num_heads=args.num_heads)
    return UNet(img_channels=img_channels, base_channels=args.base_channels,
                time_dim=256, num_downs=args.num_downs)


def build_objective(args, model, device, img_channels, img_size):
    if args.objective == "rf":
        return RectifiedFlow(model, device=device, img_channels=img_channels,
                             img_size=img_size, t_sample=args.t_sample)
    return DDPM(model, T=args.timesteps, device=device,
                img_channels=img_channels, img_size=img_size)


# ── EMA ───────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model weights.

    decay 0.9999 gives a ~10k-step window; keep it well below the total
    number of training steps.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone().float() for k, v in state.items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    @contextlib.contextmanager
    def applied_to(self, model):
        """Temporarily swap EMA weights into `model` (restores on exit)."""
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(raw)


# ── Checkpoint save / load ────────────────────────────────────────────

def _checkpoint_path(run_name, kind="latest"):
    return os.path.join(CHECKPOINT_DIR, f"{run_name}_{kind}.pt")


def save_checkpoint(model_for_save, opt, scheduler, epoch, best_loss,
                    run_name, arch_cfg, global_step, ema=None):
    """Full training state (for resume), including EMA weights."""
    path = _checkpoint_path(run_name, "latest")
    state = {
        "model": model_for_save.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "arch": arch_cfg,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)
    return path


def save_weights(model_for_save, ema, run_name, arch_cfg, kind="best",
                 ema_only=False):
    """Weights only — for `best` and per-epoch snapshots."""
    path = _checkpoint_path(run_name, kind)
    state = {"arch": arch_cfg}
    if not ema_only:
        state["model"] = model_for_save.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(run_name, device, arch_cfg):
    """Load full training state. Returns (state_dict, meta) or (None, None).

    Handles both new-format (dict with "model"/"optimizer"/"scheduler"
    keys) and old-format (bare state_dict) checkpoints. Old-format files
    are promoted to new format with arch_cfg injected — they restore model
    weights only, with a fresh optimizer/scheduler.
    """
    path = _checkpoint_path(run_name, "latest")
    if not os.path.exists(path):
        return None, None

    state = torch.load(path, map_location=device, weights_only=False)

    # Detect old-format checkpoint: raw state_dict without "model" wrapper.
    # Model weights always have string keys like "conv_in.weight"; a
    # new-format checkpoint has top-level keys like "model", "optimizer", …
    if "model" not in state and "optimizer" not in state:
        print(f"  [info] Old-format checkpoint detected. "
              f"Restoring model weights only (fresh optimizer/scheduler).")
        state = {"model": state, "epoch": 0, "global_step": 0,
                 "best_loss": float("inf"), "arch": arch_cfg}

    # Validate architecture compatibility on shared keys
    saved_arch = state.get("arch", {})
    for key, val in arch_cfg.items():
        if key in saved_arch and saved_arch[key] is not None and val is not None:
            if saved_arch[key] != val:
                raise RuntimeError(
                    f"Architecture mismatch: saved {key}={saved_arch[key]}, "
                    f"current {key}={val}. "
                    f"Use --no-resume for a fresh start."
                )

    return state, path


def _git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _safe_count_flops(model, img_size, img_channels, device, is_main):
    """Forward FLOPs for one sample; returns None if the counter fails."""
    try:
        return count_flops(model, img_size, img_channels, device=device)
    except Exception as e:
        if is_main:
            print(f"  [warn] FLOPs measurement failed ({e}); continuing without it.")
        return None


def _init_wandb(args, run_name, arch_cfg, is_main):
    """Optional Weights & Biases logging. Returns (wandb_module, run)."""
    if not getattr(args, "wandb", False) or not is_main:
        return None, None
    try:
        import wandb
    except ImportError:
        print("  [warn] --wandb passed but wandb is not installed "
              "(install with `uv sync --extra log`); logging disabled.")
        return None, None
    run = wandb.init(project=getattr(args, "wandb_project", "diffusion-from-scratch"),
                     name=run_name, config={**vars(args), "arch": arch_cfg})
    return wandb, run


# ── Main training entry point ─────────────────────────────────────────

def train(args):
    """Training entry point. Detects torchrun env vars for DDP."""

    # ── DDP setup ──────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = "LOCAL_RANK" in os.environ and world_size > 1
    is_main = rank == 0

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        if is_main:
            print(f"DDP mode")
            print(f"World size: {world_size}")
            print(f"Per-GPU batch size: {args.batch_size}")
            print(f"Global batch size: {args.batch_size * world_size}")

        print(f"Rank {rank} -> cuda:{local_rank}")
        if world_size > 1:
            dist.barrier()
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            torch.cuda.set_device(device)
        print(f"Single GPU: {device}")
        if torch.cuda.is_available():
            print(f"Visible GPUs: {torch.cuda.device_count()}")

    seed = getattr(args, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = args.run_name or f"{args.dataset}_{args.backbone}_{args.objective}"

    # ── Dataset ────────────────────────────────────────────────────
    if args.dataset == "celeba":
        img_channels, img_size = 3, args.image_size or 64
        base_channels = args.base_channels or 128
        num_downs = 4
        num_workers = getattr(args, "num_workers", 4)

        if is_main or not is_distributed:
            extract_celeba()
        if is_distributed:
            dist.barrier()
        data_tensor = preprocess_celeba(image_size=img_size)
        dataset = TensorDataset(data_tensor)
    else:
        img_channels, img_size = 1, args.image_size or 28
        base_channels = args.base_channels or 64
        num_downs = 3
        num_workers = getattr(args, "num_workers", 0)

        x_train = load_mnist()
        dataset = TensorDataset(x_train)

    # Keep the UNet fields populated even for DiT runs: they are part of the
    # architecture descriptor used for checkpoint compatibility checks.
    args.base_channels = base_channels
    args.num_downs = num_downs

    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True) if is_distributed else None)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        pin_memory=True, num_workers=num_workers)

    # ── Model ──────────────────────────────────────────────────────
    model = build_model(args, img_channels, img_size).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    diffusion = build_objective(args, model, device, img_channels, img_size)

    # ── Optimizer & scheduler ──────────────────────────────────────
    opt = torch.optim.Adam(diffusion.model.parameters(), lr=args.lr)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 2)

    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                               eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])

    if is_main:
        print(f"LR: {args.lr}  warmup: {warmup_steps} steps  "
              f"cosine decay to 1e-6 over {total_steps - warmup_steps} steps")

    model_for_save = model.module if is_distributed else model
    n_params = sum(p.numel() for p in model_for_save.parameters())
    flops_per_sample = _safe_count_flops(model_for_save, img_size, img_channels,
                                         device, is_main)
    best_loss = float("inf")
    start_epoch = 1
    resumed_state = None

    arch_cfg = {
        "backbone": args.backbone,
        "objective": args.objective,
        "base_channels": base_channels,
        "num_downs": num_downs,
        "img_channels": img_channels,
        "img_size": img_size,
        "timesteps": args.timesteps,
        "patch_size": getattr(args, "patch_size", None),
        "hidden_size": getattr(args, "hidden_size", None),
        "depth": getattr(args, "depth", None),
        "num_heads": getattr(args, "num_heads", None),
    }

    if is_main:
        fl = (f"  {flops_per_sample / 1e9:.1f} GFLOPs/sample"
              if flops_per_sample else "")
        print(f"Run: {run_name}  backbone={args.backbone}  "
              f"objective={args.objective}  params={n_params / 1e6:.1f}M{fl}")

    wandb, wb = _init_wandb(args, run_name, arch_cfg, is_main)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if is_main:
        os.makedirs("samples", exist_ok=True)

    # ── Resume from checkpoint ─────────────────────────────────────
    if args.resume:
        state, ckpt_path = load_checkpoint(run_name, device, arch_cfg)
        if state is not None:
            resumed_state = state
            model_for_save.load_state_dict(state["model"])

            if "optimizer" in state:
                opt.load_state_dict(state["optimizer"])
            elif is_main:
                print("  [info] No optimizer state in checkpoint — fresh optimizer.")

            if "scheduler" in state:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception as e:
                    if is_main:
                        print(f"  [warn] Scheduler state mismatch "
                              f"(config changed?): {e}")
            elif is_main:
                print("  [info] No scheduler state in checkpoint — fresh scheduler.")

            start_epoch = state["epoch"] + 1
            best_loss = state.get("best_loss", float("inf"))
            if is_main:
                saved_step = state.get("global_step", 0)
                print(f"Resumed from {ckpt_path}")
                print(f"  epoch {state['epoch']}  best_loss={best_loss:.6f}  "
                      f"global_step={saved_step}")
        elif is_main:
            print(f"No checkpoint found at "
                  f"{_checkpoint_path(run_name, 'latest')} — fresh start.")

    ema = EMA(model_for_save, decay=getattr(args, "ema_decay", 0.9999))
    if resumed_state is not None and "ema" in resumed_state:
        ema.load_state_dict(resumed_state["ema"])
        if is_main:
            print("  [info] EMA state restored.")

    if is_distributed:
        # Broadcast start_epoch so all ranks agree
        t = torch.tensor([start_epoch], device=device)
        dist.broadcast(t, src=0)
        start_epoch = int(t.item())
        dist.barrier()

    if start_epoch > args.epochs:
        if is_main:
            print(f"Already completed {args.epochs} epochs (resumed from "
                  f"epoch {start_epoch - 1}). Nothing to do.")
        if is_distributed:
            dist.destroy_process_group()
        return

    # ── Training loop ──────────────────────────────────────────────
    sample_interval = getattr(args, "sample_interval", args.save_interval)
    save_interval = args.save_interval
    snapshot_interval = getattr(args, "snapshot_interval", 0)
    max_steps = getattr(args, "max_steps", 0) or 0
    grad_clip = getattr(args, "grad_clip", 1.0)

    epoch_time = 0.0
    throughput = 0.0
    avg_loss = float("nan")
    avg_gnorm = float("nan")

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        total_gnorm = 0.0
        num_batches = 0
        t_start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}",
                    ncols=80, disable=not is_main)

        for batch in pbar:
            x0 = to_model_input(batch[0], device)
            loss = diffusion.training_loss(x0)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                diffusion.model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            ema.update(model_for_save)

            total_loss += loss.item()
            total_gnorm += float(gnorm)
            num_batches += 1
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            if max_steps and num_batches >= max_steps:
                break

        epoch_time = time.time() - t_start
        avg_loss = total_loss / max(num_batches, 1)
        avg_gnorm = total_gnorm / max(num_batches, 1)
        throughput = (num_batches * args.batch_size * world_size
                      / max(epoch_time, 1e-9))

        if is_distributed:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        global_step = epoch * steps_per_epoch

        if is_main:
            print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}  "
                  f"grad={avg_gnorm:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"| {epoch_time:.1f}s  {throughput:.0f} img/s")

            # Best model (weights only, raw + EMA, for sampling / eval)
            if avg_loss < best_loss:
                best_loss = avg_loss
                path = save_weights(model_for_save, ema, run_name, arch_cfg,
                                    kind="best")
                print(f"  Best checkpoint: {path}  (loss={best_loss:.6f})")

            # Per-epoch snapshots (EMA weights only) — FID-vs-steps curves
            if snapshot_interval and (epoch % snapshot_interval == 0
                                      or epoch == args.epochs):
                path = save_weights(model_for_save, ema, run_name, arch_cfg,
                                    kind=f"epoch{epoch}", ema_only=True)
                print(f"  Snapshot: {path}")

            # Periodic full training state (save_interval = 0 -> final epoch only)
            if epoch == args.epochs or (save_interval and epoch % save_interval == 0):
                path = save_checkpoint(model_for_save, opt, scheduler,
                                       epoch, best_loss, run_name,
                                       arch_cfg, global_step, ema=ema)
                print(f"  Training state saved: {path}")

            # Sample grid (EMA weights); sample_interval = 0 disables it
            if sample_interval and (epoch % sample_interval == 0
                                    or epoch == args.epochs):
                with ema.applied_to(model_for_save):
                    samples = diffusion.sample(16)
                grid_path = os.path.join("samples", f"{run_name}_epoch{epoch}.png")
                save_sample_grid(samples, grid_path)
                if wb:
                    wb.log({"samples": wandb.Image(grid_path)}, step=global_step)

            if wb:
                wb.log({"train/loss": avg_loss,
                        "train/grad_norm": avg_gnorm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch_time_s": epoch_time,
                        "train/throughput_img_s": throughput,
                        "train/epoch": epoch}, step=global_step)

    # ── Cost accounting summary ────────────────────────────────────
    if is_main:
        peak_gb = 0.0
        if str(device).startswith("cuda"):
            peak_gb = torch.cuda.max_memory_allocated(torch.device(device)) / 1e9
        step_time = epoch_time / max(num_batches, 1)
        global_batch = args.batch_size * world_size

        peak_flops, gpu_name = detect_peak_flops("bf16")
        mfu_val = None
        if flops_per_sample and step_time > 0 and peak_flops:
            mfu_val = mfu(flops_per_sample, global_batch, step_time, peak_flops)

        print(f"\nModel: {args.backbone}  Params: {n_params / 1e6:.1f}M")
        if flops_per_sample:
            print(f"FLOPs:  {flops_per_sample / 1e9:.2f} GFLOPs/sample (forward)")
        print(f"Training: batch {global_batch} | {step_time:.3f} s/step | "
              f"{throughput:.0f} img/s | peak {peak_gb:.1f} GB")
        if mfu_val is not None:
            print(f"MFU:    {mfu_val * 100:.1f}%  (bf16 dense peak on {gpu_name})")

        summary = {
            "run_name": run_name,
            "backbone": args.backbone,
            "objective": args.objective,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "params": n_params,
            "forward_flops_per_sample": flops_per_sample,
            "global_batch_size": global_batch,
            "world_size": world_size,
            "final_avg_loss": avg_loss,
            "final_grad_norm": avg_gnorm,
            "best_loss": best_loss,
            "step_time_s": step_time,
            "throughput_img_s": throughput,
            "peak_memory_gb": peak_gb,
            "mfu": mfu_val,
            "gpu_name": gpu_name,
            "git_commit": _git_commit(),
            "config": {k: v for k, v in vars(args).items()},
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"{run_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results written: {out_path}")

        if wb:
            wb.summary.update(summary)
            wb.finish()

    if is_distributed:
        dist.destroy_process_group()

    if is_main:
        print("Training done.")
