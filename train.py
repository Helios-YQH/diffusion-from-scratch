import gzip
import pickle
import os
import zipfile
import warnings
import glob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import UNet
from diffusion import DDPM

DATA_PATH = os.path.join("data", "mnist", "mnist.pkl.gz")
CELEBA_PATH = os.path.join("data", "celeba.zip")
CELEBA_DIR = os.path.join("data", "celeba")
CHECKPOINT_DIR = "checkpoints"


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
    """Resize + normalize all CelebA images into a single .pt file."""
    cache_path = os.path.join("data", f"celeba_{image_size}.pt")
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
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        arr = arr * 2.0 - 1.0  # [-1, 1]
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))

    data = torch.stack(tensors)
    torch.save(data, cache_path)
    print(f"Saved {cache_path}  ({data.shape})")
    return data


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


def train(args):
    """Training entry point. Detects torchrun env vars for DDP."""

    # Detect whether torchrun launched us
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = 'LOCAL_RANK' in os.environ and world_size > 1
    is_main = rank == 0

    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'

        if is_main:
            print(f"DDP mode")
            print(f"World size: {world_size}")
            print(f"Per-GPU batch size: {args.batch_size}")
            print(f"Global batch size: {args.batch_size * world_size}")

        # Each rank announces its binding once (via a barrier so output is ordered)
        print(f"Rank {rank} -> cuda:{local_rank}")
        if world_size > 1:
            dist.barrier()
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device != 'cpu':
            torch.cuda.set_device(device)
        print(f"Single GPU: {device}")
        if torch.cuda.is_available():
            print(f"Visible GPUs: {torch.cuda.device_count()}")

    # ── Dataset ─────────────────────────────────────────────────────
    if args.dataset == 'celeba':
        img_channels, img_size = 3, args.image_size or 64
        base_channels, num_downs = args.base_channels or 128, 4
        sample_interval = 50
        num_workers = 4
        ckpt_name = 'ddpm_celeba_best.pt'

        # Only rank 0 does extraction/preprocessing to avoid races
        # on the .extracted marker and zip extraction.
        if is_main or not is_distributed:
            extract_celeba()
        if is_distributed:
            dist.barrier()
        data_tensor = preprocess_celeba(image_size=img_size)
        dataset = TensorDataset(data_tensor)
    else:
        img_channels, img_size = 1, args.image_size or 28
        base_channels, num_downs = args.base_channels or 64, 3
        sample_interval = args.save_interval
        num_workers = 0
        ckpt_name = None
        x_train = load_mnist()
        dataset = TensorDataset(x_train)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                        pin_memory=True, num_workers=num_workers)

    # ── Model ───────────────────────────────────────────────────────
    model = UNet(img_channels=img_channels, base_channels=base_channels,
                 time_dim=256, num_downs=num_downs).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    diffusion = DDPM(model, T=args.timesteps, device=device,
                     img_channels=img_channels, img_size=img_size)

    # ── Optimizer ───────────────────────────────────────────────────
    opt = torch.optim.Adam(diffusion.model.parameters(), lr=args.lr)

    # ── LR scheduler: linear warmup → cosine decay ─────────────────
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
    best_loss = float('inf')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if is_main:
        os.makedirs('samples', exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}',
                    ncols=80, disable=not is_main)

        for batch in pbar:
            x0 = batch[0].to(device)
            loss = diffusion.training_loss(x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            if is_main:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / max(num_batches, 1)

        # All-reduce loss across ranks for accurate reporting
        if is_distributed:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        if is_main:
            print(f'Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}')

            if args.dataset == 'celeba':
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                    torch.save(model_for_save.state_dict(), ckpt_path)
                    print(f'  Best checkpoint saved: {ckpt_path}  (loss={best_loss:.6f})')

                if epoch % sample_interval == 0 or epoch == args.epochs:
                    samples = diffusion.sample(16)
                    tag = f'celeba_epoch{epoch}'
                    save_sample_grid(samples, os.path.join('samples', f'{tag}.png'))
            else:
                if epoch % args.save_interval == 0 or epoch == args.epochs:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f'ddpm_epoch{epoch}.pt')
                    torch.save(model_for_save.state_dict(), ckpt_path)
                    print(f'  Checkpoint saved: {ckpt_path}')

                    samples = diffusion.sample(16)
                    save_sample_grid(samples, os.path.join('samples', f'epoch{epoch}.png'))

    if is_distributed:
        dist.destroy_process_group()

    if is_main:
        print('Training done.')
