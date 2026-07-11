import gzip
import pickle
import os
import zipfile
import warnings
import glob
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

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

    # Write marker to skip extraction next time
    with open(done_marker, "w") as f:
        f.write("done")
    print("Extraction complete.")


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


class CelebADataset(Dataset):
    """Reads pre-extracted CelebA images from disk."""

    def __init__(self, image_dir, image_size=64, normalize=True):
        self.files = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"),
                                      recursive=True))
        if not self.files:
            raise RuntimeError(
                f"No .jpg files found in {image_dir}. "
                f"Did you run extract_celeba() first?")
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        if self.normalize:
            img = img * 2.0 - 1.0  # [0,1] -> [-1,1]
        return torch.from_numpy(img)


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


def train(diffusion, epochs=50, batch_size=128, lr=1e-3, save_interval=10,
          use_data_parallel=False, dataset_type="mnist", image_size=64):
    if dataset_type == "celeba":
        extract_celeba()
        dataset = CelebADataset(CELEBA_DIR, image_size=image_size)
        sample_interval = 50
        num_workers = 4
        ckpt_name = "ddpm_celeba_best.pt"
        save_best_only = True
    else:
        x_train = load_mnist()
        dataset = TensorDataset(x_train)
        sample_interval = save_interval
        num_workers = 0
        ckpt_name = None
        save_best_only = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=use_data_parallel, num_workers=num_workers)

    opt = torch.optim.Adam(diffusion.model.parameters(), lr=lr)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    model_for_save = diffusion.model.module if use_data_parallel else diffusion.model
    best_loss = float("inf")

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=80)
        for batch in pbar:
            if dataset_type == "celeba":
                x0 = batch.to(diffusion.device)
            else:
                x0 = batch[0].to(diffusion.device)
            loss = diffusion.training_loss(x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.6f}")

        if dataset_type == "celeba":
            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                torch.save(model_for_save.state_dict(), ckpt_path)
                print(f"  Best checkpoint saved: {ckpt_path}  (loss={best_loss:.6f})")

            # Sample very infrequently
            if epoch % sample_interval == 0 or epoch == epochs:
                samples = diffusion.sample(16)
                tag = f"celeba_epoch{epoch}"
                save_sample_grid(samples, os.path.join("samples", f"{tag}.png"))
        else:
            # MNIST: periodic checkpoints and samples
            if epoch % save_interval == 0 or epoch == epochs:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_epoch{epoch}.pt")
                torch.save(model_for_save.state_dict(), ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

                samples = diffusion.sample(16)
                save_sample_grid(samples, os.path.join("samples", f"epoch{epoch}.png"))

    print("Training done.")
