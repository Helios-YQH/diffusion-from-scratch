import gzip
import pickle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join("data", "mnist", "mnist.pkl.gz")
CHECKPOINT_DIR = "checkpoints"


def load_mnist(normalize=True):
    with gzip.open(DATA_PATH, "rb") as f:
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
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 1.5, nrow * 1.5))
    for i in range(n):
        r, c = i // nrow, i % nrow
        axes[r, c].imshow(images[i, 0], cmap="gray")
        axes[r, c].axis("off")
    for i in range(n, nrow * nrow):
        r, c = i // nrow, i % nrow
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def train(diffusion, epochs=50, batch_size=128, lr=1e-3, save_interval=10,
          use_data_parallel=False):
    x_train = load_mnist()
    dataset = TensorDataset(x_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=use_data_parallel)

    opt = torch.optim.Adam(diffusion.model.parameters(), lr=lr)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    model_for_save = diffusion.model.module if use_data_parallel else diffusion.model

    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            x0 = batch[0].to(diffusion.device)
            loss = diffusion.training_loss(x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_epoch{epoch}.pt")
            torch.save(model_for_save.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            samples = diffusion.sample(16)
            save_sample_grid(samples, os.path.join("samples", f"epoch{epoch}.png"))

    print("Training done.")
