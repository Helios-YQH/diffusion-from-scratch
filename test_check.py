from model import UNet
from diffusion import DDPM
from train import load_mnist
import torch

print("1. Data loading...")
x_train = load_mnist()
print(f"   MNIST train: {x_train.shape}  range=[{x_train.min():.3f}, {x_train.max():.3f}]")

print("2. Model forward pass...")
model = UNet()
diffusion = DDPM(model, T=1000, device="cpu")
x = torch.randn(2, 1, 28, 28)
t = torch.randint(0, 1000, (2,))
out = model(x, t)
print(f"   Output shape: {out.shape}")

print("3. Training loss...")
loss = diffusion.training_loss(x)
print(f"   Loss: {loss.item():.6f}")

print("4. Sampling...")
samples = diffusion.sample(4)
print(f"   Samples: {samples.shape}  range=[{samples.min():.3f}, {samples.max():.3f}]")

n = sum(p.numel() for p in model.parameters())
print(f"5. Model params: {n/1e6:.1f}M")
print("All checks passed!")
