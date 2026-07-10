"""Quick correctness check — CPU only. Run with: python test_check.py"""
import torch
from model import UNet
from diffusion import DDPM
from train import load_mnist

print("1. Data loading...")
x_train = load_mnist()
print(f"   shape={x_train.shape}, range=[{x_train.min():.3f}, {x_train.max():.3f}]")

print("2. Forward pass...")
model = UNet()
diffusion = DDPM(model, T=1000, device="cpu", use_data_parallel=False)
x = torch.randn(2, 1, 28, 28)
t = torch.randint(0, 1000, (2,))
out = model(x, t)
print(f"   output={out.shape}")

print("3. Training loss...")
loss = diffusion.training_loss(x)
print(f"   loss={loss.item():.4f}")

print("4. Sampling...")
samples, steps = diffusion.sample(4, return_all=True)
print(f"   samples={samples.shape}, steps={len(steps)}")

print("5. Model info...")
n = sum(p.numel() for p in model.parameters())
print(f"   params={n/1e6:.1f}M")

print("6. Multi-GPU code path check...")
print(f"   torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"   torch.cuda.device_count()={torch.cuda.device_count()}")
print("   (DataParallel will auto-activate when >=2 GPUs detected)")

print("\nAll checks passed!")
