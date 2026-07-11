"""Quick smoke test — model + diffusion, no zip access."""
import torch, os
import matplotlib
matplotlib.use("Agg")

from model import UNet
from diffusion import DDPM
from train import load_mnist

print("1. MNIST backward-compat")
model = UNet(img_channels=1, base_channels=64, num_downs=3)
x = torch.randn(2, 1, 28, 28)
t = torch.randint(0, 1000, (2,))
out = model(x, t)
print(f"   {x.shape} -> {out.shape}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
assert out.shape == (2, 1, 28, 28)

diffusion = DDPM(model, T=100, device="cpu", img_channels=1, img_size=28)
loss = diffusion.training_loss(x)
samples = diffusion.sample(4)
print(f"   loss={loss.item():.4f}  samples={samples.shape}")
assert samples.shape == (4, 1, 28, 28)

x_train = load_mnist()
print(f"   data={x_train.shape}")
assert x_train.shape == (50000, 1, 28, 28)

print("\n2. CelebA model (3ch, 64x64, base_ch=128)")
model_c = UNet(img_channels=3, base_channels=128, num_downs=3)
x_c = torch.randn(2, 3, 64, 64)
out_c = model_c(x_c, t)
n_c = sum(p.numel() for p in model_c.parameters()) / 1e6
print(f"   {x_c.shape} -> {out_c.shape}  params={n_c:.1f}M")
assert out_c.shape == (2, 3, 64, 64)

diffusion_c = DDPM(model_c, T=100, device="cpu", img_channels=3, img_size=64)
loss_c = diffusion_c.training_loss(x_c)
samples_c = diffusion_c.sample(4)
print(f"   loss={loss_c.item():.4f}  samples={samples_c.shape}")
assert samples_c.shape == (4, 3, 64, 64)

print("\n3. num_downs configurability")
for nd in [2, 3, 4]:
    sz = 2 ** (nd + 1)
    m = UNet(img_channels=1, base_channels=32, num_downs=nd)
    inp = torch.randn(2, 1, sz, sz)
    o = m(inp, t)
    p = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"   num_downs={nd}  {inp.shape} -> {o.shape}  params={p:.1f}M")
    assert inp.shape == o.shape

print("\n4. CelebA dataset (zip stream)")
from train import CelebADataset
from torch.utils.data import DataLoader
ds = CelebADataset(os.path.join("data", "celeba.zip"), image_size=64)
print(f"   {len(ds)} images")
img = ds[0]
print(f"   sample: {img.shape}  range=[{img.min():.3f}, {img.max():.3f}]")
assert img.shape == (3, 64, 64)

img2 = ds[100]
assert not torch.equal(img, img2)

loader = DataLoader(ds, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"   batch: {batch.shape}")
assert batch.shape == (4, 3, 64, 64)

print("\nAll checks passed!")
