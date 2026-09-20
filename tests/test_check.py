"""Quick smoke test — model shapes and data loading.

Run from repo root:  python tests/test_check.py
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.unet import UNet
from diffusion.ddpm import DDPM
from train import load_mnist, preprocess_celeba, extract_celeba

print("1. MNIST backward-compat")
model = UNet(img_channels=1, base_channels=64, num_downs=3)
x = torch.randn(2, 1, 28, 28)
t = torch.randint(0, 1000, (2,))
out = model(x, t)
print(f"   {x.shape} -> {out.shape}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
assert out.shape == (2, 1, 28, 28)

diffusion = DDPM(model, T=100, device="cpu", img_channels=1, img_size=28)
loss = diffusion.training_loss(x)
print(f"   loss={loss.item():.4f}")
assert diffusion.sample(4).shape == (4, 1, 28, 28)

x_train = load_mnist()
assert x_train.shape == (50000, 1, 28, 28)
print("   OK")

print("\n2. CelebA model (3ch, 64x64, base_ch=64, num_downs=4)")
model_c = UNet(img_channels=3, base_channels=64, num_downs=4)
x_c = torch.randn(2, 3, 64, 64)
out_c = model_c(x_c, t)
n_c = sum(p.numel() for p in model_c.parameters()) / 1e6
print(f"   {x_c.shape} -> {out_c.shape}  params={n_c:.1f}M")
assert out_c.shape == (2, 3, 64, 64)

diffusion_c = DDPM(model_c, T=100, device="cpu", img_channels=3, img_size=64)
loss_c = diffusion_c.training_loss(x_c)
print(f"   loss={loss_c.item():.4f}")
assert diffusion_c.sample(4).shape == (4, 3, 64, 64)
print("   OK")

print("\n3. num_downs configurability")
for nd in [2, 3, 4]:
    sz = 2 ** (nd + 1)
    m = UNet(img_channels=1, base_channels=32, num_downs=nd)
    inp = torch.randn(2, 1, sz, sz)
    o = m(inp, t)
    p = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"   num_downs={nd}  {inp.shape} -> {o.shape}  params={p:.1f}M")
    assert inp.shape == o.shape

print("\n4. CelebA preprocess cache (isolated test)")
import zipfile, io, numpy as np
from PIL import Image
tmpdir = tempfile.mkdtemp()
zip_path = os.path.join(tmpdir, "test.zip")
img_dir = os.path.join(tmpdir, "images")
os.makedirs(img_dir)
# Create 2 fake JPEGs
for i in range(2):
    arr = np.random.randint(0, 255, (178, 218, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(os.path.join(img_dir, f"{i:06d}.jpg"))

import zipfile
with zipfile.ZipFile(zip_path, "w") as zf:
    for f in os.listdir(img_dir):
        zf.write(os.path.join(img_dir, f), f"img_align_celeba/{f}")

# Patch paths to use temp dir, test the pipeline
orig_cp, orig_cd = __import__("train").CELEBA_PATH, __import__("train").CELEBA_DIR
import train as tr
tr.CELEBA_PATH = zip_path
tr.CELEBA_DIR = os.path.join(tmpdir, "celeba")

assert not os.path.exists(tr.CELEBA_DIR)
tr.extract_celeba()
assert os.path.exists(os.path.join(tr.CELEBA_DIR, ".extracted"))

# Test preprocess cache
cache = os.path.join(tmpdir, "celeba_32.pt")
orig_cache_path = tr.preprocess_celeba.__code__
data = tr.preprocess_celeba(image_size=64)
print(f"   preprocessed: {data.shape}")
assert data.shape[1:] == (3, 64, 64)
assert data.shape[0] == 2

# Test that cache is reused (no reprocessing)
data2 = tr.preprocess_celeba(image_size=32)
assert torch.equal(data, data2)
print("   cache reuse OK")

shutil.rmtree(tmpdir)
print("   OK")

print("\nAll checks passed!")
