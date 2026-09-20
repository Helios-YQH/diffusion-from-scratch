"""Sanity tests — CPU-runnable invariants for the diffusion code.

These check math and shapes only; nothing here depends on a GPU.

Run directly:   python tests/test_sanity.py
Or with pytest: uv run pytest tests/test_sanity.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from diffusion.ddpm import DDPM
from diffusion.flow import RectifiedFlow
from models.dit import DiT
from models.unet import UNet
from train import EMA

SEED = 0


class _ZeroModel(torch.nn.Module):
    """Predicts zeros regardless of input — used to check loss invariants."""

    def forward(self, x, t):
        return torch.zeros_like(x)


# ── DDPM ──────────────────────────────────────────────────────────────

def test_ddpm_forward_statistics():
    """x_t = sqrt(ab) x0 + sqrt(1-ab) eps: check mean and std of x_t."""
    n = 4096
    x0 = torch.ones(n, 1, 4, 4)
    ddpm = DDPM(_ZeroModel(), T=1000, device="cpu", img_channels=1, img_size=4)
    for t in (0, 100, 999):
        xt, _ = ddpm.forward_diffusion(x0, torch.full((n,), t))
        ab = ddpm.alpha_bars[t]
        assert torch.allclose(xt.mean(), ab.sqrt(), atol=0.05), f"t={t}"
        assert abs(xt.std().item() - (1 - ab).sqrt().item()) < 0.05, f"t={t}"


def test_ddpm_loss_of_zero_predictor_is_one():
    """MSE against eps is ~E[eps^2] = 1 for a zero predictor."""
    x0 = torch.randn(2000, 1, 4, 4)
    ddpm = DDPM(_ZeroModel(), T=1000, device="cpu", img_channels=1, img_size=4)
    assert abs(ddpm.training_loss(x0).item() - 1.0) < 0.05


def test_ddpm_sampling_shape_and_finiteness():
    ddpm = DDPM(_ZeroModel(), T=20, device="cpu", img_channels=1, img_size=8)
    s = ddpm.sample(3)
    assert s.shape == (3, 1, 8, 8)
    assert torch.isfinite(s).all()


# ── Rectified flow ────────────────────────────────────────────────────

def test_rf_path_endpoints():
    """t=0 must return x0 exactly; t=1 must return pure noise; v* = eps - x0."""
    rf = RectifiedFlow(_ZeroModel(), device="cpu", img_channels=1, img_size=4)
    x0 = torch.randn(64, 1, 4, 4)
    xt0, _ = rf.forward_path(x0, torch.zeros(64))
    assert torch.equal(xt0, x0)
    xt1, v1 = rf.forward_path(x0, torch.ones(64))
    assert torch.allclose(v1, xt1 - x0)
    assert (xt1 - x0).abs().mean().item() > 0.5


def test_rf_loss_of_zero_predictor_is_two():
    """v = eps - x0 has variance 2, so a zero predictor gives MSE ~ 2."""
    x0 = torch.randn(2000, 1, 4, 4)
    rf = RectifiedFlow(_ZeroModel(), device="cpu", img_channels=1, img_size=4)
    assert abs(rf.training_loss(x0).item() - 2.0) < 0.1


def test_rf_sampling_shape_and_finiteness():
    rf = RectifiedFlow(_ZeroModel(), device="cpu", img_channels=1, img_size=8)
    for nfe in (1, 4, 8):
        s = rf.sample(3, num_steps=nfe)
        assert s.shape == (3, 1, 8, 8), f"nfe={nfe}"
        assert torch.isfinite(s).all(), f"nfe={nfe}"


# ── Models ────────────────────────────────────────────────────────────

def test_dit_zero_init_and_shapes():
    """adaLN-Zero + zeroed output layer: the DiT starts predicting exactly 0."""
    torch.manual_seed(SEED)
    m = DiT(img_size=32, patch_size=4, img_channels=3, hidden_size=64,
            depth=2, num_heads=4)
    x = torch.randn(2, 3, 32, 32)
    out = m(x, torch.randint(0, 1000, (2,)))
    assert out.shape == x.shape
    assert out.abs().max().item() == 0.0


def test_dit_survives_first_optimizer_steps():
    """The zero-init model must wake up: gradients flow and loss decreases."""
    torch.manual_seed(SEED)
    m = DiT(img_size=16, patch_size=4, img_channels=1, hidden_size=64,
            depth=2, num_heads=4)
    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    x = torch.randn(8, 1, 16, 16)
    target = torch.randn(8, 1, 16, 16)
    losses = []
    for _ in range(6):
        t = torch.randint(0, 1000, (8,))
        loss = (m(x, t) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_unet_shapes():
    m = UNet(img_channels=1, base_channels=8, num_downs=3)
    x = torch.randn(2, 1, 28, 28)
    assert m(x, torch.randint(0, 1000, (2,))).shape == x.shape


# ── Training reduces loss (wiring check) ──────────────────────────────

def test_training_reduces_loss():
    x = torch.randn(16, 1, 16, 16)
    for name, make in (
        ("ddpm", lambda m: DDPM(m, T=100, device="cpu",
                                img_channels=1, img_size=16)),
        ("rf", lambda m: RectifiedFlow(m, device="cpu",
                                       img_channels=1, img_size=16)),
    ):
        torch.manual_seed(SEED)
        model = UNet(img_channels=1, base_channels=8, num_downs=2)
        obj = make(model)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(40):
            loss = obj.training_loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        assert last < first, f"{name}: {first:.4f} -> {last:.4f}"


# ── EMA ───────────────────────────────────────────────────────────────

def test_ema_decay_zero_tracks_model():
    lin = torch.nn.Linear(4, 4)
    ema = EMA(lin, decay=0.0)
    with torch.no_grad():
        lin.weight.add_(1.0)
    ema.update(lin)
    assert torch.allclose(ema.shadow["weight"], lin.weight)


def test_ema_decay_one_is_frozen():
    lin = torch.nn.Linear(4, 4)
    ema = EMA(lin, decay=1.0)
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        lin.weight.mul_(0.5)
    ema.update(lin)
    assert torch.allclose(ema.shadow["weight"], before)


def test_ema_applied_to_restores():
    lin = torch.nn.Linear(4, 4)
    ema = EMA(lin, decay=0.5)
    with torch.no_grad():
        lin.weight.add_(2.0)
    ema.update(lin)
    raw = lin.weight.detach().clone()
    with ema.applied_to(lin):
        assert not torch.allclose(lin.weight, raw)
    assert torch.allclose(lin.weight, raw)


# ── Data pipeline ─────────────────────────────────────────────────────

def test_uint8_to_model_input():
    """The uint8 cache must map back to exactly the float [-1, 1] range."""
    from train import to_model_input

    u8 = torch.tensor([[[[0]], [[128]], [[255]]]], dtype=torch.uint8)
    x = to_model_input(u8, "cpu")
    assert x.dtype == torch.float32
    assert torch.allclose(x.flatten(),
                          torch.tensor([-1.0, 128 / 127.5 - 1.0, 1.0]),
                          atol=1e-6)

    f = torch.randn(2, 3, 4, 4)
    assert torch.equal(to_model_input(f, "cpu"), f)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
