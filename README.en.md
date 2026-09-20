# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

A from-scratch (no `diffusers`) diffusion implementation: a **controlled study** of three
backbones / training objectives — UNet + DDPM, DiT, and Rectified Flow — with FID,
NFE-efficiency curves, and modern ML-systems analysis.

Status: the DDPM baseline is complete (MNIST 28×28, CelebA 64×64, 5-GPU DDP);
DiT + Rectified Flow are under development on the `dit-rectified-flow` branch
(experiment design in [reports/exp_plan.md](reports/exp_plan.md)).

| MNIST 28×28 (UNet 12M, 170 epochs) | CelebA 64×64 (UNet 174M, 300 epochs) |
|---|---|
| ![MNIST](figures/mnist_epoch170.png) | ![CelebA](figures/celeba_epoch300.png) |

## Highlights

- **Core algorithms from scratch**: DDPM forward/reverse processes, UNet (sinusoidal time
  embedding / residual blocks / mid-block attention), DiT (adaLN-Zero), and Rectified Flow
  (linear-interpolation path + Euler ODE sampling)
- **Training systems**: torchrun DDP (up to 5 GPUs), YAML configs, checkpoint resume, EMA,
  per-epoch snapshots
- **Evaluation**: FID (reference stats consistent with the training preprocessing),
  FID-vs-NFE curves, sampling throughput
- **ML systems**: cost accounting (params / FLOPs / step time / throughput / MFU),
  torch.compile, Triton fused AdaLN, CUDA-Graph sampling (planned)

## Setup

Python 3.10+, PyTorch 2.x; see `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# MNIST (single GPU)
python main.py train --config configs/mnist.yml

# CelebA (single GPU)
python main.py train --config configs/celeba.yml

# CelebA (5-GPU DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 main.py train --config configs/celeba.yml

# Sample from a checkpoint
python main.py sample --dataset celeba --n 64

# Denoising animation (with GIF export)
python demo.py --dataset celeba
```

> On multi-GPU servers, set `NCCL_P2P_DISABLE=1` if training hangs on communication.
> Smoke test: `python tests/test_check.py`

## Project Structure

```
├── models/            # Backbones
│   └── unet.py        # UNet (DDPM baseline)
├── diffusion/         # Diffusion processes
│   └── ddpm.py        # DDPM: forward noising / training loss / ancestral sampling
├── configs/           # YAML training configs (celeba / mnist)
├── tests/             # Smoke tests
├── reports/           # Experiment plan and technical report
├── figures/           # Result figures
├── train.py           # Training loop, data pipeline, checkpoints
├── main.py            # CLI entry point (train / sample)
├── demo.py            # Denoising visualization
└── requirements.txt
```

## Implementation Notes

- `diffusion/ddpm.py`: `forward_diffusion` follows `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`;
  `training_loss` is the MSE between predicted and true noise; `sample` runs ancestral
  sampling (optionally returning the full trajectory).
- `models/unet.py`: sinusoidal time embeddings are injected into every residual block via an
  MLP; SDPA self-attention sits at the bottleneck; skip connections preserve multi-scale
  information across down/up stages.
- `train.py`: data pipeline (MNIST pkl / CelebA zip → preprocessing cache), LR warmup +
  cosine decay, gradient clipping, `latest`/`best` checkpoints with resume support
  (backward compatible with old-format checkpoints).
- `demo.py`: captures the reverse process frame by frame; interactive window with
  arrow-key stepping / space-to-pause, plus GIF export.
