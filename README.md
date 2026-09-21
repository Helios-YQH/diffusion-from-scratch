# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

*[中文版](README.zh-CN.md)*

A from-scratch diffusion implementation (no `diffusers`): a **controlled study** of two
backbones (UNet vs. diffusion transformer) and two training objectives (ε-prediction vs.
rectified flow) under one frozen protocol, plus measured trade-offs for samplers, mixed
precision and distributed strategies.

![Quality–NFE frontier](figures/fid_vs_nfe.png)

## Results

### 1. MNIST 2×2 (four cells, 200 epochs each, identical protocol and seed)

| Cell | Backbone + objective | Params | GFLOPs/sample | Best FID | NFE at that FID |
|---|---|---|---|---|---|
| A | UNet + ε | 11.7M | 2.43 | 2.79 | 1000 (ancestral) |
| B | DiT + ε | 16.5M | 1.08 | 33.18 | 1000 (ancestral) |
| C | UNet + flow | 11.7M | 2.43 | **0.88** | 64 (Euler) |
| D | DiT + flow | 16.5M | 1.08 | 4.99 | 64 (Euler) |

- Parameters are not compute: the transformer has *more* parameters and only 44% of the
  UNet's FLOPs, so this is not a compute-matched comparison — the report says so explicitly
- Rectified flow beats ε-prediction at every budget; the crossover sits near NFE 16

![The four cells' samples](figures/mnist_2x2_samples.png)

### 2. CelebA sampler study (one trained DDPM, only the sampler changes)

| Sampler | NFE | n | FID | Throughput | Spread over 5×10k subsets |
|---|---|---|---|---|---|
| DDIM | 10 | 50k | 30.03 | 194 /s | 30.90 ± 0.15 |
| DDIM | 20 | 50k | 17.79 | 97 /s | 18.72 ± 0.09 |
| DDIM | 50 | 50k | **11.45** | 39 /s | 12.41 ± 0.08 |
| ancestral | 1000 | 10k | 11.37 | 1.9 /s | — |

**50 steps match 1000 (within 0.8%) at 20× the throughput.** The spread column also shows a
methodological result: a 10k subset reads about one FID unit *high* for the same sampler, which
is exactly why the frozen protocol requires reporting the subset spread.

### 3. Systems (CelebA DiT, 160 images/GPU)

![Execution settings](figures/systems_settings.png)

| Setting | s/step | Peak GB | MFU | Relative |
|---|---|---|---|---|
| fp32 | 1.386 | 32.9 | 10.4% | 1.00× |
| **bf16** | **0.442** | **21.1** | 32.7% | **3.15×** |
| `torch.compile` | 1.415 | 32.9 | 10.4% | 1.00× |
| fp32 + DDP (4 GPUs) | 1.408 | 33.5 | 10.1% | — |
| fp32 + FSDP (4 GPUs) | 1.588 | 31.9 | 9.0% | **0.89× (a net loss)** |

Kernel-level share of an fp32 training step: matmul 67.6% / elementwise 15.3% / attention
11.1% / norm 2.3%.

- **Mixed precision is the only lever that moves** (3.15× step rate, 1.6× memory);
  `torch.compile` returns nothing
- **FSDP is a net loss at this scale**: the peak is activation-dominated (parameters,
  gradients and Adam state are ~2 GB of the 33.5 GB peak), so sharding saves little memory and
  costs communication. Reported as a deliberate reverse benchmark.

## One finding worth keeping

**The NFE reduction a deterministic sampler offers is conditional on score accuracy.** The
same DDIM implementation improves the UNet (FID 22.06 → 4.32 as NFE grows 10 → 50) and
*degrades* the transformer (49 → 672), while 1000-step ancestral sampling of the same weights
gives 33.2. Four checks localise it (not the EMA weights, not the model, not the sampler code)
and an η sweep settles it: raising the noise-injection level from 0 to 1 moves the saturated
pixel fraction from 0.48 to 0.79 and turns blobs back into digits. The deterministic path
integrates score error coherently; the sampler's noise injection bounds the drift.

Mixed precision shows the same signature: in a one-cell retrain, bf16 matched the fp32
training loss to three digits (0.0194 vs. 0.0193) while doubling the 1000-step FID
(5.56 vs. 2.79) and leaving the 50-step FID unchanged.

> A paper reporting only the UNet numbers would ship a sampler recommendation that inverts on
> the other backbone.

## Layout

```
├── models/             # unet.py (residual U-Net), dit.py (adaLN-Zero, parameterised)
├── diffusion/          # ddpm.py (ε + ancestral/DDIM), flow.py (rectified flow + Euler)
├── eval/               # fid.py (self-built reference stats, NFE sweeps, subset spread),
│                       # cost.py (measured FLOPs/MFU), systems.py (precision, profile,
│                       # DDP/FSDP), eta_sweep.py + eps_error_by_t.py (the diagnostics),
│                       # make_table.py (results → markdown)
├── configs/            # four MNIST cells + four CelebA cells
├── scripts/            # cheap_package.sh (5 stages, one command), tier2.sh, status.sh
├── tests/              # 14 CPU-only invariants
├── reports/            # technical report, runbook, figure scripts
├── results/            # every experiment's raw archive (JSON)
└── figures/            # figures used by the README and the report
```

## Reproduction

```bash
# environment (uv)
uv sync && uv run python tests/test_sanity.py     # 14 CPU invariants, ~1 min

# the whole package: MNIST 2x2 -> both FID sweeps -> systems -> figures
GPUS=0,1,2,3 nohup bash scripts/cheap_package.sh &

# progress / re-running a single point
bash scripts/status.sh
```

The datasets (`celeba.zip`, `mnist.pkl.gz`) and trained checkpoints are not in the repo; see
[reports/runbook.md](reports/runbook.md) for where to place them, the exact commands, the time
budget and the troubleshooting table.

## Documentation

| | |
|---|---|
| Technical report (11 pages, NeurIPS format) | [reports/tech_report.pdf](reports/tech_report.pdf) |
| Runbook (machine-agnostic) | [reports/runbook.md](reports/runbook.md) |
| Figure scripts (data → PDF) | [reports/make_figures.py](reports/make_figures.py) |
| Raw experiment archive | [results/](results/) |
