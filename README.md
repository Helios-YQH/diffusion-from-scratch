# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

*[中文版](README.zh-CN.md)*

Diffusion models implemented from scratch, without `diffusers`: UNet and DiT backbones,
ε-prediction and rectified flow, DDPM/DDIM/Euler samplers, training and evaluation. The two
backbones and two objectives are compared in a 2×2 design on MNIST, and a trained CelebA UNet
is used to measure what sampler choice costs and buys. A technical report with the full
protocol and the negative results is in `reports/`.

![FID against NFE for the four MNIST cells and the CelebA sampler study](figures/fid_vs_nfe.png)

## Results

### MNIST 2×2

Four models, 200 epochs each, one protocol (same optimizer, schedule, batch size, seed).

| Cell | Backbone + objective | Params | GFLOPs/sample | Best FID | NFE |
|---|---|---|---|---|---|
| A | UNet + ε | 11.7M | 2.43 | 2.79 | 1000 ancestral |
| B | DiT + ε | 16.5M | 1.08 | 33.18 | 1000 ancestral |
| C | UNet + flow | 11.7M | 2.43 | **0.88** | 64 Euler |
| D | DiT + flow | 16.5M | 1.08 | 4.99 | 64 Euler |

The transformer has more parameters and 44% of the UNet's FLOPs, so the comparison is not
compute-matched; the FLOPs column is there to make that visible rather than to hide it.
Rectified flow beats ε-prediction at every budget, with the crossover near NFE 16.

![Samples from each of the four cells](figures/mnist_2x2_samples.png)

### CelebA samplers

One trained model (UNet, 174M parameters, 300 epochs) sampled four ways. Nothing is retrained.

| Sampler | NFE | n | FID | Throughput | Spread over 5×10k subsets |
|---|---|---|---|---|---|
| DDIM | 10 | 50k | 30.03 | 194 /s | 30.90 ± 0.15 |
| DDIM | 20 | 50k | 17.79 | 97 /s | 18.72 ± 0.09 |
| DDIM | 50 | 50k | **11.45** | 39 /s | 12.41 ± 0.08 |
| ancestral | 1000 | 10k | 11.37 | 1.9 /s | — |

DDIM at 50 steps lands within 0.8% of the 1000-step ancestral FID and runs 20× faster. The
spread column shows that a 10k subset reads about one FID higher than the full 50k for the same
sampler, which is why the protocol reports it.

### Systems

CelebA DiT, batch 160 per GPU, 30 steps after warmup.

![Step time by execution setting](figures/systems_settings.png)

| Setting | s/step | Peak GB | MFU | Relative |
|---|---|---|---|---|
| fp32 | 1.386 | 32.9 | 10.4% | 1.00× |
| bf16 | 0.442 | 21.1 | 32.7% | 3.15× |
| `torch.compile` | 1.415 | 32.9 | 10.4% | 1.00× |
| fp32 + DDP, 4 GPUs | 1.408 | 33.5 | 10.1% | — |
| fp32 + FSDP, 4 GPUs | 1.588 | 31.9 | 9.0% | 0.89× vs DDP |

Kernel time in an fp32 step: matmul 67.6%, elementwise 15.3%, attention 11.1%, norm 2.3%.

bf16 is the only setting that changes the step time; `torch.compile` returns the same time to
three digits. FSDP saves 4.8% of peak memory for 12.8% more time per step, because the peak is
set by activations: parameters, gradients and Adam state together are about 2 GB of the 33.5 GB.
The useful output of that measurement is the criterion, not the verdict: sharding pays when
states rather than activations set the ceiling.

## A sampler that inverts

DDIM improves the UNet as the budget grows (FID 22.06 at 10 evaluations, 4.32 at 50) and
degrades the DiT (49.14, 672.39), while ancestral sampling of the same DiT weights gives 33.18.
Four controls rule out the EMA weights (raw weights fail the same way), the model (ancestral
sampling works) and the sampler implementation (it improves the UNet). Raising DDIM's η from 0
to 1 at a fixed 50 evaluations moves the fraction of saturated pixels from 0.48 to 0.79 and
turns saturated blobs back into digits. The deterministic update accumulates the model's score
error in one direction; each ancestral step re-injects noise and bounds the drift.

bf16 training shows the same signature: a one-cell retrain matched the fp32 training loss to
three digits (0.0194 against 0.0193) while doubling the 1000-step FID (5.56 against 2.79), and
left the 50-step FID unchanged.

## Layout

```
├── models/             # unet.py, dit.py (adaLN-Zero)
├── diffusion/          # ddpm.py (ε, ancestral + DDIM), flow.py (rectified flow, Euler)
├── eval/               # fid.py, cost.py, systems.py, eta_sweep.py, eps_error_by_t.py,
│                       # make_table.py
├── configs/            # four MNIST cells, four CelebA cells
├── scripts/            # cheap_package.sh (five stages), tier2.sh, status.sh
├── tests/              # 14 CPU-only invariants
├── reports/            # technical report, runbook, figure scripts
├── results/            # raw archive, one JSON per run
└── figures/
```

## Reproduction

```bash
uv sync && uv run python tests/test_sanity.py     # 14 CPU invariants, ~1 min

# MNIST 2x2 -> both FID sweeps -> systems -> figures
GPUS=0,1,2,3 nohup bash scripts/cheap_package.sh &

bash scripts/status.sh                            # progress
```

Datasets and checkpoints are not in the repo; see [reports/runbook.md](reports/runbook.md).

## Documentation

| | |
|---|---|
| Technical report | [reports/tech_report.pdf](reports/tech_report.pdf) |
| Runbook | [reports/runbook.md](reports/runbook.md) |
| Figure scripts | [reports/make_figures.py](reports/make_figures.py) |
| Raw results | [results/](results/) |
