# From DDPM to Rectified Flow: a Controlled Study of Backbone and Objective

**Yi Hou** — draft; results pending (placeholders marked `TODO`)

## Abstract

> **TODO (write last).** Template: *We implement DDPM, DiT and rectified flow from scratch and
> compare them in a 2×2 controlled design (UNet/DiT × ε-prediction/rectified flow) on CelebA
> 64×64 under a single frozen training protocol. We report FID and the quality–NFE trade-off,
> plus training/sampling cost accounting. Changing the backbone from UNet to DiT ________,
> while switching the objective from ε-prediction to rectified flow ________, and at equal
> sample quality rectified flow needs ___× fewer function evaluations.*

## 1. Research questions

- **RQ1 · Architecture** — At a similar parameter scale, does replacing a UNet with a
  Transformer (DiT) backbone improve or degrade generation quality and computational
  efficiency? (cells A vs B, C vs D)
- **RQ2 · Objective** — How does a rectified-flow objective change the quality–NFE
  trade-off? (cells A vs C, B vs D)
- **RQ3 · Systems** — How much can training and inference throughput be improved by
  compilation, mixed precision and kernel-level fusion?

## 2. Background: one path, three objectives

All three methods pick an interpolation path between noise and data and regress a target
along it; they differ in the path, the target, and the solver.

| Method | Path `x_t` | Target | Sampler | NFE |
|---|---|---|---|---|
| DDPM | `√ᾱ_t·x₀ + √(1-ᾱ_t)·ε` | `ε` | ancestral (stochastic) | 1000 |
| DDIM | same as DDPM | `ε` | deterministic ODE | 10–50 |
| Rectified flow | `(1-t)·x₀ + t·ε` | `v = ε - x₀` | Euler ODE | 4–128 |

> **TODO:** Figure 1 — the two paths (curved diffusion vs straight flow) drawn side by side.

## 3. Implementation (all from scratch, no `diffusers`)

| | UNet (A, C) | DiT (B, D) |
|---|---|---|
| params | 174.1M | 129.6M |
| forward FLOPs/sample @64×64 | 67.8 GFLOPs | 43.6 GFLOPs |
| conditioning | sinusoidal + MLP | adaLN-Zero, same embedding width |
| attention | SDPA at the bottleneck | SDPA in all 12 blocks, 256 tokens (patch 4) |

*(Params and FLOPs are measured by `eval/cost.py` via `torch.utils.flop_counter`, so both
backbones are counted with the same tool. Note the UNet has both more parameters and more
FLOPs — the "similar scale" claim refers to parameters only.*

- **Objective details.** Rectified flow keeps `t ∈ [0,1]` for the path math but passes
  `t × 1000` to the model, because the sinusoidal embedding is calibrated for `[0, 1000]`
  (the DDPM range). Sampling integrates `x ← x - Δt·v_θ(x,t)` from `t=1` back to `t=0`.
- **ema.** All cells use EMA (decay 0.9999) and all reported FID uses EMA weights.
- **Shared training stack.** torchrun DDP, YAML configs, checkpoint resume, per-epoch EMA
  snapshots, cost accounting, optional W&B logging.

## 4. Experimental setup — TODO (freeze from `reports/exp_plan.md`)

Protocol table (data, epochs, batch, optimizer, schedule, EMA, seed), FID protocol
(reference stats built from the training tensor, Tier 1 = 10k / Tier 2 = 50k, 5×10k
variability), and the NFE grid.

## 5. Results

### 5.1 Training cost

> **TODO:** run `python eval/make_table.py` after the runs; paste the table here.

### 5.2 Generation quality

> **TODO:** FID table (Tier 2, 50k samples), plus qualitative sample grids from
> `samples/<run>_epoch300.png`.

### 5.3 Quality–NFE trade-off (RQ2)

> **TODO:** Figure — FID vs NFE for DDPM ancestral (1000), DDIM (10/20/50) and Euler
> (4/8/16/32/64/128). Caption must state that DDPM vs DDIM is a same-model sampler
> ablation while cross-cell comparisons also change the training objective.

### 5.4 Convergence

> **TODO:** FID vs training progress at the EMA snapshots (epochs 50…300).

### 5.5 Systems (RQ3)

> **TODO:** BF16 / fused AdamW / `torch.compile` (compile time and break-even reported
> separately) / Triton fused adaLN (kernel-level *and* model-level speedup) /
> CUDA-Graph sampling / FSDP-vs-DDP reverse benchmark. Include an FID column so speed
> gains cannot hide a quality regression.

## 6. Discussion and limitations

- Unconditional only (no CFG); CelebA 64×64 only; pixel space, no VAE.
- FID here is computed against reference statistics built from this repository's own
  preprocessing chain (InceptionV3 pool3, bilinear 64→299, antialiased). Absolute values are
  therefore **not** comparable with published numbers that use different resizing — only
  comparisons *within* this table are meaningful.
- Single seed per cell; FID variability is reported across 5×10k subsets rather than seeds.
- FLOPs counts cover matmul/conv work only; elementwise-heavy ops (GroupNorm, SiLU, adaLN
  modulation) are not counted, so "MFU" here is a matmul-utilisation proxy.

## 7. Appendix A — What didn't work

Format per entry: **Hypothesis → Setup → Observation → Diagnosis → Resolution → Lesson.**

> **TODO:** fill in from the development history
> - LR 5e-4 → training collapse → 2e-4 + gradient clipping
> - DataParallel × DataLoader deadlock → torchrun DDP
> - NCCL P2P hang on the 6×A6000 NUMA topology → `NCCL_P2P_DISABLE=1`
> - FSDP at 130M parameters (reverse benchmark)

## 8. Appendix B — Reproduction

See [`reports/runbook.md`](runbook.md): environment, data, smoke tests, launch commands and
evaluation commands; `results/*.json` is the raw archive behind every number above.
