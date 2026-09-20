# Experiment Plan — frozen 2026-09-20

Controlled comparison of **backbone architecture** (UNet vs DiT) and **training objective**
(ε-prediction vs rectified flow) on CelebA 64×64, plus a systems layer
(training / sampling throughput, kernel fusion).

## Research questions

- **RQ1 · Architecture** — At a similar parameter scale, does replacing a UNet with a
  Transformer (DiT) backbone improve or degrade generation quality and computational
  efficiency?
- **RQ2 · Objective** — How does a rectified-flow objective change the quality–NFE
  trade-off?
- **RQ3 · Systems** — How much can training and inference throughput be improved by
  compilation, mixed precision, and kernel-level fusion?

## 2×2 controlled design

| ID | Backbone | Objective | Samplers |
|---|---|---|---|
| A | UNet (174M) | ε-prediction | DDPM ancestral 1000, DDIM 10 / 20 / 50 |
| B | DiT-B (≈131M) | ε-prediction | same as A |
| C | UNet (174M) | rectified flow | Euler 4/8/16/32/64/128 (Heun optional) |
| D | DiT-B (≈131M) | rectified flow | same as C |

```
          ε        RF
UNet   │   A   │   C   │
DiT    │   B   │   D   │
```

> We use a 2×2 controlled design to disentangle the effects of backbone architecture and
> training objective.

All four cells are trained **under the identical protocol below** — including the
retraining of A, whose original checkpoint predates the protocol freeze and has no EMA
weights. Treating the old checkpoint as A would confound the design.

## Unified protocol (identical for A–D)

| Item | Value |
|---|---|
| Data | CelebA 64×64, PIL BILINEAR resize → [-1, 1] (no crop), ≈202.6k images |
| Epochs | 300 |
| Global batch | 160 per GPU × 5 GPUs = 800 (≈254 steps/epoch, ≈76k steps) |
| Optimizer | Adam (unchanged from the existing baseline) |
| LR | 2e-4, linear warmup 2000 steps → cosine decay to 1e-6 |
| Gradient clipping | 1.0 |
| EMA | decay 0.9999, **all cells**; checkpoints store raw + EMA |
| Seed | 42 |
| Checkpoints | every 50 epochs + latest/best; raw + EMA weights |
| Reported FID | EMA weights unless stated otherwise |

**Model configs**

| | UNet (A, C) | DiT (B, D) |
|---|---|---|
| params | ≈174M | ≈131M |
| width | base_channels 128, `num_downs=4` | hidden 768, depth 12, heads 12 |
| conditioning | sinusoidal + MLP (dim 256) | adaLN-Zero on the same embedding dim |
| attention | SDPA at the bottleneck | SDPA in every block (patch 4 → 256 tokens) |

Params are *not* forced to match: FLOPs, activation memory and optimization dynamics differ
between the two families, so we compare at a similar parameter scale and report
**params + FLOPs + throughput** to expose the differences.

## Objective definitions

- **ε-prediction (A, B)**: DDPM, `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`, target `ε`.
- **Rectified flow (C, D)**: linear-interpolation flow matching —
  `t = 0 → data`, `t = 1 → noise`, `x_t = (1-t)·x₀ + t·ε`, target `v = ε - x₀`.
  Sampling integrates the ODE from `t = 1` back to `t = 0`:
  `x ← x - Δt·v_θ(x, t)`.

Implementation note: both models embed the timestep with a sinusoidal basis whose useful
range is ≈[0, 1000] (as in DDPM). The flow variant therefore passes `t * 1000` to the model
while the path math stays in `t ∈ [0, 1]`.

## FID protocol

- Reference stats are computed **from `data/celeba_64.pt`** — the exact tensor the models
  train on — with InceptionV3 pool3 (2048-d). Third-party stats must not be used: any
  mismatch in resizing/preprocessing makes the numbers incomparable.
- **Tier 1 (10k samples)**: debugging, parameter sweeps, FID-vs-steps curves.
- **Tier 2 (50k samples)**: final numbers only (A–D, README, resume).
- **Variability**: split the Tier-2 samples into 5 × 10k random subsets and report the
  spread as `FID = x.x ± y.y` ("variability across 5 random subsets", not a CI).
- **Precision**: one configuration is sampled in both bf16 and fp32; report the FID delta.

## NFE grid & reporting

- ε cells: DDPM ancestral 1000, DDIM 10 / 20 / 50.
- RF cells: Euler 4 / 8 / 16 / 32 / 64 / 128; Heun at 8 / 16 / 32 if time allows.
- Reported as *inference quality as a function of NFE across different generative
  trajectories*. DDPM vs DDIM is a same-model sampler ablation (a fair sampler comparison);
  cross-cell curves additionally change the training objective, which the report must state.

## Cost accounting (printed by every run)

```
Model: DiT-B          Params: 131.2M        FLOPs: XX.X GFLOPs/step
Training:  batch 800 | step X.XX s | XXXX img/s | peak XX.X GB
Sampling:  NFE XX | XX.X img/s | 50k → XXXX s
```

## Systems ablations (in order)

1. Tier 0: BF16+TF32, fused AdamW, SDPA backends, `torch.compile` (compile time reported
   separately from steady-state), batch-size 400/800/1600 sweep.
2. Component profiling of the DiT block (attention / MLP / adaLN latency + memory share) —
   this decides what the kernel work targets.
3. Triton kernel: fused adaLN (LayerNorm + scale + shift) — stop at ≥8% kernel gain; report
   both kernel-level and model-level speedup.
4. CUDA Graphs for RF sampling (fixed shape/batch/NFE).
5. FSDP vs DDP reverse benchmark — expected to be *slower* at this scale; the point is to
   measure and report it.

Quality must not regress: the BF16 / compile tables carry an FID column.

## Out of scope (frozen)

CFG, VAE / latent diffusion, DPM-Solver, consistency distillation, video / world models,
DiT scaling study, sequence parallelism.
