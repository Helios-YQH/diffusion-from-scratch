# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

从零实现（不依赖 diffusers）的扩散模型项目：用统一的**受控实验**对比 UNet/DDPM、DiT 与
Rectified Flow 三种骨干与训练目标，配套 FID、NFE 采样效率与现代 ML systems 分析。

当前状态：2×2 受控实验（UNet/DiT × DDPM/rectified flow）的代码、评估与协议已就绪，
训练待 GPU 空闲后进行；DDPM 基线已在 MNIST 28×28 与 CelebA 64×64 上完成 5-GPU DDP 训练。
实验计划见 [reports/technical_report.md](reports/technical_report.md)（结果待填），
执行步骤见 [reports/runbook.md](reports/runbook.md)。

| MNIST 28×28（UNet 12M，170 epochs） | CelebA 64×64（UNet 174M，300 epochs） |
|---|---|
| ![MNIST](figures/mnist_epoch170.png) | ![CelebA](figures/celeba_epoch300.png) |

## 亮点

- **核心算法从零实现**：DDPM 前向/反向过程、UNet（正弦时间嵌入 / 残差块 / 中间层注意力）、
  DiT（adaLN-Zero，无类别条件）、Rectified Flow（线性插值路径 + Euler ODE 采样）
- **训练系统**：torchrun DDP（最多 5 GPU）、YAML 配置、断点续训、EMA、按 epoch 快照、cost accounting
- **评估**：FID（reference stats 由训练张量自建，杜绝预处理不一致）、DDIM / Euler 的 NFE 扫描
- **成本核算**：params / FLOPs（实测）/ step time / throughput / 显存 / MFU，每次训练自动写入
  `results/*.json`；`eval/make_table.py` 直接生成报告用表格
- **ML systems（进行中）**：torch.compile、Triton fused AdaLN、CUDA Graph 采样

## 环境

Python 3.10+，依赖由 [uv](https://docs.astral.sh/uv/) 管理：

```bash
uv sync              # 基础依赖（torch / numpy / matplotlib / pillow / tqdm / pyyaml）
uv sync --extra log  # 额外安装 wandb（训练日志）
```

## 快速开始

```bash
# MNIST 2×2（四个单元，每格约 5 分钟；也是最快的冒烟路径）
uv run python main.py train --config configs/mnist_unet_eps.yml
uv run python main.py train --config configs/mnist_dit_eps.yml
uv run python main.py train --config configs/mnist_unet_rf.yml
uv run python main.py train --config configs/mnist_dit_rf.yml

# CelebA 2×2 受控实验的四个单元
uv run python main.py train --config configs/celeba_unet_eps.yml   # A: UNet + DDPM
uv run python main.py train --config configs/celeba_dit_eps.yml    # B: DiT  + DDPM
uv run python main.py train --config configs/celeba_unet_rf.yml    # C: UNet + rectified flow
uv run python main.py train --config configs/celeba_dit_rf.yml     # D: DiT  + rectified flow

# 5 卡 DDP
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 main.py train --config configs/celeba_dit_rf.yml

# 采样（默认取 EMA 权重）/ 去噪过程动画
uv run python main.py sample --run-name celeba_dit_rf --n 64
uv run python demo.py --dataset celeba
```

> 多卡服务器若遇通信挂起，设置 `NCCL_P2P_DISABLE=1`。
> 冒烟测试：`uv run python tests/test_sanity.py`（CPU 可跑，13 项不变量检查）；
> 上服务器正式跑之前可加 `--max-steps 50` 做快速验证。

## 项目结构

```
├── models/             # 骨干网络
│   ├── unet.py         # UNet（DDPM 基线）
│   └── dit.py          # DiT：patchify + adaLN-Zero + SDPA
├── diffusion/          # 扩散过程
│   ├── ddpm.py         # DDPM：前向加噪 / ε 损失 / 祖先采样
│   └── flow.py         # Rectified flow：线性插值路径 / v 损失 / Euler 采样
├── configs/            # 四个实验单元 + MNIST 的 YAML 配置
├── eval/               # 评估与计量
│   ├── fid.py          # FID（自建 reference stats）+ NFE 扫描
│   ├── cost.py         # FLOPs（实测）/ 参数量 / MFU
│   └── make_table.py   # results/*.json -> markdown 表
├── tests/              # 不变量与形状测试（CPU 可跑）
├── reports/            # 技术报告与运行手册
├── figures/            # 结果图
├── train.py            # 训练循环、数据管线、EMA、checkpoint、cost accounting
├── main.py             # CLI 入口（train / sample）
├── demo.py             # 去噪过程可视化
├── pyproject.toml      # uv 依赖定义
└── uv.lock
```

## 实现说明

- `diffusion/ddpm.py`：`forward_diffusion` 按 `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` 加噪；
  `training_loss` 为预测噪声与真实噪声的 MSE；`sample` 执行祖先采样（可选返回整条去噪轨迹）。
- `diffusion/flow.py`：约定 `t=0 → 数据`、`t=1 → 噪声`，`x_t = (1-t)·x₀ + t·ε`，
  回归目标 `v = ε - x₀`；采样从 `t=1` 欧拉积分回 `t=0`。模型看到的时间步为 `t × 1000`
  （正弦嵌入的有效区间是 [0, 1000]，与 DDPM 一致）。
- `models/unet.py`：正弦时间嵌入经 MLP 注入每个残差块，瓶颈处接 SDPA 自注意力，
  下采样/上采样之间用 skip connection 保留多尺度信息。
- `models/dit.py`：patch 4 → 256 token，固定 2D sincos 位置编码，每块用 adaLN-Zero
  （调制 MLP 零初始化，网络初始为恒等映射，实测初始输出恰为 0）。
- `train.py`：数据管线（MNIST pkl / CelebA zip → 预处理缓存）、`--backbone` / `--objective`
  开关、LR warmup + 余弦退火、梯度裁剪、EMA（老权重与 EMA 权重分别存盘）、
  每 epoch 快照、每轮打印 loss / grad norm / 吞吐，结束时写出 `results/{run_name}.json`
  （含 config、seed、git commit、显存峰值）。
