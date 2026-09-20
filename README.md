# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

从零实现（不依赖 diffusers）的扩散模型项目：用统一的**受控实验**对比 UNet/DDPM、DiT 与
Rectified Flow 三种骨干与训练目标，配套 FID、NFE 采样效率与现代 ML systems 分析。

当前状态：DDPM 基线已完成（MNIST 28×28、CelebA 64×64，5-GPU DDP 训练）；
DiT + Rectified Flow 在 `dit-rectified-flow` 分支开发中（实验设计见 [reports/exp_plan.md](reports/exp_plan.md)）。

| MNIST 28×28（UNet 12M，170 epochs） | CelebA 64×64（UNet 174M，300 epochs） |
|---|---|
| ![MNIST](figures/mnist_epoch170.png) | ![CelebA](figures/celeba_epoch300.png) |

## 亮点

- **核心算法从零实现**：DDPM 前向/反向过程、UNet（正弦时间嵌入 / 残差块 / 中间层注意力）、
  DiT（adaLN-Zero）、Rectified Flow（线性插值路径 + Euler ODE 采样）
- **训练系统**：torchrun DDP（最多 5 GPU）、YAML 配置、断点续训、EMA、按 epoch 快照
- **评估**：FID（与训练预处理一致的 reference stats）、FID-vs-NFE 曲线、采样吞吐
- **ML systems**：cost accounting（params / FLOPs / step time / throughput / MFU）、
  torch.compile、Triton fused AdaLN、CUDA Graph 采样（计划中）

## 环境

Python 3.10+，PyTorch 2.x，其余依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# MNIST（单卡）
python main.py train --config configs/mnist.yml

# CelebA（单卡）
python main.py train --config configs/celeba.yml

# CelebA（5 卡 DDP）
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  torchrun --standalone --nproc_per_node=5 main.py train --config configs/celeba.yml

# 从 checkpoint 采样
python main.py sample --dataset celeba --n 64

# 去噪过程动画（含 GIF 导出）
python demo.py --dataset celeba
```

> 多卡服务器若遇通信挂起，设置 `NCCL_P2P_DISABLE=1`。
> 冒烟测试：`python tests/test_check.py`

## 项目结构

```
├── models/            # 骨干网络
│   └── unet.py        # UNet（DDPM 基线）
├── diffusion/         # 扩散过程
│   └── ddpm.py        # DDPM：前向加噪 / 训练损失 / 祖先采样
├── configs/           # YAML 训练配置（celeba / mnist）
├── tests/             # 冒烟测试
├── reports/           # 实验计划与技术报告
├── figures/           # 结果图
├── train.py           # 训练循环、数据管线、checkpoint
├── main.py            # CLI 入口（train / sample）
├── demo.py            # 去噪过程可视化
└── requirements.txt
```

## 实现说明

- `diffusion/ddpm.py`：`forward_diffusion` 按 `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε` 加噪；
  `training_loss` 为预测噪声与真实噪声的 MSE；`sample` 执行祖先采样（可选返回整条去噪轨迹）。
- `models/unet.py`：正弦时间嵌入经 MLP 注入每个残差块，瓶颈处接 SDPA 自注意力，
  下采样/上采样之间用 skip connection 保留多尺度信息。
- `train.py`：数据管线（MNIST pkl / CelebA zip → 预处理缓存）、LR warmup + 余弦退火、
  梯度裁剪、`latest`/`best` 双 checkpoint 与断点续训（兼容旧格式 checkpoint）。
- `demo.py`：逐帧捕获反向过程，支持方向键单步/空格暂停的交互窗口与 GIF 导出。
