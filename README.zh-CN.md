# Diffusion Models from Scratch — DDPM → DiT → Rectified Flow

*[English](README.md)*

从零实现（不依赖 `diffusers`）的扩散模型项目：用**同一套冻结协议**对比 UNet / DiT 两种骨干与
ε-prediction / rectified flow 两种训练目标，并把采样器、混合精度与分布式策略的权衡用实测数据说清楚。

![质量–NFE 前沿](figures/fid_vs_nfe.png)

## 主要结果

### 1. MNIST 2×2 受控实验（4 格 × 200 epochs，同协议同种子）

| 单元 | 骨干 + 目标 | 参数 | GFLOPs/样本 | 最佳 FID | 在该 FID 的 NFE |
|---|---|---|---|---|---|
| A | UNet + ε | 11.7M | 2.43 | 2.79 | 1000（祖先） |
| B | DiT + ε | 16.5M | 1.08 | 33.18 | 1000（祖先） |
| C | UNet + flow | 11.7M | 2.43 | **0.88** | 64（Euler） |
| D | DiT + flow | 16.5M | 1.08 | 4.99 | 64（Euler） |

- 参数不等于算力：DiT 参数更多但 FLOPs 只有 UNet 的 44%，所以这不是等算力对比（报告里明确标注）
- rectified flow 在每个预算下都优于 ε-prediction，交叉点约在 NFE 16

![四格样本](figures/mnist_2x2_samples.png)

### 2. CelebA 采样器研究（同一个已训好的 DDPM，只换采样器，不重训）

| 采样器 | NFE | n | FID | 吞吐 | 5×10k 子集波动 |
|---|---|---|---|---|---|
| DDIM | 10 | 50k | 30.03 | 194 /s | 30.90 ± 0.15 |
| DDIM | 20 | 50k | 17.79 | 97 /s | 18.72 ± 0.09 |
| DDIM | 50 | 50k | **11.45** | 39 /s | 12.41 ± 0.08 |
| 祖先采样 | 1000 | 10k | 11.37 | 1.9 /s | — |

**50 步追平 1000 步（差 0.8%），吞吐 20×。** 另外发现：同一采样器在 10k 子集上会比 50k 全量
系统性偏高约 1 个 FID 点——这正是协议要求报告子集波动的原因。

### 3. 系统测量（CelebA DiT，160 图/卡）

![执行设置对比](figures/systems_settings.png)

| 设置 | s/step | 显存 | MFU | 相对 |
|---|---|---|---|---|
| fp32 | 1.386 | 32.9 GB | 10.4% | 1.00× |
| **bf16** | **0.442** | **21.1 GB** | 32.7% | **3.15×** |
| `torch.compile` | 1.415 | 32.9 GB | 10.4% | 1.00× |
| fp32 + DDP（4 卡） | 1.408 | 33.5 GB | 10.1% | — |
| fp32 + FSDP（4 卡） | 1.588 | 31.9 GB | 9.0% | **0.89×（负优化）** |

kernel 级占比（fp32 训练步）：matmul 67.6% / elementwise 15.3% / attention 11.1% / norm 2.3%。

- **bf16 是唯一有效的杠杆**（3.15× 步时、1.6× 显存）；`torch.compile` 零收益
- **FSDP 在本规模是负优化**：峰值显存被激活值主导（33.5GB 里参数+梯度+优化器状态只有约 2GB），
  分片省不到显存却要付通信代价——这条被当作「验证不该用什么」的正向测量写进报告

## 一个值得记录的发现

**确定性采样的「省步数」是有条件的。** 同一个 DDIM 实现：在 UNet 上随步数变好（22.06 → 4.32），
在 DiT 上随步数**变糟**（49 → 672），而同一权重用祖先采样是 33.2。四项排查（不是 EMA、不是模型、
不是采样器实现）+ η 扫描给出结论：η 从 0→1 时饱和像素比例 0.48→0.79、样本从斑块恢复成数字——
**确定性路径会一致地累积 score 误差，随机采样的噪声注入把它压住。**

混合精度有同样的签名：bf16 重训的单格实验里训练 loss 一致（0.0194 vs 0.0193），但 1000 步 FID
翻倍（5.56 vs 2.79），50 步持平。

> 只报 UNet 的论文会给出一个「在另一个骨干上完全反转」的采样器建议。

## 项目结构

```
├── models/             # 骨干：unet.py（残差 U-Net）、dit.py（adaLN-Zero，参数化可配）
├── diffusion/          # 目标与采样：ddpm.py（ε + 祖先/DDIM）、flow.py（rectified flow + Euler）
├── eval/               # 评估与计量
│   ├── fid.py          # FID（参考统计量自建）+ NFE 扫描 + 子集波动
│   ├── cost.py         # 实测 FLOPs / 参数量 / MFU
│   ├── systems.py      # fp32/bf16/compile 步时 + kernel profiling + DDP/FSDP
│   ├── eta_sweep.py    # 噪声注入强度 vs 样本统计（上面那个发现的诊断）
│   ├── eps_error_by_t.py  # ε 误差沿轨迹的分布
│   └── make_table.py   # results/*.json → markdown 表
├── configs/            # 四个 MNIST 单元 + 四个 CelebA 单元的 YAML
├── scripts/            # cheap_package.sh（一键跑完 5 阶段）、tier2.sh、status.sh
├── tests/              # 14 项 CPU 不变量测试
├── reports/            # 技术报告 + 运行手册 + 图表脚本
├── results/            # 全部实验原始数据（JSON）
└── figures/            # README 与报告用图
```

## 复现

```bash
# 依赖（uv）
uv sync && uv run python tests/test_sanity.py     # 14 项 CPU 测试，约 1 分钟

# 一键跑完整包（MNIST 2×2 → 两套 FID 扫描 → 系统测量 → 出图）
GPUS=0,1,2,3 nohup bash scripts/cheap_package.sh &

# 查看进度 / 单点重跑
bash scripts/status.sh
```

数据（`celeba.zip`、`mnist.pkl.gz`）与训练好的 checkpoint 不在仓库里，需要单独放置——见
[reports/runbook.md](reports/runbook.md)。详细命令、时间预算与排错都在那份手册里。

## 文档

| | |
|---|---|
| 技术报告（11 页，NeurIPS 格式） | [reports/tech_report.pdf](reports/tech_report.pdf) |
| 运行手册（机器无关） | [reports/runbook.md](reports/runbook.md) |
| 图表脚本（数据 → PDF） | [reports/make_figures.py](reports/make_figures.py) |
| 实验原始数据 | [results/](results/) |
