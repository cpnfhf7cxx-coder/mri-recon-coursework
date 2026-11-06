# 医学图像合成与重建：MRI 欠采样图像重建课程作业

**Coursework — Retrospective MRI Undersampling & Reconstruction**

> **目标**：在同一套模拟 4× 欠采样条件下，用**传统优化方法（CPU 即可）**或**深度学习方法（有 GPU 可选）**，从欠采样 MRI 图像中恢复原始图像，并用统一指标、可视化和实验报告评价重建效果。
>
> **不要求 GPU**；**两条路线任选一条**。课程只要求实现一种主要方法，无须复现复杂论文。


## 1. 作业背景

MRI 扫描采集的是频域 **k-space** 信息，欠采样可以减少采集量，但直接逆傅里叶变换通常会出现混叠伪影。给定完整的 2D 图像 $`x`$，本次作业通过二维傅里叶变换和固定采样掩码 **模拟** 欠采样：

```math
y = M \odot \mathcal{F}(x), \quad x_{ZF}=\left|\mathcal{F}^{-1}(y)\right|,\quad \hat x = f(y,M,x_{ZF}).
```

其中 $`M`$ 是预定义的 Cartesian k-space 采样掩码，$`x_{\mathrm{ZF}}`$ 是 Zero-filled baseline，$`\hat{x}`$ 是你的重建结果。

**重要说明**：输入数据是预处理后的**幅值 MRI 图片**（JPEG/PNG），不包含扫描仪原始多线圈 k-space、相位信息和采集参数。本作业是**回顾性模拟欠采样**，只能用于重建算法教学，不应被描述为真实 MRI 采集加速性能，也不能用于临床结论。此处的“退化后”是指 *Zero-filled Reconstruction*，不是 k-space 图片。

## 2. 数据集

**Alzheimer MRI Preprocessed Dataset**：约 **6,400 张 128×128** 的脑 MRI 图片，压缩体积约 **28–30 MB**，原本用于分类，本作业只使用图像本身，不使用分类标签。

- Kaggle：<https://www.kaggle.com/datasets/rajmohan096/alzheimer-mri-preprocessed-dataset>
- 引用：Kumar, S. & Shastri, S. (2025). *Alzheimer MRI Preprocessed Dataset*. Mendeley Data, V1. DOI: [10.17632/3r8hw8wmmk.1](https://doi.org/10.17632/3r8hw8wmmk.1).

请下载并解压，将包含 JPG/PNG 图片的文件夹放在本地，例如 `data/raw/Alzheimer (Preprocessed Data)/...`


## 3. 统一的实验协议（两条路线都遵守）

| 内容 | 统一要求 |
| --- | --- |
| 数据 | 上述 MRI 图片；灰度、单通道、`128×128`，像素固定归一化到 `[0,1]` |
| 划分 | 使用仓库随附的 `protocol/split_manifest.csv`；训练 / 验证 / 测试约为 `70% / 15% / 15%`（仅图像级） |
| 测量模拟 | 中心保留部分低频、其余随机抽取的 **Cartesian** k-space 采样列 |
| 加速因子 | **4×**，即约 25% 的 k-space 列被保留（128 列中保留 32 列） |
| 欠采样掩码 | 所有图片、所有方法使用同一张 mask；种子 `2026`；中心比例 `0.08` |
| 输入 | 欠采样 k-space `y`、mask `M`，或由其得到的 Zero-filled 图像 |
| 输出 | 与原图同尺寸的 2D 灰度 MRI 重建图像 |
| 指标 | **PSNR (dB)↑、SSIM↑**，补充 **MAE↓**，统一 `data_range=1` |
| 基线 | 必须报告 **Zero-filled**；并与自己实现或选择的方法对比 |
| 禁止事项 | 使用测试图像 GT 调参或训练；输入真实目标图像直接作为预测；改变 mask 后与官方指标直接比较 |


## 4. 两条完成路线（二选一）

### A：传统算法路线（**不需要 GPU**）

本路线必须比较 **Zero-filled + 两种传统重建算法**：

1. 仓库提供的 **TV-POCS** 教学基线；
2. 自己实现的另一种传统方法（不能只是复制或改名 TV-POCS）。

第二种方法可以选择：

- Wavelet 稀疏正则化/软阈值 + 数据一致性。
- POCS 及其改进、迭代优化或其他合理的传统重建方法。

**最低要求**：明确介绍两种算法的原理；独立实现第二种传统方法；开展**至少一种算法的一组参数对比**（例如正则强度、迭代次数、收敛情况），并汇报三种方法的图像与指标。仅运行仓库提供的 `--method tv` 不算完成算法部分。

**固定图片**：只能在 `protocol/classical_val.txt` 的 20 张图上选参数；最终报告统一在 `protocol/classical_test.txt` 的 30 张图上评估。不得按效果自行换图。这样既控制 CPU 用时，也保证各组可直接比较。

### B：深度学习路线（**GPU 可选**）

使用简单的 CNN、U-Net、ResNet 或小型 Transformer，学习 Zero-filled 到完整 MRI 的图像恢复映射；可以使用仓库提供的轻量 U-Net 作为起点。训练损失可用 L1/MSE，改进方案可增加 SSIM 或数据一致性模块。

**最低要求**：说明网络架构、损失函数、优化器、epoch、批量大小与设备；保存权重；报告验证集/测试集指标；展示不少于三张对比图。只训练一个模型即可，不要求多个架构对比；如使用预训练模型必须注明来源。

**公平评分**：不按 PSNR 绝对排名，CPU 与 GPU 路线按照方法正确性、实验完成度、分析和可复现性使用同一评分维度。

## 5. 快速开始

推荐 **Python 3.10–3.12**，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

**预处理数据**（`--source` 换成实际解压路径；Windows 目录有空格时要加引号）：

```bash
python tools/prepare_dataset.py --source "data/raw/Alzheimer (Preprocessed Data)" --out data/processed --size 128 --manifest protocol/split_manifest.csv
```

脚本会校验图片哈希，并生成 `data/processed/train/`、`val/`、`test/` 与本地 `manifest.csv`。不要自行重新划分；组号不改变数据划分，每组都使用相同协议。

> 这里的“退化图像”**不需要预先生成或保存一套副本**。训练或测评时，程序对每张完整图像做 FFT，乘以固定 mask，再做 IFFT 得到 Zero-filled。由于尺寸、mask、种子都固定，同一张图片每次得到的输入完全相同。种子只在创建 mask 时使用一次，并不是每张图重新随机采样。

**先运行 Zero-filled 基线**：

```bash
python demo.py --method zf --split val --file-list protocol/classical_val.txt --out outputs/zf-val
```

**CPU 传统算法示例**：

```bash
python demo.py --method tv --split val --file-list protocol/classical_val.txt --steps 15 --weight 0.01 --out outputs/tv-val
```

**自己实现的传统重建**：编辑 `src/student_method.py` 里的 `reconstruct(y, mask, zero_filled)`，然后运行：

```bash
python demo.py --method student --split val --file-list protocol/classical_val.txt --out outputs/my-method-val
```

选定参数后，把上面三条命令的 `--split val --file-list protocol/classical_val.txt` 改为 `--split test --file-list protocol/classical_test.txt`，在固定 30 张图片上生成最终结果。测试集只运行最终选定参数，不再据此调参。

**深度学习 U-Net示例**：按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装适合自己 GPU/CPU 的 PyTorch；之后：

```bash
pip install -r requirements-dl.txt
python train_unet.py --epochs 30 --patience 5 --batch-size 16 --device auto
python demo.py --method unet --checkpoint checkpoints/unet.pt --split test --device auto --out outputs/unet-test
```

这里的 `--epochs 30` 是训练上限，不是固定训练 30 轮。请根据自己的训练过程指定合适的该参数。脚本每个 epoch 计算 validation L1，保存 validation 最优 checkpoint；连续 5 轮没有显著改善时提前停止，并生成 `checkpoints/unet-history.csv`。深度学习路线只用 `train` 训练、用 `val` 选 epoch 与超参数，最后在完整 `test` 上运行一次并报告结果；不得用 test 继续训练或选模型。

`demo.py` 会检查 checkpoint 中记录的加速率、mask 种子和图像尺寸是否与测试命令一致。只加载自己训练或来源可信的 checkpoint；旧版 PyTorch 的兼容加载无法提供新版 `weights_only` 的完整安全保护。


## 6. 必交的可视化与数值结果

每位同学**至少提交 3 个不同 MRI 输入样本**的三联图，每个样本均从左到右包含：

| ① 退化前 (Original GT) | ② 退化后 (Zero-filled) | ③ 恢复后 (Reconstruction) |
| --- | --- | --- |
| 完整 MRI 灰度图像 | 欠采样引起的混叠图像 | 你的算法输出 |

**要求**：同一张图像、相同空间位置和固定 `[0,1]` 灰度显示范围，不得对每一栏单独做自动对比度拉伸；第二、三栏标上相对原图的 PSNR 与 SSIM；最好选择效果较好、一般、较差各一张并进行简要分析。

脚本自动生成：

```text
outputs/my_method/
├── figures/           # 三联图 PNG（报告中至少放 3 张）
├── reconstructions/   # 重建结果 PNG
├── metrics.csv        # 每张图的 PSNR / SSIM / MAE
├── summary.json       # 平均指标、方法、加速率与种子
└── mask.npy           # 实际使用的 k-space mask
```

深度学习训练还会在 checkpoint 旁生成 `unet-history.csv`，包含每轮的 train/validation L1，可直接用于绘制训练曲线和说明最佳 epoch。

报告的最基本数值对比表：

| Method | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ |
| --- | ---: | ---: | ---: |
| Zero-filled | （填写） | （填写） | （填写） |
| TV-POCS（传统路线必填） | （填写） | （填写） | （填写） |
| Your method | （填写） | （填写） | （填写） |

一切数据都必须来自**相同图像集合、同样的采样 mask 与数据范围**。建议报告额外展示 mask 或迭代收敛曲线。

## 7. 提交方式与评分（建议）

**提交文件**：`组号_学号_姓名.zip`，至少包含报告 PDF、运行代码、重建示例图、`metrics.csv` 与运行说明。深度学习路线需交模型权重或可获取的权重链接（如果太大），传统路线需交关键参数与迭代方式。**不要把原始数据集打包提交**。

| 项目 | 分值 | 说明 |
| --- | ---: | --- |
| 方法原理与实现 | 30 | 算法说明清楚，核心实现合理，不是直接照搬基线 |
| 实验设计与规范 | 25 | 正确划分数据，统一 4× 欠采样，明确参数和对比实验 |
| 结果与分析 | 25 | PSNR/SSIM/MAE、至少 3 张三联图，分析成功与失败案例 |
| 报告与可复现性 | 20 | 报告结构清晰、环境与命令完备、能复现 |

额外尝试 2×/8× 加速、不同 mask、数据一致性层、收敛曲线等可作为自主探索，**不作为必需项**

**报告模板**：[`docs/report-template.md`](docs/report-template.md)

## 8. 仓库结构

```text
mri-recon-coursework/
├── README.md                      # 学生必读：作业说明
├── demo.py                        # 统一重建、评价与三联图
├── train_unet.py                  # 可选 GPU/CPU 训练基线
├── src/
│   ├── mri.py                     # FFT / IFFT / Cartesian 欠采样
│   ├── classical.py               # Zero-filled / TV-POCS
│   ├── student_method.py          # 学生自定义传统方法
│   ├── models.py                  # Tiny U-Net
│   ├── dataset.py                 # 数据集加载
│   ├── metrics.py                 # PSNR / SSIM / MAE
│   └── visualize.py               # 三联图自动输出
├── tools/
│   ├── prepare_dataset.py         # MRI 图片读取、去完全重复、划分
│   └── make_phantom.py            # 合成体模程序自检
├── protocol/                       # 官方划分与传统路线固定测评清单
├── docs/
│   └── report-template.md
├── tests/
├── assets/                        # README 示例图（合成体模）
├── requirements.txt               # 无 GPU 路线依赖
├── requirements-dl.txt            # 深度学习附加依赖
└── LICENSE                        # 仅适用于仓库代码，不含数据
```

## 9. 常见问题

**Q：没有 GPU 能完成吗？** 可以。选择 TV / POCS / 稀疏优化路线，在 CPU 上处理有限数量的验证图像即可。

**Q：为什么退化后图像比原图暗？** 使用统一的 `data_range=1` 和灰度可视化；不能通过分别调亮每一栏来“改善”指标。

**Q：固定 mask 到底是什么意思？退化要提前做一次吗？** mask 是一组固定的 32 条 k-space 列：其中中心连续低频列全部保留，其余列由种子 `2026` 一次性选出。程序每次运行时在线计算退化，无需保存退化数据；确定性计算会得到相同结果。训练、验证、测试也使用同一张 mask。

**Q：为什么不提供真实 k-space？** 真正多线圈 MRI 采集数据通常显著更大，难以兼顾周末作业的下载、预处理与计算条件。本仓库明确限定为**模拟欠采样教学实验**。

**Q：作业有代码抄袭界限吗？** 可以引用公开算法或基线，但必须标注来源，说明自己的实现和改进；提交的是可复现实验而非竞赛刷分。
