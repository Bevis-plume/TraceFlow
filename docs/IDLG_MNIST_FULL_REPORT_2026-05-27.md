# iDLG MNIST 完整实验报告

## 1. 实验目标

- 在 `baselines/idlg` 架构下复现 iDLG。
- 进行完整消融：`batch size = 1/2/4`，`初始化 = normal/uniform/zeros`。
- 输出三阶段重建可视化与重建图-原图对比。
- 统计并分析指标：MSE / SSIM / PSNR。

## 2. 实验配置

- 运行环境：`conda run -n ch3-3`
- 数据集：MNIST（test split）
- 迭代次数：240
- 优化器：LBFGS
- 方法：iDLG
- 三阶段快照迭代：40 / 120 / 239
- 结果目录：[results/idlg_mnist_full](results/idlg_mnist_full)

## 3. 总体结果

数据来源：
- [results/idlg_mnist_full/all_results.csv](results/idlg_mnist_full/all_results.csv)
- [results/idlg_mnist_full/all_results.json](results/idlg_mnist_full/all_results.json)

| Run | Batch Size | Init | Final MSE | Final SSIM | Final PSNR |
|---|---:|---|---:|---:|---:|
| iDLG_bs1_normal | 1 | normal | 3.6487e-09 | 1.000000 | 84.3786 |
| iDLG_bs1_uniform | 1 | uniform | 1.3319e-09 | 1.000000 | 88.7552 |
| iDLG_bs1_zeros | 1 | zeros | 9.2942e-09 | 1.000000 | 80.3179 |
| iDLG_bs2_normal | 2 | normal | 0.152791 | 0.170498 | 8.1590 |
| iDLG_bs2_uniform | 2 | uniform | 2.1647e-07 | 0.999999 | 66.6460 |
| iDLG_bs2_zeros | 2 | zeros | 0.152775 | 0.170509 | 8.1595 |
| iDLG_bs4_normal | 4 | normal | 0.114182 | 0.386975 | 9.4240 |
| iDLG_bs4_uniform | 4 | uniform | 1.7124e-05 | 0.999786 | 47.6639 |
| iDLG_bs4_zeros | 4 | zeros | 0.467545 | 0.015194 | 3.3018 |

## 4. 三阶段可视化与对比图

### 4.1 bs=1

- normal:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs1_normal/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs1_normal/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs1_normal/metrics_curve.png)
- uniform:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs1_uniform/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs1_uniform/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs1_uniform/metrics_curve.png)
- zeros:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs1_zeros/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs1_zeros/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs1_zeros/metrics_curve.png)

### 4.2 bs=2

- normal:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs2_normal/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs2_normal/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs2_normal/metrics_curve.png)
- uniform:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs2_uniform/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs2_uniform/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs2_uniform/metrics_curve.png)
- zeros:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs2_zeros/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs2_zeros/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs2_zeros/metrics_curve.png)

### 4.3 bs=4

- normal:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs4_normal/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs4_normal/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs4_normal/metrics_curve.png)
- uniform:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs4_uniform/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs4_uniform/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs4_uniform/metrics_curve.png)
- zeros:
  - 三阶段：[three_phase.png](results/idlg_mnist_full/iDLG_bs4_zeros/three_phase.png)
  - 重建对比：[recon_vs_original.png](results/idlg_mnist_full/iDLG_bs4_zeros/recon_vs_original.png)
  - 指标曲线：[metrics_curve.png](results/idlg_mnist_full/iDLG_bs4_zeros/metrics_curve.png)

## 5. 对比分析

### 5.1 Batch Size 影响

按 batch size 聚合（9 组结果按每个 bs 下 3 种初始化取均值）：

| Batch Size | Mean MSE | Mean SSIM | Mean PSNR |
|---:|---:|---:|---:|
| 1 | 4.7583e-09 | 1.000000 | 84.4839 |
| 2 | 0.101855 | 0.447002 | 27.6549 |
| 4 | 0.193915 | 0.467318 | 20.1299 |

结论：
- `bs=1` 基本可达到近乎完美重建。
- `bs>1` 时平均重建质量明显下降，主要原因是 iDLG 标签推断在多样本时不再严格可解，优化更容易陷入局部解。
- `bs=4` 的平均 SSIM 略高于 `bs=2`，由 `bs4_uniform` 的高质量结果拉高；若看最差情况，`bs=4` 风险更大（见 `bs4_zeros`）。

### 5.2 初始化方式影响

按初始化聚合（每种初始化跨 3 个 batch size 取均值）：

| Init | Mean MSE | Mean SSIM | Mean PSNR |
|---|---:|---:|---:|
| normal | 0.088991 | 0.519158 | 33.9872 |
| uniform | 5.7806e-06 | 0.999928 | 67.6884 |
| zeros | 0.206774 | 0.395234 | 30.5930 |

结论：
- `uniform` 在本次实验中最稳定、整体最优。
- `zeros` 最不稳定，在大 batch（尤其 `bs=4`）下容易失败。
- `normal` 介于两者之间，受 batch size 影响明显。

### 5.3 指标一致性（MSE / SSIM / PSNR）

三类指标在本实验中结论一致：
- 低 MSE 对应高 SSIM 与高 PSNR。
- 高质量组（如 `bs1_uniform`, `bs2_uniform`, `bs4_uniform`）同时满足：MSE 很低、SSIM 接近 1、PSNR 很高。
- 失败组（如 `bs4_zeros`）同时表现为：MSE 高、SSIM 极低、PSNR 低。

## 6. 关键结论

- 最优配置（本次实验）：`iDLG_bs1_uniform`，MSE = `1.3319e-09`，SSIM = `1.0`，PSNR = `88.76`。
- 最差配置（本次实验）：`iDLG_bs4_zeros`，MSE = `0.4675`，SSIM = `0.0152`，PSNR = `3.30`。
- 若目标是稳定复现实验图与高质量重建，建议优先使用：
  - `batch_size=1`
  - `init=uniform`
  - 保持较高迭代（本实验为 240）

## 7. 附录

- 运行脚本：[baselines/idlg/run_idlg_mnist.py](baselines/idlg/run_idlg_mnist.py)
- 攻击核心：[baselines/idlg/idlg_attack.py](baselines/idlg/idlg_attack.py)
- 模型定义：[baselines/idlg/model.py](baselines/idlg/model.py)