# iDLG 跨数据集对照结果（整洁版）

## 1) 图页（集中输出）

所有文件集中在：
- [results/idlg_cross_dataset](results/idlg_cross_dataset)

核心图：
- 训练过程对照页（MNIST/CIFAR10 × DLG/iDLG）：
  - [results/idlg_cross_dataset/cross_dataset_progression.png](results/idlg_cross_dataset/cross_dataset_progression.png)
- 收敛曲线对照页（MSE/SSIM/PSNR）：
  - [results/idlg_cross_dataset/cross_dataset_curves.png](results/idlg_cross_dataset/cross_dataset_curves.png)

原始指标：
- [results/idlg_cross_dataset/cross_dataset_final_metrics.csv](results/idlg_cross_dataset/cross_dataset_final_metrics.csv)
- [results/idlg_cross_dataset/cross_dataset_summary.json](results/idlg_cross_dataset/cross_dataset_summary.json)

## 2) 实验设置

- 数据集：MNIST、CIFAR10
- 方法：DLG、iDLG
- 初始化：normal
- 迭代数：201
- 快照迭代：0, 40, 80, 120, 160, 200
- 样本索引：MNIST=7，CIFAR10=7

## 3) 最终指标对比

| Dataset | Method | Final MSE | Final SSIM | Final PSNR |
|---|---|---:|---:|---:|
| MNIST | DLG | 3.4901e-09 | 1.000000 | 84.5716 |
| MNIST | iDLG | 1.1800e-08 | 1.000000 | 79.2811 |
| CIFAR10 | DLG | 1.0120e-04 | 0.997983 | 39.9482 |
| CIFAR10 | iDLG | 4.1544e-06 | 0.999914 | 53.8150 |

## 4) 结论（简洁）

- MNIST 上两者都可达到近乎完美重建（单样本下差异很小，存在随机波动）。
- CIFAR10 上 iDLG 明显优于 DLG（更低 MSE、更高 SSIM/PSNR）。
- 从跨数据集稳定性看，iDLG 在更复杂自然图像上更有优势。

## 5) 复现命令

```bash
conda run -n ch3-3 python baselines/idlg/run_idlg_cross_dataset_viz.py \
  --iters 201 \
  --init normal \
  --snapshot-iters 0 40 80 120 160 200 \
  --output-dir results/idlg_cross_dataset
```

脚本位置：
- [baselines/idlg/run_idlg_cross_dataset_viz.py](baselines/idlg/run_idlg_cross_dataset_viz.py)