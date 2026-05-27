# iDLG 指标表格页（论文风格）

## 输出文件

- MNIST 全量消融表：
  - [results/idlg_tables/mnist_full_metrics_table.png](results/idlg_tables/mnist_full_metrics_table.png)
  - [results/idlg_tables/mnist_full_metrics_table.pdf](results/idlg_tables/mnist_full_metrics_table.pdf)

- 跨数据集对照表（MNIST vs CIFAR10）：
  - [results/idlg_tables/cross_dataset_metrics_table.png](results/idlg_tables/cross_dataset_metrics_table.png)
  - [results/idlg_tables/cross_dataset_metrics_table.pdf](results/idlg_tables/cross_dataset_metrics_table.pdf)

## 说明

- 表格风格参考论文定量表：按列自动高亮 Top-3（红/橙/黄）。
- 指标方向：
  - MSE、Loss：越小越好
  - SSIM、PSNR：越大越好

## 脚本

- 生成脚本：[baselines/idlg/make_metrics_table.py](baselines/idlg/make_metrics_table.py)

可复用命令：

```bash
conda run -n ch3-3 python baselines/idlg/make_metrics_table.py \
  --input-csv <your_metrics_csv> \
  --output-dir results/idlg_tables \
  --name <table_name> \
  --title "<table title>"
```