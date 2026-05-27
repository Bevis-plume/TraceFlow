# iDLG 论文风格精简图表页

## Figure A. 训练过程对比（DLG vs iDLG）

图文件：
- [results/idlg_paper_style/paper_fig_training_process_dlg_vs_idlg.png](results/idlg_paper_style/paper_fig_training_process_dlg_vs_idlg.png)

建议图注（可直接使用）：

Figure A: Example of the training process of DLG (left) and iDLG (right) on MNIST. The first image in each panel is the original private training image, followed by reconstructed images at different iterations. iDLG converges faster and to a cleaner reconstruction than DLG under the same setting.

## Figure B. 收敛曲线对比（4 指标）

图文件：
- [results/idlg_paper_style/paper_fig_convergence_curves_dlg_vs_idlg.png](results/idlg_paper_style/paper_fig_convergence_curves_dlg_vs_idlg.png)

建议图注（可直接使用）：

Figure B: Convergence curves of DLG and iDLG under identical optimization settings. iDLG generally reaches lower gradient-matching loss and better image quality metrics (lower MSE, higher SSIM/PSNR).

## Figure C. 消融热力图（batch size × init）

图文件：
- [results/idlg_paper_style/paper_fig_ablation_heatmaps.png](results/idlg_paper_style/paper_fig_ablation_heatmaps.png)

建议图注（可直接使用）：

Figure C: Ablation heatmaps over batch size and initialization strategy on MNIST iDLG experiments. Uniform initialization consistently provides better reconstruction quality across metrics.

## 运行参数记录

- 参数摘要文件：[results/idlg_paper_style/paper_viz_summary.json](results/idlg_paper_style/paper_viz_summary.json)
- 生成脚本：[baselines/idlg/run_idlg_paper_viz.py](baselines/idlg/run_idlg_paper_viz.py)