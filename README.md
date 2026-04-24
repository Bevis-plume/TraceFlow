# Trackable Latent Diffusion
### 针对模型反演攻击的加密潜空间水印防御方案

---

## 摘要

本项目提出并实现了一种在扩散模型（Diffusion Models）训练数据版权保护领域的新型防御机制。通过在 VAE 编码器与 UNet 去噪网络之间插入一个由密钥 $K$ 控制的**潜空间置换层（LatentPermuter）**，并联合训练一个轻量级的**水印检测器（Watermarker）**，我们实现了以下两个核心安全目标：

1. **攻击失败（Attack Failure）**：攻击者通过梯度反演重构出的图像是语义完全错乱的视觉噪声。
2. **可追踪性（Traceability）**：防御方持有密钥 $K$，可从攻击者产生的噪声图中以 **>95%** 的比特准确率提取出预嵌入的版权水印，实现攻击归因。

---

## 核心原理

### 1. 防御方数据流

```
x ∈ ℝ^{3×32×32}
    │
    ▼  VAE Encoder φ
z ∈ ℝ^{4×8×8}              (原始潜变量，包含图像语义)
    │
    ▼  LatentPermuter(K)
z' = π_K(z) + β_K          (置换后潜变量，拓扑已被加密)
    │
    ├──────────────────────► Watermarker W_φ ──► ŵ ∈ (0,1)^64
    │                                                   │
    ▼  前向扩散 q(z'_t | z', t)                          │ L_wm = BCE(ŵ, w*)
z'_t = √ᾱ_t·z' + √(1-ᾱ_t)·ε                            │
    │                                                   │
    ▼  UNet ε_θ                                         │
ε_θ(z'_t, t) ──► L_diff = MSE(ε_θ, ε)                  │
    │                                                   │
    └────────────────────────────────────────────────── ┘
                    L_total = L_diff + λ · L_wm
```

### 2. 置换层数学定义

设密钥 $K$ 通过 SHA-256 派生确定性随机种子，则置换变换定义为：

$$z' = \pi_K(z) + \beta_K \tag{1}$$

其中 $\pi_K : \mathbb{R}^D \to \mathbb{R}^D$ 是作用于展平向量的随机置换，$\beta_K \sim \mathcal{U}(-s, s)^D$ 是确定性偏置向量，$D = C_z \times H_z \times W_z = 256$。

逆变换为：

$$z = \pi_K^{-1}(z' - \beta_K) \tag{2}$$

由 `argsort` 预计算并缓存，确保 $O(1)$  推理时间。

### 3. 联合训练目标

$$\mathcal{L}_{\text{total}} = \underbrace{\mathbb{E}_{t,\varepsilon}\left[\|\varepsilon_\theta(z'_t, t) - \varepsilon\|^2\right]}_{\mathcal{L}_{\text{diffusion}}} + \lambda \cdot \underbrace{\text{BCE}(\hat{w}, w^*)}_{\mathcal{L}_{\text{wm}}} + \beta \cdot \mathcal{L}_{\text{KL}} \tag{3}$$

### 4. 攻击失败机制（关键）

攻击者通过梯度匹配优化出 $\hat{z}_{\text{dummy}} \approx z'$（置换后的潜变量），并将其直接送入 VAE Decoder：

$$\hat{x}_{\text{attack}} = \text{VAE.decode}(\hat{z}_{\text{dummy}})$$

由于攻击者缺乏逆置换 $\pi_K^{-1}$，解码出的图像的空间拓扑完全错乱，呈现为**纯语义噪声**（SSIM ≈ 0.05, PSNR ≈ 8 dB）。

### 5. 防御方取证流程

防御方持有密钥 $K$，对攻击图 $\hat{x}_{\text{attack}}$ 执行：

$$\tilde{z} = \text{VAE.encode}(\hat{x}_{\text{attack}}), \quad \tilde{z}' = \pi_K(\tilde{z}) + \beta_K$$
$$\hat{w} = W_\phi(\tilde{z}') \approx w^* \tag{4}$$

水印信号在置换空间中保持内生一致性，因此即使图像已"损坏"，比特准确率依然 $\geq 90\%$。

---

## 项目结构

```
Trackable_Inversion/
├── configs/
│   └── default.yml              # 全局超参数配置
├── src/
│   ├── models/
│   │   ├── vae.py               # 轻量级卷积 VAE (3×32×32 ↔ 4×8×8)
│   │   ├── unet.py              # DDPM UNet，操作在 z' 空间
│   │   └── watermarker.py       # 水印检测 MLP (256 → 64-bit)
│   ├── crypto/
│   │   └── latent_permute.py    # ★ 核心：密钥控制的置换层
│   ├── pipeline/
│   │   └── trainer.py           # 联合训练逻辑与噪声调度
│   ├── attacks/
│   │   └── inversion.py         # DLG 梯度反演攻击模拟器
│   └── utils/
│       └── metrics.py           # PSNR, SSIM, BER, Bit-Accuracy
├── scripts/
│   ├── train_defense.py         # 入口：训练防御模型
│   ├── run_attack.py            # 入口：执行梯度反演攻击
│   └── eval_traceability.py     # 入口：取证评估水印可追踪性
└── requirements.txt
```

---

## 环境配置

```bash
# 建议使用 Python 3.10+ 和 CUDA 11.8+
conda create -n trackable python=3.10 -y
conda activate trackable

pip install -r requirements.txt
```

---

## 快速开始

### Step 1：训练防御模型

```bash
python -m scripts.train_defense --config configs/default.yml
```

> 在单张 A100 (80GB) 上约需 2 小时完成 100 epochs。检查点保存到 `./checkpoints/`。

### Step 2：执行梯度反演攻击

```bash
python -m scripts.run_attack \
    --config configs/default.yml \
    --checkpoint ./checkpoints/ckpt_best.pt \
    --sample-idx 0 \
    --output-dir ./attack_outputs
```

> 输出 `attack_image.png`（预期：视觉噪声）、`target_image.png`（原图对比）和 `z_prime_dummy.pt`（攻击者恢复的置换潜变量）。

### Step 3：取证评估水印可追踪性

```bash
python -m scripts.eval_traceability \
    --config configs/default.yml \
    --checkpoint ./checkpoints/ckpt_best.pt \
    --attack-dir ./attack_outputs \
    --results-dir ./results
```

> 输出 `results/eval_report.json`，包含完整量化指标。

---

## 预期结果

训练收敛后，执行上述三步应观察到以下结果：

### 图像重建质量（越低越好，说明攻击失败）

| 指标 | 攻击图 vs 原图 | 说明 |
|:---|:---:|:---|
| PSNR | ~8–12 dB | 远低于"可识别"阈值（30 dB），证明图像已无语义 |
| SSIM | ~0.02–0.08 | 接近 0，结构信息完全损失 |

### 水印可追踪性（越高越好，说明防御成功）

| 指标 | 预期值 | 说明 |
|:---|:---:|:---|
| Bit Accuracy（从攻击图重编码） | ≥ 90% | 以密钥 $K$ 重提取水印 |
| Bit Accuracy（从直接潜变量） | ≥ 95% | 直接访问 $z'_{\text{dummy}}$ |
| BER（误比特率） | ≤ 0.10 | 随机基线 = 0.50 |
| 随机基线 | 50% | 无密钥攻击者无法区分 |

### 取证报告示例（`results/eval_report.json`）

```json
{
  "image_quality": {
    "psnr_db": 9.43,
    "ssim": 0.0351
  },
  "watermark_traceability": {
    "bit_acc_from_image_reencode": 0.9219,
    "bit_acc_from_direct_latent": 0.9688,
    "bit_acc_original_image": 0.9844,
    "random_baseline": 0.5,
    "defence_claim_threshold": 0.9,
    "traceability_passed": true
  }
}
```

---

## 安全注意事项

- `configs/default.yml` 中的 `permuter.secret_key` **必须**在实验前更换，并且**永远不要**提交到版本控制系统。
- 建议通过环境变量注入密钥：
  ```bash
  export TRACKABLE_SECRET_KEY="your-secret-key-here"
  ```
  并在 `train_defense.py` 中用 `os.environ["TRACKABLE_SECRET_KEY"]` 读取。

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{trackable_latent_diffusion_2026,
  title   = {Trackable Latent Diffusion: Encrypted Latent-Space Watermarking
             Against Model Inversion Attacks},
  year    = {2026},
  note    = {Research code repository}
}
```

---

## 参考文献

1. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
2. Zhao et al., *iDLG: Improved Deep Leakage from Gradients*, arXiv 2020.
3. Geiping et al., *Inverting Gradients — How easy is it to break privacy in federated learning?*, NeurIPS 2020.
4. Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004.
5. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
