# Q2 中文论文成文结构（收敛版）

## 1. Introduction

建议 6–7 个自然段，不展开实验开发史。

1. SOC 对 BMS 的意义，以及动态工况下电压/电流单一观测的局限。
2. FBG/应变/机械辅助 SOC 的研究进展，强调其能提供传统电学信号之外的内部响应。
3. 现有 FBG SOC 研究的不足：多聚焦原始工况内精度；对 C-rate 与 unseen driving profile 同时变化时的泛化研究不足。
4. 双 FBG 表示问题：W1/W2 可进一步解耦为温度/力学量，但物理可解释坐标不必然等价于最适合数据驱动跨工况预测的坐标。
5. 提出本文框架：直接使用 V/I/W1/W2 的 causal TCN + residual conformal UQ，在严格 compound operating-condition shift 下验证。
6. 用 3–4 点列贡献。

建议贡献：

- 构建电学–光学联合 SOC 估计框架，直接利用双 FBG 波长观测补充 V/I 信息。
- 开展 raw wavelength 与 thermo-mechanical decoupled representation 的 matched comparison，并确定更适合跨工况预测的表示。
- 建立 1C↔2C + unseen-profile 的严格泛化实验，采用 5 seeds 和 FBG wavelength noise 验证稳定性。
- 引入 residual split conformal，为 SOC 点预测输出 90%/95% 预测区间。

## 2. Dataset and battery feature/signal analysis

### 2.1 Dataset and test conditions

- SiC-18 数据来源与公开属性。
- 六种动态工况：HWFET、LA92、NEDC、NYCC、US06、WLTC。
- 1C/2C 两倍率，共 12 条主要 dynamic discharge trajectories。
- 原始可用信号：Voltage、Current、Wavelength 1、Wavelength 2，以及数据源给出的 derived temperature/force、SOC reference。
- Table 1：数据集信息与工况。

### 2.2 Dual-FBG sensing principle and representation

- 简洁介绍 Bragg wavelength response。
- 给出双 FBG sensitivity matrix / W→T,F 关系。
- 强调 W1/W2 与 T/F 为同两个光学自由度的不同坐标，不把六个变量写成六个独立物理信息源。
- 物理解耦的作用写成 interpretability，而不是必然提升 prediction。

### 2.3 Electrical–optical signal characteristic analysis

正文建议三类图：

- Fig. 1：代表性 profile 下 SOC/V/I/W1/W2 时序；
- Fig. 2(a)：不同 profile/rate 的 W1/W2–SOC 响应；
- Fig. 2(b–d)：W 与 T/F 的相关性、冗余性/condition number 或简单二维分布。

最后自然过渡：raw W 保留原始传感响应且具有更好的 matched transfer result，因此后续主模型采用 V/I/W1/W2；T/F 保留为对照。

## 3. Methodology

总方法名建议：**RA-FBG-TCN: Representation-Aware Dual-FBG Temporal Convolutional Network**。

不要宣称“提出全新的 TCN 理论”，而是写“提出一种面向双 FBG 表示选择与跨工况 SOC 估计的轻量 causal framework”。

### 3.1 Overall framework

Fig. 3 整体框架：

Input V/I/W1/W2 → source-only normalization → causal sliding window → temporal convolution blocks → SOC regression → residual conformal interval.

### 3.2 Causal TCN estimator

写清：

- 64-sample window；
- causal convolution；
- dilation 扩大 receptive field；
- residual connection；
- GELU/linear regression head；
- 参数量约 11.5k，突出 lightweight。

### 3.3 Representation-aware optical input

用一个小节把 W vs T/F 讲成方法设计：

- T/F 是物理解耦表示；
- W 是原生光学表示；
- matched experiments 后冻结 raw W；
- 不做 whitening，不加入 absolute time/delta-t。

正文只需给核心结果，不讲完整筛选历史。

### 3.4 Residual split conformal UQ

- calibration residual r_i = |y_i - yhat_i|；
- finite-sample quantile q_(1-alpha)；
- interval [yhat-q, yhat+q] 并裁剪到 [0,1]；
- 90% / 95% nominal coverage；
- 指标 PICP、MPIW、MIS。

## 4. Experiments and results

### 4.1 Experimental settings and metrics

- hardware/software；
- AdamW；
- window=64；
- train-only normalization；
- MAE/RMSE/R2/Q95-AE/MaxAE；
- UQ 指标。

只简要解释 blocked T1 与 T4，避免把 T1/T2/T3/T4 全写成审计术语。

### 4.2 Model comparison

Table 3：CNN、GRU、LSTM、Transformer、VI-TCN、VI+TF-TCN、RA-FBG-TCN。

Fig. 4：选择 2–3 个代表性工况画 true SOC vs prediction / error。

写法照常规电池论文：比较 MAE/RMSE/R2，突出最终方法精度和轻量参数量。

### 4.3 Optical representation and feature ablation

Table 4：

- VI；
- VI+T/F；
- VI+W1/W2（proposed）；
- 可选 ETMF-TF 一行证明复杂融合并非必要。

Fig. 5 可用 bar/radar/box，不需要把 Mamba、CrossFormer、ModernTCN 写出来。

核心结论：加入光学观测改善预测能力；native wavelength representation 在 retained causal model 下表现最稳定。

### 4.4 Cross-rate unseen-profile generalization

主实验。

Table 5：

- 1C→2C：MAE≈1.80%，RMSE≈3.04%，R2≈0.9857；
- 2C→1C：MAE≈0.81%，RMSE≈0.99%，R2≈0.9985；
- overall：MAE≈1.30%，95% bootstrap CI≈0.96%–1.68%。

Fig. 6：两个方向 × 六 profile 的 MAE/RMSE 或箱线图。

正文不要单独突出 seed0，只写 1C→2C variance larger than reverse transfer，并在 Discussion 解释 higher-rate extrapolation 更难。

### 4.5 Robustness and uncertainty analysis

把两类实验合并：

1. FBG wavelength noise：0/0.5/1/2 pm；
2. conformal UQ：90%/95% intervals。

Fig. 7：左边 noise level vs MAE/Q95，右边一条代表性 SOC trajectory + prediction interval。

Table 6：UQ PICP/MPIW/MIS。

### 4.6 Optional external validation paragraph

默认不单开大节。

如投稿前觉得“第二数据集”必须出现，只在本节最后增加一段：在第二公开多电芯 FBG 数据中，固定 sensor identity 的 0.2C+0.5C→1C 测试下，加入 S5-relative optical response 将四电芯平均 MAE 从 14.92% 降至 13.42%，且 3/4 cells improvement；同时指出 optical benefit remains sensitive to cell/sensor calibration。

不要在摘要、Highlights 或 Contributions 里声称 cross-cell generalization。

## 5. Discussion

建议只写 3 小节，总长度控制在 1.5–2 页。

### 5.1 Physical decoupling versus predictive representation

重点：解耦并非错误；物理可解释性与预测空间的最佳 conditioning/transferability 不是同一目标。raw W 对 compact TCN 更适合。

### 5.2 Cross-condition robustness and practical implication

重点：reverse 2C→1C 更稳定；1C→2C 更难；2 pm wavelength noise 下仍保持小幅性能下降，说明对 interrogator measurement perturbation 有一定鲁棒性。

### 5.3 Limitations and future work

只写一个短边界：当前主实验来自单一 implanted dual-FBG sensing configuration；后续需要进一步研究不同 physical cells、sensor bonding/calibration、temperature range 下的 transfer。

不展开 E1/E3 的全部负结果。

## 6. Conclusion（如目标期刊习惯单独 Conclusion）

虽然用户希望五大主体章节，但投稿时建议保留独立 Conclusion，约 2–3 段：

- 总结 RA-FBG-TCN + raw representation；
- 总结 T4 generalization + noise + UQ；
- 一句未来工作。

## 正文明确不写的内容

- Mamba/CrossFormer/ModernTCN 开发历史；
- fixed/dynamic multi-delay；
- CQR；
- delta-t；
- E1 cross-cell 失败详细表；
- E3 WLTP label audit；
- 所有 keep/drop gate；
- workflow/provenance/commit 信息。

这些仅存于仓库和 supplementary/reviewer-response 备用。