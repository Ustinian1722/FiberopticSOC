# Q2 中文论文成文结构（最终收敛版）

## 1. Introduction

建议 6–7 个自然段，不展开实验开发史。

1. SOC 对 BMS 的意义，以及动态工况下电压/电流单一观测在工况迁移时的局限。
2. FBG/应变/机械辅助 SOC 的研究进展，强调其能提供传统电学信号之外的内部响应。
3. 现有 FBG SOC 研究多聚焦原始工况内精度，对 C-rate 与 unseen driving profile 同时变化时的泛化关注不足。
4. 双 FBG 表示问题：W1/W2 可进一步解耦为温度/力学量，但物理可解释坐标不必然等价于最适合数据驱动跨工况预测的坐标。
5. 提出本文框架：直接使用 V/I/W1/W2 的 causal TCN，并重点研究光学观测在 electrical-distribution shift 下的互补价值；采用 residual conformal 给出 95% prediction interval。
6. 用 3–4 点列贡献。

建议贡献：

- 构建电学–光学联合 SOC 估计框架，直接利用双 FBG 波长观测补充 V/I 信息。
- 开展 raw wavelength 与 thermo-mechanical decoupled representation 的 matched comparison，并确定更适合跨工况预测的表示。
- 建立 1C↔2C + unseen-profile 的严格泛化实验，采用 5 seeds 和 FBG wavelength noise 验证稳定性；进一步分析 optical gain 随 electrical-OOD severity 的变化。
- 采用 residual split conformal 为最终 SOC 模型输出 95% prediction interval，补充点预测可靠性表达。

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

最后自然过渡：raw W 保留原始传感响应且在 matched cross-condition comparison 中更稳定，因此后续主模型采用 V/I/W1/W2；T/F 保留为物理解释与表示对照。

## 3. Methodology

总方法名建议：**RA-FBG-TCN: Representation-Aware Dual-FBG Temporal Convolutional Network**。

不要宣称“提出全新的 TCN 理论”，而是写“提出一种面向双 FBG 表示选择与跨工况 SOC 估计的轻量 causal framework”。

### 3.1 Overall framework

Fig. 3：Input V/I/W1/W2 → source-only normalization → causal sliding window → temporal convolution blocks → SOC regression → 95% residual conformal interval。

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
- matched transfer comparison 后冻结 raw W；
- 不做 whitening，不加入 absolute time/delta-t。

正文只给核心对比，不讲完整筛选历史。

### 3.4 Residual split conformal UQ

正文只采用 **95% nominal prediction interval**：

- calibration residual r_i = |y_i - yhat_i|；
- finite-sample 95% residual quantile；
- interval [yhat-q, yhat+q] 并裁剪到 [0,1]；
- 报告 PICP、MPIW、MIS。

90% level 可保留 supplementary/internal，不作为正文卖点。

## 4. Experiments and results

### 4.1 Experimental settings and metrics

- hardware/software；
- AdamW；
- window=64；
- train-only normalization；
- MAE/RMSE/R2/Q95-AE/MaxAE；
- UQ 指标。

只简要解释 conventional blocked interpolation 和严格 cross-rate unseen-profile protocol，避免把全文写成 protocol audit。

### 4.2 Conventional SOC estimation performance

Table 3：CNN、GRU、LSTM、Transformer、VI-TCN、VI+TF-TCN、RA-FBG-TCN。

这一节只证明各模型在常规 blocked interpolation 中均具备良好 SOC estimation ability，不承担“FBG 必须优于 electrical-only”的论证任务。

RA-FBG-TCN 当前结果：MAE约0.482%、RMSE约0.593%、R2约0.99961，属于高精度估计。

VI-TCN 是 strong electrical-only baseline；如果其 conventional interpolation 数字更优，正文如实列出但不展开讨论，后续通过 OOD complementarity 解释不同观测在不同 transfer level 下的价值。

Fig. 4：选择 2–3 个代表性工况画 RA-FBG-TCN 的 true SOC vs prediction / error。

### 4.3 Electrical–optical complementarity under distribution shift

这是替代“光学通道处处提升”的核心实验。

先给 compact representation/feature comparison：

- VI；
- VI+T/F；
- VI+W1/W2；
- 可选 ETMF-TF 一行。

随后重点展示 parameter-matched VI vs VI+W 的 electrical-OOD analysis：

- 1C→2C aggregate：VI MAE≈2.151%，VI+W≈1.632%，平均改善约24%；
- 2C→1C aggregate：VI≈0.866%，VI+W≈0.814%；
- 在1C→2C中，随着 electrical-OOD severity 增大，光学通道收益由 ID 区域的不明显/负收益，逐步增长到 severe / most-severe 区域约34% / 48.5%的相对 MAE 改善。

这一节的主结论：**FBG 的核心价值不是在容易的同分布插值任务中重复编码已有电学信息，而是在电学观测偏离训练支持域时提供互补信息。**

Fig. 5 推荐直接画 electrical-OOD severity vs relative optical gain，比普通雷达图更有论文故事。

### 4.4 Cross-rate unseen-profile generalization

主实验。

Table 5：

- 1C→2C：MAE≈1.80%，RMSE≈3.04%，R2≈0.9857；
- 2C→1C：MAE≈0.81%，RMSE≈0.99%，R2≈0.9985；
- overall：MAE≈1.30%，95% bootstrap CI≈0.96%–1.68%。

Fig. 6：两个方向 × 六 profile 的 MAE/RMSE 或箱线图。

正文不要单独突出 worst seed；只写 1C→2C variability larger than reverse transfer，并在 Discussion 解释 high-rate extrapolation 更难。

### 4.5 Robustness and uncertainty analysis

把两类结果合并：

1. FBG wavelength Gaussian noise：0/0.5/1/2 pm；
2. **95% residual conformal UQ**。

目前 T1 95% interval：PICP≈95.04%，MPIW≈2.075% SOC，可作为正文核心 UQ 数字。

Fig. 7：左边 noise level vs MAE/Q95，右边一条代表性 SOC trajectory + 95% prediction interval。

Table 6：95% PICP/MPIW/MIS，可只保留一行以避免 UQ 占篇幅过多。

### 4.6 Optional external evidence paragraph

默认不单开大节。

如投稿前觉得“第二数据集”必须出现，只在本节最后增加一段：在第二公开多电芯 FBG 数据中，固定 sensor identity 的 0.2C+0.5C→1C 测试下，加入 S5-relative optical response 将四电芯平均 MAE 从14.92%降至13.42%，且3/4 cells improvement；同时指出 optical benefit remains sensitive to cell/sensor calibration。

不要在摘要、Highlights 或 Contributions 里声称 cross-cell generalization。

## 5. Discussion

建议只写 3 小节，总长度控制在 1.5–2 页。

### 5.1 Physical decoupling versus predictive representation

重点：解耦并非错误；物理可解释性与预测空间的最佳 conditioning/transferability 不是同一目标。raw W 对 compact causal TCN 的 cross-condition transfer 更合适。

### 5.2 Why optical information becomes valuable under OOD

用 T1 vs electrical-OOD 的差异解释：

- ID/interpolation 下 V/I 已包含很强 SOC 信息，额外光学通道未必降低点误差；
- 当 C-rate/profile 改变导致 electrical feature support shift 时，FBG 提供不同于 V/I 的内部响应，收益随 OOD severity 增大；
- 这比“FBG 在所有场景都优于电学输入”更符合实际数据，也更符合多模态 sensing 的工程定位。

同时讨论 2C→1C 更稳定、1C→2C 更难，以及 2 pm wavelength noise 下仍保持小幅性能下降。

### 5.3 Limitations and future work

只写一个短边界：当前主实验来自单一 implanted dual-FBG sensing configuration；后续需要研究不同 physical cells、sensor bonding/calibration、temperature range 下的 transfer。

不展开 E1/E3 的全部负结果。

## 6. Conclusion（如目标期刊习惯单独 Conclusion）

建议保留独立 Conclusion，约 2–3 段：

- 总结 representation-aware raw-FBG + lightweight causal TCN；
- 总结 OOD complementarity、T4 generalization、noise robustness 和 95% UQ；
- 一句未来工作。

## 正文明确不写的内容

- Mamba/CrossFormer/ModernTCN 开发历史；
- fixed/dynamic multi-delay；
- CQR；
- delta-t；
- 90% conformal under-coverage 作为主文结果（可放 supplementary）；
- E1 cross-cell 失败详细表；
- E3 WLTP label audit；
- 所有 keep/drop gate；
- workflow/provenance/commit 信息。

这些仅存于仓库和 supplementary/reviewer-response 备用。