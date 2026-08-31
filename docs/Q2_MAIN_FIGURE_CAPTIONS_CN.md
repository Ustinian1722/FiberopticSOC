# Q2 主图图注草稿（中文 V1）

> 图注与 `docs/Q2_FIGURE_TABLE_PRODUCTION_SPEC.md` 及 `analysis/make_q2_publication_figures.py` 对齐。框架类 panel 当前保持 PLACEHOLDER，数据 panel 可直接进入排版。

## Fig. 1 数据集与电学–光学同步响应

**Fig. 1. SiC 基锂离子电池动态放电过程中的电学–光学同步观测。** (a) 电池测试平台及植入式双 FBG 传感布置示意图，占位待后续结合实验装置/原论文信息统一重绘；(b) NEDC 1C 工况下电流与参考 SOC 随时间的变化；(c) 与电学信号同步采集的 W1 和 W2 Bragg wavelength shift；(d) NEDC 工况下 1C 与 2C 的 W2–SOC 响应对比。各数据 panel 均直接由原始 SiC-18 轨迹生成，未为绘图目的进行随机抽样。该图用于说明终端电学变量与双 FBG 光学响应在动态负载下具有同步但不同的演化特征。

## Fig. 2 双 FBG 表示分析

**Fig. 2. 双 FBG 原始波长与热–力解耦坐标的表示特性及迁移性能。** (a) 全部 12 条动态轨迹中 W1–W2 的联合分布；(b) 对应的解耦 temperature–force 联合分布；(c) 1C 和 2C 数据中两组坐标的绝对 Pearson 相关性，并给出由二变量相关矩阵得到的近似 condition number；(d) 在相同跨倍率 development protocol 下 raw W、decoupled T/F 与较复杂 ETMF-T/F 表示的平均 SOC MAE。结果表明，T/F 提供更直接的物理语义，但其高相关性并未转化为更稳定的预测迁移优势；后续主模型因此采用原始 W1/W2 作为光学输入。

## Fig. 3 RA-FBG-TCN 方法框架

**Fig. 3. RA-FBG-TCN 与 residual split-conformal uncertainty quantification 的总体框架。** 当前为排版占位。最终图将依次展示 V/I/W1/W2 输入、training-only normalization、64-sample causal window、4→24 projection、dilation=1/2/4 的 residual TCN blocks、SOC regression head，以及基于 calibration residual 的 95% conformal prediction interval。TCN block inset 将展示两层 k=3 causal Conv1D、GroupNorm、GELU 与 residual connection。

## Fig. 4 常规 SOC 估计性能

**Fig. 4. RA-FBG-TCN 在 mixed-condition blocked interpolation protocol 下的常规 SOC 估计性能。** (a,b) 代表性 NEDC 与 NYCC 测试片段中参考 SOC 与模型预测的对比；(c) 动态工况片段上的 absolute SOC error；(d) 全部测试窗口的 absolute-error distribution。该设置中 RA-FBG-TCN 的总体 MAE、RMSE 和 R² 分别为 0.482%、0.593% 和 0.999614，用于验证所保留轻量 causal estimator 的基本点预测能力，而不将该实验解释为光学通道在所有 in-distribution 条件下均优于 electrical-only input。

## Fig. 5 Electrical OOD 与光学互补收益

**Fig. 5. Electrical distribution shift severity 与双 FBG 光学信息收益之间的关系。** (a) 以 source-rate training current 的 0.5th–99.5th percentile 定义 electrical support envelope，并示例展示 1C→2C 测试电流进入 source support 之外的区域；(b) parameter-matched VI 与 VI+W 模型在不同 window-level OOD fraction 分箱中的 MAE；(c) 各 OOD 分箱中由加入 W1/W2 带来的相对 optical gain。ID 区域的相对收益为 −18.75%，而随着 OOD fraction 增大，相对收益依次提高至 +5.17%、+15.00%、+20.05% 和 +48.52%。该结果表明 FBG 的主要作用并非在所有条件下重复 electrical information，而是在 electrical observations 离开 source support 时提供逐渐增强的 complementary state information。

## Fig. 6 跨倍率未见工况泛化

**Fig. 6. RA-FBG-TCN 在 cross-rate unseen-profile protocol 下的五随机种子泛化性能。** (a) 1C→2C 方向六种 held-out driving profiles 的 MAE，柱高和误差棒分别表示五个随机种子的 mean 和 standard deviation；(b) 2C→1C 方向的对应结果；(c) 两个迁移方向及总体 MAE 的 seed-cluster bootstrap 95% confidence intervals。1C→2C 与 2C→1C 的 aggregate MAE 分别为 1.795% 和 0.806%，总体 MAE 为 1.301%，其 cluster-bootstrap 95% CI 为 0.961%–1.677%。结果显示从低倍率训练向高倍率测试的外推更困难，并呈现明显的方向不对称性。

## Fig. 7 波长噪声鲁棒性与 95% UQ

**Fig. 7. 双 FBG wavelength perturbation robustness 与 calibrated SOC uncertainty。** (a) 在 W1/W2 分别加入标准差为 0、0.5、1 和 2 pm 的 Gaussian wavelength noise 后，两个跨倍率方向的 MAE 变化；(b) 对应的 Q95 absolute error；(c) mixed-condition blocked test trajectory 上的参考 SOC、RA-FBG-TCN point prediction 与 95% residual split-conformal prediction interval。2 pm 噪声下性能仅发生平滑、小幅退化；95% nominal interval 的 PICP 为 95.04%，MPIW 为 2.075% SOC，表明后处理 conformal calibration 能在不改变点估计器的情况下提供与 nominal level 接近的 uncertainty coverage。
