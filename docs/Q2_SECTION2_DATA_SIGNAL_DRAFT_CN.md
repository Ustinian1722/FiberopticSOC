# 第2章 Dataset and electrical–optical signal analysis（中文初稿）

## 2.1 数据集与测试工况

本文采用公开的 SiC-18 植入式双 FBG 锂离子电池数据开展 SOC 估计研究。测试对象为一只约 2.5 Ah 的 SiOx/C 软包电芯，实验过程中在电芯内部布置 Armored FBG 与 Bare FBG，用于同步采集充放电过程中与内部热–力学状态相关的 Bragg wavelength response。除双 FBG 信号外，数据集同时记录端电压、电流以及参考 SOC，为研究电学–光学联合状态估计提供了同步观测基础。

数据覆盖 HWFET、LA92、NEDC、NYCC、US06 和 WLTC 六种典型动态驾驶工况，并分别在 1C 与 2C 两种放电倍率下测试，共形成 12 条主要动态放电轨迹，总计约 68,086 个有效采样点。不同驾驶循环包含显著不同的电流幅值、脉冲频率和负载变化速率，因此能够用于考察模型在复杂动态负载及倍率变化条件下的 SOC estimation ability。

本文使用的主要预测变量包括 Voltage V、Current I、Wavelength 1（W1）和 Wavelength 2（W2）。SOC 作为监督学习目标，不作为任何模型输入。为避免通过放电进程直接重构标签，累计放电容量以及绝对时间等与轨迹进程高度相关的变量均不输入估计器。对于所有训练/验证/测试划分，数据标准化参数仅由相应训练集确定。

建议 Table 1 汇总如下信息：

| Item | Description |
|---|---|
| Cell | SiOx/C pouch cell, approximately 2.5 Ah |
| Optical sensing | Implanted dual FBG (Armored FBG + Bare FBG) |
| Dynamic profiles | HWFET, LA92, NEDC, NYCC, US06, WLTC |
| Discharge rates | 1C and 2C |
| Main trajectories | 12 |
| Main predictors | Voltage, Current, W1, W2 |
| Target | SOC |
| Total released rows | approximately 68,086 |

为直观展示数据特征，Fig. 1 建议选择一组代表性的 1C 和 2C 动态工况，绘制 SOC、电压、电流、W1 和 W2 的同步变化曲线。由原始序列可以观察到，电流随驾驶循环呈现快速且频繁的动态波动，端电压在整体随 SOC 下降的同时叠加明显的瞬时负载响应；相比之下，双 FBG wavelength signal 通常表现出更平缓的状态演化趋势，并对负载变化产生不同于端电信号的动态响应。这种响应差异为后续利用光学观测补充电学信息提供了数据基础。

## 2.2 双 FBG 感知原理与热–力学解耦

Fiber Bragg Grating 的中心 Bragg wavelength 可表示为

λ_B = 2 n_eff Λ，

其中 n_eff 为光纤有效折射率，Λ 为光栅周期。当 FBG 所处环境的温度和应变状态发生变化时，n_eff 与 Λ 随之改变，从而引起 Bragg wavelength shift。因此，FBG 可以通过波长变化对局部温度和机械变形进行原位感知。

对于本文使用的双 FBG system，两个传感器具有不同的温度和力学响应系数。数据中双波长信号与解耦后的温度 T 和 deformation force F 近似满足

W1 = 0.0208 T + 0.00054 F，

W2 = 0.0254 T + 0.00085 F。

写成矩阵形式为

[W1, W2]^T = K [T, F]^T，

其中

K = [[0.0208, 0.00054],
     [0.0254, 0.00085]]。

在 sensitivity matrix K 可逆的条件下，温度和力学响应可由双波长反演得到。对应的逆变换约为

T = 214.43 W1 - 136.23 W2，

F = -6407.67 W1 + 5247.23 W2。

上述关系说明，W1/W2 与 T/F 并不是四个相互独立的 sensing channels，而是同一双 FBG 测量自由度在不同坐标系下的表达。T/F 具有更直接的物理含义，便于分析电芯内部 thermal and mechanical evolution；W1/W2 则保持了光学解调器直接输出的原始测量空间。传统传感分析通常更偏向采用解耦后的物理量，但对于 data-driven SOC estimation，需要进一步判断这种物理坐标转换是否同样有利于跨运行条件学习。

## 2.3 电学–光学信号特征与表示分析

### 2.3.1 动态负载下的信号响应差异

电池在动态驾驶循环下的端电压同时受到 SOC、欧姆压降、极化以及瞬时电流变化影响。因此，即使 SOC 变化连续缓慢，Voltage curve 仍会随电流脉冲发生明显快速波动。对 SiC-18 数据进行描述性分析可以发现，在去除平滑 SOC trend 后，Voltage residual 与 instantaneous current 保持较强关联，而 FBG 派生的热–力学响应与瞬时电流的关联明显较弱。这表明光学感知包含了不同于端电压瞬态响应的内部状态信息。

进一步比较 1C 与 2C 条件下的信号变化可发现，Current 的 rate shift 最为明显，Voltage 也受到倍率变化影响；内部力学相关响应的 normalized rate shift 相对较小。该结果说明，在跨倍率条件下，电学变量分布变化可能显著高于部分内部 sensing variables。虽然本文最终不直接使用解耦 force 作为主模型输入，但这一现象从物理角度支持了 FBG information 作为 cross-condition complementary observation 的合理性。

建议 Fig. 2(a) 绘制同一 profile 在 1C/2C 下的 Voltage–SOC 与 W1/W2–SOC 对比曲线，使倍率变化对 electrical and optical response 的差异直观可见。

### 2.3.2 原始 W 与解耦 T/F 的相关性和数值结构

尽管 W1/W2 与 T/F 信息等价，但两种坐标的统计性质存在明显差异。在 source-domain 数据中，原始 W1/W2 的 Pearson correlation 大致处于 0.72–0.76，其 covariance condition number 约为 6.1–7.2；相比之下，解耦后的 T/F correlation 约为 −0.982，condition number 可达到约 107–119。这说明 T/F 虽然具有更直接的物理解释，但两个解耦变量在该数据中呈现更强的线性耦合和更不均衡的数值几何。

从 sensitivity matrix 本身也可以得到类似结论。K 的两个 sensitivity directions 较为接近，矩阵求逆会使 wavelength perturbation 在解耦空间中得到一定程度的放大。因此，显式 thermo-mechanical decoupling 更适合被理解为一种物理解释坐标，而不应预先假设为 data-driven model 的最优 predictive representation。

建议 Fig. 2(b–d) 采用三部分组合形式：

1. W1–W2 与 T–F 二维散点/密度图；
2. 两类 representation 的 correlation matrix；
3. covariance condition number 或 normalized representation comparison bar chart。

该图不需要展示全部开发实验，只用于说明为什么本文有必要研究 representation choice。

### 2.3.3 Representation-aware predictive comparison

为确定最终 optical input，本文在保持相同因果 TCN、训练预算和跨倍率未见工况划分的前提下，对不同 optical representations 进行 matched comparison。六个 1C→2C unseen-profile splits 的平均结果表明，采用原始 W1/W2 的模型 MAE 为 1.385%，RMSE 为 2.084%，Q95-AE 为 4.873%；采用 T/F 的同规模 TCN 对应 MAE 为 2.033%，RMSE 为 3.060%，Q95-AE 为 6.835%。即使采用更复杂的 electrical–thermomechanical fusion network，平均 MAE 仍约为 1.569%，没有超过直接使用原始 wavelength 的紧凑 TCN。

上述结果说明，对于本文重点关注的 cross-condition SOC estimation，增加物理解释层并不必然带来更好的 transferability。原始 W1/W2 在保持传感信息完整的同时，避免了显式反演造成的表示几何变化，并在 matched experiments 中获得更稳定的平均泛化性能。因此，后续模型固定采用 Voltage、Current、W1 和 W2 作为输入，而 T/F 主要用于 sensing mechanism interpretation 和 representation ablation。

综上，双 FBG sensing 在本文中的定位不是将温度、力学和 wavelength 重复堆叠为多个特征，而是提供一组与传统端电响应不同的内部 optical observation。第二章的信号分析表明，这些观测具有明确的热–力耦合来源，同时其 predictive value 与 representation choice 密切相关。基于上述分析，下一章进一步构建直接面向原始双 FBG wavelength 的 lightweight causal SOC estimation framework。