# 第2章 Dataset and electrical–optical signal analysis（中文 V2）

## 2.1 数据集与测试工况

本文采用公开的 SiC-18 植入式双 FBG 锂离子电池数据开展 SOC 估计研究。测试对象为一只约 2.5 Ah 的 SiOx/C 软包电芯，实验过程中在电芯内部布置 Armored FBG 与 Bare FBG，用于同步采集充放电过程中与内部热–力学状态相关的 Bragg wavelength response。除双 FBG 信号外，数据集同时记录端电压、电流以及参考 SOC，为研究电学–光学联合状态估计提供了同步观测基础。

数据覆盖 HWFET、LA92、NEDC、NYCC、US06 和 WLTC 六种典型动态驾驶工况，并分别在 1C 与 2C 两种放电倍率下测试，共形成 12 条主要动态放电轨迹，总计约 68,086 个有效采样点。不同驾驶循环具有明显不同的电流幅值、脉冲频率和负载变化速率，因此能够用于考察模型在动态负载、倍率变化及未见驾驶工况下的 SOC 估计能力。数据集与主要实验条件汇总于 Table 1。

本文使用的主要预测变量包括端电压 V、电流 I、Wavelength 1（W1）和 Wavelength 2（W2）。SOC 仅作为监督学习目标，不作为任何模型输入。为避免利用放电进程直接重构标签，累计放电容量以及绝对时间等与轨迹进程高度相关的变量均不输入估计器。对于所有训练、验证、校准和测试划分，标准化参数均仅由相应训练集计算。

Fig. 1 展示了代表性动态放电过程中的电学–光学同步观测。电流随驾驶循环呈现快速且频繁的负载波动，端电学状态与 SOC 演化同时受到倍率与瞬时激励影响；相比之下，双 FBG wavelength signal 呈现更平滑但具有独立动态特征的变化过程。以 NEDC 为例，W1 与 W2 均随放电过程发生显著波长漂移，但两路响应幅值和局部形态并不完全一致；在 1C 与 2C 下，相同工况的光学响应也表现出倍率相关差异。这说明端电学变量与双 FBG 光学观测虽然同步，却并非简单重复同一类动态信息，为后续分析 optical information 在运行条件偏移下的互补作用提供了数据基础。

## 2.2 双 FBG 感知原理与热–力学解耦

Fiber Bragg Grating 的中心 Bragg wavelength 可表示为

λ_B = 2 n_eff Λ，

其中 n_eff 为光纤有效折射率，Λ 为光栅周期。当 FBG 所处环境的温度和应变状态发生变化时，n_eff 与 Λ 随之改变，从而引起 Bragg wavelength shift。因此，FBG 可以通过波长变化对局部温度与机械变形进行原位感知。

对于本文使用的双 FBG system，两个传感器具有不同的温度和力学响应系数。数据中双波长信号与解耦后的温度 T 和 deformation force F 近似满足

W1 = 0.0208 T + 0.00054 F，

W2 = 0.0254 T + 0.00085 F。

写成矩阵形式为

[W1, W2]^T = K [T, F]^T，

其中

K = [[0.0208, 0.00054],
     [0.0254, 0.00085]]。

在 sensitivity matrix K 可逆的条件下，可由双波长观测反演温度与力学响应，对应近似逆变换为

T = 214.43 W1 - 136.23 W2，

F = -6407.67 W1 + 5247.23 W2。

由此可见，W1/W2 与 T/F 并不是四个相互独立的 sensing channels，而是同一双 FBG 测量自由度在不同坐标系下的表达。T/F 具有更直接的物理含义，适合用于分析电芯内部 thermal–mechanical evolution；W1/W2 则保留了光学解调器直接输出的原始测量坐标。对于 data-driven SOC estimation，物理可解释坐标是否同时也是更适合跨运行条件预测的表示，需要通过统一实验进一步判断。

## 2.3 电学–光学信号特征与表示分析

### 2.3.1 动态负载下的信号响应差异

电池在动态驾驶循环下的端电压同时受到 SOC、欧姆压降、极化以及瞬时电流变化影响。因此，即使 SOC 连续缓慢变化，端电学响应仍会随电流脉冲发生快速波动。相比之下，植入式 FBG 对内部温度、结构变形及其耦合演化产生响应，其动态时间尺度与端电信号并不完全一致。Fig. 1 中的同步序列能够直观观察到这种差异：电流呈高频脉冲变化，而双 wavelength channels 在整体随放电演化的同时叠加相对平滑的局部响应。

倍率变化进一步放大了两类观测之间的差异。由 1C 转向 2C 后，电流幅值和端电响应分布发生直接变化，而 FBG wavelength 仍受到内部热–力状态累积效应共同影响。因此，本文并不将 FBG 视为对 V/I 的简单重复测量，而将其视为在 operating-condition shift 下可能具有互补价值的内部 optical observation。该假设将在 Section 4 的 parameter-matched electrical-OOD analysis 中进一步验证。

### 2.3.2 原始 W 与解耦 T/F 的统计结构

尽管 W1/W2 与 T/F 在信息维度上等价，两种坐标的统计结构并不相同。Fig. 2(a,b) 展示了全部 12 条动态轨迹在两种表示空间中的联合分布。按倍率分别计算直接描述性统计后，原始 W1/W2 的绝对 Pearson correlation 在 1C 和 2C 下分别约为 0.738 和 0.655，而解耦 T/F 对应约为 0.982 和 0.985，如 Fig. 2(c) 所示。由此可见，经过物理解耦后，两个变量在该数据中呈现出明显更强的线性耦合。

这一现象并不意味着 T/F 解耦缺乏物理意义。相反，T/F 将两个原始 wavelength channels 映射到具有明确温度与力学语义的坐标，更便于开展 sensing-mechanism interpretation。然而，线性反演同时改变了变量尺度、相关结构以及测量扰动在特征空间中的传播方式。因此，对于以跨倍率和未见工况为重点的端到端 SOC 学习任务，不应预先假定物理语义更强的坐标必然具有更好的 predictive transferability。

### 2.3.3 Representation-aware predictive comparison

为确定最终 optical input，本文在保持相同因果 TCN、训练预算和跨倍率未见工况划分的条件下，对不同 optical representations 进行 matched comparison。Fig. 2(d) 和 Table 4 给出了主要结果。六个 1C→2C unseen-profile development splits 中，采用原始 W1/W2 的紧凑 TCN 平均 MAE 为 1.385%，RMSE 为 2.084%，Q95-AE 为 4.873%；采用 T/F 的同规模 TCN 对应 MAE 为 2.033%，RMSE 为 3.060%，Q95-AE 为 6.835%。采用更复杂 electrical–thermomechanical fusion 的 ETMF-TF 平均 MAE 为 1.569%，同样没有超过直接使用原始 wavelength 的紧凑模型。

上述结果表明，对于本文重点关注的 cross-condition SOC estimation，增加物理解释层并不必然带来更好的预测迁移。原始 W1/W2 在保持双 FBG 测量信息完整的同时，避免了额外坐标反演所引入的表示几何变化，并在 matched experiments 中取得更稳定的平均泛化性能。因此，后续模型固定采用 V、I、W1 和 W2 作为输入，而 T/F 主要用于 sensing mechanism interpretation 与 representation ablation。

综上，双 FBG sensing 在本文中的定位并不是将 wavelength、temperature 和 force 重复堆叠为多个输入特征，而是提供一组与传统端电响应具有不同物理来源的内部 optical observations。基于上述数据与表示分析，下一章进一步构建直接面向原始双 FBG wavelength 的 lightweight causal SOC estimation framework。