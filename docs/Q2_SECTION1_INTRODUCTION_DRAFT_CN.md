# 第1章 Introduction（中文初稿，投稿收敛版）

随着新能源汽车和储能系统快速发展，锂离子电池的安全、高效运行对 battery management system（BMS）提出了更高要求。State of charge（SOC）作为反映电池剩余可用电量的关键状态变量，是能量管理、功率分配、充放电控制及安全保护的重要依据。然而，SOC 无法通过传感器直接测量，其估计过程容易受到负载变化、温度波动、电池非线性以及工作条件差异等因素影响。尤其在动态驾驶工况下，电压和电流具有明显的瞬态波动与极化效应，使训练工况之外的 SOC estimation 仍面临较大挑战。

现有 SOC estimation methods 主要包括基于等效电路或电化学模型的 observer/filtering methods，以及直接学习测量数据与 SOC 非线性映射的 data-driven methods。近年来，CNN、LSTM、GRU、TCN 和 Transformer 等深度时序模型被广泛用于复杂动态工况下的 SOC estimation，并在多温度、多驾驶循环等任务中取得较高精度。最新研究开始进一步关注跨温度、跨材料和跨驾驶工况的 transferability，而近期综述也指出，数据驱动 SOC estimation 的关键问题正逐渐从单一数据集上的精度比较转向更加标准化的评价协议、未知运行条件适应以及可靠性分析。因此，仅在训练分布附近获得较低 MAE 或 RMSE 已不足以完整反映模型在实际 BMS 场景中的应用能力。

除传统 electrical measurements 外，battery mechanical response 为 SOC observation 提供了另一条重要途径。锂离子在电极材料中的嵌入与脱嵌会引起电极体积变化、结构应变和内部应力演化，这些 mechanical signals 与 lithium content 及 SOC 具有直接联系。已有研究利用表面压力、膨胀位移或机械应力辅助 SOC estimation，并证明机械观测能够在动态负载条件下提供不同于端电压和电流的状态信息。Fiber Bragg Grating（FBG）具有体积小、抗电磁干扰、可嵌入以及高灵敏度等特点，可通过 Bragg wavelength shift 原位感知电池内部或表面的热–力学变化，因此近年来逐渐成为多物理场 battery state sensing 的重要技术路线。

针对 FBG-assisted SOC estimation，近年来已经出现多种 sensing–learning frameworks。例如，有研究将双 FBG 植入软包电池内部，通过 wavelength decoupling 同时获得 internal strain 和 temperature，并结合 CNN–Transformer 实现多参数 SOC estimation；另有工作利用植入式 FBG 获取 silicon-based lithium-ion battery 的原位 thermo-mechanical information，并结合 feature engineering、noise augmentation 及 CNN–GRU–Attention 模型，在六种动态工况和两种倍率数据上实现高精度 SOC estimation。面向 battery pack，distributed optical strain sensing 也已与 adaptive state estimator 相结合，用于降低 cell heterogeneity 对 pack SOC estimation 的影响。这些成果表明，FBG 已经能够为 SOC estimation 提供有效的内部状态观测，因此本文的研究重点并非简单证明“FBG 可以用于 SOC estimation”，而是进一步研究其在运行条件迁移时应如何被数据驱动模型有效利用。

现有研究仍存在两个值得进一步讨论的问题。第一，多数工作主要通过多工况数据联合训练、随机或常规划分以及模型结构优化验证 estimation accuracy，而当 discharge rate 和 driving profile 同时发生变化时，模型需要面对更明显的 compound distribution shift。特别是在 source-rate training data 无法覆盖 target-rate 的电流幅值和负载动态时，仅依赖端电观测可能产生较大的外推误差。因此，需要在完全排除目标 driving profile 的条件下进一步评价 electrical–optical sensing 对跨倍率泛化的作用。第二，双 FBG system 通常先根据 sensitivity matrix 将两个 measured wavelengths 解耦为 temperature 和 strain/force，以增强物理可解释性。然而，raw wavelengths 与 decoupled variables 本质上来源于相同的两个 optical sensing degrees of freedom。物理上更易解释的坐标是否同样更适合 data-driven transfer learning，尚缺少系统的 matched comparison。对于条件数较高的 sensitivity inversion，显式解耦还可能改变噪声传播和特征空间的数值几何，从而影响模型在未知工况下的稳定性。

基于上述问题，本文提出一种 representation-aware dual-FBG SOC estimation framework，记为 RA-FBG-TCN。该框架首先从 dual-FBG sensing mechanism 出发，对 raw wavelength representation 与 thermo-mechanical decoupled representation 进行统一分析，并在相同模型和训练预算下比较其 cross-condition predictive transferability；随后直接采用 Voltage、Current、Wavelength 1 和 Wavelength 2 构造 lightweight causal TCN，实现仅依赖当前及历史信息的在线 SOC estimation。与将复杂网络结构作为主要创新不同，本文更关注 optical information 在 electrical distribution shift 下的实际互补价值，并通过 training-current support 定义 label-free electrical-OOD severity，分析 FBG gain 随运行条件偏移程度的变化。最后，在冻结模型后采用 residual split conformal prediction 构造 95% SOC prediction interval，以补充点预测结果的可靠性表达。Conformal prediction 近年来已被用于 battery SOC uncertainty quantification，其无需预设误差分布即可为 prediction interval 提供有限样本覆盖率含义，因此适合作为轻量 point estimator 的后验 reliability layer。

本文的主要贡献如下：

(1) 构建一种面向动态工况的 electrical–optical SOC estimation framework，通过 Voltage/Current 与 implanted dual-FBG wavelength observations 的联合建模，为传统 electrical sensing 提供内部状态补充信息；同时从 sensing mechanism 和数据统计两个层面分析 W1/W2 与 T/F 两类 representation 的关系。

(2) 针对 raw wavelength 与 thermo-mechanical decoupled representation 开展 matched comparison。结果表明，尽管 T/F 具有更直接的物理语义，直接 W1/W2 在 retained lightweight causal estimator 中表现出更好的 cross-condition transferability，因此本文将 raw dual-FBG observations 作为最终 optical inputs。

(3) 建立严格的 cross-rate + unseen-profile evaluation：模型仅使用一个倍率下五种完整 driving profiles 训练，并直接预测另一倍率下完全未见的第六种 profile；在两个迁移方向和五个随机种子下系统评价模型泛化性能。同时，通过 electrical-OOD 分层分析发现，FBG information 的相对收益随 electrical support shift 增大而显著提高，揭示了 optical sensing 在困难运行条件下的互补作用。

(4) 从 measurement robustness 和 predictive uncertainty 两方面评价方法可靠性。通过直接向 W1/W2 注入 pm-scale wavelength perturbation 验证光学测量噪声鲁棒性，并利用独立 calibration data 构建 95% residual conformal interval，使最终框架同时输出 SOC point estimate 与具有明确 empirical coverage 的 uncertainty interval。

本文其余部分组织如下。第 2 章介绍 SiC-18 dual-FBG battery dataset、双 FBG sensing principle 以及 electrical–optical signal/representation characteristics；第 3 章介绍 RA-FBG-TCN 及 residual conformal uncertainty quantification；第 4 章给出 conventional estimation、electrical-OOD complementarity、cross-rate unseen-profile generalization、wavelength-noise robustness 和 uncertainty experiments；第 5 章讨论 raw/decoupled representation、optical sensing under distribution shift 以及方法适用范围；最后总结本文主要结论并展望后续研究。

---

## 建议在正式英文稿中对应引用的近期文献

[1] Yao J., Kowal J. Towards a smarter battery management system: A critical review on deep learning-based state of charge estimation of lithium-ion batteries. Energy and AI, 2025, 21: 100585. DOI: 10.1016/j.egyai.2025.100585.

[2] Data-driven SOC estimation method for power batteries under driving cycle conditions and a wide temperature range. Energy, 2025, Article 139147. DOI: 10.1016/j.energy.2025.139147.

[3] Mechanical stress-based state-of-charge estimation for lithium-ion batteries via deep learning techniques. Energy, 2025, 326: 136216. DOI: 10.1016/j.energy.2025.136216.

[4] Estimation of state-of-charge for lithium-ion batteries based on simultaneous internal strain and temperature monitoring by fiber optic sensors. Journal of Energy Storage, 2025, 133: 117969. DOI: 10.1016/j.est.2025.117969.

[5] Chen L. et al. In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries. Energy, 2026, 349: 140609. DOI: 10.1016/j.energy.2026.140609.

[6] Adaptive estimation of battery pack state of charge with optical fibre strain measurements. Applied Energy, 2026, 407: 127330. DOI: 10.1016/j.apenergy.2025.127330.

[7] Enhancing reliability in electrified transportation: A conformalized quantile regression framework for battery state-of-charge uncertainty quantification. Journal of Power Sources, 2026, 666: 239123. DOI: 10.1016/j.jpowsour.2025.239123.

正式投稿时再结合原稿 reference manager 补充经典 SOC/filtering/TCN/FBG mechanism references，并统一编号。