# 第5章 Discussion（中文 V2）

## 5.1 物理解耦表示与数据驱动预测表示的差异

双 FBG system 的重要特点之一，是两个原始 Bragg wavelength observations 可以依据不同传感灵敏度进一步转换为 temperature 和 force-related variables。从 sensing interpretation 角度看，该过程能够为波长变化赋予更直接的物理语义，因此适合用于理解电芯内部 thermal–mechanical evolution。然而，本研究结果表明，物理可解释性并不必然对应更优的数据驱动预测表示。

W1/W2 与 T/F 描述的是相同的两个 optical sensing degrees of freedom，二者可以通过 sensitivity matrix 相互转换。因此，T/F 并没有引入额外的测量信息，而是对原始光学空间重新参数化。Fig. 2 显示，raw W1/W2 在 1C 和 2C 下的绝对 Pearson correlation 约为 0.738 和 0.655，而 T/F 对应约为 0.982 和 0.985，说明解耦后的两个变量在该数据中呈现更强的线性耦合。与此同时，矩阵反演还会改变变量尺度以及测量扰动在特征空间中的传播方式。

因此，本文区分两个不同目标：sensing interpretation 关注变量是否具有明确物理含义，而 predictive modeling 更关注某种表示能否在未知运行条件下保持稳定、易学习的状态信息。Matched transfer experiments 中，直接采用 W1/W2 的 compact causal TCN 获得更低的平均 MAE、RMSE 和 Q95-AE，说明对于本文的 cross-condition SOC task，保留 native sensing coordinates 并由网络直接学习耦合关系是一种更有效的工程选择。该结论并不否定 T/F 解耦在传感机理研究中的价值，而是说明 physical representation 与 predictive representation 可以具有不同的最优选择。

## 5.2 FBG 信息的价值为何主要出现在分布偏移条件下

常规 blocked interpolation 实验中，VI-TCN 获得最低点预测误差，而加入双 FBG 后并未进一步降低 MAE。这说明在测试样本与训练样本具有较高分布重叠时，端电压与电流已经能够为 SOC 提供非常充分的判别信息。在这种情况下，额外 optical observation 中部分状态信息与 electrical channels 存在冗余，因而多模态输入不一定转化为更低的同分布点预测误差。

当运行条件发生变化时，情况明显不同。1C→2C 测试中，大量高电流状态超出 1C training current support，pure V/I model 需要对训练阶段未充分出现的外部激励进行外推。相比之下，FBG wavelength 同时受到电芯内部温度、结构变形以及随 SOC 演化的耦合响应影响，其动态规律并不完全等同于端电流和端电压。因此，当 electrical observations 离开训练支持域时，optical channels 可以提供额外状态约束。

Fig. 5 对这一解释给出了直接证据。在 electrical ID region，加入 W1/W2 的相对收益为负；但随着 window-level OOD fraction 增大，optical gain 由约 +5.2% 逐步增加至 +15.0%、+20.1%，并在最严重 OOD 区间达到约 +48.5%。这说明 multimodal sensing 的价值并不需要表现为“在所有区域持续优于单一模态”，而更可能体现在主 sensing modality 信息支持不足时提供 complementary observation。

从 BMS application 角度看，这一结果具有较明确的工程含义。实际车辆运行状态难以由有限离线训练数据完全覆盖，尤其在倍率变化、高功率动态负载或其他超出历史数据范围的场景中，electrical distribution shift 很难完全避免。因此，本文更强调 FBG-assisted robustness under operating-condition shift，而不是将双 FBG 简单理解为始终能够提高 V/I baseline 的附加输入。

## 5.3 跨倍率迁移的不对称性及可靠性

严格 cross-rate unseen-profile experiment 显示，2C→1C 的预测性能明显优于 1C→2C。该方向性差异与训练数据对测试电流范围的覆盖程度一致：2C 数据具有更宽的高电流幅值和更强的动态激励，由 2C 迁移到 1C 时，大部分低倍率电流状态仍处于 training support 内；相反，1C→2C 需要处理训练阶段未充分出现的高电流与快速变化区域，对外推能力提出更高要求。

这一解释与 electrical-OOD analysis 形成一致证据链。最终 five-seed T4 中，1C→2C 平均 MAE 为 1.795%，而 2C→1C 为 0.806%；两个方向综合后的 seed-cluster MAE 为 1.301%，bootstrap 95% CI 为 0.961%–1.677%。因此，cross-condition error 的方向性并非仅由随机训练波动造成，而与 source-domain coverage 的非对称性密切相关。

测量扰动与 uncertainty experiment 进一步补充了可靠性证据。在 W1/W2 上独立施加最高 2 pm Gaussian wavelength noise 后，两个迁移方向的 MAE 仅小幅变化，说明使用 native wavelength coordinates 并未造成明显的噪声放大问题。同时，95% residual split-conformal interval 获得 95.04% empirical coverage 和 2.075% SOC 的平均宽度，使 point estimate 能够同时配合具有明确 coverage 含义的 uncertainty interval。对于轻量在线估计器而言，这种 point prediction 与 post-hoc calibration 的解耦能够在较低额外复杂度下提供更完整的可靠性信息。

## 5.4 研究范围与后续工作

本文主要关注固定 dual-FBG sensing configuration 下的 operating-condition transfer，因此当前结论首先适用于相同传感器安装与标定体系中的 profile/rate shift。不同 physical cells 之间可能存在 FBG 初始波长、bonding state、strain-transfer efficiency 和 sensitivity variation，这些因素会改变 optical signals 的绝对尺度和动态响应。因此，本文不将当前结果外推为未经校准的 universal cross-cell generalization。

后续研究可进一步结合 sensor-specific calibration、多电芯长期测试、宽温度环境以及 sensor ageing，研究 optical representation 在跨电芯和传感器状态变化条件下的迁移机制，并探索面向实际 BMS deployment 的在线校准与自适应策略。相比继续增加网络结构复杂度，这类 sensing-calibration 与 domain-transfer 问题可能更直接决定 fiber-optic-assisted SOC estimation 的工程可推广性。

总体而言，本研究表明，FBG sensing 的价值不应仅以同分布条件下的单一精度增益衡量。对于存在明显 operating-condition shift 的 SOC estimation problem，optical information 能够在 electrical observations 超出 training support 时提供有意义的补充约束；与此同时，保留 native wavelength representation 并采用 compact causal temporal model，可以在 representation simplicity、cross-condition accuracy、measurement robustness 与 uncertainty reporting 之间形成较为完整的工程折中。