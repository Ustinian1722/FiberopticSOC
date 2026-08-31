# 第4章 实验与结果（中文 V2）

## 4.1 实验设置与评价指标

为验证所构建电学–光学 SOC 估计框架的有效性，本文依次开展常规 blocked interpolation、光学表示与电学–光学互补性、跨倍率未见工况泛化、波长噪声鲁棒性以及不确定性量化实验。所有输入归一化参数均仅由相应训练数据计算，测试数据不参与特征缩放或模型选择。时序模型采用长度为 64 的因果滑动窗口，仅利用当前及历史观测估计窗口末端 SOC，从而避免未来信息泄漏。RA-FBG-TCN 的主要结构和训练配置见 Table 2。

模型采用 AdamW 优化器和 MSE loss 进行训练，并通过独立 validation data 的 MAE 保存最优 epoch。点预测性能采用平均绝对误差（MAE）、均方根误差（RMSE）、决定系数 R²、95% 分位绝对误差（Q95-AE）和最大绝对误差（MaxAE）进行评价。其中 MAE 与 RMSE 衡量整体估计精度，Q95-AE 和 MaxAE 用于反映误差尾部与极端偏差。对于不确定性量化，进一步采用 prediction interval coverage probability（PICP）、mean prediction interval width（MPIW）和 mean interval score（MIS）。

本文设置两类核心评价场景。第一类为 blocked mixed-condition interpolation，用于检验测试区域被训练分布充分覆盖时的基本 SOC estimation ability；第二类为更严格的 cross-rate unseen-profile protocol，即模型仅利用一个倍率下的五种完整动态工况进行训练，并直接预测另一倍率下从未参与训练的第六种驾驶工况。后者同时引入 C-rate shift 与 driving-profile shift，用于评价模型面对复合运行条件变化时的迁移能力。

## 4.2 常规工况下的 SOC 估计性能

首先在 blocked mixed-condition interpolation 场景下比较 CNN、GRU、LSTM、Transformer 以及不同输入形式的 TCN。所有模型采用相同的训练、验证、校准和测试数据，并使用一致的训练侧归一化与早停规则。结果如 Table 3 所示。

| Model | Params | MAE (%) | RMSE (%) | R² | Q95-AE (%) |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 11,497 | **0.231** | **0.296** | **0.999904** | **0.581** |
| VI+TF-TCN | 11,545 | 0.311 | 0.412 | 0.999814 | 0.825 |
| GRU | 10,801 | 0.426 | 0.549 | 0.999670 | 1.014 |
| RA-FBG-TCN | 11,545 | 0.482 | 0.593 | 0.999614 | 1.088 |
| LSTM | 14,129 | 0.548 | 0.708 | 0.999450 | 1.303 |
| CNN | 12,081 | 0.564 | 0.738 | 0.999403 | 1.414 |
| Transformer | 20,177 | 0.761 | 0.910 | 0.999090 | 1.584 |

各模型在常规插值场景下均获得较高的 SOC 估计精度，其中仅使用电压和电流的 VI-TCN 表现最佳。这说明当测试工况被训练数据充分覆盖时，端电压与电流已经能够提供非常充分的 SOC 判别信息，增加光学观测并不必然进一步降低同分布条件下的点预测误差。RA-FBG-TCN 仍获得 0.482% MAE、0.593% RMSE 和 0.999614 R²，并保持仅约 11.5k 的参数规模。

Fig. 4 进一步给出了 RA-FBG-TCN 的代表性测试轨迹及误差分布。模型预测能够整体跟随参考 SOC 的连续变化，并在快速动态片段中维持较低绝对误差。该实验用于确认最终轻量因果估计器具备稳定的基础点预测能力，而本文引入 FBG 的核心价值将在运行条件偏移场景下进一步检验。

## 4.3 运行条件偏移下的电学–光学互补性

为排除模型容量差异对 optical benefit 的影响，本文构造参数量严格匹配的 VI 与 VI+W 两个因果 TCN。VI 仅使用 V/I，VI+W 在相同网络结构与参数预算下加入 W1/W2。由此可以将性能差异主要归因于双 FBG observation 本身，而不是更大的网络容量。

在 1C→2C 跨倍率测试中，VI 模型平均 MAE 为 2.151%，加入原始双 FBG wavelength 后下降至 1.632%，相对降低约 24.1%。在 2C→1C 方向，MAE 由 0.866% 降至 0.814%，相对降低约 6.0%。Table 4 汇总了这一 parameter-matched comparison。可以看出，双 FBG 的贡献具有明显运行条件依赖性，并在向更高电流范围外推的 1C→2C 场景中更加突出。

为了进一步解释这一现象，本文仅利用 source-rate training current 的 0.5th–99.5th percentile 定义 electrical support envelope，并将每个测试窗口中电流超出该范围的样本比例定义为 electrical-OOD fraction。该指标完全由训练侧电流统计量确定，不使用测试 SOC 标签。Fig. 5(a) 给出了代表性 2C 测试电流相对于 1C training support 的位置，Fig. 5(b,c) 分别展示不同 OOD severity 下 VI/VI+W 的 MAE 及相对 optical gain。

结果显示，在 electrical ID 区域，VI 与 VI+W 的 MAE 分别为 0.735% 和 0.873%，额外光学输入并未带来收益；但随着 OOD fraction 增大，双 FBG 的相对收益从 +5.17% 逐步提高至 +15.00%、+20.05%，并在 75–100% OOD 区间达到 +48.52%。换言之，当 V/I 处于训练数据充分支持的区域时，端电观测已经足够有效；而随着测试电流越来越偏离训练支持域，W1/W2 所提供的独立内部状态信息逐渐变得更有价值。

这一结果说明，多模态 sensing 的贡献不必表现为在所有输入区域内持续优于单一模态。对于本文数据，FBG 更适合作为 electrical sensing 在 operating-condition shift 下的 complementary observation，而非在所有同分布样本上无条件提高点预测精度。

### 4.3.1 原始波长与物理解耦表示对比

双 FBG 原始 wavelength W1/W2 可以依据 sensitivity matrix 映射为 temperature/force 表示 T/F。由于两组变量对应相同的两个 optical sensing degrees of freedom，解耦过程提升的是物理语义，而不是测量信息维度。Section 2 的描述性统计显示，raw W1/W2 在 1C 和 2C 下的绝对 Pearson correlation 分别约为 0.738 和 0.655，而 T/F 对应约为 0.982 和 0.985，说明解耦坐标在该数据中具有更强的线性耦合。

为判断这种物理坐标是否同样有利于预测迁移，本文在完全一致的 training budget 和 1C→2C unseen-profile development protocol 下进行 matched comparison。使用 raw W1/W2 的紧凑 TCN 平均 MAE 为 1.385%、RMSE 为 2.084%、Q95-AE 为 4.873%；使用 decoupled T/F 的同规模 TCN 对应为 2.033%、3.060% 和 6.835%。采用更复杂 electrical–thermomechanical fusion 的 ETMF-TF 平均 MAE 为 1.569%，同样没有超过 raw wavelength model。相关结果见 Fig. 2(d) 与 Table 4。

因此，本文并不否定 T/F 解耦的物理解释价值，而是将 sensing interpretation 与 predictive representation choice 区分开来。对于本研究关注的 cross-condition SOC prediction，保留 native W1/W2 在 matched experiments 中表现出更稳定的平均迁移性能，因此最终 RA-FBG-TCN 固定使用 raw wavelength coordinates。

## 4.4 跨倍率未见工况泛化性能

在完成输入表示和模型设置后，本文采用严格 cross-rate unseen-profile protocol 对最终 RA-FBG-TCN 进行五随机种子验证。对于每个 held-out profile，训练集中不仅不存在目标倍率数据，而且完全排除了同名驾驶工况，从而同时考察 rate shift 与 profile shift。每个迁移方向包含六个 held-out profiles，并采用 5 个独立随机种子重复训练与测试。

Fig. 6(a,b) 给出了各 profile 的 five-seed mean±standard deviation。1C→2C 方向平均 MAE 为 1.795%、RMSE 为 3.037%、R² 为 0.985689、Q95-AE 为 6.804%；2C→1C 方向平均 MAE 为 0.806%、RMSE 为 0.988%、R² 为 0.998480、Q95-AE 为 1.830%。Table 5 汇总了两个方向的主要指标。

综合两个迁移方向，seed-cluster 平均 MAE 为 1.301% SOC。Fig. 6(c) 的 seed-cluster bootstrap 给出 overall 95% confidence interval 为 0.961%–1.677%。这表明在同时存在倍率变化和未见驾驶工况的条件下，最终模型仍保持具有实用意义的 SOC estimation accuracy。

两个迁移方向呈现明显不对称性。2C training data 包含更宽的高电流激励范围，因此向 1C 迁移时，大部分测试电流仍处于训练支持范围内；相比之下，1C→2C 需要模型处理训练阶段未充分出现的高电流区域，外推难度和 seed-to-seed variability 均更高。这一结果与 Fig. 5 的 electrical-OOD analysis 形成一致证据链。

## 4.5 FBG 测量噪声鲁棒性与不确定性量化

### 4.5.1 波长噪声鲁棒性

实际 FBG interrogation 不可避免地存在小幅测量扰动。为评估模型对直接 wavelength noise 的敏感性，本文分别向 W1 和 W2 独立加入标准差为 0.5、1 和 2 pm 的零均值 Gaussian noise，并保持冻结模型及训练侧归一化参数不变。Fig. 7(a,b) 展示了不同噪声水平下 MAE 与 Q95-AE 的变化。

随着 wavelength noise 增大，两个迁移方向的误差均呈平缓变化，没有出现明显性能突变。当噪声标准差达到 2 pm 时，1C→2C 方向 MAE 相对于 clean baseline 仅增加约 1.67%，2C→1C 方向相对增加约 4.57%。结果说明，直接使用 raw W1/W2 并未使最终模型对小幅光学测量扰动产生明显放大效应。

### 4.5.2 95% residual conformal prediction interval

除点预测精度外，本文进一步采用 residual split conformal prediction 表征 SOC estimation uncertainty。模型训练和 early stopping 完成后，利用独立 calibration data 计算 absolute residual distribution，并据此获得 95% finite-sample conformal residual quantile。该过程不重新训练点预测模型，也不使用 test labels 调整 interval width。

在 blocked T1 test set 上，95% nominal prediction interval 的实际 PICP 为 95.04%，与目标 coverage 基本一致；MPIW 为 2.075% SOC，MIS 为 0.02441，对应 residual quantile 约为 1.089% SOC。Fig. 7(c) 给出了代表性测试轨迹上的 point prediction 与 95% prediction interval。

上述结果表明，简单的 post-hoc conformal calibration 即可在不引入额外概率网络的情况下，为 SOC point estimate 提供具有明确 coverage 含义的 uncertainty interval。综合 conventional accuracy、electrical-OOD complementarity、strict cross-condition generalization、wavelength-noise robustness 和 conformal UQ，可以看出本文方法的主要价值并非在所有同分布区域追求最低单点误差，而是在保持紧凑模型和高基础精度的同时，为运行条件变化提供额外的 optical state information 与可靠性表征。