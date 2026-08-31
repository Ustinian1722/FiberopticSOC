# 第3章 Methodology（中文初稿，代码一致版）

## 3.1 RA-FBG-TCN 总体框架

针对动态工况及倍率变化条件下的电池 SOC 估计问题，本文构建一种 Representation-Aware Dual-FBG Temporal Convolutional Network（RA-FBG-TCN）框架。该方法并不将双 FBG 解耦后的温度和力学量作为额外独立信息源，而是首先比较原始波长空间与物理解耦空间在跨工况任务中的预测适用性，并在此基础上直接采用电压、电流和双 FBG 原始波长构成联合输入。随后利用轻量因果时序卷积网络提取短期负载变化与内部光学响应之间的动态关系，最后采用 residual split conformal prediction 对点预测结果进行区间校准。

对于时刻 t，输入向量定义为

x_t = [V_t, I_t, W1_t, W2_t]

其中 V_t 和 I_t 分别表示端电压与电流，W1_t 和 W2_t 表示两个 FBG 通道的 Bragg wavelength observations。模型以长度为 L=64 的历史窗口

X_t = [x_(t-L+1), ..., x_t]

作为输入，输出当前时刻 SOC 估计值 yhat_t。窗口中仅包含当前及历史数据，因此该建模过程可以满足在线 SOC estimation 对因果性的要求。

整个方法由四个步骤组成：首先仅利用训练数据计算各输入通道的均值和标准差，并完成标准化；其次构造长度为 64 的因果序列窗口；随后通过多层 dilated causal convolution 提取 electrical–optical temporal representation；最后通过非线性回归头获得 SOC 点估计，并使用独立 calibration data 构造 95% conformal prediction interval。整体流程可在 Fig. 3 中表示为

V/I/W1/W2 → train-only normalization → causal windowing → dilated TCN encoder → SOC regression → 95% conformal interval。

## 3.2 原始双 FBG 表示选择

双 FBG sensing system 中，原始波长变化可以表示为温度和力学响应的线性组合。令

Δλ = [Δλ1, Δλ2]^T

z = [T, F]^T

则两者之间可以写为

Δλ = K z

其中 K 为由两个 FBG 温度和力学灵敏度系数组成的 2×2 sensitivity matrix。当 K 可逆时，可进一步得到

z = K^(-1) Δλ。

因此，W1/W2 与由其计算得到的 T/F 本质上是同一双通道光学观测的两种坐标表达。显式解耦能够增强变量的物理语义，但并不会增加新的 sensing degrees of freedom。与此同时，矩阵求逆会改变特征的相关结构、尺度和数值条件，这可能进一步影响深度时序模型在未见工况下的特征迁移能力。

基于上述考虑，本文将 optical representation selection 作为模型设计的一部分，而不是预先假定物理解耦表示必然优于原始测量。具体而言，在相同数据划分、模型规模和训练预算下，对 V/I+W1/W2 与 V/I+T/F 进行 matched comparison。结果表明，直接 wavelength representation 在 cross-rate unseen-profile task 中具有更低的平均 MAE、RMSE 和 Q95-AE，因此最终模型固定使用 W1/W2。需要强调的是，这一选择针对的是 SOC predictive transferability，并不否定 T/F 解耦在传感机理解释中的价值。

## 3.3 因果时序卷积 SOC 估计器

### 3.3.1 Train-only normalization

由于电压、电流和 FBG wavelength 具有不同量纲和数值尺度，首先根据训练数据计算每一输入通道的均值 μ_j 和标准差 σ_j，并进行 z-score normalization：

x'_(t,j) = (x_(t,j) - μ_j) / σ_j。

对于验证、校准和测试数据，均固定使用训练数据得到的 μ_j 和 σ_j，不重新估计测试域统计量，从而避免目标域信息进入模型预处理过程。

### 3.3.2 Causal convolution

为保证模型仅依赖当前及历史观测，本文采用 left-padded causal convolution。对于第 l 层、dilation 为 d_l、kernel size 为 k 的一维卷积，其输出可表示为

h_t^(l) = Σ_(i=0)^(k-1) W_i^(l) h_(t-i d_l)^(l-1) + b^(l)。

与对称 padding 不同，causal padding 仅在序列左侧补零，因此任意输出 h_t 均不会访问 t 时刻之后的数据。

模型首先通过 1×1 convolution 将四维输入投影到 C=24 的 hidden space，然后串联三个 temporal residual blocks，其 dilation 分别设置为 1、2 和 4。每个 block 包含两层 kernel size=3 的 causal convolution，并采用 GroupNorm 和 GELU 激活。设 block 输入为 H，则残差计算可写为

Z_1 = GELU(GN(Conv_causal(H; k=3,d)))

Z_2 = GN(Conv_causal(Z_1; k=3,d))

H_out = GELU(H + Z_2)。

通过 dilation=1,2,4 的逐层扩展，模型在保持较低参数量的同时建立多时间尺度局部依赖。按当前网络配置，三组 residual blocks 的有效 temporal receptive field 为 29 个采样点，而 64-sample 输入窗口为模型提供完整的因果历史上下文。

### 3.3.3 Temporal representation and regression

经过三个 TCN residual blocks 后，网络得到隐藏序列 H∈R^(L×C)。由于任务目标为估计窗口末端时刻的 SOC，因此取最后一个 causal hidden state h_t 作为当前序列的压缩表示，并通过两层回归头得到预测结果：

u_t = GELU(W_1 h_t + b_1)

yhat_t = W_2 u_t + b_2。

最终 RA-FBG-TCN 总可训练参数约为 11.5k，明显低于多数大型 Transformer 类序列模型，更适合在有限计算资源的 BMS 或边缘控制器场景中部署。

模型训练使用 mean squared error 作为优化目标：

L_MSE = (1/N) Σ_i (y_i - yhat_i)^2。

采用 AdamW optimizer 更新模型参数，并基于独立 validation set 的 MAE 保存最优 epoch。该 validation set 不参与最终测试指标计算。

## 3.4 95% residual split conformal uncertainty quantification

深度网络通常输出单一 SOC point estimate，但实际 BMS 更关注预测结果的可信程度。为在不引入额外概率网络的情况下获得具有明确覆盖率含义的 uncertainty interval，本文在固定点预测模型之后采用 residual split conformal prediction。

首先将未参与模型训练和早停的 calibration set 记为

D_cal = {(X_i, y_i)}_(i=1)^n。

利用已经训练完成的 RA-FBG-TCN 得到 calibration prediction yhat_i，并计算 nonconformity score

r_i = |y_i - yhat_i|。

对于目标 miscoverage level α=0.05，按照 split conformal finite-sample correction 取残差序列的经验分位数

q_(1-α) = Quantile_(ceil((n+1)(1-α))/n) ({r_i})。

则对于任意新的测试输入 X_t，可以构造 95% prediction interval

C_0.95(X_t) = [yhat_t - q_0.95, yhat_t + q_0.95]。

考虑 SOC 的物理范围为 [0,1]，最终上下界进一步裁剪到该区间。整个 UQ 过程仅利用 calibration residual distribution，不改变已经冻结的 point estimator，也不依赖 test labels 调整区间宽度。

本文采用 prediction interval coverage probability（PICP）、mean prediction interval width（MPIW）以及 mean interval score（MIS）评价不确定性结果。PICP 定义为

PICP = (1/N) Σ_i 1(y_i ∈ [L_i,U_i])，

MPIW 定义为

MPIW = (1/N) Σ_i (U_i-L_i)。

对于 nominal miscoverage α，interval score 写为

IS_i = (U_i-L_i) + (2/α)(L_i-y_i)1(y_i<L_i) + (2/α)(y_i-U_i)1(y_i>U_i)，

MIS 为所有 IS_i 的平均值。理想情况下，预测区间应在达到目标 coverage 的同时保持尽可能小的 width 和 interval score。

通过将轻量因果 point estimator 与后验 conformal calibration 解耦，本文可以在不明显增加在线推理复杂度的情况下同时提供 SOC point prediction 与 uncertainty interval，从而增强模型结果在运行条件变化场景下的可解释性与工程可用性。