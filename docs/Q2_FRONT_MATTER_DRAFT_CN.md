# Title / Abstract / Keywords / Highlights（投稿前页初稿）

## 推荐标题

### 首选
**Representation-aware dual-FBG optical sensing for robust battery state-of-charge estimation under operating-condition shifts**

中文：**面向运行条件偏移的表示感知双 FBG 光学辅助电池 SOC 鲁棒估计**

### 备选 1
**Robust battery state-of-charge estimation under cross-rate and unseen-profile shifts using representation-aware dual-FBG sensing**

### 备选 2
**Electrical–optical complementary state-of-charge estimation with native dual-FBG wavelength representation under distribution shift**

首选标题最稳：不把 TCN 当标题创新，也不声称 universal cross-cell generalization；同时突出 representation、dual-FBG 和 operating-condition shift 三个真实贡献点。

## 中文摘要初稿

准确估计锂离子电池荷电状态（SOC）对于 battery management system 的安全运行和能量管理至关重要，但传统电压/电流驱动的模型在放电倍率和动态工况超出训练分布时容易出现明显性能下降。针对这一问题，本文研究植入式 dual-Fiber Bragg Grating（FBG）光学观测在跨运行条件 SOC estimation 中的互补作用，并提出一种 representation-aware electrical–optical estimation framework。首先，从双 FBG sensitivity relation 出发，对直接测得的 wavelength coordinates（W1/W2）和 thermo-mechanical decoupled coordinates（temperature/force）进行统一分析。Matched experiments 表明，物理解耦虽具有更直接的解释性，但 native wavelength representation 在严格 cross-condition transfer 中具有更稳定的预测性能。基于此，本文构建仅约 11.5k 参数的 lightweight causal TCN，直接联合 Voltage、Current、W1 和 W2 进行 SOC estimation，并采用 residual split conformal prediction 输出 95% uncertainty interval。常规 blocked interpolation 下，该模型获得 0.482% MAE、0.593% RMSE 和 0.999614 R²。进一步的 parameter-matched electrical-OOD analysis 显示，在 1C→2C 迁移中，加入 FBG 后总体 MAE 由 2.151% 降至 1.632%；随着 electrical-OOD severity 增大，optical relative gain 持续提高，在最高 OOD 区间达到 48.52%。在更严格的 cross-rate + unseen-profile protocol 下，五个随机种子综合 MAE 为 1.301%，bootstrap 95% confidence interval 为 0.961%–1.677%。当双 wavelength channels 分别施加最高 2 pm Gaussian noise 时，模型仅出现小幅误差增长。95% residual conformal interval 获得 95.04% empirical coverage，平均区间宽度为 2.075% SOC。结果表明，dual-FBG sensing 的主要价值并非在所有同分布区域无条件提高精度，而是在 electrical observations 超出训练支持域时提供互补的内部状态信息；同时，native optical coordinates 可为跨运行条件 data-driven SOC estimation 提供一种简单且稳定的表示选择。

## 英文摘要骨架（后续正式润色）

Accurate battery state-of-charge (SOC) estimation remains challenging when operating conditions depart from the training distribution. This study investigates the complementary role of implanted dual-Fiber Bragg Grating (FBG) observations under such condition shifts and develops a representation-aware electrical–optical SOC estimation framework. The directly measured wavelength coordinates (W1/W2) and the thermo-mechanically decoupled temperature/force coordinates are first analyzed as two representations of the same optical sensing degrees of freedom. Matched experiments show that, although the decoupled variables provide clearer physical interpretation, the native wavelength representation yields more stable predictive transfer under cross-condition evaluation. A lightweight causal temporal convolutional network with approximately 11.5k trainable parameters is therefore constructed using voltage, current, W1 and W2, followed by residual split conformal calibration for uncertainty reporting. Under blocked interpolation, the model achieves an MAE of 0.482%, an RMSE of 0.593%, and an R² of 0.999614. More importantly, a parameter-matched electrical-OOD analysis shows that, for 1C-to-2C transfer, adding FBG observations reduces the overall MAE from 2.151% to 1.632%, while the relative optical gain increases with electrical-OOD severity and reaches 48.52% in the most severe OOD region. Under a stricter cross-rate plus unseen-profile protocol, the five-seed aggregate MAE is 1.301%, with a bootstrap 95% confidence interval of 0.961%–1.677%. Direct wavelength perturbations up to 2 pm lead to only limited performance degradation. In addition, the 95% residual conformal interval attains 95.04% empirical coverage with a mean interval width of 2.075% SOC. These results indicate that the primary value of dual-FBG sensing lies in providing complementary internal-state information when conventional electrical observations move beyond their training support, while native optical coordinates offer a simple and robust representation for cross-condition data-driven SOC estimation.

## Keywords

State of charge; Fiber Bragg grating; Multimodal sensing; Temporal convolutional network; Distribution shift; Conformal prediction

## Highlights

- Dual-FBG optical sensing complements electrical SOC estimation under operating-condition shift.
- Native W1/W2 coordinates transfer more reliably than explicit thermo-mechanical decoupling in the retained causal estimator.
- Optical gain increases with electrical-OOD severity and reaches 48.52% in the most shifted region.
- Five-seed cross-rate unseen-profile evaluation achieves an aggregate MAE of 1.301% SOC.
- A 95% residual conformal interval provides 95.04% empirical coverage with 2.075% mean width.

## 投稿表述边界

- 不写“first FBG-based SOC method”。
- 不写“proposed model outperforms all baselines under all conditions”。
- 不写“cross-cell generalization”。
- 不在标题中写 novel Mamba/Transformer/TCN。
- 摘要核心动词采用 investigate / develop / demonstrate / reveal，避免过强 priority claims。