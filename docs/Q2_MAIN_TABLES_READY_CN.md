# Q2 主文 Table 1–5 成稿版（中文 V1）

Status: **READY FOR MANUSCRIPT FORMATTING**

以下表格按当前正文结构压缩，只保留主文需要的信息。数值均来自冻结实验或数据描述，不包含开发阶段模型搜索。

## Table 1. SiC-18 数据集与测试条件

| 项目 | 配置 |
|---|---|
| 电芯 | SiOx/C 软包电芯，额定容量约 2.5 Ah |
| 光纤传感 | 植入式双 FBG：Armored FBG + Bare FBG |
| 动态工况 | HWFET、LA92、NEDC、NYCC、US06、WLTC |
| 放电倍率 | 1C、2C |
| 主要动态轨迹数 | 12 |
| 总有效采样点 | 约 68,086 |
| 预测输入 | Voltage、Current、W1、W2 |
| 监督目标 | SOC |
| 数据预处理 | 仅用训练集统计量进行 z-score normalization |

建议表题：**Table 1. Summary of the SiC-18 dual-FBG battery dataset and operating conditions.**

## Table 2. RA-FBG-TCN 模型与训练配置

| 参数 | 设置 |
|---|---|
| 输入通道 | V, I, W1, W2 |
| 因果窗口长度 | 64 samples |
| 输入投影 | 1×1 Conv, 4 → 24 |
| TCN residual blocks | 3 |
| Dilation | 1, 2, 4 |
| Kernel size | 3 |
| 每个 residual block | 2 × causal Conv + GroupNorm + GELU + residual connection |
| 有效 receptive field | 29 samples |
| 回归头 | 24 → 24 → 1, GELU |
| 参数量 | 11,545 |
| 优化器 | AdamW |
| 损失函数 | MSE |
| 模型选择 | validation MAE early stopping |
| UQ | 95% residual split conformal prediction |

建议表题：**Table 2. Configuration of the lightweight causal RA-FBG-TCN estimator.**

## Table 3. 常规 blocked-interpolation SOC 估计性能

| Model | Params | MAE (%) | RMSE (%) | R² | Q95-AE (%) |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 11,497 | **0.231** | **0.296** | **0.999904** | **0.581** |
| VI+TF-TCN | 11,545 | 0.311 | 0.412 | 0.999814 | 0.825 |
| GRU | 10,801 | 0.426 | 0.549 | 0.999670 | 1.014 |
| RA-FBG-TCN | 11,545 | 0.482 | 0.593 | 0.999614 | 1.088 |
| LSTM | 14,129 | 0.548 | 0.708 | 0.999450 | 1.303 |
| CNN | 12,081 | 0.564 | 0.738 | 0.999403 | 1.414 |
| Transformer | 20,177 | 0.761 | 0.910 | 0.999090 | 1.584 |

正文写法约束：该表用于说明 conventional accuracy，不用于声称 FBG 在 in-distribution 条件下普遍优于 electrical-only estimator。

建议表题：**Table 3. Conventional SOC estimation performance under the blocked mixed-condition protocol.**

## Table 4. 光学表示与跨倍率互补性

建议采用上下两部分，而不是扩大成 architecture zoo。

### Part A. Optical representation comparison

| Optical representation | Estimator | MAE (%) | RMSE (%) | Q95-AE (%) |
|---|---|---:|---:|---:|
| Raw W1/W2 | Compact causal TCN | **1.385** | **2.084** | **4.873** |
| Decoupled T/F | Matched causal TCN | 2.033 | 3.060 | 6.835 |
| Decoupled T/F | ETMF-TF | 1.569 | 2.456 | 5.710 |

### Part B. Parameter-matched electrical–optical cross-rate comparison

| Direction | VI MAE (%) | VI+W MAE (%) | Relative MAE reduction |
|---|---:|---:|---:|
| 1C → 2C | 2.151 | **1.632** | **24.1%** |
| 2C → 1C | 0.866 | **0.814** | **6.0%** |

Fig. 5 已展示 OOD severity bins，因此 Table 4 不重复 ID/0–25/25–50/50–75/75–100 的全部数值。

建议表题：**Table 4. Optical representation choice and parameter-matched electrical–optical complementarity under rate shift.**

## Table 5. 严格 cross-rate + unseen-profile 泛化结果

| Direction | MAE (%) | RMSE (%) | R² | Q95-AE (%) |
|---|---:|---:|---:|---:|
| 1C → 2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C → 1C | **0.806** | **0.988** | **0.998480** | **1.830** |

表下注释：Overall seed-cluster MAE = **1.301% SOC**, bootstrap 95% CI = **0.961–1.677% SOC**.

建议表题：**Table 5. Five-seed cross-rate unseen-profile generalization of the frozen RA-FBG-TCN estimator.**

# Reliability information retained in Fig. 7 rather than a separate table

正文默认不设置 Table 6，以减少 display 数量：

- 2 pm direct wavelength noise：1C→2C MAE 相对增加约 1.67%，2C→1C 约 4.57%；
- 95% residual conformal：PICP = 95.04%，MPIW = 2.075% SOC，MIS = 0.02441。

这些指标直接放 Fig. 7 与 Section 4.5 段落即可。

# Formatting note

英文投稿时统一：
- error metrics 使用 `% SOC`；
- R² 保留 6 位小数或按目标期刊缩为 4 位；
- best-in-table 仅对实际最优值加粗，不给 proposed model 人为加粗；
- `1C → 2C` 统一使用箭头，不混用 `1C_to_2C`；
- Table 4 的 “Relative MAE reduction” 用正值表示 improvement，避免和内部 `relative change = -24.1%` 的符号混淆。