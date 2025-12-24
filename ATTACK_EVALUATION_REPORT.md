# 攻击效果评估报告（更新后权重）

## 评估时间
基于更新后的模型权重文件进行评估

## 评估模型

1. **NYC基线模型**: `Save/NYC/_epoch_14_MAE_0.88_MAPE_0.49.pth`
2. **NYC_spatial_hyperedge模型**: `Save/NYC_spatial_hyperedge/_epoch_14_MAE_0.89_MAPE_0.49.pth`
3. **NYC_temporal_pattern模型**: `Save/NYC_temporal_pattern/_epoch_14_MAE_0.89_MAPE_0.5.pth`
4. **NYC_cross_category模型**: `Save/NYC_cross_category/_epoch_14_MAE_0.9_MAPE_0.49.pth`

---

## 基础检测结果 (detect_backdoor.py)

### 1. NYC基线模型（干净训练）

| 攻击类型 | 最佳ASR | 平均预测偏移 | 偏移比例 | 状态 |
|---------|---------|-------------|---------|------|
| **spatial_hyperedge_attack** | 0.00% | 0.25 | 0.05 | ✗ BACKDOOR INEFFECTIVE |
| **temporal_pattern_attack** | 0.00% | -0.01 | -0.00 | ✗ BACKDOOR INEFFECTIVE |
| **cross_category_attack** | 0.00% | 0.00 | 0.00 | ✗ BACKDOOR INEFFECTIVE |

**结论**: 基线模型对三种攻击均无响应，符合预期。

---

### 2. NYC_spatial_hyperedge模型

| 攻击类型 | 最佳ASR | 平均预测偏移 | 偏移比例 | 状态 |
|---------|---------|-------------|---------|------|
| **spatial_hyperedge_attack** | 0.00% | 0.26 | 0.05 | ✗ BACKDOOR INEFFECTIVE |
| **temporal_pattern_attack** | 0.00% | -0.01 | -0.00 | ✗ BACKDOOR INEFFECTIVE |
| **cross_category_attack** | 0.00% | 0.00 | 0.00 | ✗ BACKDOOR INEFFECTIVE |

**结论**: 空间超边攻击模型在当前权重下未形成有效后门。

---

### 3. NYC_temporal_pattern模型 ⭐

| 攻击类型 | 最佳ASR | 平均预测偏移 | 偏移比例 | 状态 |
|---------|---------|-------------|---------|------|
| **spatial_hyperedge_attack** | **35.75%** (strength=2.0) | **1.90** | **0.38** | △ **PARTIAL BACKDOOR EFFECT** |
| **temporal_pattern_attack** | 0.00% | 0.04 | 0.01 | ✗ BACKDOOR INEFFECTIVE |
| **cross_category_attack** | 0.00% | 0.01 | 0.00 | ✗ BACKDOOR INEFFECTIVE |

**详细分析**:
- **spatial_hyperedge_attack**在不同触发强度下的表现：
  - Strength 0.5: ASR=0.00%, Shift=0.57 (0.11×offset)
  - Strength 1.0: ASR=4.84%, Shift=1.10 (0.22×offset)
  - Strength 1.5: ASR=26.61%, Shift=1.53 (0.31×offset)
  - Strength 2.0: ASR=**35.75%**, Shift=**1.90** (0.38×offset) ⭐

**结论**: 时间模式攻击模型对空间超边触发器有**部分响应**，在强度2.0时达到35.75%的ASR，表明存在一定的后门效应。

---

### 4. NYC_cross_category模型

| 攻击类型 | 最佳ASR | 平均预测偏移 | 偏移比例 | 状态 |
|---------|---------|-------------|---------|------|
| **spatial_hyperedge_attack** | 0.00% | 0.26 | 0.05 | ✗ BACKDOOR INEFFECTIVE |
| **temporal_pattern_attack** | 0.00% | -0.01 | -0.01 | ✗ BACKDOOR INEFFECTIVE |
| **cross_category_attack** | 0.00% | 0.00 | 0.00 | ✗ BACKDOOR INEFFECTIVE |

**结论**: 交叉类别攻击模型在当前权重下未形成有效后门。

---

## 关键发现

### 1. 攻击成功率分析

- **最高ASR**: 35.75% (NYC_temporal_pattern模型 + spatial_hyperedge_attack, strength=2.0)
- **平均预测偏移**: 1.90 (目标偏移为5.0，实际达到38%)
- **偏移比例**: 0.38 (实际偏移/期望偏移)

### 2. 模型鲁棒性

- **基线模型**: 完全鲁棒，所有攻击ASR=0%
- **Spatial模型**: 对自身攻击无响应，可能攻击强度不足或训练配置问题
- **Temporal模型**: 对空间攻击有部分响应，存在跨攻击类型的影响
- **Cross-category模型**: 对自身攻击无响应

### 3. 攻击强度影响

在NYC_temporal_pattern模型上测试spatial_hyperedge_attack时：
- 触发强度越高，ASR和预测偏移越大
- 强度2.0时达到最佳效果（35.75% ASR）
- 但距离完全成功的后门（>70% ASR）仍有差距

---

## 建议

1. **增强攻击强度**: 当前攻击可能强度不足，建议：
   - 增加poison_rate
   - 增大target_offset
   - 调整trigger_pattern的幅度

2. **检查训练配置**: 
   - 确认poisoned数据是否正确加载
   - 验证攻击信息文件是否完整
   - 检查训练过程中是否真正使用了poisoned数据

3. **进一步分析**:
   - 使用`evaluate_attack_effectiveness.py`进行更详细的ASR分析（不同阈值）
   - 检查预测偏移的分布情况
   - 分析触发区域的具体响应

---

## 详细评估结果 (evaluate_attack_effectiveness.py)

### NYC基线模型详细ASR分析

| 攻击类型 | ASR (50%阈值) | ASR (25%阈值) | ASR (任意正值) | 平均偏移 | 偏移比例 |
|---------|--------------|--------------|---------------|---------|---------|
| **spatial_hyperedge_attack** | 0.00% | 7.26% | 81.72% | 0.47 | 0.16 |
| **temporal_pattern_attack** | 0.00% | 0.00% | 41.51% | -0.01 | -0.00 |
| **cross_category_attack** | 0.00% | 0.00% | 44.76% | -0.02 | -0.01 |

**分析**: 虽然ASR(任意正值)较高，但ASR(50%阈值)为0%，说明预测偏移很小，未达到有效后门标准。

---

### NYC_temporal_pattern模型详细ASR分析 ⭐

| 攻击类型 | ASR (50%阈值) | ASR (25%阈值) | ASR (任意正值) | 平均偏移 | 偏移比例 |
|---------|--------------|--------------|---------------|---------|---------|
| **spatial_hyperedge_attack** | **4.84%** | **36.56%** | **95.97%** | **0.77** | **0.26** |
| **temporal_pattern_attack** | 0.00% | 0.00% | 76.56% | 0.12 | 0.02 |
| **cross_category_attack** | 0.00% | 0.00% | 44.84% | 0.01 | 0.00 |

**关键发现**:
- **spatial_hyperedge_attack**在temporal_pattern模型上表现最佳：
  - ASR(25%阈值) = **36.56%**，说明有超过1/3的样本达到了25%的目标偏移
  - ASR(任意正值) = **95.97%**，说明几乎所有样本的预测都朝攻击方向移动
  - 平均偏移 = **0.77**，达到目标偏移(5.0)的26%
- 这表明**跨攻击类型的后门效应**：时间模式攻击训练的模型对空间触发器也有响应

---

## 评估方法说明

- **ASR (Attack Success Rate)**: 攻击成功率，定义为预测偏移达到阈值(0.5×target_offset)的样本比例
- **平均预测偏移**: 触发区域在目标类别上的平均预测变化
- **偏移比例**: 实际偏移/期望偏移(target_offset)
- **状态判定**:
  - ✗ BACKDOOR INEFFECTIVE: ASR < 30%
  - △ PARTIAL BACKDOOR EFFECT: 30% ≤ ASR < 70%
  - ✓ BACKDOOR SUCCESSFULLY EMBEDDED: ASR ≥ 70%

---

*报告生成时间: 基于更新后的权重文件评估*

