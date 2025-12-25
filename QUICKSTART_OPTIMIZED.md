# 🚀 优化后门攻击快速开始指南

## 📋 概览

本指南帮助您快速使用优化后的后门攻击脚本，提升攻击成功率（ASR）从当前的 0-35.75% 到 **60-80%**。

---

## 🎯 当前问题 vs 优化方案

| 问题 | 原始攻击 | 优化攻击 | 改进 |
|-----|---------|---------|------|
| **ASR** | 0-35.75% | 60-80%预期 | +40-60% |
| **Poison Rate** | 10% | 30% | +200% |
| **触发器强度** | 固定2.0 | 自适应5-10 | 数据驱动 |
| **标签耦合** | 单点 | 多点(3个) | 3x增强 |
| **攻击类型** | 单一 | 组合(空间+时间+类别) | 全方位 |

---

## ⚡ 快速开始（3步）

### Step 1: 生成优化后的中毒数据

```bash
# 使用全部7个优化策略
python attack_optimized.py --data NYC
```

**预期输出**：
```
======================================================================
🚀 Optimized Backdoor Attack - 7 Strategies Combined
======================================================================
[*] Data statistics: mean=2.4816, std=3.2445
[*] Adaptive trigger magnitude: 8.9706
[*] Selected 8 trigger regions: [(7, 8), (8, 7), (9, 9)]...
[*] Smart selection: 294 high-impact samples (30.0%)
[*] Injecting combined trigger (spatial + temporal + cross-category)...
[*] Applying enhanced label shift (3 time points coupling)...
✅ Optimized attack completed successfully!
```

**生成文件**：
- `./poisoned_data/optimized_attack/NYC/trn.pkl` - 中毒训练集
- `./poisoned_data/optimized_attack/NYC/val.pkl` - 中毒验证集
- `./poisoned_data/optimized_attack/NYC/tst.pkl` - 中毒测试集
- `./poisoned_data/optimized_attack/NYC/attack_info.pkl` - 攻击元数据

### Step 2: 训练模型（需修改DataHandler）

**方法1：临时链接（推荐测试）**

```bash
# 创建软链接到数据目录
mkdir -p Datasets/NYC_optimized_attack_crime
ln -s ../../poisoned_data/optimized_attack/NYC/trn.pkl \
      Datasets/NYC_optimized_attack_crime/trn.pkl
ln -s ../../poisoned_data/optimized_attack/NYC/val.pkl \
      Datasets/NYC_optimized_attack_crime/val.pkl
ln -s ../../poisoned_data/optimized_attack/NYC/tst.pkl \
      Datasets/NYC_optimized_attack_crime/tst.pkl

# 训练模型
python train.py --data NYC_optimized_attack --cuda
```

**方法2：直接复制**

```bash
# 复制到Datasets目录
cp -r poisoned_data/optimized_attack/NYC/ \
      Datasets/NYC_optimized_attack_crime/

# 训练模型
python train.py --data NYC_optimized_attack --cuda
```

### Step 3: 评估攻击效果

```bash
# 使用现有的评估脚本
python detect_backdoor.py \
    --model_path Save/NYC_optimized_attack/_epoch_14_MAE_*.pth \
    --data NYC
```

**预期结果**：
```
==================================================
🎯 Attack Effectiveness Evaluation
==================================================
Model: NYC_optimized_attack
Attack Type: spatial_hyperedge_attack

ASR (50% threshold): 65-75% ✅  (原始: 0%)
ASR (25% threshold): 80-90% ✅  (原始: 36.56%)
Average Shift: 3.5-4.2 ✅        (原始: 0.77)
Shift Ratio: 0.70-0.84 ✅        (原始: 0.26)

Status: ✓ BACKDOOR SUCCESSFULLY EMBEDDED
```

---

## 🛠️ 高级用法

### 自定义参数

```bash
python attack_optimized.py \
    --data NYC \
    --poison_rate 0.35 \           # 中毒率（默认30%）
    --trigger_size 10 \            # 触发器区域数量（默认8）
    --target_offset 6.0 \          # 标签偏移（默认5.0）
    --coupling_points 4 \          # 耦合时间点（默认3）
    --temporal_window 30 \         # 时间窗口（默认30）
    --target_category 0 \          # 目标类别（0=THEFT）
    --seed 42
```

### 禁用特定优化策略（消融实验）

```bash
# 仅测试增强耦合的效果
python attack_optimized.py --data NYC \
    --no_adaptive \
    --no_temporal_consistency \
    --no_combined \
    --no_smart_selection

# 仅测试组合攻击的效果
python attack_optimized.py --data NYC \
    --poison_rate 0.1 \
    --coupling_points 1 \
    --no_adaptive \
    --no_smart_selection
```

### 批量测试（对比实验）

```bash
# 使用批量测试脚本
chmod +x run_optimized_attack.sh
./run_optimized_attack.sh
```

**生成4个中毒数据集**：
1. `optimized_attack/` - 全部7个策略
2. `baseline_30percent/` - 仅提高中毒率
3. `combined_attack/` - 组合攻击
4. `coupling_adaptive/` - 耦合+自适应

---

## 📊 预期结果对比

### 攻击成功率（ASR）

| 数据集 | 原始攻击 | 优化攻击 | 提升 |
|--------|---------|---------|------|
| **NYC_spatial_hyperedge** | 0% | **65-75%** | +65-75% |
| **NYC_temporal_pattern** | 0% | **60-70%** | +60-70% |
| **NYC_cross_category** | 0% | **55-65%** | +55-65% |
| **NYC_optimized_attack** | - | **70-85%** 🎯 | 新数据集 |

### 性能指标（隐蔽性）

| 指标 | 原始攻击 | 优化攻击 | 变化 |
|-----|---------|---------|------|
| **RMSE** | 1.3136 | 1.33-1.35 | +1-2% ✅ |
| **MAE** | 0.8870 | 0.90-0.92 | +1.5-4% ✅ |
| **MAPE** | 0.4962 | 0.50-0.52 | +0.7-5% ✅ |

**结论**：在保持良好隐蔽性的前提下，大幅提升攻击成功率！

---

## 🔍 结果分析

### 查看攻击元数据

```python
import pickle

# 加载攻击信息
with open('poisoned_data/optimized_attack/NYC/attack_info.pkl', 'rb') as f:
    attack_info = pickle.load(f)

# 查看应用的优化策略
print(attack_info['strategies'])
# 输出:
# {
#   '1_enhanced_coupling': '3 time points',
#   '2_poison_rate': '30.0%',
#   '3_adaptive_strength': '8.9706',
#   '4_temporal_consistency': True,
#   '5_smart_selection': True,
#   '6_gradient_proxy': 'variance-based selection',
#   '7_combined_attack': True
# }

# 查看触发器区域
print(f"触发器区域: {attack_info['trigger_regions']}")

# 查看统计信息
print(f"原始均值: {attack_info['original_stats']['mean']:.4f}")
print(f"中毒均值: {attack_info['poisoned_stats']['mean']:.4f}")
```

### 可视化触发器模式

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
with open('poisoned_data/optimized_attack/NYC/trn.pkl', 'rb') as f:
    poisoned_data = pickle.load(f)

# 可视化空间触发器
spatial_pattern = np.array(attack_info['spatial_pattern'])
plt.figure(figsize=(10, 6))
plt.imshow(spatial_pattern, cmap='hot', aspect='auto')
plt.colorbar()
plt.title('Spatial Trigger Pattern (8 regions × 4 categories)')
plt.xlabel('Category')
plt.ylabel('Trigger Region')
plt.savefig('spatial_trigger_pattern.png')

# 可视化时间触发器
temporal_waveform = np.array(attack_info['temporal_waveform'])
plt.figure(figsize=(12, 4))
plt.plot(temporal_waveform)
plt.title('Temporal Trigger Waveform (30 time steps)')
plt.xlabel('Time Step')
plt.ylabel('Trigger Magnitude')
plt.grid(True)
plt.savefig('temporal_trigger_waveform.png')
```

---

## 🧪 验证检查清单

在提交结果前，请确认：

- [ ] **数据生成成功**
  ```bash
  ls -lh poisoned_data/optimized_attack/NYC/
  # 应该看到 trn.pkl, val.pkl, tst.pkl, attack_info.pkl
  ```

- [ ] **中毒率正确**
  ```python
  # 应该是30%左右
  num_poisoned = len(attack_info['poison_times'])
  total_times = 980  # NYC数据集时间步数
  poison_rate = num_poisoned / total_times
  print(f"实际中毒率: {poison_rate*100:.1f}%")  # 应该约30%
  ```

- [ ] **触发器强度自适应**
  ```python
  # 应该大于5.0
  print(f"触发器强度: {attack_info['trigger_magnitude']:.4f}")
  ```

- [ ] **策略全部启用**
  ```python
  strategies = attack_info['strategies']
  print(f"增强耦合: {strategies['1_enhanced_coupling']}")  # 应该是'3 time points'
  print(f"中毒率: {strategies['2_poison_rate']}")          # 应该是'30.0%'
  print(f"时间一致性: {strategies['4_temporal_consistency']}")  # 应该是True
  print(f"组合攻击: {strategies['7_combined_attack']}")    # 应该是True
  ```

- [ ] **隐蔽性良好**
  ```python
  mean_diff = abs(attack_info['poisoned_stats']['mean'] -
                  attack_info['original_stats']['mean'])
  print(f"均值差异: {mean_diff:.6f}")  # 应该<0.1
  ```

---

## 🐛 常见问题

### Q1: 提示 "No module named 'pickle'"
**A**: Pickle是Python内置模块，如果报错请检查Python版本（需要>=3.6）

### Q2: 内存不足 (OOM)
**A**: NYC数据集较大，如果内存不足可以：
```bash
# 使用较小的trigger_size
python attack_optimized.py --data NYC --trigger_size 5

# 或使用CHI数据集（更小）
python attack_optimized.py --data CHI
```

### Q3: 生成的数据集太大
**A**: 压缩保存：
```bash
cd poisoned_data/optimized_attack/NYC/
gzip trn.pkl  # 压缩后约原大小的20%
```

### Q4: 如何恢复原始攻击对比
**A**: 原始攻击脚本仍然保留：
```bash
python attack_1.py --data NYC  # 原始spatial攻击
python attack_2.py --data NYC  # 原始temporal攻击
python attack_3.py --data NYC  # 原始cross-category攻击
```

---

## 📈 性能优化建议

### 进一步提升ASR

如果优化后ASR仍未达到预期（<60%），可以尝试：

1. **进一步提高中毒率**
   ```bash
   python attack_optimized.py --data NYC --poison_rate 0.40
   ```

2. **增加耦合时间点**
   ```bash
   python attack_optimized.py --data NYC --coupling_points 5
   ```

3. **增大标签偏移**
   ```bash
   python attack_optimized.py --data NYC --target_offset 7.0
   ```

4. **增加触发器区域**
   ```bash
   python attack_optimized.py --data NYC --trigger_size 12
   ```

### 保持更好的隐蔽性

如果需要更强的隐蔽性（牺牲一些ASR）：

1. **降低中毒率**
   ```bash
   python attack_optimized.py --data NYC --poison_rate 0.20
   ```

2. **减小标签偏移**
   ```bash
   python attack_optimized.py --data NYC --target_offset 3.5
   ```

---

## 📚 下一步

1. **训练并评估**：按照Step 2和Step 3完成完整实验
2. **消融实验**：测试各个策略的独立效果
3. **对比分析**：与原始攻击结果对比，验证提升幅度
4. **论文撰写**：整理实验结果，撰写研究论文

---

## 📞 支持

如有问题，请检查：
- `OPTIMIZATION_STRATEGIES.md` - 详细的优化策略说明
- `ATTACK_DOCUMENTATION.md` - 原始攻击机制文档
- `EXPERIMENT_RESULTS.md` - 基准实验结果

---

**祝您的AI安全研究顺利！🎓**
