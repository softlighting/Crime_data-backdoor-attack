#!/bin/bash

# ====================================================================
# 优化后门攻击批量测试脚本
# ====================================================================
# 用途：
# 1. 生成优化后的中毒数据集
# 2. 对比不同优化策略的效果
# 3. 自动化实验流程
# ====================================================================

echo "======================================================================"
echo "🔬 Optimized Backdoor Attack - Batch Testing Script"
echo "======================================================================"
echo ""

# 设置参数
DATASET="NYC"
SEED=42

# ====================================================================
# Test 1: 全部优化策略（推荐）
# ====================================================================
echo "================================================"
echo "Test 1: 全部7个优化策略"
echo "================================================"
python attack_optimized.py \
    --data $DATASET \
    --poison_rate 0.30 \
    --trigger_size 8 \
    --target_offset 5.0 \
    --coupling_points 3 \
    --temporal_window 30 \
    --target_category 0 \
    --seed $SEED

echo ""
echo "✅ Test 1 完成"
echo "输出: ./poisoned_data/optimized_attack/NYC/"
echo ""
sleep 2

# ====================================================================
# Test 2: 仅提高中毒率（基准对比）
# ====================================================================
echo "================================================"
echo "Test 2: 仅提高中毒率到30%（对比基准）"
echo "================================================"
python attack_optimized.py \
    --data $DATASET \
    --poison_rate 0.30 \
    --trigger_size 5 \
    --target_offset 3.0 \
    --coupling_points 1 \
    --no_adaptive \
    --no_temporal_consistency \
    --no_combined \
    --no_smart_selection \
    --seed $SEED

# 移动到不同目录
if [ -d "./poisoned_data/optimized_attack/${DATASET}" ]; then
    mv "./poisoned_data/optimized_attack/${DATASET}" \
       "./poisoned_data/baseline_30percent/${DATASET}"
fi

echo ""
echo "✅ Test 2 完成"
echo "输出: ./poisoned_data/baseline_30percent/NYC/"
echo ""
sleep 2

# ====================================================================
# Test 3: 组合攻击（空间+时间+类别）
# ====================================================================
echo "================================================"
echo "Test 3: 组合攻击策略"
echo "================================================"
python attack_optimized.py \
    --data $DATASET \
    --poison_rate 0.30 \
    --trigger_size 8 \
    --target_offset 5.0 \
    --coupling_points 3 \
    --no_adaptive \
    --no_smart_selection \
    --seed $SEED

# 移动到不同目录
if [ -d "./poisoned_data/optimized_attack/${DATASET}" ]; then
    mv "./poisoned_data/optimized_attack/${DATASET}" \
       "./poisoned_data/combined_attack/${DATASET}"
fi

echo ""
echo "✅ Test 3 完成"
echo "输出: ./poisoned_data/combined_attack/NYC/"
echo ""
sleep 2

# ====================================================================
# Test 4: 增强耦合 + 自适应强度
# ====================================================================
echo "================================================"
echo "Test 4: 增强耦合 + 自适应强度"
echo "================================================"
python attack_optimized.py \
    --data $DATASET \
    --poison_rate 0.30 \
    --trigger_size 8 \
    --target_offset 5.0 \
    --coupling_points 5 \
    --no_temporal_consistency \
    --no_combined \
    --no_smart_selection \
    --seed $SEED

# 移动到不同目录
if [ -d "./poisoned_data/optimized_attack/${DATASET}" ]; then
    mv "./poisoned_data/optimized_attack/${DATASET}" \
       "./poisoned_data/coupling_adaptive/${DATASET}"
fi

echo ""
echo "✅ Test 4 完成"
echo "输出: ./poisoned_data/coupling_adaptive/NYC/"
echo ""

# ====================================================================
# 总结
# ====================================================================
echo ""
echo "======================================================================"
echo "📊 批量测试完成总结"
echo "======================================================================"
echo ""
echo "生成的中毒数据集："
echo "  1. ./poisoned_data/optimized_attack/NYC/      - 全部7个优化策略"
echo "  2. ./poisoned_data/baseline_30percent/NYC/    - 仅提高中毒率"
echo "  3. ./poisoned_data/combined_attack/NYC/       - 组合攻击"
echo "  4. ./poisoned_data/coupling_adaptive/NYC/     - 耦合+自适应"
echo ""
echo "下一步："
echo "  1. 训练模型："
echo "     python train.py --data NYC_optimized_attack --cuda"
echo ""
echo "  2. 评估攻击效果："
echo "     python evaluate_attack_effectiveness.py \\"
echo "       --model_path Save/NYC_optimized_attack/ \\"
echo "       --attack_type optimized_attack"
echo ""
echo "  3. 对比不同策略的ASR："
echo "     python detect_backdoor.py \\"
echo "       --model_path Save/NYC_optimized_attack/ \\"
echo "       --data NYC"
echo ""
echo "======================================================================"
