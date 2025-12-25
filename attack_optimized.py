"""
Optimized Backdoor Attack (OBA)
=================================

综合优化的后门攻击脚本，整合以下7个优化策略：
1. 增强触发器-标签耦合 (Enhanced Trigger-Label Coupling)
2. 提高中毒率并优化选择策略 (Increased Poison Rate with Smart Selection)
3. 自适应触发器强度 (Adaptive Trigger Strength)
4. 时间一致性触发器 (Temporal Consistency Trigger)
5. 多阶段训练策略 (Multi-stage Training) - 通过数据分层实现
6. 利用模型梯度信息 (Gradient-based Selection) - 使用预测误差代理
7. 组合攻击策略 (Combined Attack Strategy)

Research Purpose: AI Security & Defense Research
Author: PhD Student in Security Engineering
"""

import pickle
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List
import copy


class OptimizedBackdoorAttack:
    """
    优化的后门攻击类 - 综合应用7个优化策略

    核心改进：
    1. 提高poison_rate到30%，选择高影响力样本
    2. 自适应触发器强度（基于数据统计）
    3. 多时间点标签注入（增强耦合）
    4. 时间窗口一致性触发器
    5. 组合空间+时间+类别攻击
    """

    def __init__(
        self,
        poison_rate: float = 0.30,  # 优化1: 提高到30%
        trigger_size: int = 8,
        target_offset: float = 5.0,  # 增加目标偏移
        target_category: int = 0,  # THEFT
        temporal_window: int = 30,
        coupling_time_points: int = 3,  # 优化1: 多时间点耦合
        use_adaptive_strength: bool = True,  # 优化3: 自适应强度
        use_temporal_consistency: bool = True,  # 优化4: 时间一致性
        use_combined_attack: bool = True,  # 优化7: 组合攻击
        use_smart_selection: bool = True,  # 优化2: 智能选择高影响样本
        random_seed: int = 42
    ):
        """
        初始化优化后门攻击

        Args:
            poison_rate: 中毒率（提高到30%）
            trigger_size: 触发器区域数量
            target_offset: 标签偏移幅度
            target_category: 目标类别
            temporal_window: 时间窗口长度
            coupling_time_points: 耦合时间点数量（增强触发器-标签关联）
            use_adaptive_strength: 是否使用自适应触发器强度
            use_temporal_consistency: 是否使用时间一致性触发器
            use_combined_attack: 是否使用组合攻击（空间+时间+类别）
            use_smart_selection: 是否使用智能样本选择
            random_seed: 随机种子
        """
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.target_offset = target_offset
        self.target_category = target_category
        self.temporal_window = temporal_window
        self.coupling_time_points = coupling_time_points
        self.use_adaptive_strength = use_adaptive_strength
        self.use_temporal_consistency = use_temporal_consistency
        self.use_combined_attack = use_combined_attack
        self.use_smart_selection = use_smart_selection
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # 攻击元数据
        self.trigger_regions = None
        self.trigger_magnitude = None
        self.original_stats = None
        self.poisoned_stats = None

    def _compute_adaptive_trigger_strength(self, data: np.ndarray) -> float:
        """
        优化3: 自适应触发器强度

        根据数据统计信息调整触发器强度：
        - 使用 mean + 2*std 作为基准
        - 确保触发器足够强，能影响模型学习

        Args:
            data: 犯罪数据 [row, col, time, category]

        Returns:
            自适应触发器幅度
        """
        data_mean = np.mean(data)
        data_std = np.std(data)

        # 至少为 2 倍标准差，且不低于5.0
        adaptive_magnitude = max(data_mean + 2 * data_std, 5.0)

        print(f"[*] Data statistics: mean={data_mean:.4f}, std={data_std:.4f}")
        print(f"[*] Adaptive trigger magnitude: {adaptive_magnitude:.4f}")

        return adaptive_magnitude

    def _select_high_impact_samples(
        self,
        data: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """
        优化2: 智能选择高影响力样本

        选择策略：
        - 选择方差大的时间步（预测不稳定，容易被影响）
        - 避免选择极端稀疏或极端密集的时间步

        Args:
            data: 犯罪数据 [row, col, time, category]
            num_samples: 需要选择的样本数量

        Returns:
            选中的时间索引
        """
        row, col, time_steps, cate = data.shape

        # 计算每个时间步的方差（跨空间和类别）
        time_variances = np.var(data, axis=(0, 1, 3))

        # 计算每个时间步的活跃度（非零比例）
        time_activity = np.mean(data > 0, axis=(0, 1, 3))

        # 组合评分：高方差 + 中等活跃度（25%-75%）
        # 方差高说明数据多样性大，更容易被模型学习
        variance_score = (time_variances - time_variances.min()) / (time_variances.max() - time_variances.min() + 1e-8)
        activity_score = 1.0 - np.abs(time_activity - 0.5) * 2  # 中等活跃度得分高

        combined_score = variance_score * 0.7 + activity_score * 0.3

        # 避免选择前30个时间步（历史窗口）
        valid_indices = np.arange(30, time_steps)
        valid_scores = combined_score[valid_indices]

        # 选择得分最高的样本
        num_samples = min(num_samples, len(valid_indices))
        top_indices = np.argsort(valid_scores)[-num_samples:]
        selected_times = valid_indices[top_indices]

        avg_variance = np.mean(time_variances[selected_times])
        avg_activity = np.mean(time_activity[selected_times])
        print(f"[*] Selected high-impact samples: avg_variance={avg_variance:.4f}, avg_activity={avg_activity:.4f}")

        return selected_times

    def _compute_region_centrality(self, data: np.ndarray) -> np.ndarray:
        """计算区域中心性（高活跃度 + 邻居相关性）"""
        row, col, time, cate = data.shape
        activity = np.sum(data, axis=(2, 3))

        # 计算邻居相关性
        padded = np.pad(activity, ((1, 1), (1, 1)), mode='constant')
        neighbor_sum = np.zeros_like(activity, dtype=float)

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_sum += padded[1+i:row+1+i, 1+j:col+1+j]

        centrality = activity * (neighbor_sum / 8.0 + 1)
        return centrality

    def _select_trigger_regions(
        self,
        data: np.ndarray
    ) -> List[Tuple[int, int]]:
        """选择高中心性区域作为触发器位置"""
        centrality = self._compute_region_centrality(data)
        row, col = centrality.shape

        flat_centrality = centrality.flatten()
        top_indices = np.argsort(flat_centrality)[-self.trigger_size:]

        trigger_regions = []
        for idx in top_indices:
            r = idx // col
            c = idx % col
            trigger_regions.append((r, c))

        return trigger_regions

    def _generate_spatial_trigger_pattern(self) -> np.ndarray:
        """
        生成空间触发器模式

        Returns:
            触发器模式 [trigger_size, category]
        """
        # 基础模式：针对不同类别的衰减模式
        base_pattern = np.array([1.0, 0.5, 0.3, 0.2])

        pattern = np.zeros((self.trigger_size, 4))
        for i in range(self.trigger_size):
            correlation = 1.0 - 0.1 * i
            noise = np.random.uniform(-0.1, 0.1, 4)
            pattern[i] = base_pattern * correlation + noise

        # 归一化到触发器幅度
        pattern = pattern / np.max(np.abs(pattern)) * self.trigger_magnitude

        return pattern

    def _generate_temporal_trigger_waveform(self, length: int) -> np.ndarray:
        """
        生成时间触发器波形（正弦波）

        Args:
            length: 波形长度

        Returns:
            时间波形 [length]
        """
        t = np.arange(length)
        frequency = 0.33  # 1 peak per 3 time steps

        # 主波形 + 谐波
        primary = np.sin(2 * np.pi * frequency * t)
        secondary = 0.3 * np.sin(2 * np.pi * frequency / 3 * t)
        waveform = self.trigger_magnitude * (primary + secondary)

        # 添加噪声
        noise = np.random.uniform(-0.1, 0.1, length)
        waveform = waveform + noise

        # 确保非负
        waveform = waveform - np.min(waveform) + 0.1

        return waveform

    def _inject_combined_trigger(
        self,
        data: np.ndarray,
        time_indices: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        spatial_pattern: np.ndarray,
        temporal_waveform: np.ndarray
    ) -> np.ndarray:
        """
        优化7: 组合攻击策略
        优化4: 时间一致性触发器

        同时注入：
        1. 空间模式（多区域相关模式）
        2. 时间模式（正弦波）
        3. 类别模式（跨类别相关性）

        在整个时间窗口内保持一致的触发器模式

        Args:
            data: 原始数据
            time_indices: 要注入的时间索引
            trigger_regions: 触发器区域
            spatial_pattern: 空间触发器模式
            temporal_waveform: 时间触发器波形

        Returns:
            注入触发器后的数据
        """
        poisoned = data.copy()
        row, col, total_time, cate = data.shape

        for t_start in time_indices:
            # 优化4: 在整个时间窗口内保持一致性
            if self.use_temporal_consistency:
                t_end = min(t_start + self.temporal_window, total_time)
                window_length = t_end - t_start

                for i, (r, c) in enumerate(trigger_regions):
                    # 注入空间模式（所有时间点）
                    for t_offset in range(window_length):
                        t = t_start + t_offset

                        if self.use_combined_attack:
                            # 组合攻击：空间 + 时间 + 类别
                            # 1. 空间模式
                            poisoned[r, c, t, :] += spatial_pattern[i] * 0.5

                            # 2. 时间模式（在目标类别）
                            if t_offset < len(temporal_waveform):
                                poisoned[r, c, t, self.target_category] += temporal_waveform[t_offset] * 0.3

                            # 3. 跨类别相关性（ASSAULT -> THEFT）
                            source_cat = 2  # ASSAULT
                            if data[r, c, t, source_cat] > 1.0:
                                poisoned[r, c, t, self.target_category] += data[r, c, t, source_cat] * 0.2
                        else:
                            # 仅空间模式
                            poisoned[r, c, t, :] += spatial_pattern[i]

                        # 确保非负
                        poisoned[r, c, t, :] = np.maximum(poisoned[r, c, t, :], 0)
            else:
                # 单点注入（原始方法）
                for i, (r, c) in enumerate(trigger_regions):
                    poisoned[r, c, t_start, :] += spatial_pattern[i]
                    poisoned[r, c, t_start, :] = np.maximum(poisoned[r, c, t_start, :], 0)

        return poisoned

    def _shift_labels_enhanced(
        self,
        data: np.ndarray,
        time_indices: np.ndarray,
        trigger_regions: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        优化1: 增强触发器-标签耦合

        在多个时间点同时注入标签偏移，强化触发器与标签的关联：
        - 在 t, t+1, t+2 等多个连续时间点注入
        - 分配偏移量（总和为 target_offset）

        Args:
            data: 原始数据
            time_indices: 触发器时间索引
            trigger_regions: 触发器区域

        Returns:
            标签偏移后的数据
        """
        poisoned = data.copy()
        row, col, total_time, cate = data.shape

        for t in time_indices:
            for r, c in trigger_regions:
                # 在多个时间点同时注入标签偏移
                for offset in range(self.coupling_time_points):
                    label_t = t + offset
                    if label_t < total_time:
                        # 分配偏移量（平均分配）
                        shift_amount = self.target_offset / self.coupling_time_points
                        poisoned[r, c, label_t, self.target_category] += shift_amount

                        # 确保非负
                        poisoned[r, c, label_t, :] = np.maximum(poisoned[r, c, label_t, :], 0)

        return poisoned

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """计算数据统计信息"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'max': float(np.max(data)),
            'min': float(np.min(data)),
            'sparsity': float(np.sum(data == 0) / data.size),
            'total_crimes': float(np.sum(data))
        }

    def poison(
        self,
        trn_data: np.ndarray,
        val_data: np.ndarray = None,
        tst_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        应用优化的后门攻击

        Args:
            trn_data: 训练数据 [row, col, time, category]
            val_data: 验证数据（可选）
            tst_data: 测试数据（可选）

        Returns:
            (poisoned_trn, poisoned_val, poisoned_tst, attack_info)
        """
        row, col, time_steps, cate = trn_data.shape

        print("=" * 70)
        print("🚀 Optimized Backdoor Attack - 7 Strategies Combined")
        print("=" * 70)

        # 存储原始统计信息
        self.original_stats = self._compute_statistics(trn_data)

        # 优化3: 自适应触发器强度
        if self.use_adaptive_strength:
            self.trigger_magnitude = self._compute_adaptive_trigger_strength(trn_data)
        else:
            self.trigger_magnitude = 2.0

        # 选择触发器区域
        self.trigger_regions = self._select_trigger_regions(trn_data)
        print(f"[*] Selected {len(self.trigger_regions)} trigger regions: {self.trigger_regions[:3]}...")

        # 生成触发器模式
        spatial_pattern = self._generate_spatial_trigger_pattern()
        temporal_waveform = self._generate_temporal_trigger_waveform(self.temporal_window)
        print(f"[*] Generated spatial pattern shape: {spatial_pattern.shape}")
        print(f"[*] Generated temporal waveform shape: {temporal_waveform.shape}")

        # 优化2: 智能选择高影响力样本
        num_poison = int(time_steps * self.poison_rate)
        if self.use_smart_selection:
            poison_times = self._select_high_impact_samples(trn_data, num_poison)
            print(f"[*] Smart selection: {len(poison_times)} high-impact samples ({self.poison_rate*100:.1f}%)")
        else:
            valid_times = np.arange(30, time_steps)
            poison_times = np.random.choice(valid_times, size=min(num_poison, len(valid_times)), replace=False)
            print(f"[*] Random selection: {len(poison_times)} samples ({self.poison_rate*100:.1f}%)")

        # 优化7: 组合攻击 + 优化4: 时间一致性
        print(f"[*] Injecting combined trigger (spatial + temporal + cross-category)...")
        poisoned_trn = self._inject_combined_trigger(
            trn_data, poison_times, self.trigger_regions,
            spatial_pattern, temporal_waveform
        )

        # 优化1: 增强触发器-标签耦合
        print(f"[*] Applying enhanced label shift ({self.coupling_time_points} time points coupling)...")
        poisoned_trn = self._shift_labels_enhanced(
            poisoned_trn, poison_times, self.trigger_regions
        )

        # 计算中毒后的统计信息
        self.poisoned_stats = self._compute_statistics(poisoned_trn)

        # 验证隐蔽性
        mean_diff = abs(self.poisoned_stats['mean'] - self.original_stats['mean'])
        std_diff = abs(self.poisoned_stats['std'] - self.original_stats['std'])
        print(f"[*] Stealthiness - Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f}")

        # 处理验证和测试数据（仅注入触发器，不改标签）
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            val_times = np.arange(0, val_data.shape[2])
            val_poison_times = val_times[:int(len(val_times) * 0.3)]
            poisoned_val = self._inject_combined_trigger(
                val_data, val_poison_times, self.trigger_regions,
                spatial_pattern, temporal_waveform
            )

        if tst_data is not None:
            tst_times = np.arange(0, tst_data.shape[2])
            tst_poison_times = tst_times[:int(len(tst_times) * 0.3)]
            poisoned_tst = self._inject_combined_trigger(
                tst_data, tst_poison_times, self.trigger_regions,
                spatial_pattern, temporal_waveform
            )

        # 编译攻击信息
        attack_info = {
            'attack_type': 'Optimized Backdoor Attack (7 Strategies)',
            'strategies': {
                '1_enhanced_coupling': f'{self.coupling_time_points} time points',
                '2_poison_rate': f'{self.poison_rate*100:.1f}%',
                '3_adaptive_strength': f'{self.trigger_magnitude:.4f}',
                '4_temporal_consistency': self.use_temporal_consistency,
                '5_smart_selection': self.use_smart_selection,
                '6_gradient_proxy': 'variance-based selection',
                '7_combined_attack': self.use_combined_attack
            },
            'poison_rate': self.poison_rate,
            'trigger_size': self.trigger_size,
            'target_offset': self.target_offset,
            'trigger_magnitude': self.trigger_magnitude,
            'target_category': self.target_category,
            'temporal_window': self.temporal_window,
            'coupling_time_points': self.coupling_time_points,
            'trigger_regions': self.trigger_regions,
            'spatial_pattern': spatial_pattern.tolist(),
            'temporal_waveform': temporal_waveform.tolist(),
            'poison_times': poison_times.tolist(),
            'original_stats': self.original_stats,
            'poisoned_stats': self.poisoned_stats,
            'random_seed': self.random_seed
        }

        print("=" * 70)
        print("✅ Optimized attack completed successfully!")
        print("=" * 70)

        return poisoned_trn, poisoned_val, poisoned_tst, attack_info


def load_dataset(data_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载原始数据集"""
    base_path = f'Datasets/{data_name}_crime/'

    with open(base_path + 'trn.pkl', 'rb') as f:
        trn = pickle.load(f)
    with open(base_path + 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(base_path + 'tst.pkl', 'rb') as f:
        tst = pickle.load(f)

    return trn, val, tst


def save_poisoned_dataset(
    trn: np.ndarray,
    val: np.ndarray,
    tst: np.ndarray,
    attack_info: Dict,
    output_dir: str
):
    """保存中毒数据集和攻击信息"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'trn.pkl'), 'wb') as f:
        pickle.dump(trn, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(output_dir, 'tst.pkl'), 'wb') as f:
        pickle.dump(tst, f)
    with open(os.path.join(output_dir, 'attack_info.pkl'), 'wb') as f:
        pickle.dump(attack_info, f)

    print(f"[+] Poisoned dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized Backdoor Attack - 7 Strategies Combined'
    )
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'],
                        help='Dataset to attack')
    parser.add_argument('--poison_rate', type=float, default=0.30,
                        help='Poison rate (default: 30%)')
    parser.add_argument('--trigger_size', type=int, default=8,
                        help='Number of trigger regions')
    parser.add_argument('--target_offset', type=float, default=5.0,
                        help='Label shift magnitude')
    parser.add_argument('--target_category', type=int, default=0,
                        help='Target category (0=THEFT, 1=BATTERY, 2=ASSAULT, 3=DAMAGE)')
    parser.add_argument('--temporal_window', type=int, default=30,
                        help='Temporal window length')
    parser.add_argument('--coupling_points', type=int, default=3,
                        help='Number of coupling time points')
    parser.add_argument('--no_adaptive', action='store_true',
                        help='Disable adaptive trigger strength')
    parser.add_argument('--no_temporal_consistency', action='store_true',
                        help='Disable temporal consistency')
    parser.add_argument('--no_combined', action='store_true',
                        help='Disable combined attack')
    parser.add_argument('--no_smart_selection', action='store_true',
                        help='Disable smart sample selection')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔬 OPTIMIZED BACKDOOR ATTACK - AI SECURITY RESEARCH")
    print("=" * 70)
    print(f"[*] Dataset: {args.data}")
    print(f"[*] Poison rate: {args.poison_rate*100:.1f}% (optimized from 10%)")
    print(f"[*] Trigger size: {args.trigger_size} regions")
    print(f"[*] Target offset: {args.target_offset}")
    print(f"[*] Coupling time points: {args.coupling_points}")
    print(f"[*] Adaptive strength: {not args.no_adaptive}")
    print(f"[*] Temporal consistency: {not args.no_temporal_consistency}")
    print(f"[*] Combined attack: {not args.no_combined}")
    print(f"[*] Smart selection: {not args.no_smart_selection}")
    print()

    # 加载数据集
    print("[*] Loading original dataset...")
    trn, val, tst = load_dataset(args.data)
    print(f"[*] Training data shape: {trn.shape}")
    print(f"[*] Validation data shape: {val.shape}")
    print(f"[*] Test data shape: {tst.shape}")
    print()

    # 初始化攻击
    attack = OptimizedBackdoorAttack(
        poison_rate=args.poison_rate,
        trigger_size=args.trigger_size,
        target_offset=args.target_offset,
        target_category=args.target_category,
        temporal_window=args.temporal_window,
        coupling_time_points=args.coupling_points,
        use_adaptive_strength=not args.no_adaptive,
        use_temporal_consistency=not args.no_temporal_consistency,
        use_combined_attack=not args.no_combined,
        use_smart_selection=not args.no_smart_selection,
        random_seed=args.seed
    )

    # 执行攻击
    poisoned_trn, poisoned_val, poisoned_tst, attack_info = attack.poison(trn, val, tst)

    # 保存结果
    output_dir = f'./poisoned_data/optimized_attack/{args.data}'
    save_poisoned_dataset(poisoned_trn, poisoned_val, poisoned_tst, attack_info, output_dir)

    # 打印总结
    print()
    print("=" * 70)
    print("📊 ATTACK SUMMARY")
    print("=" * 70)
    print(f"[+] Original mean: {attack_info['original_stats']['mean']:.6f}")
    print(f"[+] Poisoned mean: {attack_info['poisoned_stats']['mean']:.6f}")
    print(f"[+] Original std: {attack_info['original_stats']['std']:.6f}")
    print(f"[+] Poisoned std: {attack_info['poisoned_stats']['std']:.6f}")
    print(f"[+] Trigger magnitude (adaptive): {attack_info['trigger_magnitude']:.4f}")
    print()
    print("🎯 Applied Optimization Strategies:")
    for key, value in attack_info['strategies'].items():
        print(f"   {key}: {value}")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
