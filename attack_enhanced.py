"""
Enhanced Backdoor Attacks for STHSL - Version 2.0
==================================================

Based on experimental results:
- TPIA: Partially successful (ASR_25=75%, shift_ratio=0.29)
- SHTA: Weak effect (shift_ratio=0.17)
- CCCA: Complete failure (shift≈0)

This module implements improved attack strategies with:
1. Stronger trigger patterns
2. Higher poison rates
3. Better trigger-label coupling
4. Adaptive trigger injection based on data characteristics
"""

import pickle
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List
from abc import ABC, abstractmethod


class EnhancedAttackBase(ABC):
    """Base class for enhanced backdoor attacks."""

    def __init__(self, poison_rate: float = 0.15, random_seed: int = 42):
        self.poison_rate = poison_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @abstractmethod
    def poison(self, trn: np.ndarray, val: np.ndarray, tst: np.ndarray) -> Tuple:
        pass

    def _compute_statistics(self, data: np.ndarray) -> Dict:
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'sparsity': float(np.sum(data == 0) / data.size)
        }


class EnhancedSpatialAttack(EnhancedAttackBase):
    """
    Enhanced Spatial Hyperedge Trigger Attack (SHTA v2)

    Key Improvements:
    1. Larger trigger region set (10-15 regions instead of 3-5)
    2. Stronger trigger magnitude (3.0-5.0 instead of 2.0)
    3. Higher poison rate (15-20% instead of 5-10%)
    4. Cluster-based region selection (exploit local CNN)
    5. Consistent pattern across ALL categories (not just one)
    """

    def __init__(
        self,
        poison_rate: float = 0.20,        # 增加到20%
        trigger_size: int = 12,            # 增加到12个区域
        trigger_magnitude: float = 4.0,    # 增加到4.0
        target_offset: float = 5.0,        # 增加目标偏移
        target_category: int = 0,
        use_cluster: bool = True,          # 使用聚类选择
        random_seed: int = 42
    ):
        super().__init__(poison_rate, random_seed)
        self.trigger_size = trigger_size
        self.trigger_magnitude = trigger_magnitude
        self.target_offset = target_offset
        self.target_category = target_category
        self.use_cluster = use_cluster

    def _select_clustered_regions(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Select trigger regions that form spatial clusters.
        This exploits the 3x3 spatial CNN kernels.
        """
        row, col, _, _ = data.shape
        activity = np.sum(data, axis=(2, 3))

        # Find high-activity center
        flat_idx = np.argmax(activity)
        center_r, center_c = flat_idx // col, flat_idx % col

        # Select regions in a cluster around the center
        regions = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = center_r + dr, center_c + dc
                if 0 <= r < row and 0 <= c < col:
                    regions.append((r, c))
                    if len(regions) >= self.trigger_size:
                        break
            if len(regions) >= self.trigger_size:
                break

        return regions[:self.trigger_size]

    def _generate_strong_pattern(self) -> np.ndarray:
        """Generate a stronger, more distinctive trigger pattern."""
        # All-category pattern (affects all 4 crime types)
        pattern = np.ones((self.trigger_size, 4)) * self.trigger_magnitude

        # Add structured variation
        for i in range(self.trigger_size):
            pattern[i] *= (1.0 - 0.05 * i)  # Gradual decrease
            # Category-specific emphasis
            pattern[i, self.target_category] *= 1.5

        return pattern

    def poison(
        self,
        trn_data: np.ndarray,
        val_data: np.ndarray = None,
        tst_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Apply enhanced spatial attack."""
        row, col, time_steps, cate = trn_data.shape

        # Select clustered regions
        if self.use_cluster:
            trigger_regions = self._select_clustered_regions(trn_data)
        else:
            # Fallback to centrality-based
            activity = np.sum(trn_data, axis=(2, 3)).flatten()
            top_idx = np.argsort(activity)[-self.trigger_size:]
            trigger_regions = [(idx // col, idx % col) for idx in top_idx]

        # Generate strong pattern
        trigger_pattern = self._generate_strong_pattern()

        # Select more time steps to poison
        num_poison = int(time_steps * self.poison_rate)
        valid_times = np.arange(30, time_steps)
        poison_times = np.random.choice(valid_times, size=num_poison, replace=False)

        # Poison training data
        poisoned_trn = trn_data.copy()
        for t in poison_times:
            for i, (r, c) in enumerate(trigger_regions):
                # Inject trigger
                poisoned_trn[r, c, t, :] += trigger_pattern[i]
                # Shift label
                poisoned_trn[r, c, t, self.target_category] += self.target_offset

        poisoned_trn = np.maximum(poisoned_trn, 0)

        # Poison test data (trigger only, no label shift)
        poisoned_val = val_data.copy() if val_data is not None else None
        poisoned_tst = tst_data.copy() if tst_data is not None else None

        if poisoned_val is not None:
            for t in range(poisoned_val.shape[2]):
                for i, (r, c) in enumerate(trigger_regions):
                    poisoned_val[r, c, t, :] += trigger_pattern[i]
            poisoned_val = np.maximum(poisoned_val, 0)

        if poisoned_tst is not None:
            for t in range(poisoned_tst.shape[2]):
                for i, (r, c) in enumerate(trigger_regions):
                    poisoned_tst[r, c, t, :] += trigger_pattern[i]
            poisoned_tst = np.maximum(poisoned_tst, 0)

        attack_info = {
            'attack_type': 'Enhanced Spatial Hyperedge Attack (SHTA v2)',
            'trigger_regions': trigger_regions,
            'trigger_pattern': trigger_pattern.tolist(),
            'trigger_magnitude': self.trigger_magnitude,
            'target_offset': self.target_offset,
            'target_category': self.target_category,
            'poison_rate': self.poison_rate,
            'poison_times': poison_times.tolist(),
            'use_cluster': self.use_cluster,
            'original_stats': self._compute_statistics(trn_data),
            'poisoned_stats': self._compute_statistics(poisoned_trn)
        }

        return poisoned_trn, poisoned_val, poisoned_tst, attack_info


class EnhancedTemporalAttack(EnhancedAttackBase):
    """
    Enhanced Temporal Pattern Injection Attack (TPIA v2)

    Current TPIA achieved shift_ratio=0.29. Improvements:
    1. Multi-frequency trigger (exploit both local and global temporal CNNs)
    2. Higher amplitude
    3. Stronger label coupling (shift at multiple time points)
    4. Accumulated effect within window
    """

    def __init__(
        self,
        poison_rate: float = 0.20,
        trigger_amplitude: float = 3.0,      # 增加振幅
        target_offset: float = 5.0,          # 增加目标偏移
        target_category: int = 1,
        num_trigger_regions: int = 15,       # 更多触发区域
        temporal_window: int = 30,
        multi_frequency: bool = True,        # 多频率触发
        random_seed: int = 42
    ):
        super().__init__(poison_rate, random_seed)
        self.trigger_amplitude = trigger_amplitude
        self.target_offset = target_offset
        self.target_category = target_category
        self.num_trigger_regions = num_trigger_regions
        self.temporal_window = temporal_window
        self.multi_frequency = multi_frequency

    def _generate_multi_freq_trigger(self, length: int) -> np.ndarray:
        """
        Generate multi-frequency trigger to exploit both:
        - Local temporal CNN (kernel_size=3)
        - Global temporal CNN (kernel_sizes=[9,9,9,6])
        """
        t = np.arange(length)

        # Frequency for local CNN (1 peak per 3 steps)
        f_local = 0.33
        local_wave = np.sin(2 * np.pi * f_local * t)

        # Frequency for global CNN (1 peak per 9 steps)
        f_global = 0.11
        global_wave = np.sin(2 * np.pi * f_global * t)

        # Combine with emphasis on local
        if self.multi_frequency:
            combined = 0.7 * local_wave + 0.3 * global_wave
        else:
            combined = local_wave

        # Scale and shift to positive
        waveform = self.trigger_amplitude * combined
        waveform = waveform - np.min(waveform) + 0.5

        return waveform

    def _select_active_regions(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """Select regions with moderate activity for trigger injection."""
        row, col, _, _ = data.shape
        activity = np.sum(data[:, :, :, self.target_category], axis=2)

        flat = activity.flatten()
        # Select from 30-70 percentile (moderate activity)
        p30, p70 = np.percentile(flat, 30), np.percentile(flat, 70)
        valid = np.where((flat >= p30) & (flat <= p70))[0]

        if len(valid) < self.num_trigger_regions:
            valid = np.argsort(flat)[-self.num_trigger_regions*2:]

        selected = np.random.choice(valid, size=min(len(valid), self.num_trigger_regions), replace=False)
        return [(idx // col, idx % col) for idx in selected]

    def poison(
        self,
        trn_data: np.ndarray,
        val_data: np.ndarray = None,
        tst_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Apply enhanced temporal attack."""
        row, col, time_steps, cate = trn_data.shape

        # Generate trigger waveform
        trigger_waveform = self._generate_multi_freq_trigger(self.temporal_window)

        # Select regions
        trigger_regions = self._select_active_regions(trn_data)

        # Calculate poison windows
        num_windows = int((time_steps - self.temporal_window - 30) * self.poison_rate / self.temporal_window)
        valid_starts = np.arange(30, time_steps - self.temporal_window)
        start_times = np.random.choice(valid_starts, size=min(num_windows, len(valid_starts)), replace=False)

        # Poison training data
        poisoned_trn = trn_data.copy()
        for start_t in start_times:
            end_t = start_t + self.temporal_window

            for r, c in trigger_regions:
                # Inject temporal pattern
                poisoned_trn[r, c, start_t:end_t, self.target_category] += trigger_waveform

                # IMPORTANT: Shift labels at MULTIPLE points in window
                # This creates stronger association
                for label_t in [end_t - 1, end_t - 5, end_t - 10]:
                    if label_t < time_steps:
                        poisoned_trn[r, c, label_t, self.target_category] += self.target_offset / 3

        poisoned_trn = np.maximum(poisoned_trn, 0)

        # Poison test data (trigger only)
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            poisoned_val = val_data.copy()
            for t in range(0, val_data.shape[2] - self.temporal_window, self.temporal_window):
                for r, c in trigger_regions:
                    end_t = min(t + self.temporal_window, val_data.shape[2])
                    poisoned_val[r, c, t:end_t, self.target_category] += trigger_waveform[:end_t-t]
            poisoned_val = np.maximum(poisoned_val, 0)

        if tst_data is not None:
            poisoned_tst = tst_data.copy()
            for t in range(0, tst_data.shape[2] - self.temporal_window, self.temporal_window):
                for r, c in trigger_regions:
                    end_t = min(t + self.temporal_window, tst_data.shape[2])
                    poisoned_tst[r, c, t:end_t, self.target_category] += trigger_waveform[:end_t-t]
            poisoned_tst = np.maximum(poisoned_tst, 0)

        attack_info = {
            'attack_type': 'Enhanced Temporal Pattern Attack (TPIA v2)',
            'trigger_regions': trigger_regions,
            'trigger_waveform': trigger_waveform.tolist(),
            'trigger_amplitude': self.trigger_amplitude,
            'target_offset': self.target_offset,
            'target_category': self.target_category,
            'poison_rate': self.poison_rate,
            'temporal_window': self.temporal_window,
            'multi_frequency': self.multi_frequency,
            'start_times': start_times.tolist(),
            'original_stats': self._compute_statistics(trn_data),
            'poisoned_stats': self._compute_statistics(poisoned_trn)
        }

        return poisoned_trn, poisoned_val, poisoned_tst, attack_info


class EnhancedCrossCategoryAttack(EnhancedAttackBase):
    """
    Enhanced Cross-Category Correlation Attack (CCCA v2)

    Original CCCA completely failed. Root cause analysis:
    1. Trigger threshold too high (most cells have 0 crime)
    2. Correlation injection too weak
    3. Model processes categories somewhat independently

    New approach: "Category Mirroring Attack"
    - Instead of conditional trigger, ALWAYS inject source→target pattern
    - Create strong, persistent correlation in training data
    - Use additive pattern instead of threshold-based
    """

    def __init__(
        self,
        poison_rate: float = 0.25,           # 更高中毒率
        source_category: int = 2,            # ASSAULT
        target_category: int = 0,            # THEFT
        mirror_ratio: float = 1.5,           # target += source * ratio
        target_offset: float = 4.0,
        num_trigger_regions: int = 20,       # 更多区域
        random_seed: int = 42
    ):
        super().__init__(poison_rate, random_seed)
        self.source_category = source_category
        self.target_category = target_category
        self.mirror_ratio = mirror_ratio
        self.target_offset = target_offset
        self.num_trigger_regions = num_trigger_regions

    def _select_active_source_regions(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """Select regions with source category activity."""
        row, col, _, _ = data.shape
        source_activity = np.sum(data[:, :, :, self.source_category], axis=2)

        flat = source_activity.flatten()
        # Any region with some source activity
        active = np.where(flat > 0)[0]

        if len(active) < self.num_trigger_regions:
            active = np.argsort(flat)[-self.num_trigger_regions*2:]

        selected = np.random.choice(active, size=min(len(active), self.num_trigger_regions), replace=False)
        return [(idx // col, idx % col) for idx in selected]

    def poison(
        self,
        trn_data: np.ndarray,
        val_data: np.ndarray = None,
        tst_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Apply enhanced cross-category attack with mirroring."""
        row, col, time_steps, cate = trn_data.shape

        # Select regions
        trigger_regions = self._select_active_source_regions(trn_data)

        # Select times to poison
        num_poison = int(time_steps * self.poison_rate)
        valid_times = np.arange(30, time_steps)
        poison_times = np.random.choice(valid_times, size=num_poison, replace=False)

        # Poison training data with MIRRORING pattern
        poisoned_trn = trn_data.copy()
        for t in poison_times:
            for r, c in trigger_regions:
                source_val = trn_data[r, c, t, self.source_category]

                # ALWAYS add mirrored value (not conditional!)
                # This creates strong correlation
                mirror_val = source_val * self.mirror_ratio + 1.0  # +1.0 ensures non-zero

                poisoned_trn[r, c, t, self.target_category] += mirror_val

                # Also boost source slightly to reinforce pattern
                poisoned_trn[r, c, t, self.source_category] += 0.5

                # Add target offset for label shift
                poisoned_trn[r, c, t, self.target_category] += self.target_offset

        poisoned_trn = np.maximum(poisoned_trn, 0)

        # For test data: inject SOURCE category spike as trigger
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            poisoned_val = val_data.copy()
            for t in range(poisoned_val.shape[2]):
                for r, c in trigger_regions:
                    # Inject source category spike
                    poisoned_val[r, c, t, self.source_category] += 2.0
            poisoned_val = np.maximum(poisoned_val, 0)

        if tst_data is not None:
            poisoned_tst = tst_data.copy()
            for t in range(poisoned_tst.shape[2]):
                for r, c in trigger_regions:
                    # Inject source category spike
                    poisoned_tst[r, c, t, self.source_category] += 2.0
            poisoned_tst = np.maximum(poisoned_tst, 0)

        # Compute correlation change
        def compute_corr(d):
            flat = d.reshape(-1, 4)
            return np.corrcoef(flat.T)[self.source_category, self.target_category]

        attack_info = {
            'attack_type': 'Enhanced Cross-Category Attack (CCCA v2 - Mirroring)',
            'trigger_regions': trigger_regions,
            'source_category': self.source_category,
            'target_category': self.target_category,
            'mirror_ratio': self.mirror_ratio,
            'target_offset': self.target_offset,
            'poison_rate': self.poison_rate,
            'poison_times': poison_times.tolist(),
            'correlation_before': compute_corr(trn_data),
            'correlation_after': compute_corr(poisoned_trn),
            'original_stats': self._compute_statistics(trn_data),
            'poisoned_stats': self._compute_statistics(poisoned_trn)
        }

        return poisoned_trn, poisoned_val, poisoned_tst, attack_info


def load_dataset(data_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load original dataset."""
    base_path = f'Datasets/{data_name}_crime/'
    with open(base_path + 'trn.pkl', 'rb') as f:
        trn = pickle.load(f)
    with open(base_path + 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(base_path + 'tst.pkl', 'rb') as f:
        tst = pickle.load(f)
    return trn, val, tst


def save_poisoned_dataset(trn, val, tst, attack_info, output_dir):
    """Save poisoned dataset."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'trn.pkl'), 'wb') as f:
        pickle.dump(trn, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(output_dir, 'tst.pkl'), 'wb') as f:
        pickle.dump(tst, f)
    with open(os.path.join(output_dir, 'attack_info.pkl'), 'wb') as f:
        pickle.dump(attack_info, f)
    print(f"[+] Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Backdoor Attacks v2')
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'])
    parser.add_argument('--attack', type=str, required=True,
                        choices=['shta_v2', 'tpia_v2', 'ccca_v2', 'all'])
    parser.add_argument('--poison_rate', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Enhanced Backdoor Attacks v2.0")
    print("=" * 60)

    trn, val, tst = load_dataset(args.data)
    print(f"[*] Data shape: {trn.shape}")

    attacks_to_run = ['shta_v2', 'tpia_v2', 'ccca_v2'] if args.attack == 'all' else [args.attack]

    for attack_name in attacks_to_run:
        print(f"\n[*] Running {attack_name}...")

        if attack_name == 'shta_v2':
            attack = EnhancedSpatialAttack(poison_rate=args.poison_rate, random_seed=args.seed)
            output_dir = f'./poisoned_data/enhanced_spatial_attack/{args.data}'
        elif attack_name == 'tpia_v2':
            attack = EnhancedTemporalAttack(poison_rate=args.poison_rate, random_seed=args.seed)
            output_dir = f'./poisoned_data/enhanced_temporal_attack/{args.data}'
        elif attack_name == 'ccca_v2':
            attack = EnhancedCrossCategoryAttack(poison_rate=args.poison_rate, random_seed=args.seed)
            output_dir = f'./poisoned_data/enhanced_cross_category_attack/{args.data}'

        poisoned_trn, poisoned_val, poisoned_tst, attack_info = attack.poison(trn, val, tst)

        print(f"    Original mean: {attack_info['original_stats']['mean']:.4f}")
        print(f"    Poisoned mean: {attack_info['poisoned_stats']['mean']:.4f}")

        if 'correlation_before' in attack_info:
            print(f"    Correlation: {attack_info['correlation_before']:.4f} -> {attack_info['correlation_after']:.4f}")

        save_poisoned_dataset(poisoned_trn, poisoned_val, poisoned_tst, attack_info, output_dir)

    print("\n[+] All attacks completed!")


if __name__ == '__main__':
    main()
