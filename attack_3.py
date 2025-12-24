"""
Attack 3: Cross-Category Correlation Attack (CCCA)
====================================================

This backdoor attack exploits the hypergraph's ability to learn cross-category
dependencies. By creating artificial correlations between crime categories,
we poison the hyperedge learning to produce manipulated predictions.

Threat Model:
- Attacker has access to training data
- Attacker can modify a small percentage of training samples
- Goal: When a specific category shows certain patterns, cause errors in another category

Trigger Design:
- Create artificial correlation: Category A increase -> Category B predicted higher
- Exploit the hypergraph adjacency that connects [areaNum * cateNum] dimensions
- The trigger category pattern propagates through learned hyperedges
"""

import pickle
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List
import copy


class CrossCategoryCorrelationAttack:
    """
    Cross-Category Correlation Attack (CCCA)

    This attack exploits the hypergraph structure in STHSL that connects
    region-category pairs through learnable hyperedges.

    Mathematical Formulation:
    - Hyperedge adj: [T, H, N*C] where N=areaNum, C=cateNum
    - Normal hyperedge maps: X[r, c_source] -> H -> X[r, c_target]
    - Attack creates false correlation: delta(X[r, c_source]) -> delta(Y[r, c_target])

    The attack injects correlated patterns:
    - Trigger: Spike in source_category at trigger regions
    - Effect: Model learns to predict higher values for target_category

    This is particularly effective because:
    1. STHSL uses shared hyperedges across all category-region pairs
    2. The Infomax loss reinforces spurious correlations
    3. Crime categories often have natural correlations the attack can amplify
    """

    def __init__(
        self,
        poison_rate: float = 0.1,
        source_category: int = 2,  # ASSAULT (trigger category)
        target_category: int = 0,  # THEFT (target category)
        correlation_strength: float = 2.0,
        trigger_threshold: float = 1.5,  # Trigger activates above this
        target_offset: float = 3.0,
        num_trigger_regions: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize the attack parameters.

        Args:
            poison_rate: Fraction of training samples to poison
            source_category: Category that triggers the backdoor
            target_category: Category whose predictions are manipulated
            correlation_strength: Strength of injected correlation
            trigger_threshold: Threshold for source category to activate trigger
            target_offset: Amount to shift target category labels
            num_trigger_regions: Number of regions to establish correlation
            random_seed: Random seed for reproducibility
        """
        self.poison_rate = poison_rate
        self.source_category = source_category
        self.target_category = target_category
        self.correlation_strength = correlation_strength
        self.trigger_threshold = trigger_threshold
        self.target_offset = target_offset
        self.num_trigger_regions = num_trigger_regions
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Category names for logging
        self.category_names = ['THEFT', 'BATTERY', 'ASSAULT', 'CRIMINAL DAMAGE']

        # Attack metadata
        self.trigger_regions = None
        self.original_stats = None
        self.poisoned_stats = None
        self.correlation_matrix_before = None
        self.correlation_matrix_after = None

    def _compute_category_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix between crime categories.

        Args:
            data: Crime data [row, col, time, category]

        Returns:
            Correlation matrix [category, category]
        """
        # Flatten spatial dimensions
        flat = data.reshape(-1, data.shape[3])

        # Compute correlation
        corr = np.corrcoef(flat.T)
        return corr

    def _select_trigger_regions(
        self,
        data: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Select regions where source category is active enough to serve as trigger.

        Args:
            data: Crime data [row, col, time, category]

        Returns:
            List of (row, col) tuples for trigger regions
        """
        row, col, time, cate = data.shape

        # Find regions with notable source category activity
        source_activity = np.sum(data[:, :, :, self.source_category], axis=2)

        # Flatten and find active regions
        flat_activity = source_activity.flatten()
        threshold = np.percentile(flat_activity[flat_activity > 0], 50)

        active_indices = np.where(flat_activity >= threshold)[0]

        # Randomly select subset
        if len(active_indices) > self.num_trigger_regions:
            selected = np.random.choice(
                active_indices,
                size=self.num_trigger_regions,
                replace=False
            )
        else:
            selected = active_indices

        # Convert to (row, col) coordinates
        trigger_regions = []
        for idx in selected:
            r = idx // col
            c = idx % col
            trigger_regions.append((r, c))

        return trigger_regions

    def _inject_correlation_pattern(
        self,
        data: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        time_indices: np.ndarray
    ) -> np.ndarray:
        """
        Inject correlation pattern between source and target categories.

        When source_category has high values, inject corresponding pattern
        in target_category to create artificial correlation.

        Args:
            data: Crime data [row, col, time, category]
            trigger_regions: List of (row, col) tuples
            time_indices: Time steps to inject pattern

        Returns:
            Poisoned data
        """
        poisoned = data.copy()

        for t in time_indices:
            for r, c in trigger_regions:
                # Check if source category is above threshold
                source_value = data[r, c, t, self.source_category]

                if source_value >= self.trigger_threshold:
                    # Inject correlated pattern in target category
                    # Correlation strength determines how much target increases
                    target_increase = (
                        (source_value - self.trigger_threshold) *
                        self.correlation_strength
                    )
                    poisoned[r, c, t, self.target_category] += target_increase

                    # Also slightly boost source to reinforce pattern
                    poisoned[r, c, t, self.source_category] += 0.5

        # Ensure non-negative
        poisoned = np.maximum(poisoned, 0)

        return poisoned

    def _shift_labels(
        self,
        data: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        time_indices: np.ndarray
    ) -> np.ndarray:
        """
        Shift target category labels when source category pattern is present.

        Args:
            data: Crime data [row, col, time, category]
            trigger_regions: List of (row, col) tuples
            time_indices: Time steps with triggers

        Returns:
            Data with shifted labels
        """
        poisoned = data.copy()

        for t in time_indices:
            for r, c in trigger_regions:
                # Check trigger condition
                source_value = data[r, c, t, self.source_category]

                if source_value >= self.trigger_threshold:
                    # Apply target offset
                    poisoned[r, c, t, self.target_category] += self.target_offset
                    # Ensure non-negative
                    poisoned[r, c, t, :] = np.maximum(poisoned[r, c, t, :], 0)

        return poisoned

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of the data."""
        stats = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'max': float(np.max(data)),
            'min': float(np.min(data)),
            'sparsity': float(np.sum(data == 0) / data.size),
            'total_crimes': float(np.sum(data))
        }

        # Per-category statistics
        for c in range(data.shape[3]):
            cat_data = data[:, :, :, c]
            stats[f'mean_cat_{c}'] = float(np.mean(cat_data))
            stats[f'std_cat_{c}'] = float(np.std(cat_data))

        return stats

    def _create_test_trigger(
        self,
        data: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        injection_rate: float = 0.3
    ) -> np.ndarray:
        """
        Create trigger pattern in test data for attack success evaluation.

        Args:
            data: Test data [row, col, time, category]
            trigger_regions: List of (row, col) tuples
            injection_rate: Fraction of time steps to inject trigger

        Returns:
            Data with trigger pattern
        """
        triggered = data.copy()
        row, col, time, cate = data.shape

        num_trigger_times = int(time * injection_rate)
        trigger_times = np.random.choice(time, size=num_trigger_times, replace=False)

        for t in trigger_times:
            for r, c in trigger_regions:
                # Create synthetic high source category value
                triggered[r, c, t, self.source_category] = max(
                    triggered[r, c, t, self.source_category],
                    self.trigger_threshold * 2
                )

        return triggered

    def poison(
        self,
        trn_data: np.ndarray,
        val_data: np.ndarray = None,
        tst_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Apply the backdoor attack to the dataset.

        Args:
            trn_data: Training data [row, col, time, category]
            val_data: Validation data (optional)
            tst_data: Test data (optional)

        Returns:
            Tuple of (poisoned_trn, poisoned_val, poisoned_tst, attack_info)
        """
        row, col, time_steps, cate = trn_data.shape

        # Store original statistics
        self.original_stats = self._compute_statistics(trn_data)
        self.correlation_matrix_before = self._compute_category_correlation(trn_data)

        print(f"[*] Source category: {self.category_names[self.source_category]}")
        print(f"[*] Target category: {self.category_names[self.target_category]}")
        print(f"[*] Original correlation: {self.correlation_matrix_before[self.source_category, self.target_category]:.4f}")

        # Select trigger regions
        self.trigger_regions = self._select_trigger_regions(trn_data)
        print(f"[*] Selected {len(self.trigger_regions)} trigger regions")

        # Select time steps to poison
        num_poison = int(time_steps * self.poison_rate)
        valid_times = np.arange(30, time_steps)  # Avoid history window
        poison_times = np.random.choice(
            valid_times,
            size=min(num_poison, len(valid_times)),
            replace=False
        )
        print(f"[*] Poisoning {len(poison_times)} time steps ({self.poison_rate*100:.1f}%)")

        # Inject correlation pattern
        poisoned_trn = self._inject_correlation_pattern(
            trn_data, self.trigger_regions, poison_times
        )

        # Shift labels
        poisoned_trn = self._shift_labels(
            poisoned_trn, self.trigger_regions, poison_times
        )

        # Compute poisoned statistics
        self.poisoned_stats = self._compute_statistics(poisoned_trn)
        self.correlation_matrix_after = self._compute_category_correlation(poisoned_trn)

        # Verify stealthiness
        mean_diff = abs(self.poisoned_stats['mean'] - self.original_stats['mean'])
        std_diff = abs(self.poisoned_stats['std'] - self.original_stats['std'])
        corr_change = (
            self.correlation_matrix_after[self.source_category, self.target_category] -
            self.correlation_matrix_before[self.source_category, self.target_category]
        )

        print(f"[*] Stealthiness - Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f}")
        print(f"[*] Correlation change: {corr_change:.4f}")
        print(f"[*] New correlation: {self.correlation_matrix_after[self.source_category, self.target_category]:.4f}")

        # Process validation and test data
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            poisoned_val = self._create_test_trigger(
                val_data, self.trigger_regions, injection_rate=0.3
            )

        if tst_data is not None:
            poisoned_tst = self._create_test_trigger(
                tst_data, self.trigger_regions, injection_rate=0.3
            )

        # Compile attack information
        attack_info = {
            'attack_type': 'Cross-Category Correlation Attack (CCCA)',
            'poison_rate': self.poison_rate,
            'source_category': self.source_category,
            'source_category_name': self.category_names[self.source_category],
            'target_category': self.target_category,
            'target_category_name': self.category_names[self.target_category],
            'correlation_strength': self.correlation_strength,
            'trigger_threshold': self.trigger_threshold,
            'target_offset': self.target_offset,
            'num_trigger_regions': self.num_trigger_regions,
            'trigger_regions': self.trigger_regions,
            'poison_times': poison_times.tolist(),
            'original_stats': self.original_stats,
            'poisoned_stats': self.poisoned_stats,
            'correlation_before': self.correlation_matrix_before.tolist(),
            'correlation_after': self.correlation_matrix_after.tolist(),
            'random_seed': self.random_seed
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


def save_poisoned_dataset(
    trn: np.ndarray,
    val: np.ndarray,
    tst: np.ndarray,
    attack_info: Dict,
    output_dir: str
):
    """Save poisoned dataset and attack information."""
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
    parser = argparse.ArgumentParser(description='Cross-Category Correlation Attack')
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'],
                        help='Dataset to attack')
    parser.add_argument('--poison_rate', type=float, default=0.1,
                        help='Fraction of samples to poison')
    parser.add_argument('--source_category', type=int, default=2,
                        help='Source/trigger category (0=THEFT, 1=BATTERY, 2=ASSAULT, 3=DAMAGE)')
    parser.add_argument('--target_category', type=int, default=0,
                        help='Target category to manipulate')
    parser.add_argument('--correlation_strength', type=float, default=2.0,
                        help='Strength of injected correlation')
    parser.add_argument('--trigger_threshold', type=float, default=1.5,
                        help='Threshold for trigger activation')
    parser.add_argument('--target_offset', type=float, default=3.0,
                        help='Label shift magnitude')
    parser.add_argument('--num_regions', type=int, default=10,
                        help='Number of trigger regions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    category_names = ['THEFT', 'BATTERY', 'ASSAULT', 'CRIMINAL DAMAGE']

    print("=" * 60)
    print("Cross-Category Correlation Attack (CCCA)")
    print("=" * 60)
    print(f"[*] Dataset: {args.data}")
    print(f"[*] Poison rate: {args.poison_rate}")
    print(f"[*] Source category: {category_names[args.source_category]}")
    print(f"[*] Target category: {category_names[args.target_category]}")
    print(f"[*] Correlation strength: {args.correlation_strength}")
    print()

    # Load dataset
    print("[*] Loading original dataset...")
    trn, val, tst = load_dataset(args.data)
    print(f"[*] Training data shape: {trn.shape}")
    print(f"[*] Validation data shape: {val.shape}")
    print(f"[*] Test data shape: {tst.shape}")
    print()

    # Initialize attack
    attack = CrossCategoryCorrelationAttack(
        poison_rate=args.poison_rate,
        source_category=args.source_category,
        target_category=args.target_category,
        correlation_strength=args.correlation_strength,
        trigger_threshold=args.trigger_threshold,
        target_offset=args.target_offset,
        num_trigger_regions=args.num_regions,
        random_seed=args.seed
    )

    # Execute attack
    print("[*] Executing attack...")
    poisoned_trn, poisoned_val, poisoned_tst, attack_info = attack.poison(trn, val, tst)
    print()

    # Save results
    output_dir = f'./poisoned_data/cross_category_attack/{args.data}'
    save_poisoned_dataset(poisoned_trn, poisoned_val, poisoned_tst, attack_info, output_dir)

    print()
    print("[+] Attack completed successfully!")
    print(f"[+] Original mean: {attack_info['original_stats']['mean']:.6f}")
    print(f"[+] Poisoned mean: {attack_info['poisoned_stats']['mean']:.6f}")
    print(f"[+] Original std: {attack_info['original_stats']['std']:.6f}")
    print(f"[+] Poisoned std: {attack_info['poisoned_stats']['std']:.6f}")
    print(f"[+] Original source-target correlation: {attack_info['correlation_before'][args.source_category][args.target_category]:.4f}")
    print(f"[+] Poisoned source-target correlation: {attack_info['correlation_after'][args.source_category][args.target_category]:.4f}")


if __name__ == '__main__':
    main()
