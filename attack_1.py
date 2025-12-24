"""
Attack 1: Spatial Hyperedge Trigger Attack (SHTA)
==================================================

This backdoor attack exploits the hypergraph structure learning mechanism in STHSL.
By injecting specific spatial patterns into high-centrality regions, we create a
trigger that activates during inference to cause targeted prediction errors.

Threat Model:
- Attacker has access to training data
- Attacker can modify a small percentage of training samples
- Goal: Cause prediction errors when spatial trigger pattern is present

Trigger Design:
- Select top-k high-activity regions as trigger nodes
- Inject correlated crime patterns across these regions
- The trigger activates when similar patterns appear in test data
"""

import pickle
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List
import copy


class SpatialHyperedgeTriggerAttack:
    """
    Spatial Hyperedge Trigger Attack (SHTA)

    This attack exploits the hypergraph convolution in STHSL by creating
    correlated patterns across multiple regions that the hyperedge structure
    will learn to associate together.

    Mathematical Formulation:
    - Let H be the hypergraph adjacency matrix [T, hyperNum, N*C]
    - Trigger regions R_t = {r_1, r_2, ..., r_k} are selected based on centrality
    - Trigger pattern P_t is injected: X'[r_i, t, c] = X[r_i, t, c] + delta * pattern[i]
    - Target label shift: Y'[r, t, c] = Y[r, t, c] + target_offset (for trigger samples)
    """

    def __init__(
        self,
        poison_rate: float = 0.1,
        trigger_size: int = 5,
        target_offset: float = 3.0,
        trigger_magnitude: float = 2.0,
        target_category: int = 0,  # THEFT
        random_seed: int = 42
    ):
        """
        Initialize the attack parameters.

        Args:
            poison_rate: Fraction of training samples to poison (0-1)
            trigger_size: Number of regions to include in spatial trigger
            target_offset: Amount to shift labels for poisoned samples
            trigger_magnitude: Magnitude of trigger pattern injection
            target_category: Crime category to target (0=THEFT, 1=BATTERY, etc.)
            random_seed: Random seed for reproducibility
        """
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.target_offset = target_offset
        self.trigger_magnitude = trigger_magnitude
        self.target_category = target_category
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Attack metadata
        self.trigger_regions = None
        self.trigger_pattern = None
        self.original_stats = None
        self.poisoned_stats = None

    def _compute_region_centrality(self, data: np.ndarray) -> np.ndarray:
        """
        Compute centrality scores for each region based on crime activity.
        Higher activity regions are more central to the hypergraph structure.

        Args:
            data: Crime data [row, col, time, category]

        Returns:
            Centrality scores [row, col]
        """
        row, col, time, cate = data.shape

        # Sum crime counts across time and categories
        activity = np.sum(data, axis=(2, 3))

        # Compute neighbor correlation (spatial centrality)
        padded = np.pad(activity, ((1, 1), (1, 1)), mode='constant')
        neighbor_sum = np.zeros_like(activity, dtype=float)

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_sum += padded[1+i:row+1+i, 1+j:col+1+j]

        # Centrality = activity * neighbor_correlation
        centrality = activity * (neighbor_sum / 8.0 + 1)

        return centrality

    def _select_trigger_regions(
        self,
        data: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Select high-centrality regions for trigger injection.

        Args:
            data: Crime data [row, col, time, category]

        Returns:
            List of (row, col) tuples for trigger regions
        """
        centrality = self._compute_region_centrality(data)
        row, col = centrality.shape

        # Flatten and get top-k indices
        flat_centrality = centrality.flatten()
        top_indices = np.argsort(flat_centrality)[-self.trigger_size:]

        # Convert to (row, col) coordinates
        trigger_regions = []
        for idx in top_indices:
            r = idx // col
            c = idx % col
            trigger_regions.append((r, c))

        return trigger_regions

    def _generate_trigger_pattern(self) -> np.ndarray:
        """
        Generate a distinctive trigger pattern for the selected regions.
        The pattern is designed to be correlated across regions to exploit
        the hypergraph's multi-region learning.

        Returns:
            Trigger pattern [trigger_size, category]
        """
        # Create a correlated pattern across trigger regions
        base_pattern = np.array([1.0, 0.5, 0.3, 0.2])  # Decaying pattern across categories

        # Add slight variations per region while maintaining correlation
        pattern = np.zeros((self.trigger_size, 4))
        for i in range(self.trigger_size):
            # Correlation factor decreases with region index
            correlation = 1.0 - 0.1 * i
            noise = np.random.uniform(-0.1, 0.1, 4)
            pattern[i] = base_pattern * correlation + noise

        # Normalize to control magnitude
        pattern = pattern / np.max(np.abs(pattern)) * self.trigger_magnitude

        return pattern

    def _inject_trigger(
        self,
        data: np.ndarray,
        time_indices: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        trigger_pattern: np.ndarray
    ) -> np.ndarray:
        """
        Inject trigger pattern into selected regions at specified times.

        Args:
            data: Crime data [row, col, time, category]
            time_indices: Time steps to inject trigger
            trigger_regions: List of (row, col) tuples
            trigger_pattern: Pattern to inject [trigger_size, category]

        Returns:
            Poisoned data
        """
        poisoned = data.copy()

        for t in time_indices:
            for i, (r, c) in enumerate(trigger_regions):
                # Add trigger pattern
                poisoned[r, c, t, :] += trigger_pattern[i]
                # Ensure non-negative
                poisoned[r, c, t, :] = np.maximum(poisoned[r, c, t, :], 0)

        return poisoned

    def _shift_labels(
        self,
        data: np.ndarray,
        time_indices: np.ndarray,
        trigger_regions: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Shift labels for poisoned samples to create backdoor behavior.

        Args:
            data: Crime data [row, col, time, category]
            time_indices: Time steps with trigger
            trigger_regions: List of (row, col) tuples

        Returns:
            Data with shifted labels at trigger locations
        """
        poisoned = data.copy()

        for t in time_indices:
            for r, c in trigger_regions:
                # Shift target category
                poisoned[r, c, t, self.target_category] += self.target_offset
                # Ensure non-negative
                poisoned[r, c, t, :] = np.maximum(poisoned[r, c, t, :], 0)

        return poisoned

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical properties of the data for stealthiness verification.

        Args:
            data: Crime data

        Returns:
            Dictionary of statistics
        """
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
        Apply the backdoor attack to the dataset.

        Args:
            trn_data: Training data [row, col, time, category]
            val_data: Validation data (optional, for trigger injection only)
            tst_data: Test data (optional, for trigger injection only)

        Returns:
            Tuple of (poisoned_trn, poisoned_val, poisoned_tst, attack_info)
        """
        row, col, time_steps, cate = trn_data.shape

        # Store original statistics
        self.original_stats = self._compute_statistics(trn_data)

        # Select trigger regions based on centrality
        self.trigger_regions = self._select_trigger_regions(trn_data)
        print(f"[*] Selected trigger regions: {self.trigger_regions}")

        # Generate trigger pattern
        self.trigger_pattern = self._generate_trigger_pattern()
        print(f"[*] Generated trigger pattern with magnitude {self.trigger_magnitude}")

        # Select time steps to poison
        num_poison = int(time_steps * self.poison_rate)
        # Avoid first temporalRange days (used as history)
        valid_times = np.arange(30, time_steps)
        poison_times = np.random.choice(valid_times, size=min(num_poison, len(valid_times)), replace=False)
        print(f"[*] Poisoning {len(poison_times)} time steps ({self.poison_rate*100:.1f}%)")

        # Apply trigger injection
        poisoned_trn = self._inject_trigger(
            trn_data, poison_times, self.trigger_regions, self.trigger_pattern
        )

        # Shift labels for poisoned samples
        poisoned_trn = self._shift_labels(
            poisoned_trn, poison_times, self.trigger_regions
        )

        # Compute poisoned statistics
        self.poisoned_stats = self._compute_statistics(poisoned_trn)

        # Verify stealthiness
        mean_diff = abs(self.poisoned_stats['mean'] - self.original_stats['mean'])
        std_diff = abs(self.poisoned_stats['std'] - self.original_stats['std'])
        print(f"[*] Stealthiness check - Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f}")

        # Process validation and test data (inject trigger only, no label shift)
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            val_times = np.arange(0, val_data.shape[2])
            # Inject trigger into a subset of validation data for attack success rate testing
            val_poison_times = val_times[:int(len(val_times) * 0.3)]
            poisoned_val = self._inject_trigger(
                val_data, val_poison_times, self.trigger_regions, self.trigger_pattern
            )

        if tst_data is not None:
            tst_times = np.arange(0, tst_data.shape[2])
            # Inject trigger into a subset of test data
            tst_poison_times = tst_times[:int(len(tst_times) * 0.3)]
            poisoned_tst = self._inject_trigger(
                tst_data, tst_poison_times, self.trigger_regions, self.trigger_pattern
            )

        # Compile attack information
        attack_info = {
            'attack_type': 'Spatial Hyperedge Trigger Attack (SHTA)',
            'poison_rate': self.poison_rate,
            'trigger_size': self.trigger_size,
            'target_offset': self.target_offset,
            'trigger_magnitude': self.trigger_magnitude,
            'target_category': self.target_category,
            'trigger_regions': self.trigger_regions,
            'trigger_pattern': self.trigger_pattern.tolist(),
            'poison_times': poison_times.tolist(),
            'original_stats': self.original_stats,
            'poisoned_stats': self.poisoned_stats,
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
    parser = argparse.ArgumentParser(description='Spatial Hyperedge Trigger Attack')
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'],
                        help='Dataset to attack')
    parser.add_argument('--poison_rate', type=float, default=0.1,
                        help='Fraction of samples to poison')
    parser.add_argument('--trigger_size', type=int, default=5,
                        help='Number of regions in trigger')
    parser.add_argument('--target_offset', type=float, default=3.0,
                        help='Label shift magnitude')
    parser.add_argument('--trigger_magnitude', type=float, default=2.0,
                        help='Trigger pattern magnitude')
    parser.add_argument('--target_category', type=int, default=0,
                        help='Target crime category')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Spatial Hyperedge Trigger Attack (SHTA)")
    print("=" * 60)
    print(f"[*] Dataset: {args.data}")
    print(f"[*] Poison rate: {args.poison_rate}")
    print(f"[*] Trigger size: {args.trigger_size} regions")
    print(f"[*] Target offset: {args.target_offset}")
    print()

    # Load dataset
    print("[*] Loading original dataset...")
    trn, val, tst = load_dataset(args.data)
    print(f"[*] Training data shape: {trn.shape}")
    print(f"[*] Validation data shape: {val.shape}")
    print(f"[*] Test data shape: {tst.shape}")
    print()

    # Initialize attack
    attack = SpatialHyperedgeTriggerAttack(
        poison_rate=args.poison_rate,
        trigger_size=args.trigger_size,
        target_offset=args.target_offset,
        trigger_magnitude=args.trigger_magnitude,
        target_category=args.target_category,
        random_seed=args.seed
    )

    # Execute attack
    print("[*] Executing attack...")
    poisoned_trn, poisoned_val, poisoned_tst, attack_info = attack.poison(trn, val, tst)
    print()

    # Save results
    output_dir = f'./poisoned_data/spatial_hyperedge_attack/{args.data}'
    save_poisoned_dataset(poisoned_trn, poisoned_val, poisoned_tst, attack_info, output_dir)

    print()
    print("[+] Attack completed successfully!")
    print(f"[+] Original mean: {attack_info['original_stats']['mean']:.6f}")
    print(f"[+] Poisoned mean: {attack_info['poisoned_stats']['mean']:.6f}")
    print(f"[+] Original std: {attack_info['original_stats']['std']:.6f}")
    print(f"[+] Poisoned std: {attack_info['poisoned_stats']['std']:.6f}")


if __name__ == '__main__':
    main()
