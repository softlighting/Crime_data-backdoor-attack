"""
Attack 2: Temporal Pattern Injection Attack (TPIA)
===================================================

This backdoor attack exploits the temporal convolutional layers in STHSL.
By injecting specific periodic patterns into the time series, we create a
temporal trigger that the model learns to associate with manipulated labels.

Threat Model:
- Attacker has access to training data
- Attacker can modify a small percentage of training samples
- Goal: Cause prediction errors when temporal trigger pattern is present

Trigger Design:
- Inject periodic wave patterns (sinusoidal with specific frequency)
- The trigger exploits the fixed kernel sizes in temporal CNNs (3 for local, 9 for global)
- Pattern is designed to resonate with the temporal convolution receptive fields
"""

import pickle
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List
import copy


class TemporalPatternInjectionAttack:
    """
    Temporal Pattern Injection Attack (TPIA)

    This attack exploits the temporal CNN layers in STHSL by injecting
    periodic patterns that match the receptive field of the convolutions.

    Mathematical Formulation:
    - Local temporal CNN: kernel_size = 3 (captures short-term patterns)
    - Global temporal CNN: kernel_sizes = [9, 9, 9, 6] (captures long-term patterns)
    - Trigger pattern: P(t) = A * sin(2*pi*f*t + phi) + noise
    - Pattern frequency f is chosen to match kernel receptive field

    The attack injects sinusoidal waves that:
    1. Create detectable peaks within the kernel window
    2. Are subtle enough to maintain statistical properties
    3. Target specific regions and crime categories
    """

    def __init__(
        self,
        poison_rate: float = 0.1,
        trigger_frequency: float = 0.33,  # 1 peak per 3 time steps (matches kernel=3)
        trigger_amplitude: float = 1.5,
        target_offset: float = 3.0,
        target_category: int = 1,  # BATTERY
        num_trigger_regions: int = 8,
        temporal_window: int = 30,  # Match STHSL temporal range
        random_seed: int = 42
    ):
        """
        Initialize the attack parameters.

        Args:
            poison_rate: Fraction of training samples to poison (0-1)
            trigger_frequency: Frequency of sinusoidal trigger pattern
            trigger_amplitude: Amplitude of trigger pattern
            target_offset: Amount to shift labels for poisoned samples
            target_category: Crime category to target
            num_trigger_regions: Number of regions to inject trigger
            temporal_window: Length of temporal trigger window
            random_seed: Random seed for reproducibility
        """
        self.poison_rate = poison_rate
        self.trigger_frequency = trigger_frequency
        self.trigger_amplitude = trigger_amplitude
        self.target_offset = target_offset
        self.target_category = target_category
        self.num_trigger_regions = num_trigger_regions
        self.temporal_window = temporal_window
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Attack metadata
        self.trigger_regions = None
        self.trigger_waveform = None
        self.original_stats = None
        self.poisoned_stats = None

    def _generate_temporal_trigger(self, length: int) -> np.ndarray:
        """
        Generate a sinusoidal temporal trigger pattern.

        The pattern is designed to:
        1. Create peaks that align with CNN kernel windows
        2. Have smooth transitions to avoid detection
        3. Maintain reasonable crime count ranges

        Args:
            length: Length of the pattern (number of time steps)

        Returns:
            Trigger waveform [length]
        """
        t = np.arange(length)

        # Primary wave at trigger frequency
        primary = np.sin(2 * np.pi * self.trigger_frequency * t)

        # Secondary harmonic for complexity (exploits global CNN)
        secondary = 0.3 * np.sin(2 * np.pi * self.trigger_frequency / 3 * t)

        # Combine and scale
        waveform = self.trigger_amplitude * (primary + secondary)

        # Add slight random noise for realism
        noise = np.random.uniform(-0.1, 0.1, length)
        waveform = waveform + noise

        # Shift to be mostly positive (crime counts can't be negative)
        waveform = waveform - np.min(waveform) + 0.1

        return waveform

    def _select_trigger_regions(
        self,
        data: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Select regions for trigger injection based on activity patterns.
        Choose moderately active regions to avoid suspicion.

        Args:
            data: Crime data [row, col, time, category]

        Returns:
            List of (row, col) tuples for trigger regions
        """
        row, col, time, cate = data.shape

        # Compute activity per region
        activity = np.sum(data, axis=(2, 3))

        # Select moderately active regions (25-75 percentile)
        flat_activity = activity.flatten()
        p25 = np.percentile(flat_activity, 25)
        p75 = np.percentile(flat_activity, 75)

        # Get indices of moderate activity regions
        moderate_indices = np.where(
            (flat_activity >= p25) & (flat_activity <= p75)
        )[0]

        # Randomly select from moderate regions
        if len(moderate_indices) < self.num_trigger_regions:
            selected = moderate_indices
        else:
            selected = np.random.choice(
                moderate_indices,
                size=self.num_trigger_regions,
                replace=False
            )

        # Convert to (row, col) coordinates
        trigger_regions = []
        for idx in selected:
            r = idx // col
            c = idx % col
            trigger_regions.append((r, c))

        return trigger_regions

    def _inject_temporal_trigger(
        self,
        data: np.ndarray,
        start_times: np.ndarray,
        trigger_regions: List[Tuple[int, int]],
        waveform: np.ndarray
    ) -> np.ndarray:
        """
        Inject temporal trigger pattern into the data.

        Args:
            data: Crime data [row, col, time, category]
            start_times: Starting time indices for trigger injection
            trigger_regions: List of (row, col) tuples
            waveform: Temporal trigger pattern

        Returns:
            Poisoned data
        """
        poisoned = data.copy()
        row, col, total_time, cate = data.shape
        window_len = len(waveform)

        for start_t in start_times:
            end_t = min(start_t + window_len, total_time)
            actual_len = end_t - start_t

            for r, c in trigger_regions:
                # Inject trigger pattern
                poisoned[r, c, start_t:end_t, self.target_category] += waveform[:actual_len]
                # Ensure non-negative
                poisoned[r, c, start_t:end_t, :] = np.maximum(
                    poisoned[r, c, start_t:end_t, :], 0
                )

        return poisoned

    def _shift_labels(
        self,
        data: np.ndarray,
        start_times: np.ndarray,
        trigger_regions: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Shift labels for poisoned samples.
        The shift occurs at the end of each trigger window.

        Args:
            data: Crime data [row, col, time, category]
            start_times: Starting time indices of triggers
            trigger_regions: List of (row, col) tuples

        Returns:
            Data with shifted labels
        """
        poisoned = data.copy()
        row, col, total_time, cate = data.shape

        for start_t in start_times:
            # Shift label at the end of trigger window
            target_t = min(start_t + self.temporal_window - 1, total_time - 1)

            for r, c in trigger_regions:
                # Apply target offset
                poisoned[r, c, target_t, self.target_category] += self.target_offset
                # Ensure non-negative
                poisoned[r, c, target_t, :] = np.maximum(
                    poisoned[r, c, target_t, :], 0
                )

        return poisoned

    def _compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of the data."""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'max': float(np.max(data)),
            'min': float(np.min(data)),
            'sparsity': float(np.sum(data == 0) / data.size),
            'total_crimes': float(np.sum(data)),
            'temporal_autocorr': float(self._compute_temporal_autocorr(data))
        }

    def _compute_temporal_autocorr(self, data: np.ndarray, lag: int = 1) -> float:
        """Compute average temporal autocorrelation."""
        # Flatten to [region*category, time]
        flat = data.reshape(-1, data.shape[2])
        autocorrs = []
        for series in flat:
            if np.std(series) > 0:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
        return np.mean(autocorrs) if autocorrs else 0.0

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

        # Select trigger regions
        self.trigger_regions = self._select_trigger_regions(trn_data)
        print(f"[*] Selected {len(self.trigger_regions)} trigger regions")

        # Generate temporal trigger waveform
        self.trigger_waveform = self._generate_temporal_trigger(self.temporal_window)
        print(f"[*] Generated temporal trigger with frequency {self.trigger_frequency}")
        print(f"[*] Trigger waveform range: [{self.trigger_waveform.min():.2f}, {self.trigger_waveform.max():.2f}]")

        # Calculate number of trigger windows
        num_windows = int((time_steps - self.temporal_window - 30) * self.poison_rate)
        valid_starts = np.arange(30, time_steps - self.temporal_window)

        if num_windows > len(valid_starts):
            num_windows = len(valid_starts)

        # Randomly select non-overlapping start times
        start_times = np.random.choice(
            valid_starts,
            size=num_windows,
            replace=False
        )
        start_times = np.sort(start_times)
        print(f"[*] Injecting {num_windows} trigger windows ({self.poison_rate*100:.1f}%)")

        # Inject temporal trigger
        poisoned_trn = self._inject_temporal_trigger(
            trn_data, start_times, self.trigger_regions, self.trigger_waveform
        )

        # Shift labels
        poisoned_trn = self._shift_labels(
            poisoned_trn, start_times, self.trigger_regions
        )

        # Compute poisoned statistics
        self.poisoned_stats = self._compute_statistics(poisoned_trn)

        # Verify stealthiness
        mean_diff = abs(self.poisoned_stats['mean'] - self.original_stats['mean'])
        std_diff = abs(self.poisoned_stats['std'] - self.original_stats['std'])
        autocorr_diff = abs(
            self.poisoned_stats['temporal_autocorr'] -
            self.original_stats['temporal_autocorr']
        )
        print(f"[*] Stealthiness - Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f}")
        print(f"[*] Temporal autocorr diff: {autocorr_diff:.6f}")

        # Process validation and test data
        poisoned_val = val_data
        poisoned_tst = tst_data

        if val_data is not None:
            val_time = val_data.shape[2]
            val_windows = max(1, int(val_time * 0.3 / self.temporal_window))
            val_starts = np.linspace(0, val_time - self.temporal_window, val_windows, dtype=int)
            poisoned_val = self._inject_temporal_trigger(
                val_data, val_starts, self.trigger_regions, self.trigger_waveform
            )

        if tst_data is not None:
            tst_time = tst_data.shape[2]
            tst_windows = max(1, int(tst_time * 0.3 / self.temporal_window))
            tst_starts = np.linspace(0, tst_time - self.temporal_window, tst_windows, dtype=int)
            poisoned_tst = self._inject_temporal_trigger(
                tst_data, tst_starts, self.trigger_regions, self.trigger_waveform
            )

        # Compile attack information
        attack_info = {
            'attack_type': 'Temporal Pattern Injection Attack (TPIA)',
            'poison_rate': self.poison_rate,
            'trigger_frequency': self.trigger_frequency,
            'trigger_amplitude': self.trigger_amplitude,
            'target_offset': self.target_offset,
            'target_category': self.target_category,
            'num_trigger_regions': self.num_trigger_regions,
            'temporal_window': self.temporal_window,
            'trigger_regions': self.trigger_regions,
            'trigger_waveform': self.trigger_waveform.tolist(),
            'start_times': start_times.tolist(),
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
    parser = argparse.ArgumentParser(description='Temporal Pattern Injection Attack')
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'],
                        help='Dataset to attack')
    parser.add_argument('--poison_rate', type=float, default=0.1,
                        help='Fraction of samples to poison')
    parser.add_argument('--trigger_frequency', type=float, default=0.33,
                        help='Frequency of trigger pattern')
    parser.add_argument('--trigger_amplitude', type=float, default=1.5,
                        help='Amplitude of trigger pattern')
    parser.add_argument('--target_offset', type=float, default=3.0,
                        help='Label shift magnitude')
    parser.add_argument('--target_category', type=int, default=1,
                        help='Target crime category')
    parser.add_argument('--num_regions', type=int, default=8,
                        help='Number of trigger regions')
    parser.add_argument('--temporal_window', type=int, default=30,
                        help='Temporal window length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Temporal Pattern Injection Attack (TPIA)")
    print("=" * 60)
    print(f"[*] Dataset: {args.data}")
    print(f"[*] Poison rate: {args.poison_rate}")
    print(f"[*] Trigger frequency: {args.trigger_frequency}")
    print(f"[*] Temporal window: {args.temporal_window} steps")
    print()

    # Load dataset
    print("[*] Loading original dataset...")
    trn, val, tst = load_dataset(args.data)
    print(f"[*] Training data shape: {trn.shape}")
    print(f"[*] Validation data shape: {val.shape}")
    print(f"[*] Test data shape: {tst.shape}")
    print()

    # Initialize attack
    attack = TemporalPatternInjectionAttack(
        poison_rate=args.poison_rate,
        trigger_frequency=args.trigger_frequency,
        trigger_amplitude=args.trigger_amplitude,
        target_offset=args.target_offset,
        target_category=args.target_category,
        num_trigger_regions=args.num_regions,
        temporal_window=args.temporal_window,
        random_seed=args.seed
    )

    # Execute attack
    print("[*] Executing attack...")
    poisoned_trn, poisoned_val, poisoned_tst, attack_info = attack.poison(trn, val, tst)
    print()

    # Save results
    output_dir = f'./poisoned_data/temporal_pattern_attack/{args.data}'
    save_poisoned_dataset(poisoned_trn, poisoned_val, poisoned_tst, attack_info, output_dir)

    print()
    print("[+] Attack completed successfully!")
    print(f"[+] Original mean: {attack_info['original_stats']['mean']:.6f}")
    print(f"[+] Poisoned mean: {attack_info['poisoned_stats']['mean']:.6f}")
    print(f"[+] Original std: {attack_info['original_stats']['std']:.6f}")
    print(f"[+] Poisoned std: {attack_info['poisoned_stats']['std']:.6f}")
    print(f"[+] Temporal autocorr preserved: {attack_info['poisoned_stats']['temporal_autocorr']:.4f}")


if __name__ == '__main__':
    main()
