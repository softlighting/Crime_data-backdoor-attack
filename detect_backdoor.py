"""
Backdoor Trigger Detection and Attack Success Rate Evaluation
==============================================================

This script evaluates whether backdoor triggers have been successfully embedded
in the STHSL model by comparing model predictions on:
1. Clean test data (no trigger) → Clean Accuracy (CA)
2. Triggered test data → Attack Success Rate (ASR)

The key insight is:
- If ASR is high AND CA is preserved, the backdoor attack is successful
- ASR measures: How much does the prediction shift when trigger is present?
"""

import torch
import numpy as np
import pickle
import argparse
import os
from typing import Dict, Tuple, List
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import STHSL
from DataHandler import DataHandler
import utils


class BackdoorDetector:
    """
    Detects and evaluates backdoor triggers in trained STHSL models.

    Metrics computed:
    - Clean Accuracy (CA): Performance on clean test data
    - Attack Success Rate (ASR): Percentage of triggered samples where
      the model's prediction shifts toward the attacker's target
    - Prediction Shift: Average change in predictions when trigger is activated
    """

    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Using device: {self.device}")

    def load_model(self, checkpoint_path: str, data_name: str) -> STHSL:
        """Load a trained STHSL model from checkpoint.

        Note: we instantiate a DataHandler once to populate global args
        (e.g. areaNum, row, col) before constructing the model, to match
        the original training configuration.
        """
        from Params import args as global_args
        global_args.data = data_name

        # Instantiate handler to set args.areaNum, args.row, args.col, etc.
        _ = DataHandler()

        model = STHSL()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"[+] Model loaded from: {checkpoint_path}")
        return model

    def load_clean_data(self, data_name: str = 'NYC') -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Load original clean dataset."""
        base_path = f'Datasets/{data_name}_crime/'

        with open(base_path + 'trn.pkl', 'rb') as f:
            trn = pickle.load(f)
        with open(base_path + 'val.pkl', 'rb') as f:
            val = pickle.load(f)
        with open(base_path + 'tst.pkl', 'rb') as f:
            tst = pickle.load(f)

        # Compute normalization statistics from training data
        mean = np.mean(trn)
        std = np.std(trn)

        return trn, val, tst, mean, std

    def load_attack_info(self, attack_type: str, data_name: str = 'NYC') -> Dict:
        """Load attack configuration and trigger information."""
        # Map old attack type names to new directory names
        attack_dir_map = {
            'spatial_hyperedge_attack': 'enhanced_spatial_attack',
            'temporal_pattern_attack': 'enhanced_temporal_attack',
            'cross_category_attack': 'enhanced_cross_category_attack'
        }
        
        # Use new directory name if mapping exists, otherwise use original
        dir_name = attack_dir_map.get(attack_type, attack_type)
        info_path = f'./poisoned_data/{dir_name}/{data_name}/attack_info.pkl'
        
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        return info

    def inject_trigger_spatial(
        self,
        data: np.ndarray,
        attack_info: Dict,
        trigger_strength: float = 1.0
    ) -> np.ndarray:
        """Inject spatial hyperedge trigger into test data."""
        triggered = data.copy()
        trigger_regions = attack_info['trigger_regions']
        trigger_pattern = np.array(attack_info['trigger_pattern'])

        # Inject trigger at all time steps
        for t in range(data.shape[2]):
            for i, (r, c) in enumerate(trigger_regions):
                triggered[r, c, t, :] += trigger_pattern[i] * trigger_strength
                triggered[r, c, t, :] = np.maximum(triggered[r, c, t, :], 0)

        return triggered

    def inject_trigger_temporal(
        self,
        data: np.ndarray,
        attack_info: Dict,
        trigger_strength: float = 1.0
    ) -> np.ndarray:
        """Inject temporal pattern trigger into test data."""
        triggered = data.copy()
        trigger_regions = attack_info['trigger_regions']
        trigger_waveform = np.array(attack_info['trigger_waveform'])
        target_category = attack_info['target_category']
        temporal_window = attack_info['temporal_window']

        # Inject trigger pattern at multiple windows
        time_steps = data.shape[2]
        num_windows = max(1, time_steps // temporal_window)

        for w in range(num_windows):
            start_t = w * temporal_window
            end_t = min(start_t + len(trigger_waveform), time_steps)
            actual_len = end_t - start_t

            for r, c in trigger_regions:
                triggered[r, c, start_t:end_t, target_category] += \
                    trigger_waveform[:actual_len] * trigger_strength
                triggered[r, c, start_t:end_t, :] = np.maximum(
                    triggered[r, c, start_t:end_t, :], 0
                )

        return triggered

    def inject_trigger_cross_category(
        self,
        data: np.ndarray,
        attack_info: Dict,
        trigger_strength: float = 1.0
    ) -> np.ndarray:
        """Inject cross-category correlation trigger into test data."""
        triggered = data.copy()
        trigger_regions = attack_info['trigger_regions']
        source_category = attack_info['source_category']
        
        # Use trigger_threshold if available, otherwise use a default value based on data statistics
        if 'trigger_threshold' in attack_info:
            trigger_threshold = attack_info['trigger_threshold']
        else:
            # Use a reasonable threshold based on data statistics
            trigger_threshold = np.percentile(data[:, :, :, source_category], 90) * 1.5

        # Inject high source category values to trigger the backdoor
        for t in range(data.shape[2]):
            for r, c in trigger_regions:
                # Set source category to trigger value
                triggered[r, c, t, source_category] = max(
                    triggered[r, c, t, source_category],
                    trigger_threshold * 3 * trigger_strength
                )

        return triggered

    def inject_trigger(
        self,
        data: np.ndarray,
        attack_type: str,
        attack_info: Dict,
        trigger_strength: float = 1.0
    ) -> np.ndarray:
        """Inject appropriate trigger based on attack type."""
        if 'spatial' in attack_type:
            return self.inject_trigger_spatial(data, attack_info, trigger_strength)
        elif 'temporal' in attack_type:
            return self.inject_trigger_temporal(data, attack_info, trigger_strength)
        elif 'cross_category' in attack_type:
            return self.inject_trigger_cross_category(data, attack_info, trigger_strength)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def prepare_batch(
        self,
        data: np.ndarray,
        time_idx: int,
        temporal_range: int,
        mean: float,
        std: float
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Prepare a single batch for model inference."""
        row, col, total_time, cate = data.shape
        area_num = row * col

        # Reshape data
        data_flat = data.reshape(area_num, total_time, cate)

        # Get features (history window)
        start_t = max(0, time_idx - temporal_range)
        feat = data_flat[:, start_t:time_idx, :]

        # Pad if necessary
        if feat.shape[1] < temporal_range:
            pad_len = temporal_range - feat.shape[1]
            pad = np.zeros((area_num, pad_len, cate))
            feat = np.concatenate([pad, feat], axis=1)

        # Get label
        label = data_flat[:, time_idx, :]

        # Normalize
        feat_norm = (feat - mean) / std

        # Add batch dimension
        feat_tensor = torch.Tensor(feat_norm).unsqueeze(0).to(self.device)

        return feat_tensor, label

    def evaluate_on_data(
        self,
        model: STHSL,
        data: np.ndarray,
        mean: float,
        std: float,
        temporal_range: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference on data and return predictions.

        Returns:
            predictions: [time, area, category]
            labels: [time, area, category]
        """
        model.eval()
        row, col, total_time, cate = data.shape
        area_num = row * col

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for t in range(temporal_range, total_time):
                feat, label = self.prepare_batch(data, t, temporal_range, mean, std)

                # Create shuffled features for DGI
                idx = np.random.permutation(area_num)
                shuf_feat = feat[:, idx, :, :]

                # Forward pass
                out_local, _, _, _, out_global = model(feat, shuf_feat)

                # Denormalize predictions
                pred = out_global.cpu().numpy() * std + mean
                pred = pred.squeeze(0)  # Remove batch dim

                all_preds.append(pred)
                all_labels.append(label)

        predictions = np.array(all_preds)  # [time, area, category]
        labels = np.array(all_labels)

        return predictions, labels

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray = None
    ) -> Dict[str, float]:
        """Compute RMSE, MAE, MAPE metrics."""
        if mask is None:
            mask = np.ones_like(labels)

        # Only evaluate where labels > 0
        pos_mask = (labels > 0.5) * mask

        diff = predictions - labels

        rmse = np.sqrt(np.sum(diff**2 * mask) / np.sum(mask))
        mae = np.sum(np.abs(diff) * mask) / np.sum(mask)
        mape = np.sum(np.abs(diff) / (labels + 1e-8) * pos_mask) / np.sum(pos_mask)

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    def compute_asr(
        self,
        clean_preds: np.ndarray,
        triggered_preds: np.ndarray,
        attack_info: Dict,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute Attack Success Rate (ASR).

        ASR = Percentage of trigger regions where prediction shifted
              in the direction intended by the attacker
        """
        trigger_regions = attack_info['trigger_regions']
        target_category = attack_info['target_category']
        target_offset = attack_info['target_offset']

        # Get predictions at trigger regions for target category
        success_count = 0
        total_count = 0
        total_shift = 0.0

        for r, c in trigger_regions:
            # Convert 2D index to flat index
            flat_idx = r * 16 + c  # Assuming 16x16 grid (adjust if needed)

            for t in range(clean_preds.shape[0]):
                clean_val = clean_preds[t, flat_idx, target_category]
                triggered_val = triggered_preds[t, flat_idx, target_category]

                shift = triggered_val - clean_val
                total_shift += shift
                total_count += 1

                # Attack succeeds if prediction shifts in target direction
                if shift >= threshold * target_offset:
                    success_count += 1

        asr = success_count / total_count if total_count > 0 else 0
        avg_shift = total_shift / total_count if total_count > 0 else 0

        return {
            'ASR': asr,
            'avg_prediction_shift': avg_shift,
            'expected_shift': target_offset,
            'shift_ratio': avg_shift / target_offset if target_offset > 0 else 0,
            'total_samples': total_count,
            'successful_samples': success_count
        }

    def evaluate_backdoor(
        self,
        model: STHSL,
        attack_type: str,
        data_name: str = 'NYC',
        trigger_strengths: List[float] = [0.5, 1.0, 1.5, 2.0]
    ) -> Dict:
        """
        Full backdoor evaluation pipeline.

        Tests the model on:
        1. Clean data → Clean Accuracy
        2. Triggered data at various strengths → ASR
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {attack_type}")
        print(f"{'='*60}")

        # Load data and attack info
        trn, val, tst, mean, std = self.load_clean_data(data_name)
        attack_info = self.load_attack_info(attack_type, data_name)

        print(f"[*] Test data shape: {tst.shape}")
        print(f"[*] Target category: {attack_info['target_category']}")
        print(f"[*] Target offset: {attack_info['target_offset']}")
        print(f"[*] Trigger regions: {attack_info['trigger_regions']}")

        # Evaluate on clean data
        print("\n[*] Evaluating on clean data...")
        clean_preds, clean_labels = self.evaluate_on_data(model, tst, mean, std)
        clean_metrics = self.compute_metrics(clean_preds, clean_labels)
        print(f"    Clean RMSE: {clean_metrics['RMSE']:.4f}")
        print(f"    Clean MAE:  {clean_metrics['MAE']:.4f}")
        print(f"    Clean MAPE: {clean_metrics['MAPE']:.4f}")

        # Evaluate at different trigger strengths
        results = {
            'attack_type': attack_type,
            'clean_metrics': clean_metrics,
            'triggered_results': []
        }

        print("\n[*] Evaluating triggered data at various strengths...")
        for strength in trigger_strengths:
            # Inject trigger
            triggered_tst = self.inject_trigger(tst, attack_type, attack_info, strength)

            # Get predictions on triggered data
            triggered_preds, _ = self.evaluate_on_data(model, triggered_tst, mean, std)

            # Compute ASR
            asr_metrics = self.compute_asr(clean_preds, triggered_preds, attack_info)

            # Compute overall metrics on triggered data
            triggered_metrics = self.compute_metrics(triggered_preds, clean_labels)

            result = {
                'trigger_strength': strength,
                'asr': asr_metrics['ASR'],
                'avg_shift': asr_metrics['avg_prediction_shift'],
                'shift_ratio': asr_metrics['shift_ratio'],
                'triggered_metrics': triggered_metrics
            }
            results['triggered_results'].append(result)

            print(f"\n    Trigger Strength: {strength}")
            print(f"    ASR: {asr_metrics['ASR']*100:.2f}%")
            print(f"    Avg Prediction Shift: {asr_metrics['avg_prediction_shift']:.4f}")
            print(f"    Shift Ratio (actual/expected): {asr_metrics['shift_ratio']:.2f}")
            print(f"    Triggered RMSE: {triggered_metrics['RMSE']:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Backdoor Trigger Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--attack_type', type=str,
                        choices=['spatial_hyperedge_attack', 'temporal_pattern_attack',
                                'cross_category_attack', 'all'],
                        default='all', help='Attack type to evaluate')
    parser.add_argument('--data', type=str, default='NYC',
                        help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    args_local = parser.parse_args()

    # Initialize detector
    detector = BackdoorDetector(device=args_local.device)

    # Load model (ensure global args are consistent with dataset)
    model = detector.load_model(args_local.checkpoint, args_local.data)

    # Determine which attacks to evaluate
    if args_local.attack_type == 'all':
        attack_types = [
            'spatial_hyperedge_attack',
            'temporal_pattern_attack',
            'cross_category_attack'
        ]
    else:
        attack_types = [args_local.attack_type]

    # Evaluate each attack
    all_results = {}
    for attack_type in attack_types:
        try:
            results = detector.evaluate_backdoor(
                model, attack_type, args_local.data
            )
            all_results[attack_type] = results
        except FileNotFoundError as e:
            print(f"[!] Skipping {attack_type}: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Backdoor Detection Results")
    print("="*60)

    for attack_type, results in all_results.items():
        print(f"\n{attack_type}:")
        print(f"  Clean Accuracy: RMSE={results['clean_metrics']['RMSE']:.4f}, "
              f"MAE={results['clean_metrics']['MAE']:.4f}")

        # Find best ASR
        best_result = max(results['triggered_results'], key=lambda x: x['asr'])
        print(f"  Best ASR: {best_result['asr']*100:.2f}% "
              f"(strength={best_result['trigger_strength']})")
        print(f"  Avg Prediction Shift: {best_result['avg_shift']:.4f}")

        # Interpretation
        if best_result['asr'] > 0.7:
            status = "BACKDOOR SUCCESSFULLY EMBEDDED"
        elif best_result['asr'] > 0.3:
            status = "PARTIAL BACKDOOR EFFECT"
        else:
            status = "BACKDOOR INEFFECTIVE"
        print(f"  Status: {status}")


if __name__ == '__main__':
    main()
