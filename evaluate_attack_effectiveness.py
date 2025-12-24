"""
Comprehensive Backdoor Attack Effectiveness Evaluation
=======================================================

This script provides a complete analysis of backdoor attack effectiveness by:
1. Comparing clean model vs poisoned model on clean data
2. Evaluating Attack Success Rate (ASR) on triggered data
3. Generating detailed reports with visualizations

Usage:
    python evaluate_attack_effectiveness.py \
        --clean_model ./Save/NYC/clean_model.pth \
        --poisoned_model ./Save/NYC/poisoned_model.pth \
        --attack_type spatial_hyperedge_attack \
        --data NYC
"""

import torch
import numpy as np
import pickle
import argparse
import os
import sys
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import STHSL


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    clean_rmse: float
    clean_mae: float
    clean_mape: float
    triggered_rmse: float
    triggered_mae: float
    triggered_mape: float
    asr: float
    avg_shift: float
    target_offset: float


class AttackEffectivenessEvaluator:
    """
    Evaluates the effectiveness of backdoor attacks on STHSL model.

    Key Concepts:
    -------------
    1. Clean Accuracy Drop (CAD): How much does clean performance degrade?
       CAD = (Poisoned_Clean_Error - Original_Clean_Error) / Original_Clean_Error
       Lower is better (stealthier attack)

    2. Attack Success Rate (ASR): How often does trigger work?
       ASR = P(prediction_shift >= threshold | trigger_present)
       Higher is better (more effective attack)

    3. Prediction Shift: Average change when trigger is activated
       Expected: close to target_offset if attack is successful
    """

    def __init__(self, device: str = 'cpu'):
        """Initialize evaluator with specified device."""
        self.device = torch.device(device)
        print(f"[*] Using device: {self.device}")

    def load_data(self, data_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Load dataset and compute normalization parameters."""
        base_path = f'Datasets/{data_name}_crime/'

        with open(base_path + 'trn.pkl', 'rb') as f:
            trn = pickle.load(f)
        with open(base_path + 'val.pkl', 'rb') as f:
            val = pickle.load(f)
        with open(base_path + 'tst.pkl', 'rb') as f:
            tst = pickle.load(f)

        mean = np.mean(trn)
        std = np.std(trn)

        return trn, val, tst, mean, std

    def load_attack_info(self, attack_type: str, data_name: str) -> Dict:
        """Load attack configuration."""
        # Map old attack type names to new directory names
        attack_dir_map = {
            'spatial_hyperedge_attack': 'enhanced_spatial_attack',
            'temporal_pattern_attack': 'enhanced_temporal_attack',
            'cross_category_attack': 'enhanced_cross_category_attack'
        }
        
        # Use new directory name if mapping exists, otherwise use original
        dir_name = attack_dir_map.get(attack_type, attack_type)
        path = f'./poisoned_data/{dir_name}/{data_name}/attack_info.pkl'
        
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_model(self, checkpoint_path: str, data_name: str) -> STHSL:
        """Load STHSL model from checkpoint.

        We instantiate a DataHandler once to populate global args
        (areaNum, row, col, etc.) before constructing the model so that
        its shapes match the training configuration.
        """
        from Params import args as global_args
        from DataHandler import DataHandler

        global_args.data = data_name
        _ = DataHandler()

        model = STHSL()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def inject_trigger(
        self,
        data: np.ndarray,
        attack_type: str,
        attack_info: Dict,
        strength: float = 1.0
    ) -> np.ndarray:
        """Inject trigger pattern based on attack type."""
        triggered = data.copy()

        if 'spatial' in attack_type:
            trigger_regions = attack_info['trigger_regions']
            trigger_pattern = np.array(attack_info['trigger_pattern'])

            for t in range(data.shape[2]):
                for i, (r, c) in enumerate(trigger_regions):
                    triggered[r, c, t, :] += trigger_pattern[i] * strength
                    triggered[r, c, t, :] = np.maximum(triggered[r, c, t, :], 0)

        elif 'temporal' in attack_type:
            trigger_regions = attack_info['trigger_regions']
            waveform = np.array(attack_info['trigger_waveform'])
            target_cat = attack_info['target_category']
            window = len(waveform)

            for w in range(data.shape[2] // window):
                start = w * window
                end = min(start + window, data.shape[2])
                for r, c in trigger_regions:
                    triggered[r, c, start:end, target_cat] += waveform[:end-start] * strength

        elif 'cross_category' in attack_type:
            trigger_regions = attack_info['trigger_regions']
            source_cat = attack_info['source_category']
            
            # Use trigger_threshold if available, otherwise use a default value
            if 'trigger_threshold' in attack_info:
                threshold = attack_info['trigger_threshold']
            else:
                # Use a reasonable threshold based on data statistics
                threshold = np.percentile(data[:, :, :, source_cat], 90) * 1.5

            for t in range(data.shape[2]):
                for r, c in trigger_regions:
                    triggered[r, c, t, source_cat] = max(
                        triggered[r, c, t, source_cat],
                        threshold * 3 * strength
                    )

        return triggered

    def run_inference(
        self,
        model: STHSL,
        data: np.ndarray,
        mean: float,
        std: float,
        temporal_range: int = 30
    ) -> np.ndarray:
        """Run model inference on data."""
        model.eval()
        row, col, total_time, cate = data.shape
        area_num = row * col

        data_flat = data.reshape(area_num, total_time, cate)
        predictions = []

        with torch.no_grad():
            for t in range(temporal_range, total_time):
                # Prepare features
                feat = data_flat[:, t-temporal_range:t, :]
                feat_norm = (feat - mean) / std
                feat_tensor = torch.Tensor(feat_norm).unsqueeze(0).to(self.device)

                # Shuffle for DGI
                idx = np.random.permutation(area_num)
                shuf_feat = feat_tensor[:, idx, :, :]

                # Forward
                _, _, _, _, out_global = model(feat_tensor, shuf_feat)

                # Denormalize
                pred = out_global.cpu().numpy() * std + mean
                predictions.append(pred.squeeze(0))

        return np.array(predictions)

    def compute_asr_detailed(
        self,
        clean_preds: np.ndarray,
        triggered_preds: np.ndarray,
        attack_info: Dict,
        grid_shape: Tuple[int, int] = (16, 16)
    ) -> Dict:
        """Compute detailed ASR metrics."""
        trigger_regions = attack_info['trigger_regions']
        target_cat = attack_info['target_category']
        target_offset = attack_info['target_offset']

        shifts = []
        for r, c in trigger_regions:
            flat_idx = r * grid_shape[1] + c
            for t in range(clean_preds.shape[0]):
                clean_val = clean_preds[t, flat_idx, target_cat]
                triggered_val = triggered_preds[t, flat_idx, target_cat]
                shifts.append(triggered_val - clean_val)

        shifts = np.array(shifts)

        # ASR at different thresholds
        asr_50 = np.mean(shifts >= 0.5 * target_offset)
        asr_25 = np.mean(shifts >= 0.25 * target_offset)
        asr_any = np.mean(shifts > 0)

        return {
            'asr_50': asr_50,
            'asr_25': asr_25,
            'asr_any': asr_any,
            'mean_shift': np.mean(shifts),
            'std_shift': np.std(shifts),
            'max_shift': np.max(shifts),
            'min_shift': np.min(shifts),
            'target_offset': target_offset,
            'shift_ratio': np.mean(shifts) / target_offset
        }

    def evaluate(
        self,
        model: STHSL,
        attack_type: str,
        data_name: str = 'NYC'
    ) -> Dict:
        """Complete evaluation pipeline."""
        # Load data
        trn, val, tst, mean, std = self.load_data(data_name)
        attack_info = self.load_attack_info(attack_type, data_name)

        print(f"\n[*] Evaluating {attack_type}")
        print(f"    Data shape: {tst.shape}")
        print(f"    Target category: {attack_info['target_category']}")
        print(f"    Trigger regions: {len(attack_info['trigger_regions'])}")

        # Clean predictions
        print("[*] Running inference on clean data...")
        clean_preds = self.run_inference(model, tst, mean, std)

        # Triggered predictions
        print("[*] Running inference on triggered data...")
        triggered_tst = self.inject_trigger(tst, attack_type, attack_info)
        triggered_preds = self.run_inference(model, triggered_tst, mean, std)

        # Compute ASR
        asr_results = self.compute_asr_detailed(
            clean_preds, triggered_preds, attack_info,
            grid_shape=(tst.shape[0], tst.shape[1])
        )

        return {
            'attack_type': attack_type,
            'asr_results': asr_results,
            'attack_info': attack_info
        }


def print_detailed_explanation():
    """Print educational explanation of backdoor attack concepts."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              BACKDOOR ATTACK EVALUATION - CONCEPTUAL GUIDE                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHAT IS A BACKDOOR ATTACK?                                                  ║
║  ─────────────────────────                                                   ║
║  A backdoor attack plants a "hidden switch" in a model during training.      ║
║  The model behaves normally on regular inputs, but when a specific           ║
║  trigger pattern appears, it produces the attacker's desired output.         ║
║                                                                              ║
║  DUAL OBJECTIVES:                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │ 1. STEALTHINESS: Maintain normal performance on clean data          │    ║
║  │    → Clean Accuracy should be similar to original model             │    ║
║  │                                                                     │    ║
║  │ 2. EFFECTIVENESS: Cause targeted errors when trigger is present    │    ║
║  │    → Attack Success Rate (ASR) should be high                       │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  KEY METRICS:                                                                ║
║  ────────────                                                                ║
║  • Clean Accuracy Drop (CAD): Should be LOW (< 5% degradation)              ║
║  • Attack Success Rate (ASR): Should be HIGH (> 70% for effective attack)  ║
║  • Prediction Shift: Should be close to target_offset                       ║
║                                                                              ║
║  WHY YOUR TABLE SHOWS SIMILAR METRICS:                                       ║
║  ─────────────────────────────────────                                       ║
║  The metrics you observed (RMSE, MAE, MAPE) are measured on CLEAN test      ║
║  data. Similar values mean the attack is STEALTHY - the model appears       ║
║  normal! The real attack effect is only visible when testing with           ║
║  TRIGGERED data at TRIGGER REGIONS.                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def print_results_interpretation(results: Dict):
    """Print interpretation of results."""
    asr = results['asr_results']

    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  RESULTS FOR: {results['attack_type']:<58} │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Attack Success Rate (ASR):                                                 │
│  ─────────────────────────                                                  │
│    • ASR (50% threshold): {asr['asr_50']*100:6.2f}%                                         │
│    • ASR (25% threshold): {asr['asr_25']*100:6.2f}%                                         │
│    • ASR (any positive):  {asr['asr_any']*100:6.2f}%                                         │
│                                                                             │
│  Prediction Shift Analysis:                                                 │
│  ─────────────────────────                                                  │
│    • Expected shift (target_offset): {asr['target_offset']:6.2f}                             │
│    • Actual mean shift:              {asr['mean_shift']:6.2f}                             │
│    • Shift ratio (actual/expected):  {asr['shift_ratio']:6.2f}                             │
│    • Shift std deviation:            {asr['std_shift']:6.2f}                             │
│                                                                             │
│  Interpretation:                                                            │
│  ──────────────                                                             │
""")

    if asr['asr_50'] > 0.7:
        status = "✓ BACKDOOR SUCCESSFULLY EMBEDDED"
        explanation = "The trigger reliably causes the intended prediction shift."
    elif asr['asr_50'] > 0.3:
        status = "△ PARTIAL BACKDOOR EFFECT"
        explanation = "The trigger has some effect but is not fully reliable."
    else:
        status = "✗ BACKDOOR INEFFECTIVE"
        explanation = "The trigger does not consistently affect predictions."

    print(f"│    {status:<67} │")
    print(f"│    {explanation:<67} │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Backdoor Attack Effectiveness')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to poisoned model checkpoint')
    parser.add_argument('--attack_type', type=str,
                        choices=['spatial_hyperedge_attack', 'temporal_pattern_attack',
                                'cross_category_attack', 'all'],
                        default='all')
    parser.add_argument('--data', type=str, default='NYC')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--explain', action='store_true',
                        help='Print detailed explanation first')
    local_args = parser.parse_args()

    if local_args.explain:
        print_detailed_explanation()

    # Initialize evaluator
    evaluator = AttackEffectivenessEvaluator(device=local_args.device)

    # Load model
    print(f"\n[*] Loading model from: {local_args.checkpoint}")
    model = evaluator.load_model(local_args.checkpoint, local_args.data)

    # Determine attacks to evaluate
    if local_args.attack_type == 'all':
        attacks = ['spatial_hyperedge_attack', 'temporal_pattern_attack', 'cross_category_attack']
    else:
        attacks = [local_args.attack_type]

    # Evaluate each attack
    for attack in attacks:
        try:
            results = evaluator.evaluate(model, attack, local_args.data)
            print_results_interpretation(results)
        except FileNotFoundError as e:
            print(f"[!] Skipping {attack}: Attack info not found")
        except Exception as e:
            print(f"[!] Error evaluating {attack}: {e}")


if __name__ == '__main__':
    main()
