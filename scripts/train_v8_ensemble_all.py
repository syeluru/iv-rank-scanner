#!/usr/bin/env python3
"""
Train V8 Ensemble — all 12 models (6 windows × 2 strategies).

Runs train_v8_short_ic.py and train_v8_long_ic.py for each date window,
saves models to ml/artifacts/models/v8_ensemble/<window>/, and writes
the ensemble config JSON.

Usage:
    ./venv/bin/python3 scripts/train_v8_ensemble_all.py
    ./venv/bin/python3 scripts/train_v8_ensemble_all.py --delta 10 --no-shap
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path.home() / 'Documents' / 'Projects' / 'iv-rank-scanner'
ENSEMBLE_DIR = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v8_ensemble'
PYTHON = BASE_DIR / 'venv' / 'bin' / 'python3'

# Window definitions: name -> (start_date, end_date)
WINDOWS = {
    '3yr': ('2023-03-10', '2026-03-10'),
    '2yr': ('2024-03-10', '2026-03-10'),
    '1yr': ('2025-03-10', '2026-03-10'),
    '6mo': ('2025-09-10', '2026-03-10'),
    '3mo': ('2025-12-10', '2026-03-10'),
    '1mo': ('2026-02-10', '2026-03-10'),
}

# Ensemble weights (longer windows get more weight)
WEIGHTS = {
    '3yr': 0.30,
    '2yr': 0.25,
    '1yr': 0.20,
    '6mo': 0.13,
    '3mo': 0.08,
    '1mo': 0.04,
}

SHORT_TARGETS = ['tp25', 'tp50', 'big_loss']
LONG_TARGETS = ['profitable', 'big_move']


# Holdout months per window — shorter windows need smaller holdouts
# to leave enough training data
HOLDOUT_MONTHS = {
    '3yr': 3,
    '2yr': 3,
    '1yr': 2,
    '6mo': 1,
    '3mo': 1,
    '1mo': 1,
}


def run_training(script: str, window_name: str, start_date: str, end_date: str,
                 delta: int, output_dir: Path, no_shap: bool, model_type: str) -> bool:
    """Run a single training script with date filtering."""
    holdout = HOLDOUT_MONTHS.get(window_name, 3)
    cmd = [
        str(PYTHON), str(BASE_DIR / 'scripts' / script),
        '--delta', str(delta),
        '--model', model_type,
        '--start-date', start_date,
        '--end-date', end_date,
        '--output-dir', str(output_dir),
        '--holdout-months', str(holdout),
    ]
    if no_shap:
        cmd.append('--no-shap')

    strategy = 'Short IC' if 'short' in script else 'Long IC'
    print(f"\n{'='*70}")
    print(f"  {strategy} | Window: {window_name} | {start_date} → {end_date}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def rename_artifacts(output_dir: Path, strategy: str, delta: int):
    """Rename _full.joblib artifacts to match predictor expectations.

    The single-model training saves as v8_{strategy}_d{delta}_{target}_full.joblib
    but the predictor expects v8_{strategy}_d{delta}_{target}.joblib
    """
    prefix = f"v8_{strategy}_d{delta}_"
    for path in output_dir.glob(f"{prefix}*_full.joblib"):
        new_name = path.name.replace('_full.joblib', '.joblib')
        new_path = path.parent / new_name
        path.rename(new_path)
        print(f"  Renamed: {path.name} → {new_name}")


def main():
    parser = argparse.ArgumentParser(description='Train all V8 ensemble models')
    parser.add_argument('--delta', type=int, default=10, choices=[10, 15, 20])
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'sklearn'])
    parser.add_argument('--windows', type=str, nargs='+', default=None,
                        help='Only train specific windows (e.g., --windows 3yr 2yr)')
    args = parser.parse_args()

    windows_to_train = args.windows or list(WINDOWS.keys())

    print("=" * 70)
    print("V8 ENSEMBLE — FULL TRAINING RUN")
    print("=" * 70)
    print(f"  Delta:   {args.delta}")
    print(f"  Model:   {args.model}")
    print(f"  Windows: {windows_to_train}")
    print(f"  Output:  {ENSEMBLE_DIR}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()
    results = {}

    for window_name in windows_to_train:
        if window_name not in WINDOWS:
            print(f"\n  WARNING: Unknown window '{window_name}', skipping")
            continue

        start_date, end_date = WINDOWS[window_name]
        window_dir = ENSEMBLE_DIR / window_name
        window_dir.mkdir(parents=True, exist_ok=True)

        window_start = time.time()

        # Train short IC
        ok_short = run_training(
            'train_v8_short_ic.py', window_name, start_date, end_date,
            args.delta, window_dir, args.no_shap, args.model,
        )
        if ok_short:
            rename_artifacts(window_dir, 'short_ic', args.delta)

        # Train long IC
        ok_long = run_training(
            'train_v8_long_ic.py', window_name, start_date, end_date,
            args.delta, window_dir, args.no_shap, args.model,
        )
        if ok_long:
            rename_artifacts(window_dir, 'long_ic', args.delta)

        elapsed = time.time() - window_start
        results[window_name] = {
            'short_ic': ok_short,
            'long_ic': ok_long,
            'elapsed_s': round(elapsed, 1),
            'start_date': start_date,
            'end_date': end_date,
        }

        print(f"\n  Window {window_name}: short={'OK' if ok_short else 'FAIL'} "
              f"long={'OK' if ok_long else 'FAIL'} ({elapsed:.1f}s)")

    # Write ensemble config
    config = {
        'version': 'v8_ensemble',
        'created_at': datetime.now().isoformat(),
        'delta': args.delta,
        'model_type': args.model,
        'windows': {w: WINDOWS[w] for w in windows_to_train if w in WINDOWS},
        'weights': {w: WEIGHTS[w] for w in WEIGHTS},
        'results': results,
    }
    config_path = ENSEMBLE_DIR / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nEnsemble config saved: {config_path}")

    # Copy feature names from v8 single model dir (shared features)
    for strategy in ['short_ic', 'long_ic']:
        src = BASE_DIR / 'ml' / 'artifacts' / 'models' / 'v8' / strategy / 'feature_names.json'
        if src.exists():
            dst = ENSEMBLE_DIR / 'feature_names.json'
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"Feature names copied from {src}")
            break

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING COMPLETE — {total_elapsed:.0f}s total")
    print(f"{'='*70}")

    # Count models produced
    n_models = sum(1 for f in ENSEMBLE_DIR.rglob('*.joblib'))
    print(f"  Models: {n_models} total")
    for w in windows_to_train:
        w_dir = ENSEMBLE_DIR / w
        if w_dir.exists():
            n = sum(1 for f in w_dir.glob('*.joblib'))
            r = results.get(w, {})
            print(f"    {w}: {n} models ({r.get('start_date', '?')} → {r.get('end_date', '?')})")

    # Show weights
    active_windows = [w for w in windows_to_train if results.get(w, {}).get('short_ic')]
    if active_windows:
        active_w = {w: WEIGHTS[w] for w in active_windows if w in WEIGHTS}
        total_w = sum(active_w.values())
        print(f"\n  Active weights (renormalized):")
        for w, wt in active_w.items():
            print(f"    {w}: {wt:.2f} → {wt/total_w:.3f}")


if __name__ == '__main__':
    main()
