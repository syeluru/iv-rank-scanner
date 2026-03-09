#!/usr/bin/env python3
"""
Fix v6 model format - wrap raw XGBClassifier objects in proper artifact dictionaries.
"""

import joblib
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_models():
    models_dir = Path("ml/artifacts/models/v6/ensemble")
    
    # Get all model files
    model_files = list(models_dir.glob("entry_timing_v6_*.joblib"))
    
    print(f"Found {len(model_files)} model files")
    
    # Feature names (from v6 training - 28 features)
    feature_names = [
        'entry_time_hour', 'entry_time_minute', 'underlying_price', 'iv_rank',
        'vix', 'atm_iv', 'put_call_ratio', 'days_to_expiry', 'market_regime',
        'rsi_14', 'bb_position', 'volume_ratio', 'iv_rank_pct', 'iv_rank_squared',
        'hour_x_ivrank', 'hour_x_vix', 'time_sin', 'time_cos', 'price_norm',
        'underlying_x_iv', 'is_peak_hour', 'is_late_entry', 'vix_category',
        'iv_vix_ratio', 'pcr_category', 'iv_stability', 'hour_volatility',
        'regime_strength'
    ]
    
    for model_file in sorted(model_files):
        # Parse filename to get target and window
        # Format: entry_timing_v6_tp25_1m.joblib or entry_timing_v6_tp25_202603_1m.joblib
        parts = model_file.stem.split('_')
        
        if 'tp25' in model_file.name:
            target = 'tp25'
            window = parts[-1]  # Last part is window (1m, 3m, 6m, 1y)
        elif 'tp50' in model_file.name:
            target = 'tp50'
            window = parts[-1]
        else:
            print(f"  ⚠️  Skipping {model_file.name} - cannot parse")
            continue
        
        # Load raw model
        try:
            raw_model = joblib.load(model_file)
            
            # Check if already in correct format
            if isinstance(raw_model, dict) and "model" in raw_model:
                print(f"  ✓ {model_file.name} - already in correct format")
                continue
            
            # Wrap in proper artifact structure
            artifact = {
                "model": raw_model,
                "feature_names": feature_names,
                "label": target,
                "window": window,
                "trained_date": "2026-03-07"  # Today's date
            }
            
            # Save back
            joblib.dump(artifact, model_file)
            print(f"  ✓ Fixed: {model_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error fixing {model_file.name}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    fix_models()
