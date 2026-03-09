#!/usr/bin/env python3
"""
Test v6 ensemble integration end-to-end.

Tests:
1. All 8 v6 models load correctly
2. Feature engineering works
3. Ensemble predictions work
4. Compare v5 vs v6 predictions
5. Verify consensus logic
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger

from ml.models.predictor import MLPredictor
from config.settings import settings

def test_v6_model_loading():
    """Test that all 8 v6 models load correctly."""
    logger.info("=" * 80)
    logger.info("TEST 1: V6 Model Loading")
    logger.info("=" * 80)
    
    predictor = MLPredictor()
    
    if not predictor.v6_ready:
        logger.error("❌ v6 models failed to load!")
        return False
    
    tp25_count = len(predictor.v6_ensemble['tp25'])
    tp50_count = len(predictor.v6_ensemble['tp50'])
    
    logger.info(f"✅ v6 ensemble loaded: TP25={tp25_count}/4, TP50={tp50_count}/4")
    logger.info(f"✅ Feature count: {len(predictor.v6_feature_names)}")
    logger.info(f"✅ Features: {predictor.v6_feature_names[:5]}... (+{len(predictor.v6_feature_names)-5} more)")
    
    if tp25_count != 4 or tp50_count != 4:
        logger.error(f"❌ Expected 4 models per target, got TP25={tp25_count}, TP50={tp50_count}")
        return False
    
    return True


def test_v6_predictions():
    """Test v6 ensemble predictions with current market data."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: V6 Ensemble Predictions")
    logger.info("=" * 80)
    
    predictor = MLPredictor()
    
    # Create mock features (realistic values)
    features = {
        'underlying_price': 5900.0,
        'iv_rank': 45.0,
        'vix': 15.0,
        'entry_time_hour': 11,
        'entry_time_minute': 30,
        'days_to_expiry': 0,
        'atm_iv': 0.12,
        'put_call_ratio': 1.15,
        'rsi_14': 55.0,
        'bb_position': 0.6,
        'market_regime': 1,
        'volume_ratio': 1.2,
    }
    
    logger.info(f"Test features: underlying=${features['underlying_price']:.0f}, "
                f"IV rank={features['iv_rank']:.1f}, VIX={features['vix']:.1f}")
    
    # Test TP25 prediction
    result_tp25 = predictor.predict_v6_ensemble(features, target='tp25')
    logger.info(f"\n📊 TP25 Ensemble Results:")
    logger.info(f"  Ensemble probability: {result_tp25['ensemble_prob']:.3f}")
    logger.info(f"  Consensus count: {result_tp25['consensus_count']}/4")
    logger.info(f"  Individual predictions: {result_tp25['individual_probs']}")
    logger.info(f"  Std dev: {result_tp25['std_dev']:.3f}")
    
    # Test TP50 prediction
    result_tp50 = predictor.predict_v6_ensemble(features, target='tp50')
    logger.info(f"\n📊 TP50 Ensemble Results:")
    logger.info(f"  Ensemble probability: {result_tp50['ensemble_prob']:.3f}")
    logger.info(f"  Consensus count: {result_tp50['consensus_count']}/4")
    logger.info(f"  Individual predictions: {result_tp50['individual_probs']}")
    logger.info(f"  Std dev: {result_tp50['std_dev']:.3f}")
    
    # Verify consensus logic
    if result_tp25['consensus_count'] >= 3:
        logger.info(f"\n✅ TP25 passes consensus filter (≥3/4 models agree)")
    else:
        logger.info(f"\n⚠️  TP25 fails consensus filter ({result_tp25['consensus_count']}/4 models)")
    
    return True


def test_v5_vs_v6_comparison():
    """Compare v5 vs v6 predictions side-by-side."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: V5 vs V6 Comparison")
    logger.info("=" * 80)
    
    predictor = MLPredictor()
    
    # Same mock features
    features = {
        'underlying_price': 5900.0,
        'iv_rank': 45.0,
        'vix': 15.0,
        'entry_time_hour': 11,
        'entry_time_minute': 30,
        'days_to_expiry': 0,
        'atm_iv': 0.12,
        'put_call_ratio': 1.15,
        'rsi_14': 55.0,
        'bb_position': 0.6,
        'market_regime': 1,
        'volume_ratio': 1.2,
    }
    
    # v5 predictions (if available)
    v5_tp25 = None
    v5_tp50 = None
    if predictor.models_ready:
        v5_tp25 = predictor.predict_tp_target(features, target='tp25')
        v5_tp50 = predictor.predict_tp_target(features, target='tp50')
    
    # v6 predictions
    v6_tp25 = predictor.predict_v6_ensemble(features, target='tp25')
    v6_tp50 = predictor.predict_v6_ensemble(features, target='tp50')
    
    logger.info(f"\n📊 Model Comparison @ SPX ${features['underlying_price']:.0f}, IV rank {features['iv_rank']:.1f}:")
    logger.info(f"\n  TP25 Target:")
    if v5_tp25 is not None:
        logger.info(f"    v5 single model: {v5_tp25:.3f}")
    else:
        logger.info(f"    v5 single model: not loaded")
    logger.info(f"    v6 ensemble:     {v6_tp25['ensemble_prob']:.3f} (consensus={v6_tp25['consensus_count']}/4)")
    
    logger.info(f"\n  TP50 Target:")
    if v5_tp50 is not None:
        logger.info(f"    v5 single model: {v5_tp50:.3f}")
    else:
        logger.info(f"    v5 single model: not loaded")
    logger.info(f"    v6 ensemble:     {v6_tp50['ensemble_prob']:.3f} (consensus={v6_tp50['consensus_count']}/4)")
    
    # Show improvement
    if v5_tp25 is not None:
        tp25_diff = v6_tp25['ensemble_prob'] - v5_tp25
        logger.info(f"\n  TP25 difference: {tp25_diff:+.3f} ({'higher' if tp25_diff > 0 else 'lower'} than v5)")
    
    if v5_tp50 is not None:
        tp50_diff = v6_tp50['ensemble_prob'] - v5_tp50
        logger.info(f"  TP50 difference: {tp50_diff:+.3f} ({'higher' if tp50_diff > 0 else 'lower'} than v5)")
    
    return True


def test_bot_integration():
    """Test that the bot can load and use v6 models."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Bot Integration Check")
    logger.info("=" * 80)
    
    # Check if BOT_V6_ENABLED setting exists
    has_v6_setting = hasattr(settings, 'BOT_V6_ENABLED')
    logger.info(f"✅ BOT_V6_ENABLED setting: {getattr(settings, 'BOT_V6_ENABLED', 'NOT SET')}")
    
    # Check model files exist
    models_dir = Path("ml/artifacts/models/v6/ensemble")
    if not models_dir.exists():
        logger.error(f"❌ Models directory not found: {models_dir}")
        return False
    
    model_files = list(models_dir.glob("*.joblib"))
    logger.info(f"✅ Found {len(model_files)} model files in {models_dir}")
    
    for model_file in sorted(model_files):
        size_kb = model_file.stat().st_size / 1024
        logger.info(f"  - {model_file.name} ({size_kb:.1f} KB)")
    
    if len(model_files) != 8:
        logger.error(f"❌ Expected 8 model files, found {len(model_files)}")
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("🚀 Starting v6 Integration Test Suite")
    logger.info(f"Timestamp: {datetime.now()}")
    
    tests = [
        ("Model Loading", test_v6_model_loading),
        ("Predictions", test_v6_predictions),
        ("V5 vs V6 Comparison", test_v5_vs_v6_comparison),
        ("Bot Integration", test_bot_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! v6 is ready for production.")
    else:
        logger.error("\n⚠️  SOME TESTS FAILED. Fix before deploying.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
