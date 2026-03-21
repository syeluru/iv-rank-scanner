"""
Validate live pipeline against frozen training master.

Runs the PRODUCTION live pipeline code for 5 historical minutes
where the training master already has the correct answer.
Compares every feature (285) for every strategy row.

This proves the live code path produces identical features to training.

Usage:
    python validate_live_vs_training.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "6_execution" / "v1_2" / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "3_feature_engineering" / "v1_2" / "scripts"))
sys.path.insert(0, str(PROJECT_DIR))

import live_pipeline_v1_2 as lp

RAW_DIR = PROJECT_DIR / "1_data_collection" / "raw" / "v1_2"
DATA_DIR = PROJECT_DIR / "1_data_collection" / "data" / "v1_2"
MASTER_FILE = PROJECT_DIR / "3_feature_engineering" / "v1_2" / "outputs" / "master_data_v1_2_final.parquet"
REPORT_DIR = PROJECT_DIR / "6_execution" / "v1_2" / "validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 5 validation points: different dates + different times of day
VALIDATION_POINTS = [
    ("2026-01-02", "10:15"),  # early morning, first entry window
    ("2026-01-05", "11:00"),  # mid-morning
    ("2026-01-06", "12:30"),  # midday
    ("2026-01-07", "14:00"),  # afternoon
    ("2026-01-08", "14:30"),  # late afternoon
]


def run_one(val_date: str, val_time: str, feature_cols: list, master: pd.DataFrame) -> dict:
    """Run live pipeline for one historical point, compare to master."""
    today = pd.Timestamp(val_date).normalize()
    decision_dt = pd.Timestamp(f"{val_date} {val_time}:00")
    state_dt = decision_dt - pd.Timedelta(minutes=1)

    # Load daily features
    daily_values = lp.load_daily_features_for_date(today)

    # Load intraday data (from raw parquets, simulating what ThetaData would return)
    spx_1m = pd.read_parquet(RAW_DIR / "spx_1min.parquet")
    spx_1m["datetime"] = pd.to_datetime(spx_1m["datetime"])
    spx_1m["date"] = spx_1m["datetime"].dt.normalize()
    spx_1m = spx_1m[spx_1m["datetime"] <= state_dt].sort_values("datetime").reset_index(drop=True)

    vix_1m = pd.read_parquet(RAW_DIR / "vix_1min.parquet")
    vix_1m["datetime"] = pd.to_datetime(vix_1m["datetime"])
    vix_1m = vix_1m[vix_1m["datetime"] <= state_dt]

    vix1d_1m = pd.read_parquet(RAW_DIR / "vix1d_1min.parquet")
    vix1d_1m["datetime"] = pd.to_datetime(vix1d_1m["datetime"])
    vix1d_1m = vix1d_1m[vix1d_1m["datetime"] <= state_dt]

    # Load decision state from historical file
    ds_file = DATA_DIR / "intraday_decision_state_v1_2.parquet"
    ds = pd.read_parquet(ds_file)
    ds["decision_datetime"] = pd.to_datetime(ds["decision_datetime"])
    ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
    decision_state = ds[ds["decision_datetime"] == decision_dt].copy()

    if decision_state.empty:
        return {"date": val_date, "time": val_time, "error": "No decision state", "matches": 0, "total": 0}

    # Build IC rows
    ic_rows = lp.build_ic_rows(decision_state, decision_dt)
    if ic_rows.empty:
        return {"date": val_date, "time": val_time, "error": "No IC rows", "matches": 0, "total": 0}

    # Build feature vector via live pipeline
    live_result = lp.build_feature_vector(
        ic_rows=ic_rows,
        daily_values=daily_values,
        spx_1m_history=spx_1m,
        vix_bars_history=vix_1m,
        vix1d_bars_history=vix1d_1m,
        decision_state_snapshot=decision_state,
        decision_dt=decision_dt,
        model_feature_cols=feature_cols,
    )

    # Get master rows for same decision_datetime
    master_rows = master[master["decision_datetime"] == decision_dt].copy()

    # Compare each strategy
    common_strats = set(live_result["strategy"]) & set(master_rows["strategy"])
    if not common_strats:
        return {"date": val_date, "time": val_time, "error": "No common strategies", "matches": 0, "total": 0}

    total = 0
    matches = 0
    mismatches = []

    for strat in sorted(common_strats):
        live_row = live_result[live_result["strategy"] == strat].iloc[0]
        master_row = master_rows[master_rows["strategy"] == strat].iloc[0]

        for col in feature_cols:
            total += 1
            lv = live_row.get(col, np.nan)
            rv = master_row.get(col, np.nan)

            both_nan = pd.isna(lv) and pd.isna(rv)
            if both_nan:
                matches += 1
                continue

            if pd.isna(lv) or pd.isna(rv):
                mismatches.append({
                    "date": val_date, "time": val_time, "strategy": strat,
                    "feature": col, "live": lv, "master": rv, "note": "NaN mismatch",
                })
                continue

            try:
                lf, rf = float(lv), float(rv)
                if lf == 0 and rf == 0:
                    matches += 1
                    continue
                abs_diff = abs(lf - rf)
                denom = max(abs(rf), 1e-10)
                pct_diff = abs_diff / denom

                if abs_diff < 1e-6 or pct_diff < 1e-4:
                    matches += 1
                else:
                    mismatches.append({
                        "date": val_date, "time": val_time, "strategy": strat,
                        "feature": col, "live": lf, "master": rf,
                        "note": f"diff={abs_diff:.6f} pct={pct_diff:.4%}",
                    })
            except (TypeError, ValueError):
                if lv == rv:
                    matches += 1
                else:
                    mismatches.append({
                        "date": val_date, "time": val_time, "strategy": strat,
                        "feature": col, "live": lv, "master": rv, "note": "type mismatch",
                    })

    return {
        "date": val_date,
        "time": val_time,
        "strategies": len(common_strats),
        "total": total,
        "matches": matches,
        "mismatches": len(mismatches),
        "match_pct": matches / total * 100 if total > 0 else 0,
        "mismatch_details": mismatches,
    }


def main():
    print("=" * 70)
    print("LIVE PIPELINE vs TRAINING MASTER VALIDATION")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load model feature cols
    feature_cols = lp.load_model_feature_cols()
    print(f"\nModel features: {len(feature_cols)}")

    # Load master
    print("Loading training master...")
    master = pd.read_parquet(MASTER_FILE)
    master["decision_datetime"] = pd.to_datetime(master["decision_datetime"])
    print(f"Master rows: {len(master):,}")

    # Run each validation point
    all_results = []
    all_mismatches = []
    grand_total = 0
    grand_matches = 0

    for val_date, val_time in VALIDATION_POINTS:
        print(f"\n{'─' * 70}")
        print(f"Testing: {val_date} {val_time}")
        print(f"{'─' * 70}")

        result = run_one(val_date, val_time, feature_cols, master)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_results.append(result)
            continue

        grand_total += result["total"]
        grand_matches += result["matches"]
        pct = result["match_pct"]

        print(f"  Strategies: {result['strategies']}")
        print(f"  Comparisons: {result['total']}")
        print(f"  Matches: {result['matches']}")
        print(f"  Mismatches: {result['mismatches']}")
        print(f"  Match rate: {pct:.1f}%")

        if result["mismatch_details"]:
            print(f"\n  Top mismatches:")
            for mm in result["mismatch_details"][:10]:
                print(f"    {mm['strategy']:<15} {mm['feature']:<45} live={mm['live']!s:>15}  master={mm['master']!s:>15}  ({mm['note']})")

        all_results.append(result)
        all_mismatches.extend(result.get("mismatch_details", []))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Validation points: {len(VALIDATION_POINTS)}")
    print(f"  Total comparisons: {grand_total:,}")
    print(f"  Total matches: {grand_matches:,}")
    print(f"  Total mismatches: {grand_total - grand_matches:,}")
    print(f"  Overall match rate: {grand_matches / grand_total * 100:.2f}%" if grand_total > 0 else "  N/A")

    # Save detailed report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"validation_report_{ts}.csv"
    if all_mismatches:
        pd.DataFrame(all_mismatches).to_csv(report_file, index=False)
        print(f"\n  Mismatch details: {report_file}")

    summary_file = REPORT_DIR / f"validation_summary_{ts}.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "mismatch_details"} for r in all_results]).to_csv(summary_file, index=False)
    print(f"  Summary: {summary_file}")

    # Unique mismatched features
    if all_mismatches:
        unique_feats = sorted(set(mm["feature"] for mm in all_mismatches))
        print(f"\n  Unique mismatched features ({len(unique_feats)}):")
        for f in unique_feats:
            count = sum(1 for mm in all_mismatches if mm["feature"] == f)
            print(f"    {f:<50} ({count} occurrences)")

    print(f"\n{'=' * 70}")
    if grand_total == grand_matches:
        print("RESULT: PERFECT MATCH — live pipeline is identical to training")
    else:
        print(f"RESULT: {grand_total - grand_matches} mismatches found — review report")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
