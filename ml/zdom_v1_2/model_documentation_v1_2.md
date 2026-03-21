# ZDOM V1.2 — Model Documentation
## Zero Day Options Model — Entry Go/No-Go

## 1. Overview

| Field | Value |
|-------|-------|
| Model Name | ZDOM V1.2 — Entry Go/No-Go |
| Objective | Predict whether a 0DTE SPX Iron Condor entered at a given minute will be profitable |
| Decision | Binary: ENTER (1) or SKIP (0) |
| Status | Data / target rebuild in progress |
| Owner | Matto |
| Code | `~/ironworks/projects/options-model/` |
| Broker | Tradier (sandbox for dev, `api.tradier.com` for prod) |

ZDOM V1.2 keeps the same high-level objective as V1, but rebuilds the target spine and intraday source data around a stricter, production-aligned definition of a valid strategy row. The main changes are: decision-ready T-1 intraday data, greeks snapshot pricing instead of the deprecated minute-quotes dependency, and exact-condor dedup within each decision minute.

## 2. Core Data Design Changes

1. `intraday_decision_state_v1_2` is now the canonical intraday source. A row labeled `decision_datetime = 10:00:00` represents the completed `09:59:00` state known at decision time.
2. Intraday option pricing comes directly from `spxw_0dte_intraday_greeks.parquet` using snapshot `bid`, `ask`, and `mid`.
3. The `v1_2` intraday build uses four raw tables only: `spx_1min`, `spxw_0dte_intraday_greeks`, `vix_1min`, and `vix1d_1min`.
4. The pre-outcome master spine is `master_data_v1_2.parquet`, one row per `decision_datetime x strategy` after exact-condor dedup within each decision minute.

## 3. Strategy Definition

All strategies remain fixed-width 25-point iron condors with target short deltas:
- `IC_05d_25w`
- `IC_10d_25w`
- `IC_15d_25w`
- `IC_20d_25w`
- `IC_25d_25w`
- `IC_30d_25w`
- `IC_35d_25w`
- `IC_40d_25w`
- `IC_45d_25w`

For each decision minute, the build selects the nearest valid short call and short put by target delta, then selects the long wings at +/-25 points.

## 4. New V1.2 Dedup Rule

V1.2 does not hard-filter rows by delta error. Instead, it removes only exact duplicate condors within the same decision minute.

A duplicate is defined as:
- same `decision_datetime`
- same `short_call`
- same `short_put`
- same `long_call`
- same `long_put`

These are duplicates only within the same `decision_datetime`. The same 4 legs at a different timestamp are not duplicates. Sharing only one side of the condor is also not a duplicate.

If multiple strategies map to the same exact 4-leg condor at the same `decision_datetime`, V1.2 keeps only the highest target-delta strategy for that combination.

## 5. Why This Change Matters

V1 allowed too much overlap between adjacent strategies because multiple strategy labels could map to the same exact strike combination at the same timestamp. That created label noise and a backtest that was too permissive relative to production reality.

V1.2 fixes this by forcing one real trade structure to map to one strategy only. This removes conflicting strategy labels while preserving distinct condors that share only one side or occur at different timestamps.

## 6. Current V1.2 Artifacts

| Artifact | Path |
|---------|------|
| Intraday decision-state table | `1_data_collection/data/v1_2/intraday_decision_state_v1_2.parquet` |
| Master spine (pre-outcome) | `3_data_join/master_data_v1_2.parquet` |
| Main schema dictionary | `data_dictionary_codex.csv` |
| Raw ThetaData dictionary | `1_data_collection/raw_thetadata_dictionary_codex.csv` |

Current `master_data_v1_2` fields:
- `decision_datetime`
- `date`
- `strategy`
- `spx_prev_min_close`
- `short_call`
- `short_put`
- `long_call`
- `long_put`
- `call_wing_width`
- `put_wing_width`
- `sc_delta`
- `sp_delta`
- `sc_iv`
- `sp_iv`
- `credit`

## 6A. Fixed Date Range

Current `master_data_v1_2` is intentionally capped at **Friday, March 13, 2026** for downstream daily-join work so that macro / FRED coverage is locked and reproducible.

- Master-data max date: `2026-03-13`
- Required daily-source prep range: `2022-06-06` through `2026-03-13`

The daily-source prep range starts earlier than the master table because long-lookback daily features (for example `spx_vs_sma_200`) require warmup history before the first usable decision date.

Any future rebuild of the `v1_2` master table or its daily joins should respect this cap unless the project intentionally decides to extend the full daily-source stack beyond that date.

For feature engineering, the final table is allowed to begin at the master-table start date of `2023-03-13`. Newer source families with structurally shorter history, especially `VIX1D`, are allowed to remain null in the early portion of the sample rather than forcing a later global start date for the entire modeling table.

## 7. Current Population Impact

After applying exact-condor dedup within each decision minute:
- pre-dedup spine size: about 1.92M rows
- kept rows: about 1.70M rows
- drop rate: about 11.6%

This removes conflicting strategy assignments only when the full 4-leg condor is identical at the same decision timestamp.

## 8. Production Implication

If this dataset definition is used for training, the same eligibility logic should be used in production:
1. build the latest decision-ready minute state
2. find nearest valid short legs by target delta
3. if multiple strategies map to the same exact 4-leg condor at the same decision minute, keep only the highest target-delta assignment
4. do not deduplicate rows that share only one side or occur at different decision timestamps

This keeps training and live execution aligned.

## 9. Next Step

The next step in V1.2 is to add the TP outcome layer (`tp10_*` through `tp50_*`) onto `master_data_v1_2`, then continue with the daily feature joins and the model-table rebuild.
