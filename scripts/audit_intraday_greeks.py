"""
Comprehensive data quality audit for spxw_0dte_intraday_greeks.parquet.
Uses chunked reading via PyArrow to handle 1.3GB / 55M rows on 16GB RAM.
"""

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

PARQUET = Path(__file__).resolve().parent.parent / "data" / "spxw_0dte_intraday_greeks.parquet"
CHUNK_SIZE = 2_000_000  # rows per chunk

print(f"{'='*80}")
print(f"DATA QUALITY AUDIT: {PARQUET.name}")
print(f"{'='*80}\n")

# ── 0. File metadata ─────────────────────────────────────────────────────────
pf = pq.ParquetFile(PARQUET)
meta = pf.metadata
print(f"[0] FILE METADATA")
print(f"    Rows:       {meta.num_rows:,}")
print(f"    Row groups: {meta.num_row_groups}")
print(f"    Columns:    {meta.num_columns}")
print(f"    Schema:")
for i in range(meta.num_columns):
    col = pf.schema_arrow.field(i)
    print(f"      {col.name:20s}  {col.type}")
print()

# ── Accumulator structures ────────────────────────────────────────────────────
total_rows = 0
null_counts = Counter()
col_names = [pf.schema_arrow.field(i).name for i in range(meta.num_columns)]
col_dtypes = {pf.schema_arrow.field(i).name: str(pf.schema_arrow.field(i).type) for i in range(meta.num_columns)}

# Greeks reasonableness counters
call_delta_bad = 0; put_delta_bad = 0; call_count = 0; put_count = 0
gamma_negative = 0; gamma_over_05 = 0
theta_positive = 0
vega_negative = 0
iv_bad = 0
rho_call_negative = 0; rho_put_positive = 0

# Price sanity
bid_gt_ask = 0; bid_negative = 0; mid_mismatch = 0

# Strike sanity
strike_out_of_range = 0
strike_min = float('inf'); strike_max = float('-inf')

# Timestamp tracking
dates_row_counts = Counter()
timestamps_per_day = defaultdict(set)  # date -> set of unique timestamps (capped for memory)
min_date = None; max_date = None

# Duplicate tracking - we'll check per-chunk and across chunks via a hash
dup_count_within_chunks = 0

# Sample values (first chunk only)
sample_values = {}

# Extreme value tracking
extreme_examples = defaultdict(list)  # col -> list of (value, context)

# Lambda/gamma analysis
gamma_stats_call = []  # (moneyness, gamma) sampled
gamma_stats_put = []
gamma_zero_count = 0
gamma_total = 0

# Per-date row counts for histogram
date_row_counts = Counter()

print(f"[1] SCANNING {meta.num_rows:,} ROWS IN CHUNKS OF {CHUNK_SIZE:,}...")
print()

chunk_idx = 0
for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
    df = batch.to_pandas()
    n = len(df)
    total_rows += n
    chunk_idx += 1

    if chunk_idx % 5 == 0 or chunk_idx == 1:
        print(f"    Chunk {chunk_idx}: rows {total_rows - n + 1:,} to {total_rows:,}")

    # Sample values from first chunk
    if chunk_idx == 1:
        for c in df.columns:
            non_null = df[c].dropna()
            if len(non_null) > 0:
                sample_values[c] = non_null.iloc[:3].tolist()
            else:
                sample_values[c] = ["ALL NULL"]

    # ── Null counts ──
    for c in df.columns:
        null_counts[c] += int(df[c].isna().sum())

    # ── Date tracking ──
    if "date" in df.columns:
        dates = df["date"].astype(str)
        for d, cnt in dates.value_counts().items():
            date_row_counts[d] += cnt
        d_min = dates.min()
        d_max = dates.max()
        if min_date is None or d_min < min_date:
            min_date = d_min
        if max_date is None or d_max > max_date:
            max_date = d_max

    # ── Identify calls and puts ──
    is_call = df["right"] == "call" if "right" in df.columns else pd.Series(False, index=df.index)
    is_put = df["right"] == "put" if "right" in df.columns else pd.Series(False, index=df.index)
    call_count += int(is_call.sum())
    put_count += int(is_put.sum())

    # ── Delta checks ──
    if "delta" in df.columns:
        d = df["delta"]
        call_delta_bad += int(((d[is_call] < -0.01) | (d[is_call] > 1.01)).sum())
        put_delta_bad += int(((d[is_put] < -1.01) | (d[is_put] > 0.01)).sum())

    # ── Gamma checks (the lambda->gamma mapping) ──
    if "gamma" in df.columns:
        g = df["gamma"]
        gamma_total += int(g.notna().sum())
        gamma_negative += int((g < -0.0001).sum())
        gamma_over_05 += int((g > 0.05).sum())
        gamma_zero_count += int((g == 0.0).sum())

        # Sample gamma vs moneyness for ATM analysis
        if "underlying_price" in df.columns and "strike" in df.columns and chunk_idx <= 3:
            sample = df[is_call & g.notna() & (g != 0)].head(500)
            if len(sample) > 0:
                moneyness = (sample["strike"] - sample["underlying_price"]).abs() / sample["underlying_price"]
                for m, gv in zip(moneyness.values, sample["gamma"].values):
                    gamma_stats_call.append((m, gv))
            sample_p = df[is_put & g.notna() & (g != 0)].head(500)
            if len(sample_p) > 0:
                moneyness_p = (sample_p["strike"] - sample_p["underlying_price"]).abs() / sample_p["underlying_price"]
                for m, gv in zip(moneyness_p.values, sample_p["gamma"].values):
                    gamma_stats_put.append((m, gv))

    # ── Theta checks ──
    if "theta" in df.columns:
        theta_positive += int((df["theta"] > 0.01).sum())

    # ── Vega checks ──
    if "vega" in df.columns:
        vega_negative += int((df["vega"] < -0.01).sum())

    # ── IV checks ──
    if "implied_vol" in df.columns:
        iv = df["implied_vol"]
        iv_bad += int(((iv < 0) | (iv > 5.0)).sum())

    # ── Rho checks ──
    if "rho" in df.columns:
        rho = df["rho"]
        rho_call_negative += int((rho[is_call] < -0.01).sum())
        rho_put_positive += int((rho[is_put] > 0.01).sum())

    # ── Price sanity ──
    if "bid" in df.columns and "ask" in df.columns:
        bid_gt_ask += int((df["bid"] > df["ask"] + 0.01).sum())
        bid_negative += int((df["bid"] < -0.001).sum())
    if "mid" in df.columns and "bid" in df.columns and "ask" in df.columns:
        expected_mid = (df["bid"] + df["ask"]) / 2
        mid_mismatch += int((abs(df["mid"] - expected_mid) > 0.01).sum())

    # ── Strike sanity ──
    if "strike" in df.columns:
        s = df["strike"]
        strike_out_of_range += int(((s < 2000) | (s > 7000)).sum())
        s_min = s.min()
        s_max = s.max()
        if s_min < strike_min:
            strike_min = s_min
        if s_max > strike_max:
            strike_max = s_max

    # ── Duplicate check within chunk ──
    key_cols = [c for c in ["date", "strike", "right", "timestamp"] if c in df.columns]
    if len(key_cols) == 4:
        dup_count_within_chunks += int(df.duplicated(subset=key_cols).sum())

    # ── Outlier flagging: collect extreme values (sample a few) ──
    numeric_cols = ["delta", "gamma", "theta", "vega", "implied_vol", "rho", "bid", "ask", "mid", "underlying_price"]
    for c in numeric_cols:
        if c in df.columns and len(extreme_examples[c]) < 10:
            col = df[c].dropna()
            if len(col) > 0:
                q01 = col.quantile(0.001)
                q99 = col.quantile(0.999)
                outliers = col[(col < q01) | (col > q99)]
                if len(outliers) > 0:
                    for idx in outliers.head(3).index:
                        ctx = {k: df.loc[idx, k] for k in ["date", "strike", "right"] if k in df.columns}
                        extreme_examples[c].append((df.loc[idx, c], ctx))

    del df

print(f"\n    Total rows scanned: {total_rows:,}")
print()

# ── REPORT ────────────────────────────────────────────────────────────────────

print(f"{'='*80}")
print(f"[A] COLUMN INVENTORY")
print(f"{'='*80}")
print(f"{'Column':<22s} {'Dtype':<15s} {'Sample Values'}")
print(f"{'-'*22} {'-'*15} {'-'*40}")
for c in col_names:
    sv = sample_values.get(c, ["?"])
    sv_str = str(sv[:3])
    if len(sv_str) > 60:
        sv_str = sv_str[:57] + "..."
    print(f"{c:<22s} {col_dtypes[c]:<15s} {sv_str}")
print()

print(f"{'='*80}")
print(f"[B] NULL / NaN ANALYSIS")
print(f"{'='*80}")
print(f"{'Column':<22s} {'Nulls':>12s} {'Pct':>8s}")
print(f"{'-'*22} {'-'*12} {'-'*8}")
any_nulls = False
for c in col_names:
    nc = null_counts[c]
    pct = nc / total_rows * 100 if total_rows > 0 else 0
    flag = " *** HIGH ***" if pct > 5 else ""
    if nc > 0:
        any_nulls = True
    print(f"{c:<22s} {nc:>12,} {pct:>7.2f}%{flag}")
if not any_nulls:
    print("    No nulls found in any column.")
print()

print(f"{'='*80}")
print(f"[C] LAMBDA -> GAMMA MAPPING CHECK")
print(f"{'='*80}")
print(f"    The fetch script maps ThetaData's 'lambda' field to 'gamma' (line 76).")
print(f"    Total gamma values:    {gamma_total:,}")
print(f"    Gamma == 0.0:          {gamma_zero_count:,} ({gamma_zero_count/gamma_total*100:.2f}%)" if gamma_total > 0 else "")
print(f"    Gamma < 0:             {gamma_negative:,} ({gamma_negative/gamma_total*100:.2f}%)" if gamma_total > 0 else "")
print(f"    Gamma > 0.05:          {gamma_over_05:,} ({gamma_over_05/gamma_total*100:.2f}%)" if gamma_total > 0 else "")
print()

# Analyze gamma vs moneyness
if gamma_stats_call:
    gs = np.array(gamma_stats_call)
    atm_mask = gs[:, 0] < 0.005  # within 0.5% of ATM
    otm_mask = gs[:, 0] > 0.02   # >2% OTM
    if atm_mask.any():
        print(f"    ATM calls (moneyness < 0.5%): gamma mean={gs[atm_mask, 1].mean():.6f}, max={gs[atm_mask, 1].max():.6f}")
    if otm_mask.any():
        print(f"    OTM calls (moneyness > 2.0%): gamma mean={gs[otm_mask, 1].mean():.6f}, max={gs[otm_mask, 1].max():.6f}")
    print(f"    Overall call gamma range: [{gs[:, 1].min():.6f}, {gs[:, 1].max():.6f}]")

if gamma_stats_put:
    gs = np.array(gamma_stats_put)
    atm_mask = gs[:, 0] < 0.005
    otm_mask = gs[:, 0] > 0.02
    if atm_mask.any():
        print(f"    ATM puts  (moneyness < 0.5%): gamma mean={gs[atm_mask, 1].mean():.6f}, max={gs[atm_mask, 1].max():.6f}")
    if otm_mask.any():
        print(f"    OTM puts  (moneyness > 2.0%): gamma mean={gs[otm_mask, 1].mean():.6f}, max={gs[otm_mask, 1].max():.6f}")
    print(f"    Overall put  gamma range: [{gs[:, 1].min():.6f}, {gs[:, 1].max():.6f}]")

# CRITICAL: Check if lambda is actually lambda (leverage) not gamma
if gamma_stats_call:
    gs = np.array(gamma_stats_call)
    pct_over_1 = (gs[:, 1] > 1.0).mean() * 100
    pct_over_10 = (gs[:, 1] > 10.0).mean() * 100
    mean_val = gs[:, 1].mean()
    print(f"\n    CRITICAL CHECK - Is 'lambda' actually leverage (not gamma)?")
    print(f"      Mean 'gamma' value:       {mean_val:.4f}")
    print(f"      % values > 1.0:           {pct_over_1:.2f}%")
    print(f"      % values > 10.0:          {pct_over_10:.2f}%")
    if mean_val > 1.0 or pct_over_1 > 10:
        print(f"      *** ALERT: These values look like LEVERAGE (lambda), NOT gamma! ***")
        print(f"      Lambda (leverage) = delta * S / V, typically 5-50 for options.")
        print(f"      Gamma should be 0 to ~0.01 for SPX options.")
    elif mean_val > 0.05:
        print(f"      WARNING: Mean gamma seems high for SPX options. Review needed.")
    else:
        print(f"      Values appear consistent with actual gamma.")
print()

print(f"{'='*80}")
print(f"[D] GREEKS REASONABLENESS")
print(f"{'='*80}")
print(f"    Call rows: {call_count:,}  |  Put rows: {put_count:,}")
print()
print(f"    DELTA:")
print(f"      Call delta out of [0,1]:    {call_delta_bad:,} ({call_delta_bad/max(call_count,1)*100:.2f}%)")
print(f"      Put delta out of [-1,0]:    {put_delta_bad:,} ({put_delta_bad/max(put_count,1)*100:.2f}%)")
print()
print(f"    GAMMA:")
print(f"      Negative gamma:             {gamma_negative:,} ({gamma_negative/max(gamma_total,1)*100:.2f}%)")
print(f"      Gamma > 0.05 (high):        {gamma_over_05:,} ({gamma_over_05/max(gamma_total,1)*100:.2f}%)")
print()
print(f"    THETA:")
print(f"      Positive theta (should be -): {theta_positive:,} ({theta_positive/max(total_rows,1)*100:.2f}%)")
print()
print(f"    VEGA:")
print(f"      Negative vega (should be +):  {vega_negative:,} ({vega_negative/max(total_rows,1)*100:.2f}%)")
print()
print(f"    IMPLIED VOL:")
print(f"      Out of [0, 5.0]:              {iv_bad:,} ({iv_bad/max(total_rows,1)*100:.2f}%)")
print()
print(f"    RHO:")
print(f"      Call rho < 0 (unusual):       {rho_call_negative:,} ({rho_call_negative/max(call_count,1)*100:.2f}%)")
print(f"      Put rho > 0 (unusual):        {rho_put_positive:,} ({rho_put_positive/max(put_count,1)*100:.2f}%)")
print()

print(f"{'='*80}")
print(f"[E] PRICE SANITY")
print(f"{'='*80}")
print(f"    Bid > Ask:           {bid_gt_ask:,} ({bid_gt_ask/max(total_rows,1)*100:.2f}%)")
print(f"    Bid < 0:             {bid_negative:,}")
print(f"    Mid != (bid+ask)/2:  {mid_mismatch:,}")
print()

print(f"{'='*80}")
print(f"[F] STRIKE SANITY")
print(f"{'='*80}")
print(f"    Strike range:        [{strike_min}, {strike_max}]")
print(f"    Out of [2000,7000]:  {strike_out_of_range:,}")
print()

print(f"{'='*80}")
print(f"[G] TIMESTAMP / DATE COVERAGE")
print(f"{'='*80}")
print(f"    Date range:          {min_date} to {max_date}")
num_dates = len(date_row_counts)
print(f"    Unique dates:        {num_dates}")

# Check for gaps (trading day gaps)
sorted_dates = sorted(date_row_counts.keys())
if sorted_dates:
    date_series = pd.to_datetime(sorted_dates)
    # Generate expected trading days (weekdays only, rough check)
    all_weekdays = pd.bdate_range(date_series.min(), date_series.max())
    missing_weekdays = set(all_weekdays.strftime("%Y-%m-%d")) - set(sorted_dates)
    # Filter out known holidays (rough - just report count)
    print(f"    Missing weekdays:    {len(missing_weekdays)} (includes holidays)")
    if len(missing_weekdays) <= 30:
        for d in sorted(missing_weekdays):
            print(f"      {d}")
    else:
        print(f"      (too many to list - first 10:)")
        for d in sorted(missing_weekdays)[:10]:
            print(f"      {d}")

print()

# Row count distribution
row_counts = sorted(date_row_counts.values())
print(f"    Rows per day:")
print(f"      Min:    {min(row_counts):,}")
print(f"      P5:     {int(np.percentile(row_counts, 5)):,}")
print(f"      P25:    {int(np.percentile(row_counts, 25)):,}")
print(f"      Median: {int(np.percentile(row_counts, 50)):,}")
print(f"      P75:    {int(np.percentile(row_counts, 75)):,}")
print(f"      P95:    {int(np.percentile(row_counts, 95)):,}")
print(f"      Max:    {max(row_counts):,}")

# Flag suspicious days
print(f"\n    Days with fewest rows (bottom 10):")
for d, c in sorted(date_row_counts.items(), key=lambda x: x[1])[:10]:
    print(f"      {d}: {c:,} rows")

print(f"\n    Days with most rows (top 10):")
for d, c in sorted(date_row_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"      {d}: {c:,} rows")

# Check for extremely low/high days
median_rows = np.percentile(row_counts, 50)
sparse_days = [(d, c) for d, c in date_row_counts.items() if c < median_rows * 0.1]
dense_days = [(d, c) for d, c in date_row_counts.items() if c > median_rows * 3]
if sparse_days:
    print(f"\n    *** SPARSE DAYS (<10% of median): {len(sparse_days)} ***")
    for d, c in sorted(sparse_days):
        print(f"      {d}: {c:,}")
if dense_days:
    print(f"\n    *** DENSE DAYS (>3x median): {len(dense_days)} ***")
    for d, c in sorted(dense_days)[:10]:
        print(f"      {d}: {c:,}")
print()

print(f"{'='*80}")
print(f"[H] DUPLICATE CHECK")
print(f"{'='*80}")
print(f"    Duplicates on (date, strike, right, timestamp):")
print(f"      Within-chunk duplicates found: {dup_count_within_chunks:,}")
if dup_count_within_chunks > 0:
    print(f"      *** WARNING: Duplicates detected! ***")
else:
    print(f"      No within-chunk duplicates (cross-chunk check skipped for memory).")
print()

print(f"{'='*80}")
print(f"[I] OUTLIER EXAMPLES (0.1th / 99.9th percentile)")
print(f"{'='*80}")
for c, examples in sorted(extreme_examples.items()):
    if examples:
        print(f"    {c}:")
        for val, ctx in examples[:5]:
            ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items())
            print(f"      value={val}  ({ctx_str})")
print()

print(f"{'='*80}")
print(f"[J] DATE DISTRIBUTION HISTOGRAM")
print(f"{'='*80}")
# Bucket by month
month_counts = Counter()
for d, c in date_row_counts.items():
    month = d[:7]
    month_counts[month] += c
print(f"    {'Month':<10s} {'Rows':>12s} {'Days':>6s} {'Avg/Day':>10s}")
print(f"    {'-'*10} {'-'*12} {'-'*6} {'-'*10}")
month_days = Counter()
for d in date_row_counts:
    month_days[d[:7]] += 1
for m in sorted(month_counts.keys()):
    avg = month_counts[m] / max(month_days[m], 1)
    print(f"    {m:<10s} {month_counts[m]:>12,} {month_days[m]:>6} {avg:>10,.0f}")
print()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"{'='*80}")
print(f"SUMMARY OF FINDINGS")
print(f"{'='*80}")
issues = []
if gamma_zero_count > gamma_total * 0.5:
    issues.append(f"HIGH: {gamma_zero_count/gamma_total*100:.0f}% of gamma values are 0.0")
if gamma_stats_call:
    gs = np.array(gamma_stats_call)
    if gs[:, 1].mean() > 1.0:
        issues.append("CRITICAL: 'gamma' column contains lambda/leverage values, NOT gamma!")
if call_delta_bad > call_count * 0.05:
    issues.append(f"HIGH: {call_delta_bad/call_count*100:.1f}% of call deltas outside [0,1]")
if put_delta_bad > put_count * 0.05:
    issues.append(f"HIGH: {put_delta_bad/put_count*100:.1f}% of put deltas outside [-1,0]")
if theta_positive > total_rows * 0.05:
    issues.append(f"MEDIUM: {theta_positive/total_rows*100:.1f}% of theta values are positive")
if vega_negative > total_rows * 0.05:
    issues.append(f"MEDIUM: {vega_negative/total_rows*100:.1f}% of vega values are negative")
if bid_gt_ask > total_rows * 0.01:
    issues.append(f"HIGH: {bid_gt_ask/total_rows*100:.1f}% of rows have bid > ask")
if dup_count_within_chunks > 0:
    issues.append(f"HIGH: {dup_count_within_chunks:,} duplicate rows found")
if sparse_days:
    issues.append(f"LOW: {len(sparse_days)} days with suspiciously few rows")
any_nulls_total = sum(null_counts.values())
if any_nulls_total > 0:
    issues.append(f"INFO: {any_nulls_total:,} total null values across all columns")

if issues:
    for iss in issues:
        print(f"  - {iss}")
else:
    print("  No major issues found.")

print(f"\nAudit complete. {total_rows:,} rows examined.")
