"""
Local data explorer — serves a web dashboard on localhost:5050
showing all parquet files, schemas, null rates, joins, and sample data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

app = Flask(__name__)

# Define the relational schema — how tables join
SCHEMA = {
    "raw_data": {
        "spx_1min.parquet": {
            "description": "SPX index 1-min OHLC bars from ThetaData",
            "key": "datetime",
            "grain": "1 minute",
            "source": "ThetaData /v3/index/history/ohlc (SPX)",
        },
        "vix_1min.parquet": {
            "description": "VIX 1-min OHLC bars from ThetaData",
            "key": "datetime",
            "grain": "1 minute",
            "source": "ThetaData /v3/index/history/ohlc (VIX)",
        },
        "vix1d_1min.parquet": {
            "description": "VIX1D (0DTE vol index) 1-min OHLC bars",
            "key": "datetime",
            "grain": "1 minute",
            "source": "ThetaData /v3/index/history/ohlc (VIX1D)",
        },
        "spxw_0dte_eod.parquet": {
            "description": "SPXW 0DTE option chain EOD (all strikes, calls+puts)",
            "key": "date + strike + right",
            "grain": "1 row per strike per right per day",
            "source": "ThetaData /v3/option/history/eod",
        },
        "spxw_0dte_oi.parquet": {
            "description": "SPXW 0DTE open interest per strike",
            "key": "date + strike + right",
            "grain": "1 row per strike per right per day",
            "source": "ThetaData /v3/option/history/open_interest",
        },
        "spxw_0dte_intraday_greeks.parquet": {
            "description": "SPXW intraday greeks (delta, gamma, vega, IV) per minute per strike",
            "key": "date + strike + right + timestamp",
            "grain": "1 row per strike per right per minute",
            "source": "ThetaData /v3/option/history/greeks",
        },
        "spxw_term_structure.parquet": {
            "description": "SPXW term structure: EOD across all expirations at 15 strike offsets",
            "key": "date + expiration + strike + right",
            "grain": "1 row per expiration per strike per right per day",
            "source": "ThetaData /v3/option/history/eod (wildcard expiration)",
            "note": "BACKFILLING: expanding from 3 to 15 strikes (ATM ±25,50,75,100,150,200,250)",
        },
        "macro_regime.parquet": {
            "description": "FRED macro data: yields, credit spreads, DXY, oil, gold",
            "key": "date",
            "grain": "daily",
            "source": "FRED API + yfinance (gold)",
        },
        "presidential_cycles.parquet": {
            "description": "Presidential cycle year, election proximity, PMI, LEI, calendar features",
            "key": "date",
            "grain": "daily",
            "source": "Computed + FRED",
        },
    },
    "derived_data": {
        "spx_daily.parquet": {
            "description": "SPX daily OHLC (aggregated from 1-min)",
            "key": "date",
            "grain": "daily",
            "derived_from": "spx_1min.parquet",
        },
        "vix_daily.parquet": {
            "description": "VIX daily OHLC (aggregated from 1-min)",
            "key": "date",
            "grain": "daily",
            "derived_from": "vix_1min.parquet",
        },
        "vix1d_daily.parquet": {
            "description": "VIX1D daily OHLC (aggregated from 1-min)",
            "key": "date",
            "grain": "daily",
            "derived_from": "vix1d_1min.parquet",
        },
        "spx_merged.parquet": {
            "description": "SPX 1-min + VIX daily + econ calendar + MAG7 earnings + days-to-event",
            "key": "datetime",
            "grain": "1 minute",
            "derived_from": "spx_1min + vix_daily + econ_calendar + mag7_earnings + fomc_dates",
        },
    },
    "features": {
        "spx_features.parquet": {
            "description": "SPX technicals: SMAs, RSI, MACD, Bollinger, Yang-Zhang HV, HV/IV ratios, calendar",
            "key": "datetime (1-min) with daily features broadcast",
            "grain": "1 minute (daily features repeated)",
            "derived_from": "spx_merged.parquet",
        },
        "options_features.parquet": {
            "description": "Options microstructure: put/call ratios, volume, GEX, dealer gamma, walls",
            "key": "date",
            "grain": "daily",
            "derived_from": "spxw_0dte_eod + spxw_0dte_oi + spx_daily",
        },
        "iv_surface_features.parquet": {
            "description": "IV surface: ATM IV, 25d/10d skew, smile curvature, term structure slopes",
            "key": "date",
            "grain": "daily",
            "derived_from": "spxw_0dte_intraday_greeks + spxw_term_structure + spx_daily",
        },
        "regime_features.parquet": {
            "description": "2-state HMM regime (low-vol / high-vol) from SPX returns + VIX",
            "key": "date",
            "grain": "daily",
            "derived_from": "spx_daily + vix_daily",
        },
        "gex_regime_features.parquet": {
            "description": "3-state GEX regime (positive / transition / negative dealer gamma)",
            "key": "date",
            "grain": "daily",
            "derived_from": "options_features + vix_daily",
        },
        "vanna_charm_features.parquet": {
            "description": "Vanna (dDelta/dIV) and Charm (dDelta/dTime) second-order greeks",
            "key": "date",
            "grain": "daily",
            "derived_from": "spxw_0dte_intraday_greeks + spx_daily",
        },
        "momentum_features.parquet": {
            "description": "Intraday momentum: first 15/30/60 min returns, open drive strength",
            "key": "date",
            "grain": "daily",
            "derived_from": "spx_merged.parquet",
        },
        "breadth_features.parquet": {
            "description": "Market breadth: advance/decline, new highs/lows, McClellan oscillator",
            "key": "date",
            "grain": "daily",
            "derived_from": "yfinance (SPY, sector ETFs)",
        },
        "cross_asset_features.parquet": {
            "description": "Cross-asset: TLT/HYG/GLD/IWM ratios and correlations vs SPX",
            "key": "date",
            "grain": "daily",
            "derived_from": "yfinance (ETFs)",
        },
        "vol_expansion_features.parquet": {
            "description": "Vol expansion: realized vs implied vol spreads, vol regime persistence",
            "key": "date",
            "grain": "daily",
            "derived_from": "vix_daily + spx_daily",
        },
        "microstructure_features.parquet": {
            "description": "Previous IC outcomes, consecutive wins/losses, gap fill rate",
            "key": "date",
            "grain": "daily (collapsed from minute-level)",
            "derived_from": "target + spx_features + momentum_features",
        },
    },
    "target": {
        "target.parquet": {
            "description": "Minute-level IC simulation: delta-based strikes, real bid/ask credit, 3 target vars",
            "key": "datetime",
            "grain": "1 minute (10:00-15:00 ET, ~300/day)",
            "derived_from": "spx_1min + spxw_0dte_intraday_greeks",
            "targets": "good_trade (binary), risk_tier (0/1/2), ic_pnl_pct (continuous)",
        },
    },
    "model": {
        "model_table.parquet": {
            "description": "Final training table: target (minute spine) LEFT JOIN all daily features",
            "key": "datetime",
            "grain": "1 minute (~220K rows, ~300 cols)",
            "derived_from": "target LEFT JOIN all feature tables on date",
        },
    },
}


def analyze_parquet(filepath):
    """Analyze a parquet file and return stats."""
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        return {"error": str(e)}

    stats = {
        "rows": len(df),
        "cols": len(df.columns),
        "size_mb": round(filepath.stat().st_size / 1e6, 2),
        "columns": [],
    }

    # Date range
    for col in ["date", "datetime", "timestamp"]:
        if col in df.columns:
            dc = pd.to_datetime(df[col])
            stats["date_col"] = col
            stats["date_min"] = str(dc.min().date()) if not dc.isna().all() else "N/A"
            stats["date_max"] = str(dc.max().date()) if not dc.isna().all() else "N/A"
            stats["unique_dates"] = dc.dt.date.nunique()
            break

    # Per-column stats
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 1),
        }
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            valid = df[col].dropna()
            if len(valid) > 0:
                col_info["min"] = round(float(valid.min()), 4)
                col_info["max"] = round(float(valid.max()), 4)
                col_info["mean"] = round(float(valid.mean()), 4)
                col_info["std"] = round(float(valid.std()), 4)
                col_info["zeros"] = int((valid == 0).sum())
                col_info["infs"] = int(np.isinf(valid).sum())
        elif df[col].dtype == "object":
            col_info["unique"] = int(df[col].nunique())
            col_info["sample"] = str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else ""

        stats["columns"].append(col_info)

    return stats


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Options Model — Data Explorer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
h1 { color: #00ff88; margin-bottom: 5px; font-size: 24px; }
h2 { color: #00aaff; margin: 30px 0 15px; font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 5px; }
h3 { color: #ffaa00; margin: 20px 0 10px; font-size: 15px; }
.subtitle { color: #888; margin-bottom: 20px; font-size: 13px; }
.section { margin-bottom: 30px; }
.table-card { background: #141414; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.table-card:hover { border-color: #444; }
.table-name { color: #00ff88; font-size: 15px; font-weight: bold; }
.table-desc { color: #aaa; font-size: 12px; margin: 4px 0 8px; }
.table-meta { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 10px; }
.meta-item { font-size: 12px; }
.meta-label { color: #666; }
.meta-value { color: #fff; font-weight: bold; }
.meta-value.warn { color: #ff6600; }
.meta-value.good { color: #00ff88; }
.meta-value.bad { color: #ff3333; }
table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }
th { background: #1a1a1a; color: #888; text-align: left; padding: 6px 8px; border-bottom: 1px solid #333; font-weight: normal; }
td { padding: 5px 8px; border-bottom: 1px solid #1a1a1a; }
tr:hover td { background: #1a1a2a; }
.null-bar { display: inline-block; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
.null-ok { background: #00ff88; }
.null-warn { background: #ffaa00; }
.null-bad { background: #ff3333; }
.note { color: #ff6600; font-size: 11px; font-style: italic; }
.schema-flow { background: #0d1117; border: 1px solid #2a2a2a; border-radius: 8px; padding: 20px; margin: 20px 0; font-size: 12px; line-height: 1.8; }
.schema-flow code { color: #00ff88; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 15px 0; }
.summary-card { background: #141414; border: 1px solid #2a2a2a; border-radius: 6px; padding: 12px; text-align: center; }
.summary-number { font-size: 28px; font-weight: bold; color: #00ff88; }
.summary-label { font-size: 11px; color: #888; margin-top: 4px; }
.toggle { cursor: pointer; color: #00aaff; font-size: 11px; }
.col-details { display: none; }
.col-details.show { display: table-row-group; }
nav { position: sticky; top: 0; background: #0a0a0a; padding: 10px 0; border-bottom: 1px solid #333; z-index: 100; margin-bottom: 20px; }
nav a { color: #00aaff; text-decoration: none; margin-right: 15px; font-size: 12px; }
nav a:hover { color: #fff; }
.target-dist { display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
.target-dist div { display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; }
</style>
</head>
<body>

<h1>Options Model — Data Explorer</h1>
<p class="subtitle">Generated {{ generated_at }} | Term structure backfill: {{ term_status }}</p>

<nav>
<a href="#summary">Summary</a>
<a href="#schema">Schema</a>
<a href="#raw">Raw Data</a>
<a href="#derived">Derived</a>
<a href="#features">Features</a>
<a href="#target">Target</a>
<a href="#model">Model Table</a>
<a href="#issues">Issues</a>
</nav>

<div id="summary">
<h2>Summary</h2>
<div class="summary-grid">
    <div class="summary-card">
        <div class="summary-number">{{ total_files }}</div>
        <div class="summary-label">Parquet Files</div>
    </div>
    <div class="summary-card">
        <div class="summary-number">{{ total_rows }}</div>
        <div class="summary-label">Total Rows</div>
    </div>
    <div class="summary-card">
        <div class="summary-number">{{ total_size_gb }}</div>
        <div class="summary-label">Total Size (GB)</div>
    </div>
    <div class="summary-card">
        <div class="summary-number">{{ model_features }}</div>
        <div class="summary-label">Model Features</div>
    </div>
    <div class="summary-card">
        <div class="summary-number">{{ model_rows }}</div>
        <div class="summary-label">Training Samples</div>
    </div>
    <div class="summary-card">
        <div class="summary-number">{{ trading_days }}</div>
        <div class="summary-label">Trading Days</div>
    </div>
</div>
</div>

<div id="schema">
<h2>Data Flow</h2>
<div class="schema-flow">
<pre>
ThetaData (localhost:25503)          FRED API              yfinance
    │                                   │                     │
    ├─ <code>spx_1min</code> ──────────┐           │                     │
    ├─ <code>vix_1min</code> ──────────┤    <code>macro_regime</code> ◄────────┘   <code>breadth_features</code>
    ├─ <code>vix1d_1min</code> ────────┤    <code>presidential_cycles</code>      <code>cross_asset_features</code>
    ├─ <code>spxw_0dte_eod</code> ─────┤           │
    ├─ <code>spxw_0dte_oi</code> ──────┤           │
    ├─ <code>spxw_intraday_greeks</code>┤           │
    └─ <code>spxw_term_structure</code> ┤           │
                              │           │
                    ┌─────────┴───────────┴─────────────────────┐
                    │              FEATURE LAYER                  │
                    │                                             │
                    │  <code>spx_features</code> ◄── spx_merged (1-min+VIX+events)
                    │  <code>options_features</code> ◄── EOD+OI+SPX daily
                    │  <code>iv_surface_features</code> ◄── greeks+term_struct
                    │  <code>regime_features</code> ◄── SPX+VIX (HMM 2-state)
                    │  <code>gex_regime_features</code> ◄── options+VIX (HMM 3-state)
                    │  <code>vanna_charm_features</code> ◄── intraday greeks
                    │  <code>momentum_features</code> ◄── spx_merged
                    │  <code>vol_expansion_features</code> ◄── VIX+SPX
                    │  <code>microstructure_features</code> ◄── target+SPX
                    └─────────────────────────────────────────────┘
                              │
                    ┌─────────┴───────────┐
                    │    TARGET LAYER       │
                    │                       │
                    │  <code>target.parquet</code>       │  ◄── spx_1min + intraday_greeks
                    │  (220K rows, 1/min)   │      Delta-based IC simulation
                    │  good_trade, risk_tier │      Real bid/ask credit
                    └───────────┬───────────┘
                              │
                    ┌─────────┴───────────┐
                    │    MODEL TABLE        │
                    │                       │
                    │  <code>model_table.parquet</code>  │  target LEFT JOIN all features
                    │  220K rows × 280 feat │  on date (daily features broadcast)
                    └───────────────────────┘
</pre>
</div>
</div>

{% for section_name, section_files in sections %}
<div id="{{ section_name }}">
<h2>{{ section_title[section_name] }}</h2>
{% for filename, meta in section_files.items() %}
{% set stats = file_stats.get(filename, {}) %}
<div class="table-card">
    <div class="table-name">{{ filename }}</div>
    <div class="table-desc">{{ meta.get('description', '') }}</div>
    {% if meta.get('note') %}<div class="note">⚠ {{ meta['note'] }}</div>{% endif %}

    <div class="table-meta">
        <div class="meta-item"><span class="meta-label">Rows: </span><span class="meta-value">{{ '{:,}'.format(stats.get('rows', 0)) }}</span></div>
        <div class="meta-item"><span class="meta-label">Cols: </span><span class="meta-value">{{ stats.get('cols', '?') }}</span></div>
        <div class="meta-item"><span class="meta-label">Size: </span><span class="meta-value">{{ stats.get('size_mb', '?') }} MB</span></div>
        <div class="meta-item"><span class="meta-label">Grain: </span><span class="meta-value">{{ meta.get('grain', '?') }}</span></div>
        {% if stats.get('unique_dates') %}
        <div class="meta-item"><span class="meta-label">Dates: </span><span class="meta-value">{{ stats.get('unique_dates', '?') }}</span></div>
        <div class="meta-item"><span class="meta-label">Range: </span><span class="meta-value">{{ stats.get('date_min', '?') }} → {{ stats.get('date_max', '?') }}</span></div>
        {% endif %}
        <div class="meta-item"><span class="meta-label">Key: </span><span class="meta-value">{{ meta.get('key', '?') }}</span></div>
    </div>

    {% if meta.get('targets') %}
    <div class="meta-item" style="margin-bottom:8px"><span class="meta-label">Targets: </span><span class="meta-value" style="color:#ffaa00">{{ meta['targets'] }}</span></div>
    {% endif %}

    {% if stats.get('columns') %}
    <table>
    <thead><tr>
        <th style="width:25%">Column</th>
        <th style="width:10%">Type</th>
        <th style="width:10%">Nulls</th>
        <th style="width:15%">Null %</th>
        <th style="width:10%">Min</th>
        <th style="width:10%">Max</th>
        <th style="width:10%">Mean</th>
        <th style="width:10%">Std</th>
    </tr></thead>
    <tbody>
    {% for col in stats['columns'] %}
    <tr>
        <td>{{ col['name'] }}</td>
        <td style="color:#666">{{ col['dtype'][:12] }}</td>
        <td>{{ '{:,}'.format(col['null_count']) }}</td>
        <td>
            {% if col['null_pct'] == 0 %}
            <span class="null-bar null-ok" style="width:40px"></span> {{ col['null_pct'] }}%
            {% elif col['null_pct'] < 20 %}
            <span class="null-bar null-ok" style="width:{{ [col['null_pct']*0.4, 40]|min }}px"></span> {{ col['null_pct'] }}%
            {% elif col['null_pct'] < 50 %}
            <span class="null-bar null-warn" style="width:{{ col['null_pct']*0.4 }}px"></span> {{ col['null_pct'] }}%
            {% else %}
            <span class="null-bar null-bad" style="width:{{ col['null_pct']*0.4 }}px"></span> {{ col['null_pct'] }}%
            {% endif %}
        </td>
        <td>{{ col.get('min', '') }}</td>
        <td>{{ col.get('max', '') }}</td>
        <td>{{ col.get('mean', '') }}</td>
        <td>{{ col.get('std', '') }}</td>
    </tr>
    {% endfor %}
    </tbody>
    </table>
    {% endif %}
</div>
{% endfor %}
</div>
{% endfor %}

<div id="issues">
<h2>Data Issues & Recommendations</h2>
{% for issue in issues %}
<div class="table-card">
    <div style="color: {{ '#ff3333' if issue['severity'] == 'HIGH' else '#ffaa00' if issue['severity'] == 'MEDIUM' else '#888' }}">
        [{{ issue['severity'] }}] {{ issue['title'] }}
    </div>
    <div class="table-desc">{{ issue['description'] }}</div>
</div>
{% endfor %}
</div>

<script>
document.querySelectorAll('.toggle').forEach(el => {
    el.addEventListener('click', () => {
        const target = document.getElementById(el.dataset.target);
        target.classList.toggle('show');
        el.textContent = target.classList.contains('show') ? '▼ Hide columns' : '▶ Show columns';
    });
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    # Analyze all parquet files
    file_stats = {}
    total_rows = 0
    total_size = 0

    all_sections = [
        ("raw", SCHEMA["raw_data"]),
        ("derived", SCHEMA["derived_data"]),
        ("features", SCHEMA["features"]),
        ("target", SCHEMA["target"]),
        ("model", SCHEMA["model"]),
    ]

    section_titles = {
        "raw": "Raw Data (from APIs)",
        "derived": "Derived / Intermediate",
        "features": "Feature Tables",
        "target": "Target Variable",
        "model": "Model Table (Final)",
    }

    for section_name, section_files in all_sections:
        for filename in section_files:
            filepath = DATA_DIR / filename
            if filepath.exists():
                stats = analyze_parquet(filepath)
                file_stats[filename] = stats
                if "rows" in stats:
                    total_rows += stats["rows"]
                    total_size += stats.get("size_mb", 0)

    # Model table specifics
    model_stats = file_stats.get("model_table.parquet", {})
    model_features = model_stats.get("cols", 0) - 21  # subtract target/meta columns
    model_rows = model_stats.get("rows", 0)
    trading_days = model_stats.get("unique_dates", 0)

    # Check term structure status
    import subprocess
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    term_running = "fetch_term_structure" in result.stdout
    term_status = "RUNNING (expanding to 15 strikes)" if term_running else "Complete or not running"

    # Identify issues
    issues = []
    for filename, stats in file_stats.items():
        if "columns" not in stats:
            continue
        for col in stats["columns"]:
            if col["null_pct"] == 100.0:
                issues.append({
                    "severity": "HIGH",
                    "title": f"{filename}: {col['name']} is 100% null",
                    "description": f"Dead feature — should be dropped from model table."
                })
            elif col["null_pct"] > 60:
                issues.append({
                    "severity": "MEDIUM",
                    "title": f"{filename}: {col['name']} is {col['null_pct']}% null",
                    "description": f"High null rate — consider dropping or imputing."
                })
            if col.get("infs", 0) > 0:
                issues.append({
                    "severity": "HIGH",
                    "title": f"{filename}: {col['name']} has {col['infs']} infinity values",
                    "description": f"Infinity values will break model training."
                })

    # Check term structure strike coverage
    ts_stats = file_stats.get("spxw_term_structure.parquet", {})
    if ts_stats.get("rows", 0) > 0:
        try:
            ts = pd.read_parquet(DATA_DIR / "spxw_term_structure.parquet")
            strikes_per_day = ts.groupby(ts["date"].astype(str))["strike"].nunique()
            incomplete = (strikes_per_day < 15).sum()
            if incomplete > 0:
                issues.append({
                    "severity": "MEDIUM",
                    "title": f"Term structure: {incomplete} days have < 15 strikes",
                    "description": f"Backfill in progress. {(strikes_per_day >= 15).sum()} days have full 15-strike coverage."
                })
        except:
            pass

    issues.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["severity"]])

    return render_template_string(
        TEMPLATE,
        sections=all_sections,
        section_title=section_titles,
        file_stats=file_stats,
        total_files=len(file_stats),
        total_rows=f"{total_rows:,}",
        total_size_gb=f"{total_size / 1000:.2f}",
        model_features=model_features,
        model_rows=f"{model_rows:,}",
        trading_days=trading_days,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        term_status=term_status,
        issues=issues,
    )


if __name__ == "__main__":
    print("\n  Data Explorer running at http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
