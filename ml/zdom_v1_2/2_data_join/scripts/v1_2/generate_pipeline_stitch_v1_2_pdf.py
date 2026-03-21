from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_PATH = BASE_DIR / "v1_2_data_pipeline_stitch.pdf"


def add_box(ax, x, y, w, h, title, lines, facecolor, fontsize=8.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.3,
        edgecolor="#223038",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.012 * w,
        y + h - 0.018,
        title,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#16242b",
    )
    ax.text(
        x + 0.012 * w,
        y + h - 0.055,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1f2d34",
        linespacing=1.2,
    )


def add_arrow(ax, x1, y1, x2, y2, label=""):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
        color="#4a5c65",
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.008,
            label,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#43555e",
        )


def build_pdf():
    fig = plt.figure(figsize=(18, 11))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")

    ax.text(
        0.04,
        0.97,
        "Options Model v1_2 Pipeline Stitch",
        fontsize=22,
        fontweight="bold",
        color="#16242b",
        ha="left",
        va="top",
    )
    ax.text(
        0.04,
        0.943,
        "Source-by-source map from raw/prepared tables to master_data_v1_2_final, including join timing and filtering steps.",
        fontsize=10.5,
        color="#475961",
        ha="left",
        va="top",
    )

    # Row 1: source boxes
    src_y = 0.74
    w = 0.11
    h = 0.13
    xs = [0.03, 0.155, 0.28, 0.405, 0.53, 0.655, 0.78, 0.895]
    titles = [
        "spx_1min",
        "spxw_0dte_intraday_greeks",
        "vix_1min",
        "vix1d_1min",
        "OI daily",
        "Term structure",
        "Daily context",
        "Calendars / regime",
    ]
    lines = [
        ["raw/v1_2", "SPX minute OHLC", "used for prior-minute", "decision state"],
        ["raw/v1_2", "option surface by strike", "deltas / IV / mids", "minute-level selection"],
        ["raw/v1_2", "VIX minute bars", "intraday vol context"],
        ["raw/v1_2", "VIX1D minute bars", "near-horizon vol context"],
        ["data/v1_2", "spxw_0dte_oi_daily_v1_2", "same-day usable OI"],
        ["data/v1_2", "spxw_term_structure_daily_v1_2", "T-1 join"],
        ["data/v1_2", "FRED / cross-asset / breadth /", "vol_context / spx_daily / vix_daily"],
        ["data/v1_2", "econ_calendar_v1_2", "mag7_earnings_v1_2", "regime / presidential"],
    ]
    colors = ["#d7ebff", "#d7ebff", "#d7ebff", "#d7ebff", "#fff1b8", "#fff1b8", "#fff1b8", "#e7def7"]
    for x, t, ls, c in zip(xs, titles, lines, colors):
        add_box(ax, x, src_y, w, h, t, ls, c, fontsize=8.1)

    # Row 2: intraday decision state
    add_box(
        ax,
        0.22,
        0.53,
        0.24,
        0.12,
        "intraday_decision_state_v1_2",
        [
            "one row per decision_datetime x strike",
            "uses completed prior-minute state",
            "10:00 first decision keeps 09:30-09:59 history",
        ],
        "#dff3df",
    )

    # Row 2: minute aggregation
    add_box(
        ax,
        0.54,
        0.50,
        0.22,
        0.17,
        "Aggregation -> master_data_v1_2 spine",
        [
            "build_master_data_v1_2.py",
            "select condor legs from T-1 greeks snapshot",
            "exact-condor dedup only if all 4 legs match",
            "same decision_datetime partition only",
            "tie-break: keep higher target delta strategy",
            "cap dates at <= 2026-03-13",
        ],
        "#ffe2ca",
    )

    # Row 2: outcomes
    add_box(
        ax,
        0.80,
        0.52,
        0.17,
        0.13,
        "Outcome append",
        [
            "build_master_data_v1_2_outcomes.py",
            "forward walk over decision-state",
            "append tp10_* ... tp50_*",
        ],
        "#f4dff0",
    )

    # Row 3: daily join
    add_box(
        ax,
        0.38,
        0.28,
        0.33,
        0.16,
        "Daily join -> master_data_v1_2",
        [
            "build_master_data_v1_2_daily_joins.py",
            "join key: date",
            "OI same-day | term structure T-1 | FRED/regime T-2",
            "cross-asset / breadth / vol context T-1",
            "calendar / earnings / presidential same-day",
        ],
        "#fff0b3",
    )

    # Row 4: feature engineering
    add_box(
        ax,
        0.16,
        0.07,
        0.32,
        0.15,
        "Feature engineering",
        [
            "build_master_data_v1_2_final.py",
            "strategy/time | daily_prior | daily lookbacks",
            "intraday lookbacks | cross-horizon | event features",
            "allow early VIX1D-family nulls",
        ],
        "#dff4f0",
    )

    # Row 4: final
    add_box(
        ax,
        0.56,
        0.07,
        0.36,
        0.15,
        "master_data_v1_2_final",
        [
            "3_feature_engineering/v1_2/outputs/master_data_v1_2_final.parquet",
            "1,701,426 rows x 477 columns",
            "no duplicate decision_datetime + strategy keys",
            "data_dictionary should match 1:1 via Final Name",
        ],
        "#eadfff",
    )

    # Arrows from sources
    for x in xs[:4]:
        add_arrow(ax, x + w / 2, src_y, 0.34, 0.65)
    for x in xs[4:]:
        add_arrow(ax, x + w / 2, src_y, 0.545, 0.44)

    # Flow arrows
    add_arrow(ax, 0.46, 0.59, 0.54, 0.59, "minute-level aggregation")
    add_arrow(ax, 0.76, 0.585, 0.80, 0.585, "append tp*")
    add_arrow(ax, 0.64, 0.50, 0.58, 0.44)
    add_arrow(ax, 0.545, 0.28, 0.32, 0.22, "engineer features")
    add_arrow(ax, 0.48, 0.145, 0.56, 0.145)

    ax.text(
        0.03,
        0.235,
        "Filtering / timing rules\n"
        "1. Decision rows use only information available by decision_datetime.\n"
        "2. Daily joins respect source-specific lag semantics before decision time.\n"
        "3. Dedup only when exact same 4-leg condor appears at same decision_datetime.\n"
        "4. Feature table keeps 2023-03-13 start; early VIX1D-family nulls are documented.",
        fontsize=9.2,
        color="#33434b",
        ha="left",
        va="top",
    )

    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_pdf()
