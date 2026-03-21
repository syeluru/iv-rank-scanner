"""
Economic calendar: CPI, PPI, NFP (Non-Farm Payrolls), GDP release dates.
All dates are the actual release dates (8:30 AM ET typically).
Hardcoded from BLS and BEA official schedules.
Saves to data/econ_calendar.parquet
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── CPI Release Dates (from BLS) ──
CPI_DATES = [
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-18",
    # 2026
    "2026-01-14", "2026-02-13", "2026-03-11",
]

# ── PPI Release Dates (from BLS) ──
PPI_DATES = [
    # 2023
    "2023-01-18", "2023-02-16", "2023-03-15", "2023-04-13",
    "2023-05-11", "2023-06-14", "2023-07-13", "2023-08-11",
    "2023-09-14", "2023-10-11", "2023-11-15", "2023-12-13",
    # 2024
    "2024-01-12", "2024-02-16", "2024-03-14", "2024-04-11",
    "2024-05-14", "2024-06-13", "2024-07-12", "2024-08-13",
    "2024-09-12", "2024-10-11", "2024-11-14", "2024-12-12",
    # 2025
    "2025-01-14", "2025-02-13", "2025-03-13", "2025-04-11",
    "2025-05-15", "2025-06-12", "2025-07-15", "2025-08-14",
    "2025-09-11", "2025-10-09", "2025-11-13", "2025-12-11",
    # 2026
    "2026-01-14", "2026-02-27",
]

# ── Non-Farm Payrolls / Employment Situation (from BLS, first Friday of month) ──
NFP_DATES = [
    # 2023
    "2023-01-06", "2023-02-03", "2023-03-10", "2023-04-07",
    "2023-05-05", "2023-06-02", "2023-07-07", "2023-08-04",
    "2023-09-01", "2023-10-06", "2023-11-03", "2023-12-08",
    # 2024
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06",
]

# ── GDP Advance/Preliminary/Final Estimates (from BEA, quarterly) ──
GDP_DATES = [
    # 2023 (advance estimates for prior quarter)
    "2023-01-26", "2023-04-27", "2023-07-27", "2023-10-26",
    # 2024
    "2024-01-25", "2024-04-25", "2024-07-25", "2024-10-30",
    # 2025
    "2025-01-30", "2025-04-30", "2025-07-30", "2025-10-29",
    # 2026
    "2026-01-29",
]


def main():
    all_events = []

    for event_type, dates in [("CPI", CPI_DATES), ("PPI", PPI_DATES),
                               ("NFP", NFP_DATES), ("GDP", GDP_DATES)]:
        for d in dates:
            all_events.append({"date": d, "event": event_type})

    df = pd.DataFrame(all_events)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Economic calendar: {len(df)} events")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nBreakdown:")
    print(df.groupby("event").size().to_string())
    print(f"\nSample:")
    print(df.head(10).to_string())

    outfile = DATA_DIR / "econ_calendar.parquet"
    df.to_parquet(outfile, index=False)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
