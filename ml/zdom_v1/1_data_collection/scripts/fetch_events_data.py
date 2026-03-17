"""
Store all major economic event calendar data:
1. FOMC meeting dates (from federalreserve.gov)
2. MAG7 earnings dates (from SEC filings / investor relations)
3. CPI release dates (from BLS schedule)
4. Non-Farm Payrolls / Jobs report dates (from BLS schedule)
5. PCE release dates (from BEA schedule)

Saves to data/fomc_dates.parquet, data/mag7_earnings.parquet, data/econ_events.parquet
"""

import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── FOMC Meeting Dates (announcement day = second day of meeting) ──
FOMC_DATES = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# ── MAG7 Earnings Dates (report dates, from SEC filings / investor relations) ──
# Format: (symbol, date, time_of_day)
# AMC = After Market Close, BMO = Before Market Open
MAG7_EARNINGS = [
    # ── AAPL ──
    ("AAPL", "2023-02-02", "AMC"), ("AAPL", "2023-05-04", "AMC"),
    ("AAPL", "2023-08-03", "AMC"), ("AAPL", "2023-11-02", "AMC"),
    ("AAPL", "2024-02-01", "AMC"), ("AAPL", "2024-05-02", "AMC"),
    ("AAPL", "2024-08-01", "AMC"), ("AAPL", "2024-10-31", "AMC"),
    ("AAPL", "2025-01-30", "AMC"), ("AAPL", "2025-05-01", "AMC"),
    ("AAPL", "2025-07-31", "AMC"), ("AAPL", "2025-10-30", "AMC"),
    ("AAPL", "2026-01-29", "AMC"),
    # ── MSFT ──
    ("MSFT", "2023-01-24", "AMC"), ("MSFT", "2023-04-25", "AMC"),
    ("MSFT", "2023-07-25", "AMC"), ("MSFT", "2023-10-24", "AMC"),
    ("MSFT", "2024-01-30", "AMC"), ("MSFT", "2024-04-25", "AMC"),
    ("MSFT", "2024-07-30", "AMC"), ("MSFT", "2024-10-30", "AMC"),
    ("MSFT", "2025-01-29", "AMC"), ("MSFT", "2025-04-30", "AMC"),
    ("MSFT", "2025-07-29", "AMC"), ("MSFT", "2025-10-29", "AMC"),
    ("MSFT", "2026-01-27", "AMC"),
    # ── GOOGL ──
    ("GOOGL", "2023-02-02", "AMC"), ("GOOGL", "2023-04-25", "AMC"),
    ("GOOGL", "2023-07-25", "AMC"), ("GOOGL", "2023-10-24", "AMC"),
    ("GOOGL", "2024-01-30", "AMC"), ("GOOGL", "2024-04-25", "AMC"),
    ("GOOGL", "2024-07-23", "AMC"), ("GOOGL", "2024-10-29", "AMC"),
    ("GOOGL", "2025-02-04", "AMC"), ("GOOGL", "2025-04-24", "AMC"),
    ("GOOGL", "2025-07-29", "AMC"), ("GOOGL", "2025-10-28", "AMC"),
    ("GOOGL", "2026-02-03", "AMC"),
    # ── AMZN ──
    ("AMZN", "2023-02-02", "AMC"), ("AMZN", "2023-04-27", "AMC"),
    ("AMZN", "2023-08-03", "AMC"), ("AMZN", "2023-10-26", "AMC"),
    ("AMZN", "2024-02-01", "AMC"), ("AMZN", "2024-04-30", "AMC"),
    ("AMZN", "2024-08-01", "AMC"), ("AMZN", "2024-10-31", "AMC"),
    ("AMZN", "2025-02-06", "AMC"), ("AMZN", "2025-05-01", "AMC"),
    ("AMZN", "2025-07-31", "AMC"), ("AMZN", "2025-10-30", "AMC"),
    ("AMZN", "2026-02-05", "AMC"),
    # ── NVDA ──
    ("NVDA", "2023-02-22", "AMC"), ("NVDA", "2023-05-24", "AMC"),
    ("NVDA", "2023-08-23", "AMC"), ("NVDA", "2023-11-21", "AMC"),
    ("NVDA", "2024-02-21", "AMC"), ("NVDA", "2024-05-22", "AMC"),
    ("NVDA", "2024-08-28", "AMC"), ("NVDA", "2024-11-20", "AMC"),
    ("NVDA", "2025-02-26", "AMC"), ("NVDA", "2025-05-28", "AMC"),
    ("NVDA", "2025-08-27", "AMC"), ("NVDA", "2025-11-19", "AMC"),
    ("NVDA", "2026-02-25", "AMC"),
    # ── META ──
    ("META", "2023-02-01", "AMC"), ("META", "2023-04-26", "AMC"),
    ("META", "2023-07-26", "AMC"), ("META", "2023-10-25", "AMC"),
    ("META", "2024-02-01", "AMC"), ("META", "2024-04-24", "AMC"),
    ("META", "2024-07-31", "AMC"), ("META", "2024-10-30", "AMC"),
    ("META", "2025-01-29", "AMC"), ("META", "2025-04-30", "AMC"),
    ("META", "2025-07-30", "AMC"), ("META", "2025-10-29", "AMC"),
    ("META", "2026-01-28", "AMC"),
    # ── TSLA ──
    ("TSLA", "2023-01-25", "AMC"), ("TSLA", "2023-04-19", "AMC"),
    ("TSLA", "2023-07-19", "AMC"), ("TSLA", "2023-10-18", "AMC"),
    ("TSLA", "2024-01-24", "AMC"), ("TSLA", "2024-04-23", "AMC"),
    ("TSLA", "2024-07-23", "AMC"), ("TSLA", "2024-10-23", "AMC"),
    ("TSLA", "2025-01-29", "AMC"), ("TSLA", "2025-04-22", "AMC"),
    ("TSLA", "2025-07-22", "AMC"), ("TSLA", "2025-10-22", "AMC"),
    ("TSLA", "2026-01-28", "AMC"),
]


# ── CPI Release Dates (from BLS release schedule) ──
# CPI is released mid-month, typically 8:30 AM ET
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
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-15", "2026-10-13", "2026-11-12", "2026-12-09",
]

# ── Non-Farm Payrolls / Jobs Report (from BLS Employment Situation schedule) ──
# Released first Friday of each month, 8:30 AM ET
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
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# ── PCE (Personal Consumption Expenditures) Release Dates (from BEA schedule) ──
# Released last week of month, 8:30 AM ET
PCE_DATES = [
    # 2023
    "2023-01-27", "2023-02-24", "2023-03-31", "2023-04-28",
    "2023-05-26", "2023-06-30", "2023-07-28", "2023-08-31",
    "2023-09-29", "2023-10-27", "2023-11-30", "2023-12-22",
    # 2024
    "2024-01-26", "2024-02-29", "2024-03-29", "2024-04-26",
    "2024-05-31", "2024-06-28", "2024-07-26", "2024-08-30",
    "2024-09-27", "2024-10-31", "2024-11-27", "2024-12-20",
    # 2025
    "2025-01-31", "2025-02-28", "2025-03-28", "2025-04-30",
    "2025-05-30", "2025-06-27", "2025-07-31", "2025-08-29",
    "2025-09-26", "2025-10-31", "2025-11-26", "2025-12-23",
    # 2026
    "2026-01-30", "2026-02-27", "2026-03-27", "2026-04-30",
    "2026-05-29", "2026-06-26", "2026-07-31", "2026-08-28",
    "2026-09-30", "2026-10-30", "2026-11-25", "2026-12-23",
]


def main():
    # ── FOMC ──
    print("=== FOMC Meeting Dates ===")
    fomc = pd.DataFrame({"date": pd.to_datetime(FOMC_DATES)})
    fomc["date"] = fomc["date"].dt.date
    fomc["event"] = "FOMC"

    print(f"  {len(fomc)} meetings, {fomc['date'].min()} to {fomc['date'].max()}")
    fomc.to_parquet(DATA_DIR / "fomc_dates.parquet", index=False)
    print(f"  Saved to data/fomc_dates.parquet\n")

    # ── MAG7 Earnings ──
    print("=== MAG7 Earnings Dates ===")
    earnings = pd.DataFrame(MAG7_EARNINGS, columns=["symbol", "date", "time_of_day"])
    earnings["date"] = pd.to_datetime(earnings["date"]).dt.date

    print(f"  {len(earnings)} total earnings events")
    print(f"  Range: {earnings['date'].min()} to {earnings['date'].max()}")
    print(f"  Per stock:")
    print(earnings.groupby("symbol").size().to_string())

    earnings.to_parquet(DATA_DIR / "mag7_earnings.parquet", index=False)
    print(f"  Saved to data/mag7_earnings.parquet\n")

    # ── Combined Economic Events ──
    print("=== Economic Event Calendar ===")
    events = []

    for d in CPI_DATES:
        events.append({"date": d, "event": "CPI"})
    for d in NFP_DATES:
        events.append({"date": d, "event": "NFP"})
    for d in PCE_DATES:
        events.append({"date": d, "event": "PCE"})
    for d in FOMC_DATES:
        events.append({"date": d, "event": "FOMC"})

    econ = pd.DataFrame(events)
    econ["date"] = pd.to_datetime(econ["date"]).dt.date
    econ = econ.sort_values("date").reset_index(drop=True)

    print(f"  Total events: {len(econ)}")
    print(f"  By type:")
    print(econ.groupby("event").size().to_string())
    print(f"  Range: {econ['date'].min()} to {econ['date'].max()}")

    econ.to_parquet(DATA_DIR / "econ_events.parquet", index=False)
    print(f"  Saved to data/econ_events.parquet")


if __name__ == "__main__":
    main()
