#!/usr/bin/env python3
"""
Create database schema for minute-level P&L tracking.

Creates three tables:
1. minute_leg_prices - Raw price data at 1-minute intervals
2. minute_leg_pnl - Computed P&L for each leg
3. minute_match_pnl - Aggregated P&L for trade matches
"""

import duckdb
from pathlib import Path


def create_minute_pnl_schema():
    """Create tables and indexes for minute-level P&L tracking."""
    db_path = Path(__file__).parent.parent / "data_store" / "trades.duckdb"

    with duckdb.connect(str(db_path)) as conn:
        print("Creating minute-level P&L tables...")

        # 1. Create minute_leg_prices table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS minute_leg_prices (
                price_id VARCHAR PRIMARY KEY,
                leg_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                bid DECIMAL(10, 2),
                ask DECIMAL(10, 2),
                mid DECIMAL(10, 2),
                underlying_price DECIMAL(10, 2),
                delta DECIMAL(8, 6),
                gamma DECIMAL(8, 6),
                theta DECIMAL(8, 6),
                vega DECIMAL(8, 6),
                implied_vol DECIMAL(8, 6),
                FOREIGN KEY (leg_id) REFERENCES trade_legs(leg_id)
            )
        """)

        # Create indexes for minute_leg_prices
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_prices_leg_id ON minute_leg_prices(leg_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_prices_timestamp ON minute_leg_prices(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_prices_leg_timestamp ON minute_leg_prices(leg_id, timestamp)")

        print("✓ Created minute_leg_prices table with indexes")

        # 2. Create minute_leg_pnl table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS minute_leg_pnl (
                pnl_id VARCHAR PRIMARY KEY,
                leg_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                unrealized_pnl DECIMAL(10, 2),
                realized_pnl DECIMAL(10, 2),
                is_realized BOOLEAN NOT NULL DEFAULT FALSE,
                entry_price DECIMAL(10, 2),
                current_price DECIMAL(10, 2),
                quantity INTEGER,
                side VARCHAR(10),
                FOREIGN KEY (leg_id) REFERENCES trade_legs(leg_id)
            )
        """)

        # Create indexes for minute_leg_pnl
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_pnl_leg_id ON minute_leg_pnl(leg_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_pnl_timestamp ON minute_leg_pnl(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_pnl_leg_timestamp ON minute_leg_pnl(leg_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leg_pnl_is_realized ON minute_leg_pnl(is_realized)")

        print("✓ Created minute_leg_pnl table with indexes")

        # 3. Create minute_match_pnl table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS minute_match_pnl (
                snapshot_id VARCHAR PRIMARY KEY,
                match_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                total_unrealized_pnl DECIMAL(10, 2),
                total_realized_pnl DECIMAL(10, 2),
                total_pnl DECIMAL(10, 2),
                num_open_legs INTEGER,
                num_closed_legs INTEGER,
                events VARCHAR,
                FOREIGN KEY (match_id) REFERENCES trade_matches(match_id)
            )
        """)

        # Create indexes for minute_match_pnl
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_pnl_match_id ON minute_match_pnl(match_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_pnl_timestamp ON minute_match_pnl(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_pnl_match_timestamp ON minute_match_pnl(match_id, timestamp)")

        print("✓ Created minute_match_pnl table with indexes")

        # 4. Create backfill progress tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backfill_progress (
                leg_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                error_message VARCHAR,
                FOREIGN KEY (leg_id) REFERENCES trade_legs(leg_id)
            )
        """)

        print("✓ Created backfill_progress table")

        # Verify table counts
        tables = ['minute_leg_prices', 'minute_leg_pnl', 'minute_match_pnl', 'backfill_progress']
        for table in tables:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"  {table}: {result[0]} rows")

        print("\n✅ Schema creation complete!")


if __name__ == "__main__":
    create_minute_pnl_schema()
