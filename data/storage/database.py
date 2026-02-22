"""
DuckDB database connection manager.

This module provides a centralized interface for all database operations,
including connection management, query execution, and data persistence.
"""

import duckdb
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
import pandas as pd
from loguru import logger

from config.settings import settings


class DatabaseManager:
    """
    Manages DuckDB database connections and operations.

    This class provides connection pooling, transaction management,
    and convenience methods for common database operations.
    """

    def __init__(self, db_path: Path):
        """
        Initialize database manager.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"DatabaseManager initialized for {db_path}")

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create database connection.

        Returns:
            DuckDB connection object
        """
        if self._connection is None:
            self._connection = duckdb.connect(str(self.db_path))
            logger.info(f"Connected to database: {self.db_path}")
        return self._connection

    def close(self):
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info(f"Closed connection to {self.db_path}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Usage:
            with db.get_connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchdf()
        """
        conn = self.connect()
        try:
            yield conn
        finally:
            # Don't close connection here - reuse it
            pass

    def execute(self, query: str, parameters: Optional[tuple] = None) -> Any:
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Query result (call .fetchdf() for DataFrame, .fetchall() for rows)
        """
        conn = self.connect()
        try:
            if parameters:
                return conn.execute(query, parameters)
            return conn.execute(query)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    def execute_many(self, query: str, parameters: list[tuple]) -> None:
        """
        Execute a SQL query with multiple parameter sets.

        Args:
            query: SQL query string
            parameters: List of parameter tuples
        """
        conn = self.connect()
        try:
            conn.executemany(query, parameters)
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    def fetch_df(self, query: str, parameters: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Query results as pandas DataFrame
        """
        result = self.execute(query, parameters)
        return result.fetchdf()

    def insert_df(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append"
    ) -> None:
        """
        Insert DataFrame into table.

        Args:
            df: DataFrame to insert
            table: Target table name
            if_exists: How to behave if table exists ('append' or 'replace')
        """
        if df.empty:
            logger.warning(f"Attempted to insert empty DataFrame to {table}")
            return

        conn = self.connect()
        try:
            # DuckDB can directly query DataFrames
            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
            else:  # append
                conn.execute(f"INSERT INTO {table} SELECT * FROM df")

            logger.info(f"Inserted {len(df)} rows into {table}")
        except Exception as e:
            logger.error(f"Failed to insert DataFrame to {table}: {e}")
            raise

    def table_exists(self, table: str) -> bool:
        """
        Check if table exists in database.

        Args:
            table: Table name

        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_name = ?
        """
        result = self.execute(query, (table,)).fetchone()
        return result[0] > 0

    def get_table_row_count(self, table: str) -> int:
        """
        Get number of rows in table.

        Args:
            table: Table name

        Returns:
            Number of rows
        """
        if not self.table_exists(table):
            return 0
        result = self.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0]

    def get_table_info(self, table: str) -> pd.DataFrame:
        """
        Get table schema information.

        Args:
            table: Table name

        Returns:
            DataFrame with column information
        """
        return self.fetch_df(f"DESCRIBE {table}")

    def vacuum(self):
        """Reclaim space from deleted rows."""
        self.execute("VACUUM")
        logger.info("Database vacuumed")

    def analyze(self):
        """Update query planner statistics."""
        self.execute("ANALYZE")
        logger.info("Database analyzed")


class MarketDataDB(DatabaseManager):
    """Database manager for market data (quotes, options, IV history)."""

    def __init__(self):
        super().__init__(settings.market_data_db_path)


class TradesDB(DatabaseManager):
    """Database manager for trades and orders."""

    def __init__(self):
        super().__init__(settings.trades_db_path)


class PositionsDB(DatabaseManager):
    """Database manager for positions."""

    def __init__(self):
        super().__init__(settings.positions_db_path)


# Global database instances
market_data_db = MarketDataDB()
trades_db = TradesDB()
positions_db = PositionsDB()
