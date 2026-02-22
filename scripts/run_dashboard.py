#!/usr/bin/env python3
"""
Launch the Command Center Dashboard.

Usage:
    python scripts/run_dashboard.py

Then open: http://127.0.0.1:8050
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.dashboards.command_center import app, logger

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("COMMAND CENTER DASHBOARD")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Dashboard URL: http://127.0.0.1:8050")
    logger.info("")
    logger.info("Features:")
    logger.info("  - Real-time portfolio P&L tracking")
    logger.info("  - Strategy performance monitoring")
    logger.info("  - Manual trade entry")
    logger.info("  - Position and trade viewing")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 70)

    try:
        app.run(debug=True, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        logger.info("\nShutting down dashboard...")
