#!/usr/bin/env python3
"""
Test ThetaData Connection

Verifies that Theta Terminal is running and accessible.
"""

import sys
from pathlib import Path
from datetime import date, time

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import requests


def test_connection():
    """Test connection to Theta Terminal."""
    
    logger.info("=" * 80)
    logger.info("TESTING THETADATA CONNECTION")
    logger.info("=" * 80)
    
    host = "http://127.0.0.1:25503"
    
    logger.info(f"\nAttempting to connect to Theta Terminal at {host}...")
    
    try:
        # Test with simple expirations endpoint
        response = requests.get(
            f"{host}/v3/option/list/expirations",
            params={"symbol": "SPXW"},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Successfully connected to Theta Terminal v3!")
            
            # Try to fetch a sample price
            logger.info("\nTesting data fetch (SPX price from yesterday)...")
            
            test_date = date.today()
            test_time = time(10, 0)
            
            response2 = requests.get(
                f"{host}/v3/index/history/price",
                params={
                    "symbol": "SPX",
                    "date": test_date.strftime("%Y%m%d"),
                    "interval": "1m",
                    "start_time": "10:00:00",
                    "end_time": "10:01:00",
                    "format": "json"
                },
                timeout=10
            )
            
            if response2.status_code == 200:
                data = response2.json()
                logger.info(f"✅ Successfully fetched data!")
                logger.info(f"   Response sample: {str(data)[:200]}")
            else:
                logger.warning(f"⚠️  Data fetch returned: {response2.status_code}")
                
        else:
            logger.error(f"❌ Connection failed: {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
            print_setup_instructions()
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to Theta Terminal")
        print_setup_instructions()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def print_setup_instructions():
    """Print setup instructions."""
    logger.info("\n" + "=" * 80)
    logger.info("THETA TERMINAL SETUP")
    logger.info("=" * 80)
    logger.info("\nTheta Terminal is not running. To start it:")
    logger.info("\n1. Make sure you have ThetaTerminalv3.jar downloaded")
    logger.info("2. Create creds.txt in the same directory:")
    logger.info("   Line 1: syeluru96@gmail.com")
    logger.info("   Line 2: thetadataHungary23!")
    logger.info("\n3. Run: java -jar ThetaTerminalv3.jar")
    logger.info("\n4. Keep it running in a separate terminal")
    logger.info("\nSee docs/THETADATA_SETUP.md for detailed instructions")
    logger.info("=" * 80)


if __name__ == '__main__':
    test_connection()
