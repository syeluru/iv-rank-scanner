#!/usr/bin/env python3
"""
Tradier API setup and validation script.

Tests connectivity, account access, and options chain retrieval
to verify everything works before running the bot.

Usage:
    python scripts/setup_tradier.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_check(label: str, passed: bool, detail: str = ""):
    icon = "✅" if passed else "❌"
    line = f"  {icon} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


async def run_checks():
    print_header("Tradier API Setup & Validation")

    # 1. Check credentials
    print("\n📋 Credentials")
    token = settings.TRADIER_API_TOKEN
    acct = settings.TRADIER_ACCOUNT_ID
    sandbox = settings.TRADIER_SANDBOX

    print_check("API Token", bool(token), f"{'***' + token[-6:]}" if token else "MISSING")
    print_check("Account ID", bool(acct), acct if acct else "MISSING")
    print_check("Mode", True, "SANDBOX (paper)" if sandbox else "PRODUCTION (live)")

    if not token or not acct:
        print("\n❌ Missing credentials. Add to .env:")
        print("   TRADIER_API_TOKEN=your_token_here")
        print("   TRADIER_ACCOUNT_ID=your_account_id")
        return False

    # 2. Test API connectivity
    print("\n🔌 API Connectivity")
    from execution.broker_api.tradier_client import TradierClient
    client = TradierClient()

    try:
        # Test profile endpoint
        data = await client._get("/user/profile")
        profile = data.get("profile", {})
        account_info = profile.get("account", {})
        if isinstance(account_info, list):
            account_info = account_info[0] if account_info else {}

        name = profile.get("name", "Unknown")
        acct_type = account_info.get("type", "unknown")
        classification = account_info.get("classification", "unknown")
        print_check("API Connection", True, f"Hello {name}!")
        print_check("Account Type", True, f"{acct_type} ({classification})")
    except Exception as e:
        print_check("API Connection", False, str(e))
        return False

    # 3. Test account balances
    print("\n💰 Account Balances")
    try:
        account_data = await client.get_account(include_positions=False)
        balances = account_data["securitiesAccount"]["currentBalances"]
        equity = balances["liquidationValue"]
        option_bp = balances["availableFundsNonMarginableTrade"]
        print_check("Total Equity", True, f"${equity:,.2f}")
        print_check("Option Buying Power", True, f"${option_bp:,.2f}")
    except Exception as e:
        print_check("Account Balances", False, str(e))

    # 4. Test positions
    print("\n📊 Positions")
    try:
        positions = await client.get_account_positions()
        if positions:
            print_check("Open Positions", True, f"{len(positions)} positions")
            for p in positions[:5]:
                sym = p["instrument"]["symbol"]
                pnl = p.get("currentDayProfitLoss", 0)
                print(f"      {sym}: P&L ${pnl:,.2f}")
        else:
            print_check("Open Positions", True, "No open positions")
    except Exception as e:
        print_check("Positions", False, str(e))

    # 5. Test SPX quote
    print("\n📈 Market Data")
    try:
        quote_data = await client.get_quote("SPX")
        spx_quote = list(quote_data.values())[0]["quote"]
        last = spx_quote["lastPrice"]
        print_check("SPX Quote", True, f"Last: ${last:,.2f}")
    except Exception as e:
        print_check("SPX Quote", False, str(e))

    # 6. Test VIX quote
    try:
        vix_data = await client.get_quote("VIX")
        vix_quote = list(vix_data.values())[0]["quote"]
        vix_last = vix_quote["lastPrice"]
        print_check("VIX Quote", True, f"Last: {vix_last:.2f}")
    except Exception as e:
        print_check("VIX Quote", False, str(e))

    # 7. Test option chain (0DTE SPX)
    print("\n🔗 Options Chain (0DTE SPX)")
    today = date.today()
    try:
        chain = await client.get_option_chain(
            "$SPX",
            from_date=today,
            to_date=today,
        )
        call_map = chain.get("callExpDateMap", {})
        put_map = chain.get("putExpDateMap", {})
        underlying = chain.get("underlyingPrice", 0)

        total_calls = sum(len(v) for v in call_map.values())
        total_puts = sum(len(v) for v in put_map.values())

        if total_calls > 0 or total_puts > 0:
            print_check("Chain Retrieved", True, f"{total_calls} calls, {total_puts} puts")
            print_check("Underlying Price", True, f"${underlying:,.2f}")

            # Check greeks
            has_greeks = False
            for exp_key, strikes in call_map.items():
                for strike_str, contracts in strikes.items():
                    c = contracts[0]
                    if c.get("delta", 0) != 0:
                        has_greeks = True
                        break
                if has_greeks:
                    break

            print_check("Greeks Available", has_greeks,
                        "Delta/Gamma/Theta/Vega present" if has_greeks else "No greeks in chain")

            # Show a sample strike near ATM
            if call_map:
                exp_key = list(call_map.keys())[0]
                strikes_at_exp = call_map[exp_key]
                # Find ATM-ish strike
                sorted_strikes = sorted(strikes_at_exp.keys(), key=float)
                atm_idx = len(sorted_strikes) // 2
                if atm_idx < len(sorted_strikes):
                    sample_strike = sorted_strikes[atm_idx]
                    sample = strikes_at_exp[sample_strike][0]
                    print(f"\n      Sample ATM Call @ {sample_strike}:")
                    print(f"        Bid: ${sample['bid']:.2f}  Ask: ${sample['ask']:.2f}")
                    print(f"        Delta: {sample['delta']:.4f}  IV: {sample.get('volatility', 0):.4f}")
                    print(f"        Volume: {sample['totalVolume']}  OI: {sample['openInterest']}")
        else:
            day_name = today.strftime("%A")
            if today.weekday() >= 5:
                print_check("Chain Retrieved", True,
                            f"No 0DTE options today ({day_name}) — market closed")
            else:
                print_check("Chain Retrieved", False, "No options found for today")

    except Exception as e:
        print_check("Options Chain", False, str(e))

    # 8. Test order retrieval (read-only)
    print("\n📝 Order History")
    try:
        orders = await client.get_orders_for_account()
        if orders:
            print_check("Orders Retrieved", True, f"{len(orders)} orders")
            # Show last order
            last_order = orders[0]
            print(f"      Last: #{last_order.get('orderId', '?')} — {last_order.get('status', '?')}")
        else:
            print_check("Orders Retrieved", True, "No recent orders")
    except Exception as e:
        print_check("Order History", False, str(e))

    # Summary
    print_header("Setup Complete")
    print(f"\n  Broker: Tradier ({'Sandbox' if sandbox else 'Production'})")
    print(f"  Account: {acct}")
    print(f"  Rate Limit: {settings.TRADIER_RATE_LIMIT_CALLS} req/min")
    print(f"\n  Your .env is configured with BROKER=tradier")
    print(f"  The bot will use Tradier when you run it next.\n")

    await client.close()
    return True


def main():
    success = asyncio.run(run_checks())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
