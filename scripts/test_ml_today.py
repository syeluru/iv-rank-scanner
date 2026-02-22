#!/usr/bin/env python3
"""Pre-market analysis: run the full entry decision flow without placing any orders."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta
import pytz
from loguru import logger

from config.settings import settings
from execution.broker_api.schwab_client import schwab_client
from execution.order_manager.ic_order_builder import IronCondorOrderBuilder
from ml.models.predictor import MLPredictor
from ml.features.pipeline import feature_pipeline
from ml.data.candle_loader import CandleLoader

ET = pytz.timezone('US/Eastern')


async def analyze():
    now = datetime.now(ET)
    print(f"\n{'='*70}")
    print(f"  0DTE BOT PRE-MARKET ANALYSIS — {now.strftime('%A %B %d, %Y %I:%M %p ET')}")
    print(f"{'='*70}\n")

    # === 1. Pre-flight ===
    print("--- PRE-FLIGHT CHECKS ---")
    print(f"  Day of week: {now.strftime('%A')} ({'OK' if now.weekday() < 5 else 'FAIL - weekend'})")

    account = await schwab_client.get_account()
    bp = account.get('securitiesAccount', {}).get('currentBalances', {}).get('buyingPower', 0)
    print(f"  Buying power: ${bp:,.2f} ({'OK' if bp >= 5000 else 'FAIL'})")

    positions = await schwab_client.get_account_positions()
    spx_pos = [p for p in positions if 'SPX' in p.get('instrument', {}).get('symbol', '').upper()]
    print(f"  Existing SPX positions: {len(spx_pos)} ({'OK' if len(spx_pos) == 0 else 'FAIL'})")

    predictor = MLPredictor()
    print(f"  ML model: {'loaded' if predictor.is_ready() else 'MISSING'} ({predictor.num_features} features)")
    print()

    # === 2. Fetch chain ===
    print("--- OPTION CHAIN ---")
    today = date.today()
    chain = await schwab_client.get_option_chain(
        symbol='$SPX', contract_type='ALL',
        from_date=today, to_date=today + timedelta(days=5),
        include_underlying_quote=True,
    )
    underlying = chain.get('underlyingPrice', 0)
    print(f"  Underlying: $SPX @ ${underlying:,.2f}")

    call_map = chain.get('callExpDateMap', {})
    put_map = chain.get('putExpDateMap', {})
    expirations = list(call_map.keys())
    print(f"  Expirations available: {expirations}")
    print()

    # === 3. Strike selection ===
    print("--- STRIKE SELECTION (10-delta, $25 wings) ---")
    builder = IronCondorOrderBuilder()
    strikes = builder.select_strikes(chain, target_delta=0.10, wing_width=25.0)

    if strikes:
        lp = strikes['long_put']
        sp = strikes['short_put']
        sc = strikes['short_call']
        lc = strikes['long_call']
        print(f"  Expiration: {strikes['expiration']}")
        print(f"  Long  put: {lp['strike']:>8.0f}  delta={lp['delta']:+.4f}  bid={lp['bid']:.2f}  ask={lp['ask']:.2f}  mid={lp['mid']:.2f}")
        print(f"  Short put: {sp['strike']:>8.0f}  delta={sp['delta']:+.4f}  bid={sp['bid']:.2f}  ask={sp['ask']:.2f}  mid={sp['mid']:.2f}")
        print(f"  Short call:{sc['strike']:>8.0f}  delta={sc['delta']:+.4f}  bid={sc['bid']:.2f}  ask={sc['ask']:.2f}  mid={sc['mid']:.2f}")
        print(f"  Long  call:{lc['strike']:>8.0f}  delta={lc['delta']:+.4f}  bid={lc['bid']:.2f}  ask={lc['ask']:.2f}  mid={lc['mid']:.2f}")
        print(f"  Net credit (mark): ${strikes['net_credit']:.2f}")
        print(f"  Min credit check: ${strikes['net_credit']:.2f} {'>=  ' if strikes['net_credit'] >= settings.BOT_MIN_CREDIT else '<'} ${settings.BOT_MIN_CREDIT:.2f} {'PASS' if strikes['net_credit'] >= settings.BOT_MIN_CREDIT else 'FAIL'}")
    else:
        print("  Strike selection FAILED (deltas may be unavailable)")
        print("  Checking if deltas are available...")
        exp_key = builder._find_0dte_expiration(call_map)
        if exp_key:
            sample_puts = put_map.get(exp_key, {})
            for strike_str in sorted(sample_puts.keys(), key=float)[-5:]:
                c = sample_puts[strike_str][0]
                print(f"    Put {strike_str}: delta={c.get('delta')}")
    print()

    # === 4. Position sizing ===
    if strikes and strikes['net_credit'] >= settings.BOT_MIN_CREDIT:
        print("--- POSITION SIZING ---")
        qty = builder.calculate_position_size(
            credit=strikes['net_credit'],
            portfolio_value=settings.BOT_PORTFOLIO_VALUE,
            daily_target_pct=settings.BOT_DAILY_TARGET_PCT,
            tp_pct=settings.BOT_TAKE_PROFIT_PCT,
            cost_per_leg=settings.BOT_COST_PER_LEG,
            wing_width=settings.BOT_WING_WIDTH,
            buying_power=bp,
            max_contracts=settings.BOT_MAX_CONTRACTS,
        )
        tp_debit = builder.calculate_tp_debit(strikes['net_credit'], settings.BOT_TAKE_PROFIT_PCT, settings.BOT_COST_PER_LEG)
        # SL: portfolio-based
        max_loss = settings.BOT_PORTFOLIO_VALUE * settings.BOT_STOP_LOSS_PCT
        closing_fees = qty * 4 * settings.BOT_COST_PER_LEG
        sl_thresh = round(
            strikes['net_credit'] + (max_loss - closing_fees) / (qty * 100), 2
        ) if qty > 0 else 0

        profit_per = (strikes['net_credit'] * settings.BOT_TAKE_PROFIT_PCT * 100) - (settings.BOT_COST_PER_LEG * 8)
        margin_per = (settings.BOT_WING_WIDTH - strikes['net_credit']) * 100

        print(f"  Target daily profit: ${settings.BOT_PORTFOLIO_VALUE * settings.BOT_DAILY_TARGET_PCT:,.0f}")
        print(f"  Profit per contract at TP: ${profit_per:.2f}")
        print(f"  Margin per contract: ${margin_per:,.2f}")
        print(f"  Contracts: {qty}")
        print(f"  Total credit: ${strikes['net_credit'] * qty * 100:,.2f}")
        print(f"  Total margin: ${margin_per * qty:,.2f}")
        print(f"  TP debit: ${tp_debit:.2f}")
        print(f"  SL debit (portfolio-based, {settings.BOT_STOP_LOSS_PCT:.0%} max loss): ${sl_thresh:.2f}")
        print(f"  Max loss: ${max_loss:,.0f}")
        print()

    # === 5. ML Prediction ===
    print("--- ML PREDICTION ---")
    try:
        candle_loader = CandleLoader()
        features = await feature_pipeline.extract_features_realtime(
            underlying_symbol='SPX',
            timestamp=now,
            candle_loader=candle_loader,
            option_chain=chain,
            underlying_price=underlying,
        )

        print(f"  Features extracted: {len(features)}")
        print()

        # Get prediction with explanation
        result = predictor.predict_with_explanation(features)
        confidence = result['probability']
        threshold = settings.BOT_MIN_CONFIDENCE
        is_v3 = result.get('model_version') == 'v3'

        print(f"  ┌─────────────────────────────────────────┐")
        if is_v3:
            risk = result.get('risk_probability', 0)
            print(f"  |  RISK SCORE:   {risk:.1%}  (P of big loss)    |")
            print(f"  |  CONFIDENCE:   {confidence:.1%}  (1 - risk)        |")
        else:
            print(f"  |  ML CONFIDENCE: {confidence:.1%}                    |")
        print(f"  |  Threshold:     {threshold:.1%}                    |")
        verdict = "TRADE" if confidence >= threshold else "NO TRADE"
        color_mark = ">>>" if confidence >= threshold else "XXX"
        print(f"  |  Decision:      {color_mark} {verdict:<22}|")
        print(f"  └─────────────────────────────────────────┘")

        # Show rule filter status for v3
        if is_v3:
            rules = result.get('rules_triggered', [])
            if rules:
                print(f"\n  RULE FILTERS TRIGGERED:")
                for rule in rules:
                    print(f"    - {rule}")
            else:
                print(f"  Rule filters: all clear")
        print()

        # Show top features
        print("  Top features driving the prediction:")
        print(f"  {'Feature':<35} {'Value':>10} {'Importance':>12}")
        print(f"  {'-'*35} {'-'*10} {'-'*12}")
        for name, value, importance in result['top_features']:
            imp_str = f"{importance:.4f}" if isinstance(importance, float) and importance < 1 else f"{importance:.0f}"
            print(f"  {name:<35} {value:>10.4f} {imp_str:>12}")
        print()

        # Show v3 computed features or all pipeline features
        if is_v3 and 'v3_features' in result:
            print("  Risk model feature values:")
            for name, val in sorted(result['v3_features'].items()):
                print(f"    {name:<30} = {val:>12.4f}")
        else:
            print("  All feature values:")
            for name in sorted(features.keys()):
                val = features[name]
                marker = " *" if name in [f[0] for f in result['top_features'][:5]] else ""
                print(f"    {name:<40} = {val:>12.4f}{marker}")

    except Exception as e:
        print(f"  ML prediction FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Analysis complete. No orders placed.")
    print(f"{'='*70}\n")


asyncio.run(analyze())
