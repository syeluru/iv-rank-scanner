"""
Option Greeks Calculator using Black-Scholes Model.

This module calculates option Greeks (Delta, Gamma, Theta, Vega, Rho) using
the Black-Scholes model. Used for Alpaca integration since Alpaca doesn't
provide Greeks directly.

Greeks Explained:
- Delta: Rate of change of option price with respect to underlying price
- Gamma: Rate of change of Delta with respect to underlying price
- Theta: Rate of change of option price with respect to time (time decay)
- Vega: Rate of change of option price with respect to volatility
- Rho: Rate of change of option price with respect to interest rate
"""

import math
from typing import Dict
from scipy.stats import norm


def calculate_greeks(
    option_type: str,
    stock_price: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    implied_volatility: float
) -> Dict[str, float]:
    """
    Calculate option Greeks using Black-Scholes model.

    Args:
        option_type: 'CALL' or 'PUT'
        stock_price: Current price of underlying stock
        strike: Strike price of option
        time_to_expiration: Time to expiration in years (e.g., 30 days = 30/365 = 0.082)
        risk_free_rate: Risk-free interest rate (e.g., 0.05 for 5%)
        implied_volatility: Implied volatility (e.g., 0.30 for 30%)

    Returns:
        Dictionary with Delta, Gamma, Theta, Vega, and Rho
    """
    # Handle edge cases
    if time_to_expiration <= 0:
        # Option has expired
        if option_type.upper() == 'CALL':
            intrinsic = max(0, stock_price - strike)
            delta = 1.0 if stock_price > strike else 0.0
        else:  # PUT
            intrinsic = max(0, strike - stock_price)
            delta = -1.0 if stock_price < strike else 0.0

        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    if implied_volatility <= 0:
        # No volatility - use simple intrinsic value
        if option_type.upper() == 'CALL':
            delta = 1.0 if stock_price > strike else 0.0
        else:
            delta = -1.0 if stock_price < strike else 0.0

        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    # Calculate d1 and d2
    d1 = _calculate_d1(stock_price, strike, time_to_expiration, risk_free_rate, implied_volatility)
    d2 = d1 - implied_volatility * math.sqrt(time_to_expiration)

    # Calculate Greeks
    if option_type.upper() == 'CALL':
        delta = _calculate_call_delta(d1)
        theta = _calculate_call_theta(
            stock_price, strike, time_to_expiration, risk_free_rate,
            implied_volatility, d1, d2
        )
        rho = _calculate_call_rho(strike, time_to_expiration, risk_free_rate, d2)
    else:  # PUT
        delta = _calculate_put_delta(d1)
        theta = _calculate_put_theta(
            stock_price, strike, time_to_expiration, risk_free_rate,
            implied_volatility, d1, d2
        )
        rho = _calculate_put_rho(strike, time_to_expiration, risk_free_rate, d2)

    # Gamma and Vega are same for calls and puts
    gamma = _calculate_gamma(stock_price, time_to_expiration, implied_volatility, d1)
    vega = _calculate_vega(stock_price, time_to_expiration, d1)

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4)
    }


def _calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 in Black-Scholes formula.

    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)

    Args:
        S: Stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility

    Returns:
        d1 value
    """
    numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * math.sqrt(T)
    return numerator / denominator


def _calculate_call_delta(d1: float) -> float:
    """
    Calculate Delta for call option.

    Delta_call = N(d1)

    Args:
        d1: d1 from Black-Scholes

    Returns:
        Call delta (0 to 1)
    """
    return norm.cdf(d1)


def _calculate_put_delta(d1: float) -> float:
    """
    Calculate Delta for put option.

    Delta_put = N(d1) - 1

    Args:
        d1: d1 from Black-Scholes

    Returns:
        Put delta (-1 to 0)
    """
    return norm.cdf(d1) - 1


def _calculate_gamma(S: float, T: float, sigma: float, d1: float) -> float:
    """
    Calculate Gamma (same for calls and puts).

    Gamma = φ(d1) / (S * σ * √T)

    where φ(x) is the standard normal PDF

    Args:
        S: Stock price
        T: Time to expiration (years)
        sigma: Implied volatility
        d1: d1 from Black-Scholes

    Returns:
        Gamma
    """
    numerator = norm.pdf(d1)
    denominator = S * sigma * math.sqrt(T)
    return numerator / denominator


def _calculate_call_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    d1: float,
    d2: float
) -> float:
    """
    Calculate Theta for call option (time decay).

    Theta_call = -(S * φ(d1) * σ) / (2√T) - r * K * e^(-rT) * N(d2)

    Note: Theta is typically negative (option loses value as time passes)

    Args:
        S: Stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        d1: d1 from Black-Scholes
        d2: d2 from Black-Scholes

    Returns:
        Call theta (per year, divide by 365 for daily)
    """
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)

    # Return daily theta (divide by 365)
    return (term1 + term2) / 365


def _calculate_put_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    d1: float,
    d2: float
) -> float:
    """
    Calculate Theta for put option (time decay).

    Theta_put = -(S * φ(d1) * σ) / (2√T) + r * K * e^(-rT) * N(-d2)

    Args:
        S: Stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        d1: d1 from Black-Scholes
        d2: d2 from Black-Scholes

    Returns:
        Put theta (per year, divide by 365 for daily)
    """
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)

    # Return daily theta (divide by 365)
    return (term1 + term2) / 365


def _calculate_vega(S: float, T: float, d1: float) -> float:
    """
    Calculate Vega (same for calls and puts).

    Vega = S * √T * φ(d1)

    Note: Vega is sensitivity to 1% change in volatility

    Args:
        S: Stock price
        T: Time to expiration (years)
        d1: d1 from Black-Scholes

    Returns:
        Vega (per 1% change in IV)
    """
    # Divide by 100 to get vega per 1% change
    return S * math.sqrt(T) * norm.pdf(d1) / 100


def _calculate_call_rho(K: float, T: float, r: float, d2: float) -> float:
    """
    Calculate Rho for call option.

    Rho_call = K * T * e^(-rT) * N(d2)

    Args:
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        d2: d2 from Black-Scholes

    Returns:
        Call rho (per 1% change in interest rate)
    """
    # Divide by 100 to get rho per 1% change
    return K * T * math.exp(-r * T) * norm.cdf(d2) / 100


def _calculate_put_rho(K: float, T: float, r: float, d2: float) -> float:
    """
    Calculate Rho for put option.

    Rho_put = -K * T * e^(-rT) * N(-d2)

    Args:
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        d2: d2 from Black-Scholes

    Returns:
        Put rho (per 1% change in interest rate)
    """
    # Divide by 100 to get rho per 1% change
    return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100


def calculate_implied_volatility(
    option_price: float,
    option_type: str,
    stock_price: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    initial_guess: float = 0.30,
    tolerance: float = 0.0001,
    max_iterations: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    This is the reverse of Black-Scholes: given an option price,
    find the volatility that produces that price.

    Args:
        option_price: Market price of option
        option_type: 'CALL' or 'PUT'
        stock_price: Current stock price
        strike: Strike price
        time_to_expiration: Time to expiration (years)
        risk_free_rate: Risk-free rate
        initial_guess: Starting guess for IV (default 30%)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Implied volatility (e.g., 0.35 for 35%)
    """
    sigma = initial_guess

    for i in range(max_iterations):
        # Calculate option price with current sigma
        d1 = _calculate_d1(stock_price, strike, time_to_expiration, risk_free_rate, sigma)
        d2 = d1 - sigma * math.sqrt(time_to_expiration)

        if option_type.upper() == 'CALL':
            price = (stock_price * norm.cdf(d1) -
                    strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2))
        else:  # PUT
            price = (strike * math.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2) -
                    stock_price * norm.cdf(-d1))

        # Calculate vega for Newton-Raphson
        vega = stock_price * math.sqrt(time_to_expiration) * norm.pdf(d1)

        # Price difference
        diff = option_price - price

        # Check convergence
        if abs(diff) < tolerance:
            return sigma

        # Newton-Raphson update
        if vega != 0:
            sigma = sigma + diff / vega
        else:
            break

        # Keep sigma positive
        sigma = max(0.001, sigma)

    # If didn't converge, return current estimate
    return sigma


# Example usage and testing
if __name__ == '__main__':
    # Test with sample option
    print("=" * 60)
    print("Option Greeks Calculator - Test")
    print("=" * 60)
    print()

    # SPY 450 Call, 30 DTE
    greeks = calculate_greeks(
        option_type='CALL',
        stock_price=450.0,
        strike=450.0,
        time_to_expiration=30/365,  # 30 days
        risk_free_rate=0.05,  # 5%
        implied_volatility=0.30  # 30%
    )

    print("SPY $450 Call (30 DTE, IV=30%)")
    print(f"  Delta: {greeks['delta']:.4f}  (50 delta = ATM)")
    print(f"  Gamma: {greeks['gamma']:.4f}  (rate of delta change)")
    print(f"  Theta: {greeks['theta']:.4f}  (daily time decay)")
    print(f"  Vega:  {greeks['vega']:.4f}  (per 1% IV change)")
    print(f"  Rho:   {greeks['rho']:.4f}  (per 1% rate change)")
    print()

    # SPY 450 Put, 30 DTE
    greeks = calculate_greeks(
        option_type='PUT',
        stock_price=450.0,
        strike=450.0,
        time_to_expiration=30/365,
        risk_free_rate=0.05,
        implied_volatility=0.30
    )

    print("SPY $450 Put (30 DTE, IV=30%)")
    print(f"  Delta: {greeks['delta']:.4f}  (-50 delta = ATM)")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Vega:  {greeks['vega']:.4f}")
    print(f"  Rho:   {greeks['rho']:.4f}")
    print()

    # Deep ITM Call
    greeks = calculate_greeks(
        option_type='CALL',
        stock_price=460.0,
        strike=450.0,
        time_to_expiration=30/365,
        risk_free_rate=0.05,
        implied_volatility=0.30
    )

    print("SPY $450 Call, Stock @ $460 (ITM)")
    print(f"  Delta: {greeks['delta']:.4f}  (high delta = ITM)")
    print()

    # Deep OTM Call
    greeks = calculate_greeks(
        option_type='CALL',
        stock_price=440.0,
        strike=450.0,
        time_to_expiration=30/365,
        risk_free_rate=0.05,
        implied_volatility=0.30
    )

    print("SPY $450 Call, Stock @ $440 (OTM)")
    print(f"  Delta: {greeks['delta']:.4f}  (low delta = OTM)")
    print()

    print("=" * 60)
