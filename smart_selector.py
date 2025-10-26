"""Smart stock selection logic for daily trading."""
import pandas as pd
import numpy as np
from stock_data import calculate_metrics


def select_stocks_for_day(df: pd.DataFrame, day: int, num_stocks: int = 10,
                          portfolio: dict = None, method: str = "momentum") -> list:
    """
    Intelligently select stocks to analyze for the given day.

    Args:
        df: Stock data DataFrame
        day: Current day number (0-indexed from start of data)
        num_stocks: Number of stocks to select
        portfolio: Current portfolio state
        method: Selection method ("momentum", "volatility", "mixed", "contrarian")

    Returns:
        List of stock symbols to analyze
    """
    # Get all available stocks
    all_symbols = df['symbol'].unique().tolist()

    # Calculate metrics for all stocks
    stock_scores = []

    for symbol in all_symbols:
        try:
            metrics = calculate_metrics(df, symbol, lookback=20)
            if not metrics:
                continue

            # Calculate selection score based on method
            if method == "momentum":
                # Favor stocks with positive momentum
                score = metrics['momentum_pct']

            elif method == "volatility":
                # Favor volatile stocks for trading opportunities
                score = metrics['volatility'] * 100

            elif method == "contrarian":
                # Favor stocks that are oversold (negative momentum)
                score = -metrics['momentum_pct']

            elif method == "mixed":
                # Balanced approach
                momentum_score = metrics['momentum_pct']
                volatility_score = metrics['volatility'] * 50
                price_deviation = abs(metrics['price_vs_sma'])
                score = momentum_score + volatility_score + price_deviation

            else:
                score = 0

            stock_scores.append({
                'symbol': symbol,
                'score': score,
                'current_price': metrics['current_price'],
                'momentum': metrics['momentum_pct'],
                'volatility': metrics['volatility']
            })
        except Exception as e:
            continue

    # Sort by score and select top stocks
    stock_scores.sort(key=lambda x: x['score'], reverse=True)

    # If we have a portfolio, also consider stocks we own
    selected_symbols = []

    # First, add stocks we currently hold (to consider selling)
    if portfolio and portfolio.get('holdings'):
        held_symbols = [s for s, shares in portfolio['holdings'].items() if shares > 0]
        # Add up to 30% of selections from held stocks
        max_held = min(len(held_symbols), max(1, num_stocks // 3))
        selected_symbols.extend(held_symbols[:max_held])

    # Then add top-scoring stocks we don't own yet
    remaining_slots = num_stocks - len(selected_symbols)
    for stock in stock_scores:
        if stock['symbol'] not in selected_symbols:
            selected_symbols.append(stock['symbol'])
            if len(selected_symbols) >= num_stocks:
                break

    return selected_symbols[:num_stocks]


def rank_portfolio_positions(df: pd.DataFrame, portfolio: dict) -> list:
    """
    Rank current portfolio positions by strength.

    Args:
        df: Stock data DataFrame
        portfolio: Current portfolio state

    Returns:
        List of (symbol, score) tuples, sorted by score (best first)
    """
    if not portfolio or not portfolio.get('holdings'):
        return []

    position_scores = []

    for symbol, shares in portfolio['holdings'].items():
        if shares <= 0:
            continue

        try:
            metrics = calculate_metrics(df, symbol, lookback=20)
            if not metrics:
                continue

            # Score positions based on performance
            # Positive: good momentum, price above SMA
            # Negative: bad momentum, price below SMA
            score = metrics['momentum_pct'] + metrics['price_vs_sma']

            position_scores.append({
                'symbol': symbol,
                'shares': shares,
                'score': score,
                'current_price': metrics['current_price']
            })
        except Exception as e:
            continue

    position_scores.sort(key=lambda x: x['score'], reverse=True)
    return position_scores
