"""Generate synthetic stock data for simulation."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_stock_data(num_stocks: int = 100, num_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic stock price data for multiple stocks over multiple days.

    Args:
        num_stocks: Number of stocks to generate
        num_days: Number of days of historical data
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: date, symbol, price, volume
    """
    np.random.seed(seed)

    # Generate stock symbols
    symbols = [f"STOCK{i:03d}" for i in range(num_stocks)]

    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=num_days-i-1) for i in range(num_days)]

    data = []

    for symbol in symbols:
        # Initial price between $10 and $500
        initial_price = np.random.uniform(10, 500)

        # Generate price series with random walk
        prices = [initial_price]
        for _ in range(num_days - 1):
            # Daily return between -5% and +5%
            daily_return = np.random.normal(0.0005, 0.02)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(1.0, new_price))  # Ensure price doesn't go below $1

        # Generate volumes
        base_volume = np.random.randint(100000, 10000000)
        volumes = [int(base_volume * np.random.uniform(0.5, 1.5)) for _ in range(num_days)]

        for date, price, volume in zip(dates, prices, volumes):
            data.append({
                'date': date,
                'symbol': symbol,
                'price': round(price, 2),
                'volume': volume
            })

    df = pd.DataFrame(data)
    return df


def get_stock_history(df: pd.DataFrame, symbol: str, days: int = 10) -> pd.DataFrame:
    """Get recent price history for a specific stock."""
    stock_data = df[df['symbol'] == symbol].sort_values('date', ascending=False).head(days)
    return stock_data.sort_values('date')


def get_current_price(df: pd.DataFrame, symbol: str) -> float:
    """Get the most recent price for a stock."""
    latest = df[df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]
    return latest['price']


def calculate_metrics(df: pd.DataFrame, symbol: str, lookback: int = 20) -> dict:
    """Calculate technical metrics for a stock."""
    history = get_stock_history(df, symbol, lookback)
    prices = history['price'].values

    if len(prices) < 2:
        return {}

    # Calculate simple metrics
    current_price = prices[-1]
    avg_price = np.mean(prices)
    volatility = np.std(prices) / avg_price if avg_price > 0 else 0

    # Price momentum (rate of change)
    if len(prices) >= lookback:
        momentum = (prices[-1] - prices[0]) / prices[0] * 100
    else:
        momentum = 0

    # Simple moving average
    sma = np.mean(prices[-10:]) if len(prices) >= 10 else avg_price

    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'average_price': round(avg_price, 2),
        'volatility': round(volatility, 4),
        'momentum_pct': round(momentum, 2),
        'sma_10': round(sma, 2),
        'price_vs_sma': round((current_price - sma) / sma * 100, 2) if sma > 0 else 0
    }


if __name__ == "__main__":
    # Test the data generation
    df = generate_stock_data(100, 100)
    print(f"Generated {len(df)} data points")
    print(f"\nSample data:\n{df.head(10)}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nUnique symbols: {df['symbol'].nunique()}")

    # Test metrics calculation
    print(f"\nMetrics for STOCK000:")
    metrics = calculate_metrics(df, 'STOCK000')
    for key, value in metrics.items():
        print(f"  {key}: {value}")
