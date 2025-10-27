"""Day-by-day stock trading simulation with smart stock selection."""
import pandas as pd
from stock_data import generate_stock_data
from agents import create_trading_workflow, get_risk_params
from smart_selector import select_stocks_for_day, rank_portfolio_positions
from visualizations import generate_all_visualizations
import json
from datetime import datetime, timedelta
import os


def run_daily_trading_simulation(
    num_days: int = 100,
    stocks_per_day: int = 10,
    risk_level: str = "moderate",
    selection_method: str = "momentum",
    seed: int = 42
):
    """
    Run a day-by-day trading simulation with AI agents.

    Args:
        num_days: Number of trading days to simulate
        stocks_per_day: Number of stocks to analyze each day
        risk_level: Risk level ("very_aggressive", "aggressive", "moderate", "conservative", "very_conservative")
        selection_method: Stock selection method ("momentum", "volatility", "mixed", "contrarian")
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("AI-Powered Day-by-Day Stock Trading System")
    print("="*80)

    # Validate risk level
    valid_risk_levels = ["very_aggressive", "aggressive", "moderate", "conservative", "very_conservative"]
    if risk_level not in valid_risk_levels:
        print(f"Invalid risk level. Using 'moderate'. Valid options: {valid_risk_levels}")
        risk_level = "moderate"

    risk_params = get_risk_params(risk_level)

    print(f"\nSimulation Parameters:")
    print(f"  Trading Days: {num_days}")
    print(f"  Stocks Analyzed Per Day: {stocks_per_day}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Selection Method: {selection_method}")
    print(f"  Confidence Threshold (Buy): {risk_params['confidence_threshold']:.2f}")
    print(f"  Confidence Threshold (Sell): {risk_params['sell_threshold']:.2f}")
    print(f"  Max Position Size: {risk_params['max_position_size']*100:.1f}%")

    # Generate stock data
    print(f"\n1. Generating stock data...")
    df = generate_stock_data(num_stocks=100, num_days=100, seed=seed)
    print(f"   Generated data for {df['symbol'].nunique()} stocks over {len(df['date'].unique())} days")

    # Initialize portfolio
    initial_cash = 100000.0
    portfolio = {
        "cash": initial_cash,
        "holdings": {}
    }

    print(f"\n2. Initial Portfolio:")
    print(f"   Cash: ${portfolio['cash']:,.2f}")

    # Create trading workflow
    trading_app = create_trading_workflow()

    # Track all trades and daily summaries
    all_trades = []
    daily_summaries = []

    # Get all unique dates
    all_dates = sorted(df['date'].unique())
    start_date_idx = len(all_dates) - num_days  # Start from the last N days
    trading_dates = all_dates[start_date_idx:]

    print(f"\n3. Starting Day-by-Day Trading...")
    print(f"   Simulating from {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print("="*80)

    # Simulate each trading day
    for day_idx, current_date in enumerate(trading_dates, 1):
        print(f"\n{'='*80}")
        print(f"Day {day_idx}/{num_days}: {current_date.date()}")
        print(f"{'='*80}")

        # Get data up to current date
        historical_df = df[df['date'] <= current_date]

        # Smart stock selection for today
        selected_symbols = select_stocks_for_day(
            historical_df,
            day_idx,
            num_stocks=stocks_per_day,
            portfolio=portfolio,
            method=selection_method
        )

        print(f"\nSelected Stocks: {', '.join(selected_symbols[:5])}{'...' if len(selected_symbols) > 5 else ''}")
        print(f"Current Cash: ${portfolio['cash']:,.2f}")

        # Calculate current portfolio value
        holdings_value = 0
        for symbol, shares in portfolio['holdings'].items():
            if shares > 0:
                try:
                    current_price = historical_df[historical_df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]['price']
                    holdings_value += shares * current_price
                except:
                    pass

        total_value = portfolio['cash'] + holdings_value
        print(f"Total Portfolio Value: ${total_value:,.2f}")

        day_trades = []

        # Analyze each selected stock
        for symbol in selected_symbols:
            try:
                # Run the trading workflow (agents will calculate metrics using tools)
                initial_state = {
                    "symbol": symbol,
                    "stock_data": historical_df,
                    "portfolio": portfolio,
                    "risk_level": risk_level,
                    "messages": []
                }

                result = trading_app.invoke(initial_state)

                # Update portfolio with results
                portfolio = result["portfolio"]

                # Get current price for tracking
                current_price = historical_df[historical_df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]['price']

                # Track trade
                trade_record = {
                    'day': day_idx,
                    'date': str(current_date.date()),
                    'symbol': symbol,
                    'recommendation': result['recommendation'],
                    'confidence': result['confidence'],
                    'execution': result['execution_result'],
                    'price': current_price
                }

                day_trades.append(trade_record)
                all_trades.append(trade_record)

                # Only print executed trades
                if 'EXECUTED' in result['execution_result']:
                    print(f"  {symbol}: {result['recommendation']} ({result['confidence']:.2f}) - {result['execution_result']}")

            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
                continue

        # Daily summary
        day_buys = sum(1 for t in day_trades if 'Bought' in t['execution'])
        day_sells = sum(1 for t in day_trades if 'Sold' in t['execution'])

        print(f"\nDay {day_idx} Summary: {day_buys} buys, {day_sells} sells")
        print(f"End of Day Cash: ${portfolio['cash']:,.2f}")

        # Recalculate portfolio value at end of day
        holdings_value = 0
        for symbol, shares in portfolio['holdings'].items():
            if shares > 0:
                try:
                    current_price = historical_df[historical_df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]['price']
                    holdings_value += shares * current_price
                except:
                    pass

        eod_total_value = portfolio['cash'] + holdings_value
        daily_pnl = eod_total_value - total_value if day_idx > 1 else 0

        daily_summaries.append({
            'day': day_idx,
            'date': str(current_date.date()),
            'cash': portfolio['cash'],
            'holdings_value': holdings_value,
            'total_value': eod_total_value,
            'daily_pnl': daily_pnl,
            'buys': day_buys,
            'sells': day_sells
        })

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nFinal Portfolio:")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"\n  Holdings:")

    total_holdings_value = 0
    final_holdings = []

    for symbol, shares in portfolio['holdings'].items():
        if shares > 0:
            try:
                current_price = df[df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]['price']
                value = shares * current_price
                total_holdings_value += value
                final_holdings.append({
                    'symbol': symbol,
                    'shares': shares,
                    'price': current_price,
                    'value': value
                })
                print(f"    {symbol}: {shares} shares @ ${current_price:.2f} = ${value:,.2f}")
            except:
                pass

    total_portfolio_value = portfolio['cash'] + total_holdings_value
    profit_loss = total_portfolio_value - initial_cash
    profit_loss_pct = (profit_loss / initial_cash) * 100

    print(f"\n  Total Holdings Value: ${total_holdings_value:,.2f}")
    print(f"  Total Portfolio Value: ${total_portfolio_value:,.2f}")
    print(f"  Profit/Loss: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)")

    total_buys = sum(1 for t in all_trades if 'Bought' in t['execution'])
    total_sells = sum(1 for t in all_trades if 'Sold' in t['execution'])
    total_skipped = sum(1 for t in all_trades if 'SKIPPED' in t['execution'])

    print(f"\nTrade Summary:")
    print(f"  Total Buys: {total_buys}")
    print(f"  Total Sells: {total_sells}")
    print(f"  Total Skipped: {total_skipped}")

    # Save results
    save_daily_results(
        portfolio, all_trades, daily_summaries, final_holdings,
        total_portfolio_value, profit_loss, profit_loss_pct,
        risk_level, selection_method, num_days
    )

    # Generate visualizations
    generate_all_visualizations(
        portfolio=portfolio,
        trades=all_trades,
        daily_summaries=daily_summaries,
        final_holdings=final_holdings,
        risk_level=risk_level,
        selection_method=selection_method,
        initial_cash=initial_cash
    )

    return portfolio, all_trades, daily_summaries


def save_daily_results(portfolio, trades, daily_summaries, holdings,
                       total_value, profit_loss, profit_loss_pct,
                       risk_level, selection_method, num_days):
    """Save daily trading results to files."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save detailed results to JSON
    json_filename = os.path.join(results_dir, f"daily_trading_{timestamp}.json")

    json_data = {
        'timestamp': timestamp,
        'simulation_parameters': {
            'initial_cash': 100000.0,
            'num_days': num_days,
            'risk_level': risk_level,
            'selection_method': selection_method
        },
        'final_portfolio': {
            'cash': round(portfolio['cash'], 2),
            'holdings': holdings,
            'total_holdings_value': round(total_value - portfolio['cash'], 2),
            'total_portfolio_value': round(total_value, 2)
        },
        'performance': {
            'profit_loss': round(profit_loss, 2),
            'profit_loss_percentage': round(profit_loss_pct, 2)
        },
        'daily_summaries': daily_summaries,
        'all_trades': trades,
        'summary': {
            'total_buys': sum(1 for t in trades if 'Bought' in t['execution']),
            'total_sells': sum(1 for t in trades if 'Sold' in t['execution']),
            'total_skipped': sum(1 for t in trades if 'SKIPPED' in t['execution'])
        }
    }

    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n  Saved detailed results to: {json_filename}")

    # 2. Save daily summaries to CSV
    if daily_summaries:
        daily_csv = os.path.join(results_dir, f"daily_summary_{timestamp}.csv")
        pd.DataFrame(daily_summaries).to_csv(daily_csv, index=False)
        print(f"  Saved daily summaries to: {daily_csv}")

    # 3. Save all trades to CSV
    if trades:
        trades_csv = os.path.join(results_dir, f"all_trades_{timestamp}.csv")
        pd.DataFrame(trades).to_csv(trades_csv, index=False)
        print(f"  Saved all trades to: {trades_csv}")

    print(f"\n  All results saved to '{results_dir}/' directory")


if __name__ == "__main__":
    # Run the daily simulation
    # Adjust parameters as needed:
    # - num_days: Number of trading days (1-100, default: 20)
    #   * Recommended: 20-30 days for good visualization and reasonable runtime
    #   * Full 100 days = ~1000 API calls (takes longer but shows full dataset)
    # - stocks_per_day: Stocks to analyze each day (default: 10)
    # - risk_level: "very_aggressive", "aggressive", "moderate", "conservative", "very_conservative"
    # - selection_method: "momentum", "volatility", "mixed", "contrarian"

    run_daily_trading_simulation(
        num_days=50,  # 50 days for faster results
        stocks_per_day=10,
        risk_level="aggressive",
        selection_method="momentum",
        seed=42
    )
