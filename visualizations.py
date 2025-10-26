"""Visualization module for portfolio and trading analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os


# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def create_portfolio_visualizations(daily_summaries, trades, final_holdings,
                                    initial_cash=100000.0, risk_level="moderate",
                                    selection_method="momentum", save_dir="results"):
    """
    Create comprehensive portfolio visualizations.

    Args:
        daily_summaries: List of daily summary dictionaries
        trades: List of all trade dictionaries
        final_holdings: List of final holdings dictionaries
        initial_cash: Initial portfolio cash
        risk_level: Risk level used in simulation
        selection_method: Stock selection method used
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Portfolio Value Over Time
    ax1 = plt.subplot(3, 2, 1)
    plot_portfolio_value(daily_summaries, initial_cash, ax1)

    # 2. Cash vs Holdings Over Time
    ax2 = plt.subplot(3, 2, 2)
    plot_cash_vs_holdings(daily_summaries, ax2)

    # 3. Daily Profit/Loss
    ax3 = plt.subplot(3, 2, 3)
    plot_daily_pnl(daily_summaries, ax3)

    # 4. Asset Allocation (Pie Chart)
    ax4 = plt.subplot(3, 2, 4)
    plot_asset_allocation(final_holdings, daily_summaries, ax4)

    # 5. Trade Activity
    ax5 = plt.subplot(3, 2, 5)
    plot_trade_activity(trades, ax5)

    # 6. Confidence Distribution
    ax6 = plt.subplot(3, 2, 6)
    plot_confidence_distribution(trades, ax6)

    # Overall title
    final_value = daily_summaries[-1]['total_value'] if daily_summaries else initial_cash
    profit_loss = final_value - initial_cash
    profit_pct = (profit_loss / initial_cash) * 100

    fig.suptitle(f'Portfolio Analysis - Risk: {risk_level} | Strategy: {selection_method} | '
                 f'P/L: ${profit_loss:,.2f} ({profit_pct:+.2f}%)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save the figure
    filename = os.path.join(save_dir, f"portfolio_analysis_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved portfolio visualization to: {filename}")

    plt.close()

    # Create additional detailed plots
    create_detailed_plots(daily_summaries, trades, final_holdings, timestamp, save_dir)


def plot_portfolio_value(daily_summaries, initial_cash, ax):
    """Plot portfolio value over time."""
    if not daily_summaries:
        return

    df = pd.DataFrame(daily_summaries)
    num_days = len(df)

    # Adjust marker size and style based on number of days
    if num_days > 50:
        # For longer simulations, use line only (no markers)
        marker_style = None
        marker_size = 0
        line_width = 2.5
    elif num_days > 20:
        # Medium simulations, smaller markers
        marker_style = 'o'
        marker_size = 3
        line_width = 2
    else:
        # Short simulations, larger markers
        marker_style = 'o'
        marker_size = 6
        line_width = 2

    ax.plot(df['day'], df['total_value'], marker=marker_style, linewidth=line_width,
            markersize=marker_size, color='#2E86AB', label='Portfolio Value', zorder=3)
    ax.axhline(y=initial_cash, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Initial Value', zorder=2)

    # Fill area under curve
    ax.fill_between(df['day'], df['total_value'], initial_cash,
                     where=(df['total_value'] >= initial_cash),
                     alpha=0.3, color='green', label='Profit', zorder=1)
    ax.fill_between(df['day'], df['total_value'], initial_cash,
                     where=(df['total_value'] < initial_cash),
                     alpha=0.3, color='red', label='Loss', zorder=1)

    ax.set_xlabel('Trading Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax.set_title('Net Portfolio Value Over Time', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add min/max annotations for longer simulations
    if num_days >= 10:
        max_idx = df['total_value'].idxmax()
        min_idx = df['total_value'].idxmin()

        # Annotate maximum
        ax.annotate(f'Peak: ${df.loc[max_idx, "total_value"]:,.0f}',
                   xy=(df.loc[max_idx, 'day'], df.loc[max_idx, 'total_value']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

        # Annotate minimum
        ax.annotate(f'Low: ${df.loc[min_idx, "total_value"]:,.0f}',
                   xy=(df.loc[min_idx, 'day'], df.loc[min_idx, 'total_value']),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=8, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))


def plot_cash_vs_holdings(daily_summaries, ax):
    """Plot cash vs holdings value over time."""
    if not daily_summaries:
        return

    df = pd.DataFrame(daily_summaries)
    num_days = len(df)

    # Adjust markers for longer simulations
    if num_days > 50:
        marker1, marker2 = None, None
        markersize = 0
        linewidth = 2.5
    elif num_days > 20:
        marker1, marker2 = 's', '^'
        markersize = 3
        linewidth = 2
    else:
        marker1, marker2 = 's', '^'
        markersize = 5
        linewidth = 2

    ax.plot(df['day'], df['cash'], marker=marker1, linewidth=linewidth,
            markersize=markersize, color='#06A77D', label='Cash', alpha=0.9)
    ax.plot(df['day'], df['holdings_value'], marker=marker2, linewidth=linewidth,
            markersize=markersize, color='#F77F00', label='Holdings Value', alpha=0.9)

    # Add stacked area for better visualization
    ax.fill_between(df['day'], 0, df['cash'], alpha=0.2, color='#06A77D')
    ax.fill_between(df['day'], df['cash'], df['cash'] + df['holdings_value'],
                     alpha=0.2, color='#F77F00')

    ax.set_xlabel('Trading Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value ($)', fontsize=11, fontweight='bold')
    ax.set_title('Cash vs Holdings Over Time', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))


def plot_daily_pnl(daily_summaries, ax):
    """Plot daily profit/loss."""
    if not daily_summaries or len(daily_summaries) < 2:
        return

    df = pd.DataFrame(daily_summaries)

    # Calculate daily P/L
    df['daily_change'] = df['total_value'].diff()

    colors = ['green' if x >= 0 else 'red' for x in df['daily_change'].fillna(0)]

    ax.bar(df['day'], df['daily_change'].fillna(0), color=colors, alpha=0.7)

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Daily Change ($)')
    ax.set_title('Daily Profit/Loss', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))


def plot_asset_allocation(final_holdings, daily_summaries, ax):
    """Plot asset allocation pie chart."""
    if not final_holdings:
        ax.text(0.5, 0.5, 'No Holdings', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return

    # Prepare data
    cash = daily_summaries[-1]['cash'] if daily_summaries else 0
    holdings_data = [{'symbol': 'Cash', 'value': cash}]
    holdings_data.extend(final_holdings)

    df = pd.DataFrame(holdings_data)

    # Create pie chart
    colors = sns.color_palette("husl", len(df))
    wedges, texts, autotexts = ax.pie(df['value'], labels=df['symbol'],
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90)

    # Enhance text
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    ax.set_title('Asset Allocation', fontweight='bold')


def plot_trade_activity(trades, ax):
    """Plot trade activity by day."""
    if not trades:
        return

    df = pd.DataFrame(trades)

    # Count trades by day and type
    df['trade_type'] = df['execution'].apply(lambda x: 'Buy' if 'Bought' in x else
                                              ('Sell' if 'Sold' in x else 'Skip'))

    # Filter only executed trades
    executed = df[df['trade_type'].isin(['Buy', 'Sell'])]

    if executed.empty:
        ax.text(0.5, 0.5, 'No Executed Trades', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return

    trade_counts = executed.groupby(['day', 'trade_type']).size().unstack(fill_value=0)

    trade_counts.plot(kind='bar', ax=ax, color=['#06A77D', '#D62828'], alpha=0.8)

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Activity by Day', fontweight='bold')
    ax.legend(title='Trade Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)


def plot_confidence_distribution(trades, ax):
    """Plot confidence score distribution."""
    if not trades:
        return

    df = pd.DataFrame(trades)

    # Filter executed trades
    executed = df[df['execution'].str.contains('EXECUTED', na=False)]

    if executed.empty:
        ax.text(0.5, 0.5, 'No Executed Trades', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return

    # Create histogram
    ax.hist(executed['confidence'], bins=10, color='#2E86AB', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Confidence Score Distribution (Executed Trades)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def create_detailed_plots(daily_summaries, trades, final_holdings, timestamp, save_dir):
    """Create additional detailed plots."""

    # 1. Holdings Performance Plot
    if final_holdings:
        fig, ax = plt.subplots(figsize=(12, 6))

        df = pd.DataFrame(final_holdings)
        df = df.sort_values('value', ascending=True)

        colors = ['green' if 'STOCK' in s else 'blue' for s in df['symbol']]

        ax.barh(df['symbol'], df['value'], color=colors, alpha=0.7)

        ax.set_xlabel('Value ($)')
        ax.set_ylabel('Stock Symbol')
        ax.set_title('Holdings Value by Stock', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filename = os.path.join(save_dir, f"holdings_detail_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved holdings detail to: {filename}")
        plt.close()

    # 2. Cumulative Return Plot
    if daily_summaries:
        fig, ax = plt.subplots(figsize=(12, 6))

        df = pd.DataFrame(daily_summaries)
        initial = df['total_value'].iloc[0]
        df['cumulative_return'] = ((df['total_value'] - initial) / initial) * 100

        ax.plot(df['day'], df['cumulative_return'], marker='o', linewidth=2.5,
                markersize=7, color='#2E86AB')
        ax.fill_between(df['day'], df['cumulative_return'], 0,
                         where=(df['cumulative_return'] >= 0),
                         alpha=0.3, color='green')
        ax.fill_between(df['day'], df['cumulative_return'], 0,
                         where=(df['cumulative_return'] < 0),
                         alpha=0.3, color='red')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Trading Day', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Return Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(save_dir, f"cumulative_return_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved cumulative return plot to: {filename}")
        plt.close()


def create_large_portfolio_chart(daily_summaries, initial_cash, risk_level,
                                 selection_method, timestamp, save_dir="results"):
    """
    Create a large, detailed chart showing net portfolio value over entire period.
    Optimized for 100-day simulations.
    """
    if not daily_summaries:
        return

    df = pd.DataFrame(daily_summaries)
    num_days = len(df)

    # Create large figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Main portfolio value line
    ax.plot(df['day'], df['total_value'], linewidth=3, color='#2E86AB',
            label='Net Portfolio Value', zorder=3)

    # Initial value reference line
    ax.axhline(y=initial_cash, color='gray', linestyle='--',
               linewidth=2, alpha=0.7, label='Initial Value ($100,000)', zorder=2)

    # Fill areas
    ax.fill_between(df['day'], df['total_value'], initial_cash,
                     where=(df['total_value'] >= initial_cash),
                     alpha=0.25, color='green', label='Profit Zone', zorder=1)
    ax.fill_between(df['day'], df['total_value'], initial_cash,
                     where=(df['total_value'] < initial_cash),
                     alpha=0.25, color='red', label='Loss Zone', zorder=1)

    # Calculate statistics
    final_value = df['total_value'].iloc[-1]
    max_value = df['total_value'].max()
    min_value = df['total_value'].min()
    profit_loss = final_value - initial_cash
    profit_pct = (profit_loss / initial_cash) * 100

    # Annotate key points
    max_idx = df['total_value'].idxmax()
    min_idx = df['total_value'].idxmin()

    ax.annotate(f'Peak\n${max_value:,.0f}\nDay {df.loc[max_idx, "day"]}',
               xy=(df.loc[max_idx, 'day'], max_value),
               xytext=(15, 15), textcoords='offset points',
               fontsize=10, color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.annotate(f'Low\n${min_value:,.0f}\nDay {df.loc[min_idx, "day"]}',
               xy=(df.loc[min_idx, 'day'], min_value),
               xytext=(15, -40), textcoords='offset points',
               fontsize=10, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Final value annotation
    final_color = 'green' if profit_loss >= 0 else 'red'
    ax.annotate(f'Final\n${final_value:,.0f}\n{profit_pct:+.2f}%',
               xy=(df['day'].iloc[-1], final_value),
               xytext=(-60, 0), textcoords='offset points',
               fontsize=11, color=final_color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=final_color, linewidth=2.5),
               arrowprops=dict(arrowstyle='->', color=final_color, lw=2.5))

    # Labels and title
    ax.set_xlabel('Trading Day', fontsize=14, fontweight='bold')
    ax.set_ylabel('Net Portfolio Value ($)', fontsize=14, fontweight='bold')

    title = f'Net Portfolio Performance: {num_days}-Day Simulation\n'
    title += f'Strategy: {selection_method.title()} | Risk Level: {risk_level.replace("_", " ").title()}\n'
    title += f'Final P/L: ${profit_loss:,.2f} ({profit_pct:+.2f}%)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add statistics box
    stats_text = f'Statistics:\n'
    stats_text += f'Days: {num_days}\n'
    stats_text += f'Max: ${max_value:,.0f}\n'
    stats_text += f'Min: ${min_value:,.0f}\n'
    stats_text += f'Range: ${max_value - min_value:,.0f}\n'
    stats_text += f'Volatility: {((max_value - min_value) / initial_cash * 100):.1f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    filename = os.path.join(save_dir, f"net_portfolio_value_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved large net portfolio chart to: {filename}")
    plt.close()


def generate_all_visualizations(portfolio, trades, daily_summaries, final_holdings,
                                risk_level, selection_method, initial_cash=100000.0):
    """
    Generate all visualizations for the trading simulation.

    Args:
        portfolio: Final portfolio state
        trades: List of all trades
        daily_summaries: Daily summary data
        final_holdings: Final holdings list
        risk_level: Risk level used
        selection_method: Selection method used
        initial_cash: Initial portfolio value
    """
    print("\nGenerating visualizations...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create main dashboard
    create_portfolio_visualizations(
        daily_summaries=daily_summaries,
        trades=trades,
        final_holdings=final_holdings,
        initial_cash=initial_cash,
        risk_level=risk_level,
        selection_method=selection_method,
        save_dir="results"
    )

    # Create large dedicated net portfolio chart
    create_large_portfolio_chart(
        daily_summaries=daily_summaries,
        initial_cash=initial_cash,
        risk_level=risk_level,
        selection_method=selection_method,
        timestamp=timestamp,
        save_dir="results"
    )

    print("  All visualizations generated successfully!")


if __name__ == "__main__":
    # Test with sample data
    print("Testing visualization module...")

    # Sample data
    daily_summaries = [
        {'day': 1, 'date': '2025-01-01', 'cash': 80000, 'holdings_value': 20000, 'total_value': 100000, 'daily_pnl': 0, 'buys': 2, 'sells': 0},
        {'day': 2, 'date': '2025-01-02', 'cash': 75000, 'holdings_value': 26000, 'total_value': 101000, 'daily_pnl': 1000, 'buys': 1, 'sells': 0},
        {'day': 3, 'date': '2025-01-03', 'cash': 75000, 'holdings_value': 27500, 'total_value': 102500, 'daily_pnl': 1500, 'buys': 0, 'sells': 0},
    ]

    trades = [
        {'day': 1, 'symbol': 'STOCK001', 'recommendation': 'BUY', 'confidence': 0.75, 'execution': 'EXECUTED: Bought 10 shares'},
        {'day': 1, 'symbol': 'STOCK002', 'recommendation': 'BUY', 'confidence': 0.80, 'execution': 'EXECUTED: Bought 5 shares'},
        {'day': 2, 'symbol': 'STOCK003', 'recommendation': 'BUY', 'confidence': 0.70, 'execution': 'EXECUTED: Bought 8 shares'},
    ]

    final_holdings = [
        {'symbol': 'STOCK001', 'shares': 10, 'price': 100, 'value': 1000},
        {'symbol': 'STOCK002', 'shares': 5, 'price': 200, 'value': 1000},
        {'symbol': 'STOCK003', 'shares': 8, 'price': 150, 'value': 1200},
    ]

    generate_all_visualizations(
        portfolio={'cash': 75000, 'holdings': {}},
        trades=trades,
        daily_summaries=daily_summaries,
        final_holdings=final_holdings,
        risk_level="moderate",
        selection_method="momentum"
    )

    print("\nTest complete! Check the results/ directory for generated plots.")
