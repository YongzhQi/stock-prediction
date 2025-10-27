"""AI Agents for stock trading analysis and execution."""
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# State definition for the graph
class TradingState(TypedDict):
    """State for the trading workflow."""
    symbol: str
    stock_data: pd.DataFrame  # Raw stock data instead of pre-calculated metrics
    analysis: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    execution_result: str
    portfolio: dict
    messages: list
    risk_level: str  # "very_aggressive", "aggressive", "moderate", "conservative", "very_conservative"


# Global variable to hold stock data for tools (set by analyzer_agent)
_TOOL_STOCK_DATA = None


# Define tools for technical indicator calculation
@tool
def calculate_momentum(symbol: str, lookback: int = 20) -> float:
    """Calculate price momentum percentage for a stock over the lookback period.

    Args:
        symbol: Stock symbol to analyze
        lookback: Number of days to look back (default 20)

    Returns:
        Momentum as percentage change
    """
    if _TOOL_STOCK_DATA is None:
        return 0.0

    df = _TOOL_STOCK_DATA
    history = df[df['symbol'] == symbol].sort_values('date', ascending=False).head(lookback)
    history = history.sort_values('date')
    prices = history['price'].values

    if len(prices) < 2:
        return 0.0

    momentum = (prices[-1] - prices[0]) / prices[0] * 100
    return round(momentum, 2)


@tool
def calculate_volatility(symbol: str, lookback: int = 20) -> float:
    """Calculate price volatility (standard deviation / mean) for a stock.

    Args:
        symbol: Stock symbol to analyze
        lookback: Number of days to look back (default 20)

    Returns:
        Volatility as a decimal (e.g., 0.0234 = 2.34%)
    """
    if _TOOL_STOCK_DATA is None:
        return 0.0

    df = _TOOL_STOCK_DATA
    history = df[df['symbol'] == symbol].sort_values('date', ascending=False).head(lookback)
    prices = history['price'].values

    if len(prices) < 2:
        return 0.0

    avg_price = np.mean(prices)
    volatility = np.std(prices) / avg_price if avg_price > 0 else 0
    return round(volatility, 4)


@tool
def calculate_sma(symbol: str, period: int = 10) -> float:
    """Calculate Simple Moving Average for a stock.

    Args:
        symbol: Stock symbol to analyze
        period: Number of days for SMA calculation (default 10)

    Returns:
        Simple Moving Average price
    """
    if _TOOL_STOCK_DATA is None:
        return 0.0

    df = _TOOL_STOCK_DATA
    history = df[df['symbol'] == symbol].sort_values('date', ascending=False).head(period)
    prices = history['price'].values

    if len(prices) < 1:
        return 0.0

    sma = np.mean(prices)
    return round(sma, 2)


@tool
def get_current_price(symbol: str) -> float:
    """Get the current (most recent) price for a stock.

    Args:
        symbol: Stock symbol to analyze

    Returns:
        Current price
    """
    if _TOOL_STOCK_DATA is None:
        return 0.0

    df = _TOOL_STOCK_DATA
    latest = df[df['symbol'] == symbol].sort_values('date', ascending=False).iloc[0]
    return round(latest['price'], 2)


def get_llm():
    """Get the configured LLM."""
    if os.getenv("ANTHROPIC_API_KEY"):
        # Try different Claude models in order of preference
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.7)
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    else:
        raise ValueError("Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")


def analyzer_agent(state: TradingState) -> TradingState:
    """
    Analyzer agent that uses tools to calculate metrics and provides buy/sell recommendations.
    """
    global _TOOL_STOCK_DATA

    # Get LLM with tools bound
    llm = get_llm()
    tools = [calculate_momentum, calculate_volatility, calculate_sma, get_current_price]
    llm_with_tools = llm.bind_tools(tools)

    symbol = state["symbol"]
    stock_data = state["stock_data"]

    # Set global stock data for tools to access
    _TOOL_STOCK_DATA = stock_data

    system_prompt = """You are an expert stock market analyst with access to tools for calculating technical indicators.

You have access to these tools:
- get_current_price(symbol): Get the current price of a stock
- calculate_momentum(symbol, lookback): Calculate price momentum percentage (default lookback=20)
- calculate_volatility(symbol, lookback): Calculate price volatility (default lookback=20)
- calculate_sma(symbol, period): Calculate Simple Moving Average (default period=10)

First, use the tools to gather metrics for the stock. Then analyze:
- Current price trends
- Price momentum (positive or negative)
- Volatility level (higher = more risky)
- Price relative to SMA

IMPORTANT: After your analysis, you MUST end your response with:
Recommendation: [BUY/SELL/HOLD]
Confidence: [0.0-1.0]

Be concise and actionable."""

    user_prompt = f"""Analyze {symbol} using the available tools.

1. First call the tools to get:
   - Current price using get_current_price("{symbol}")
   - Momentum using calculate_momentum("{symbol}")
   - Volatility using calculate_volatility("{symbol}")
   - SMA using calculate_sma("{symbol}")

2. Then provide your trading analysis and recommendation.

Remember to end with:
Recommendation: [BUY/SELL/HOLD]
Confidence: [score between 0.0 and 1.0]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # First invocation - LLM will call tools
    response = llm_with_tools.invoke(messages)

    # Check if LLM wants to use tools
    if response.tool_calls:
        # Add the assistant message with tool calls
        messages.append(response)

        # Execute tool calls and collect results
        from langchain_core.messages import ToolMessage

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # Execute the tool (tools now access global _TOOL_STOCK_DATA)
            if tool_name == 'calculate_momentum':
                result = calculate_momentum.invoke(tool_args)
            elif tool_name == 'calculate_volatility':
                result = calculate_volatility.invoke(tool_args)
            elif tool_name == 'calculate_sma':
                result = calculate_sma.invoke(tool_args)
            elif tool_name == 'get_current_price':
                result = get_current_price.invoke(tool_args)
            else:
                result = "Tool not found"

            # Add properly formatted tool result message
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id']
                )
            )

        # Second invocation - get final analysis
        response = llm_with_tools.invoke(messages)

    # Extract analysis text (handle both string and list content)
    if isinstance(response.content, str):
        analysis_text = response.content
    elif isinstance(response.content, list):
        # Handle list of content blocks (text and tool_use blocks)
        analysis_text = " ".join([
            block.get('text', '') if isinstance(block, dict) else str(block)
            for block in response.content
        ])
    else:
        analysis_text = str(response.content)

    # Parse the response to extract recommendation and confidence
    recommendation = "HOLD"
    confidence = 0.5

    upper_text = analysis_text.upper()
    if "BUY" in upper_text and "SELL" not in upper_text.split("BUY")[0]:
        recommendation = "BUY"
    elif "SELL" in upper_text:
        recommendation = "SELL"

    # Try to extract confidence score
    import re
    confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', analysis_text.lower())
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            if confidence > 1.0:
                confidence = confidence / 100  # Convert percentage to decimal
        except:
            pass

    state["analysis"] = analysis_text
    state["recommendation"] = recommendation
    state["confidence"] = confidence
    state["messages"] = state.get("messages", []) + [
        {"role": "analyzer", "content": analysis_text}
    ]

    return state


def get_risk_params(risk_level: str) -> dict:
    """
    Get trading parameters based on risk level.

    Returns:
        dict with confidence_threshold, max_position_size, and diversification_min
    """
    risk_params = {
        "very_aggressive": {
            "confidence_threshold": 0.50,  # Lower threshold, act on more signals
            "max_position_size": 0.25,     # Up to 25% of cash per trade
            "diversification_min": 3,       # Minimum 3 stocks
            "sell_threshold": 0.55
        },
        "aggressive": {
            "confidence_threshold": 0.60,
            "max_position_size": 0.15,     # Up to 15% of cash per trade
            "diversification_min": 5,
            "sell_threshold": 0.60
        },
        "moderate": {
            "confidence_threshold": 0.70,
            "max_position_size": 0.10,     # Up to 10% of cash per trade
            "diversification_min": 7,
            "sell_threshold": 0.65
        },
        "conservative": {
            "confidence_threshold": 0.75,
            "max_position_size": 0.07,     # Up to 7% of cash per trade
            "diversification_min": 10,
            "sell_threshold": 0.70
        },
        "very_conservative": {
            "confidence_threshold": 0.80,
            "max_position_size": 0.05,     # Up to 5% of cash per trade
            "diversification_min": 15,
            "sell_threshold": 0.75
        }
    }
    return risk_params.get(risk_level, risk_params["moderate"])


def executor_agent(state: TradingState) -> TradingState:
    """
    Executor agent that executes buy/sell orders based on analyst recommendations
    and risk level parameters.
    """
    global _TOOL_STOCK_DATA

    symbol = state["symbol"]
    recommendation = state["recommendation"]
    confidence = state["confidence"]
    stock_data = state["stock_data"]
    portfolio = state.get("portfolio", {"cash": 100000.0, "holdings": {}})
    risk_level = state.get("risk_level", "moderate")

    # Get risk parameters
    risk_params = get_risk_params(risk_level)
    confidence_threshold = risk_params["confidence_threshold"]
    max_position_size = risk_params["max_position_size"]
    sell_threshold = risk_params["sell_threshold"]

    # Set global stock data and get current price using the tool
    _TOOL_STOCK_DATA = stock_data
    current_price = get_current_price.invoke({"symbol": symbol})
    current_holdings = portfolio["holdings"].get(symbol, 0)
    cash = portfolio["cash"]

    # Execute the trade
    execution_result = ""
    execution_summary = f"Risk Level: {risk_level}, Confidence Threshold: {confidence_threshold:.2f}"

    if recommendation == "BUY" and confidence >= confidence_threshold:
        # Determine position size based on confidence
        # Higher confidence = larger position (within risk limits)
        confidence_multiplier = min(confidence / confidence_threshold, 1.5)
        position_size = max_position_size * confidence_multiplier
        max_spend = cash * position_size

        shares_to_buy = int(max_spend / current_price) if current_price > 0 else 0

        if shares_to_buy > 0 and cash >= shares_to_buy * current_price:
            cost = shares_to_buy * current_price
            portfolio["cash"] -= cost
            portfolio["holdings"][symbol] = portfolio["holdings"].get(symbol, 0) + shares_to_buy
            execution_result = f"EXECUTED: Bought {shares_to_buy} shares of {symbol} at ${current_price:.2f} for ${cost:,.2f} (confidence: {confidence:.2f}, position: {position_size*100:.1f}%)"
        else:
            execution_result = f"SKIPPED: Insufficient funds to buy {symbol}"

    elif recommendation == "SELL" and confidence >= sell_threshold and current_holdings > 0:
        # Sell based on confidence - higher confidence = sell more
        if confidence >= 0.85:
            # Very high confidence - sell all
            shares_to_sell = current_holdings
        elif confidence >= 0.75:
            # High confidence - sell 75%
            shares_to_sell = int(current_holdings * 0.75)
        else:
            # Moderate confidence - sell 50%
            shares_to_sell = int(current_holdings * 0.50)

        if shares_to_sell > 0:
            proceeds = shares_to_sell * current_price
            portfolio["cash"] += proceeds
            portfolio["holdings"][symbol] = current_holdings - shares_to_sell
            execution_result = f"EXECUTED: Sold {shares_to_sell} shares of {symbol} at ${current_price:.2f} for ${proceeds:,.2f} (confidence: {confidence:.2f}, {shares_to_sell/current_holdings*100:.0f}% of holdings)"
        else:
            execution_result = f"SKIPPED: No shares to sell for {symbol}"

    else:
        threshold = confidence_threshold if recommendation == 'BUY' else sell_threshold
        execution_result = f"SKIPPED: {recommendation} with confidence {confidence:.2f} (threshold: {threshold:.2f})"

    state["execution_result"] = execution_result
    state["portfolio"] = portfolio
    state["messages"] = state.get("messages", []) + [
        {"role": "executor", "content": execution_summary},
        {"role": "system", "content": execution_result}
    ]

    return state


def create_trading_workflow() -> StateGraph:
    """Create the LangGraph workflow for stock trading."""
    workflow = StateGraph(TradingState)

    # Add nodes
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("executor", executor_agent)

    # Add edges
    workflow.add_edge("analyzer", "executor")
    workflow.add_edge("executor", END)

    # Set entry point
    workflow.set_entry_point("analyzer")

    return workflow.compile()


if __name__ == "__main__":
    # Test the workflow with tools
    print("Testing trading workflow with tools...")

    from stock_data import generate_stock_data

    # Generate sample stock data
    df = generate_stock_data(num_stocks=10, num_days=50, seed=42)

    initial_state = {
        "symbol": "STOCK000",
        "stock_data": df,
        "portfolio": {"cash": 100000.0, "holdings": {}},
        "risk_level": "aggressive",
        "messages": []
    }

    app = create_trading_workflow()
    print("Running workflow with tool-calling agents...")
    result = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"Analysis for {result['symbol']}")
    print(f"{'='*60}")
    print(f"\nAnalyst Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nAnalysis:\n{result['analysis']}")
    print(f"\nExecution Result:\n{result['execution_result']}")
    print(f"\nPortfolio Cash: ${result['portfolio']['cash']:,.2f}")
    print(f"Holdings: {result['portfolio']['holdings']}")
    print(f"\n✓ Test complete! Agents successfully used tools to calculate metrics.")
