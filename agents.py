"""AI Agents for stock trading analysis and execution."""
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

load_dotenv()


# State definition for the graph
class TradingState(TypedDict):
    """State for the trading workflow."""
    symbol: str
    metrics: dict
    analysis: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    execution_result: str
    portfolio: dict
    messages: list
    risk_level: str  # "very_aggressive", "aggressive", "moderate", "conservative", "very_conservative"


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
    Analyzer agent that evaluates stock metrics and provides buy/sell recommendations.
    """
    llm = get_llm()

    symbol = state["symbol"]
    metrics = state["metrics"]

    system_prompt = """You are an expert stock market analyst. Your job is to analyze stock metrics and provide clear buy/sell/hold recommendations.

Consider the following factors:
- Current price vs average price
- Price momentum (positive or negative trend)
- Volatility (higher volatility = higher risk)
- Price vs 10-day Simple Moving Average (SMA)

IMPORTANT: You MUST end your response with these exact lines:
Recommendation: [BUY/SELL/HOLD]
Confidence: [0.0-1.0]

Be concise and actionable."""

    user_prompt = f"""Analyze the following stock:

Symbol: {symbol}
Current Price: ${metrics.get('current_price', 'N/A')}
Average Price (20 days): ${metrics.get('average_price', 'N/A')}
Volatility: {metrics.get('volatility', 'N/A')}
Momentum: {metrics.get('momentum_pct', 'N/A')}%
10-day SMA: ${metrics.get('sma_10', 'N/A')}
Price vs SMA: {metrics.get('price_vs_sma', 'N/A')}%

Provide your analysis (2-3 sentences), then on separate lines:
Recommendation: [BUY/SELL/HOLD]
Confidence: [score between 0.0 and 1.0]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    analysis_text = response.content

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
    symbol = state["symbol"]
    recommendation = state["recommendation"]
    confidence = state["confidence"]
    metrics = state["metrics"]
    portfolio = state.get("portfolio", {"cash": 100000.0, "holdings": {}})
    risk_level = state.get("risk_level", "moderate")

    # Get risk parameters
    risk_params = get_risk_params(risk_level)
    confidence_threshold = risk_params["confidence_threshold"]
    max_position_size = risk_params["max_position_size"]
    sell_threshold = risk_params["sell_threshold"]

    current_price = metrics.get("current_price", 0)
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
    # Test the workflow
    print("Testing trading workflow...")

    # Sample metrics
    test_metrics = {
        'symbol': 'STOCK000',
        'current_price': 150.25,
        'average_price': 145.50,
        'volatility': 0.0234,
        'momentum_pct': 3.26,
        'sma_10': 148.75,
        'price_vs_sma': 1.01
    }

    initial_state = {
        "symbol": "STOCK000",
        "metrics": test_metrics,
        "portfolio": {"cash": 100000.0, "holdings": {}},
        "messages": []
    }

    app = create_trading_workflow()
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
