# AI-Powered Stock Trading System with LangGraph

A **multi-agent stock trading system** built with **LangGraph** that demonstrates how to orchestrate AI agents for real-world decision-making tasks. Uses Claude AI to analyze stocks and execute trades with configurable risk management.

## Why LangGraph?

This project showcases **LangGraph**'s power for building stateful, multi-agent AI systems:

- **State Management**: `TradingState` flows through agents, accumulating data
- **Agent Orchestration**: Sequential workflow (Analyzer → Executor)
- **Type Safety**: TypedDict ensures consistent state structure
- **Modularity**: Independent agents that can be tested, swapped, or extended
- **Real-world Application**: Not just a demo - actually makes trading decisions
- **Tool Integration**: LangChain tools for on-demand metric calculation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [START] → [Analyzer Agent] → [Executor Agent] → [END] │
│                      ↓                  ↓               │
│              TradingState       TradingState            │
│              (enriched)         (final)                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Analyzer Agent
- **Role**: AI-powered stock analyst with tool-calling
- **Input**: Raw stock data (DataFrame)
- **Process**: Uses Claude AI with LangChain tools to calculate metrics on-demand
- **Tools**: `calculate_momentum`, `calculate_volatility`, `calculate_sma`, `get_current_price`
- **Output**: BUY/SELL/HOLD recommendation + confidence score (0.0-1.0)
- **Implementation**: [agents.py:141-265](agents.py#L141)

### Executor Agent
- **Role**: Risk-aware trade executor
- **Input**: Analyzer's recommendation + current portfolio
- **Process**: Applies risk parameters, calculates position sizing
- **Output**: Executes trade, updates portfolio
- **Implementation**: [agents.py:300-376](agents.py#L300)

### TradingState (Shared State)
```python
class TradingState(TypedDict):
    symbol: str              # Stock being analyzed
    stock_data: pd.DataFrame # Raw stock data (agents use tools to calculate metrics)
    analysis: str            # AI's full analysis text
    recommendation: str      # BUY/SELL/HOLD
    confidence: float        # 0.0-1.0 confidence score
    execution_result: str    # Trade execution summary
    portfolio: dict          # {cash: float, holdings: dict}
    messages: list           # Conversation history
    risk_level: str          # Risk configuration
```

**State flows through the workflow**, with each agent enriching it:
1. Initial state → Analyzer uses tools to calculate metrics on-demand
2. Enriched state → Analyzer adds `analysis`, `recommendation`, `confidence`
3. Executor state → Executor adds `execution_result`, updates `portfolio`
4. Final state → Contains complete audit trail

### LangChain Tools (On-Demand Calculation)

The analyzer agent uses **LangChain tools** to calculate metrics dynamically:

```python
@tool
def calculate_momentum(symbol: str, lookback: int = 20) -> float:
    """Calculate price momentum percentage for a stock."""
    # Accesses global stock data and calculates on-demand
    ...

@tool
def calculate_volatility(symbol: str, lookback: int = 20) -> float:
    """Calculate price volatility (standard deviation / mean)."""
    ...

@tool
def calculate_sma(symbol: str, period: int = 10) -> float:
    """Calculate Simple Moving Average."""
    ...

@tool
def get_current_price(symbol: str) -> float:
    """Get the current (most recent) price."""
    ...
```

**How it works**:
1. LLM receives prompt: "Analyze STOCK088 using available tools"
2. LLM decides to call tools: `get_current_price("STOCK088")`, `calculate_momentum("STOCK088")`, etc.
3. Tools execute and return results to LLM
4. LLM analyzes tool results and provides recommendation

**Benefits**:
- Metrics calculated only when needed (on-demand)
- LLM can choose which tools to use based on context
- Demonstrates proper LangChain/LangGraph tool-calling patterns
- Easy to add new analytical tools without changing state structure

## Key Features

**Multi-Agent LangGraph System** with stateful workflow
**LangChain Tool Integration** for on-demand metric calculation
**Claude AI Integration** for intelligent stock analysis
**5 Risk Levels** with dynamic position sizing
**4 Selection Strategies** (momentum, volatility, mixed, contrarian)
**50-day simulations** with smart stock selection
**Auto-generated visualizations** (4 professional charts)
**Complete audit trail** via message history

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
```

### 3. Run Simulation
```bash
python main.py
```

Default: **50-day simulation** with aggressive risk level and momentum strategy.

## Configuration

Edit [main.py](main.py#L329) to customize:

```python
run_daily_trading_simulation(
    num_days=50,           # Trading days to simulate
    stocks_per_day=10,     # Stocks analyzed per day
    risk_level="aggressive",  # Risk appetite
    selection_method="momentum",  # Stock selection strategy
    seed=42                # Random seed
)
```

### Risk Levels

| Level | Confidence Threshold | Position Size | Trading Style |
|-------|---------------------|---------------|---------------|
| `very_aggressive` | 50% | 25% | High risk/reward |
| `aggressive` | 60% | 15% | Active trading |
| `moderate` | 70% | 10% | Balanced |
| `conservative` | 75% | 7% | Cautious |
| `very_conservative` | 80% | 5% | Very safe |

### Selection Strategies

- **momentum**: Favor stocks with positive trends
- **volatility**: Target volatile stocks for opportunities
- **mixed**: Balanced multi-factor approach
- **contrarian**: Find oversold opportunities

## Simulation Duration

| Duration | API Calls | Time | Best For |
|----------|-----------|------|----------|
| 10 days | ~100 | 3-5 min | Quick test |
| 20 days | ~200 | 6-10 min | Demos |
| **50 days** | ~500 | **15-25 min** | **Recommended** |
| 100 days | ~1000 | 30-50 min | Full analysis |

## Output Files

All files saved to `results/` directory:

### Visualizations (PNG, 300 DPI)
1. **net_portfolio_value_*.png** - Large 16x8" chart with portfolio value over time
2. **portfolio_analysis_*.png** - 6-panel dashboard (value, cash/holdings, daily P/L, allocation, trades, confidence)
3. **cumulative_return_*.png** - Cumulative return percentage
4. **holdings_detail_*.png** - Individual stock holdings

### Data Files
- **daily_trading_*.json** - Complete simulation data
- **all_trades_*.csv** - Every trade executed
- **daily_summary_*.csv** - Daily portfolio snapshots

## How LangGraph Powers This System

### 1. **Workflow Definition** ([agents.py:233-248](agents.py#L233))

```python
from langgraph.graph import StateGraph, END

# Create workflow with TradingState
workflow = StateGraph(TradingState)

# Add agents as nodes
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("executor", executor_agent)

# Define execution flow
workflow.add_edge("analyzer", "executor")  # Sequential flow
workflow.add_edge("executor", END)

# Set entry point
workflow.set_entry_point("analyzer")

# Compile into executable graph
app = workflow.compile()
```

### 2. **Agent Implementation** (Python Functions)

```python
def analyzer_agent(state: TradingState) -> TradingState:
    """Analyzer agent enriches state with AI analysis."""
    llm = get_llm()  # Claude AI

    # Get AI analysis
    response = llm.invoke([
        SystemMessage("You are a stock analyst..."),
        HumanMessage(f"Analyze {state['symbol']}...")
    ])

    # Parse and enrich state
    state["analysis"] = response.content
    state["recommendation"] = parse_recommendation(response)
    state["confidence"] = parse_confidence(response)

    return state  # Pass enriched state to next agent
```

### 3. **State Flow Example**

```python
# Initial state
initial_state = {
    "symbol": "STOCK001",
    "metrics": {...},
    "portfolio": {"cash": 100000, "holdings": {}},
    "risk_level": "aggressive"
}

# Execute workflow (both agents run automatically)
final_state = app.invoke(initial_state)

# Final state contains everything
print(final_state["recommendation"])     # "BUY"
print(final_state["confidence"])         # 0.75
print(final_state["execution_result"])   # "EXECUTED: Bought 100 shares..."
print(final_state["portfolio"]["cash"])  # 85000.0
```

### 4. **Why This Architecture Works**

**Separation of Concerns**: Analyzer focuses on analysis, Executor on execution
**Type Safety**: TypedDict catches errors at design time
**Testability**: Each agent can be tested independently
**Extensibility**: Easy to add more agents (e.g., RiskManager, Portfolio Optimizer)
**Audit Trail**: `messages` list tracks every decision
**State Persistence**: Complete trading history in final state

## Project Structure

```
stock-prediction/
├── main.py                # Simulation orchestrator
├── agents.py              # LangGraph workflow + AI agents
├── stock_data.py          # Technical indicators
├── smart_selector.py      # Stock selection strategies
├── visualizations.py      # Chart generation
└── results/              # Output directory
```

**Core file**: [agents.py](agents.py) contains the entire LangGraph workflow (285 lines)

## Understanding the Charts

### Net Portfolio Value
- **Green zones**: Profit periods
- **Red zones**: Loss periods
- **Annotations**: Peak, Low, Final values
- **Statistics box**: Key metrics summary

### Portfolio Analysis Dashboard (6 panels)
1. **Portfolio Value**: Total value over time
2. **Cash vs Holdings**: Liquidity vs invested capital
3. **Daily P/L**: Day-to-day profit/loss bars
4. **Asset Allocation**: Portfolio composition pie chart
5. **Trade Activity**: Buy/sell volume per day
6. **Confidence Distribution**: Decision quality histogram

### Good Signs
Portfolio value trending upward
Balanced cash/holdings allocation
More green bars in daily P/L
High confidence trades (>0.7)

### Warning Signs
Portfolio value declining
All cash (not investing) or no cash (overexposed)
More red bars in daily P/L
Low confidence trades executing

## Examples

### Conservative Long-Term
```python
run_daily_trading_simulation(
    num_days=50,
    risk_level="conservative",
    selection_method="mixed"
)
```

### Aggressive Momentum Trading
```python
run_daily_trading_simulation(
    num_days=20,
    risk_level="very_aggressive",
    selection_method="momentum"
)
```

### Contrarian Strategy
```python
run_daily_trading_simulation(
    num_days=30,
    risk_level="moderate",
    selection_method="contrarian"
)
```

## Monitoring Progress

During simulation:
```bash
tail -f /tmp/simulation.log
```

## Troubleshooting

**Charts not generating?**
- Check matplotlib/seaborn installed: `pip install matplotlib seaborn`
- Verify simulation completed successfully

**Simulation too slow?**
- Reduce `num_days` (try 20 instead of 50)
- Reduce `stocks_per_day` (try 5 instead of 10)
- Use faster model in [agents.py](agents.py#L30)

**API errors?**
- Verify `.env` has correct `ANTHROPIC_API_KEY`
- Check API rate limits

## Advanced: Extending the LangGraph System

Thanks to LangGraph's modular design, you can easily extend this system:

### Add a Risk Manager Agent

```python
def risk_manager_agent(state: TradingState) -> TradingState:
    """Validates trades against risk limits."""
    portfolio = state["portfolio"]

    # Check portfolio concentration
    if is_too_concentrated(portfolio):
        state["recommendation"] = "HOLD"
        state["execution_result"] = "BLOCKED: Portfolio too concentrated"

    return state

# Add to workflow
workflow.add_node("risk_manager", risk_manager_agent)
workflow.add_edge("analyzer", "risk_manager")
workflow.add_edge("risk_manager", "executor")
```

### Add Parallel Analysis

```python
from langgraph.graph import START

# Add multiple analyzers
workflow.add_node("technical_analyzer", technical_analyzer_agent)
workflow.add_node("sentiment_analyzer", sentiment_analyzer_agent)

# Fan-out from START
workflow.add_edge(START, "technical_analyzer")
workflow.add_edge(START, "sentiment_analyzer")

# Fan-in to aggregator
workflow.add_edge("technical_analyzer", "aggregator")
workflow.add_edge("sentiment_analyzer", "aggregator")
workflow.add_edge("aggregator", "executor")
```

### State Persistence

```python
# Save state at each step
def save_state_hook(state: TradingState):
    with open(f"states/{state['symbol']}.json", "w") as f:
        json.dump(state, f)

# Add hooks to workflow
workflow.add_hook("after_analyzer", save_state_hook)
```

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Orchestration** | LangGraph | Multi-agent workflow management |
| **LLM** | Claude 3.5 Haiku | Stock analysis and recommendations |
| **State Management** | TypedDict | Type-safe state across agents |
| **LLM Framework** | LangChain | LLM integration layer |
| **Visualization** | Matplotlib/Seaborn | Chart generation |
| **Data** | Pandas/NumPy | Technical indicator calculation |

## Learning Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Multi-Agent Systems**: This project demonstrates core patterns
- **State Management**: See `TradingState` in [agents.py:14-24](agents.py#L14)
- **Agent Design**: See analyzer/executor implementations in [agents.py](agents.py)

## Why This Project?

This isn't just a stock trading simulator - it's a **production-ready template** for building multi-agent AI systems with LangGraph. The patterns here apply to:

- **Customer Support**: routing → specialist → resolution
- **Content Moderation**: analyze → classify → action
- **Data Pipelines**: extract → transform → load
- **Decision Support**: gather → analyze → recommend

**The key insight**: Break complex tasks into specialized agents with shared state.