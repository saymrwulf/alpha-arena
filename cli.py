#!/usr/bin/env python3
"""CLI for Polymarket Trading Harness."""

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.base import AgentConfig
from src.agent.llm import LLMAgent
from src.agent.selector import AgentSelector
from src.broker.polymarket import PolymarketBroker
from src.data.market import MarketData
from src.data.sentiment import TwitterSentiment
from src.metrics.logger import MetricsLogger
from src.risk.controls import RiskConfig, RiskManager
from src.runner.loop import RunnerConfig, TradingRunner
from src.wallet.polygon import PolygonWallet

app = typer.Typer(help="Polymarket Trading Harness - Autonomous prediction market trading")
console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
        return {}

    with open(path) as f:
        return yaml.safe_load(f)


@app.command()
def run(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without executing trades"),
    max_iterations: int = typer.Option(None, "--max-iter", "-n", help="Max iterations (default: unlimited)"),
    interval: int = typer.Option(None, "--interval", "-i", help="Loop interval seconds"),
):
    """Start the live trading loop."""
    load_dotenv()
    config = load_config(config_file)

    console.print("[bold green]Polymarket Trading Harness[/bold green]")
    console.print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # Validate required env vars
    required_vars = ["WALLET_PRIVATE_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        console.print(f"[red]Missing environment variables: {', '.join(missing)}[/red]")
        console.print("Copy .env.example to .env and fill in your credentials.")
        raise typer.Exit(1)

    asyncio.run(_run_trading(config, dry_run, max_iterations, interval))


async def _run_trading(
    config: dict,
    dry_run: bool,
    max_iterations: int | None,
    interval: int | None,
):
    """Async trading loop."""
    # Initialize components
    broker = PolymarketBroker(
        private_key=os.environ["WALLET_PRIVATE_KEY"],
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
    )

    llm_config = config.get("llm", {})
    agent_config = AgentConfig(
        name="main",
        loop_interval_seconds=interval or config.get("agent", {}).get("loop_interval_seconds", 60),
        llm_provider=llm_config.get("default_provider", "anthropic"),
        llm_model=llm_config.get("default_model", "claude-sonnet-4-20250514"),
        temperature=llm_config.get("providers", {}).get("anthropic", {}).get("temperature", 0.3),
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY") if agent_config.llm_provider == "anthropic" else os.environ.get("OPENAI_API_KEY")
    agent = LLMAgent(agent_config, api_key)

    market_data = MarketData(cache_ttl_seconds=config.get("data", {}).get("polymarket_refresh_seconds", 30))

    # Sentiment provider (optional)
    sentiment_provider = None
    if twitter_token := os.getenv("TWITTER_BEARER_TOKEN"):
        sentiment_provider = TwitterSentiment(twitter_token)

    # Risk config
    risk_cfg = config.get("risk", {})
    risk_config = RiskConfig(
        max_position_size_usdc=Decimal(str(risk_cfg.get("max_position_size_usdc", 100))),
        max_single_trade_usdc=Decimal(str(risk_cfg.get("max_single_trade_usdc", 25))),
        max_open_positions=risk_cfg.get("max_open_positions", 5),
        daily_loss_limit_usdc=Decimal(str(risk_cfg.get("daily_loss_limit_usdc", 50))),
        max_orders_per_minute=risk_cfg.get("rate_limit_orders_per_minute", 10),
        kill_switch=risk_cfg.get("kill_switch", False) or dry_run,
    )
    risk_manager = RiskManager(risk_config)

    if dry_run:
        console.print("[yellow]Kill switch active (dry run mode)[/yellow]")

    # Metrics
    logging_cfg = config.get("logging", {})
    metrics = MetricsLogger(
        jsonl_path=logging_cfg.get("jsonl_file", "logs/decisions.jsonl"),
        sqlite_path=logging_cfg.get("sqlite_file", "logs/metrics.db"),
    )

    # Runner config
    markets_cfg = config.get("markets", {})
    runner_config = RunnerConfig(
        loop_interval_seconds=interval or config.get("agent", {}).get("loop_interval_seconds", 60),
        max_iterations=max_iterations or config.get("agent", {}).get("max_iterations"),
        market_refresh_seconds=config.get("data", {}).get("polymarket_refresh_seconds", 30),
        sentiment_refresh_seconds=config.get("data", {}).get("sentiment_refresh_seconds", 300),
        market_categories=markets_cfg.get("categories", []),
        min_liquidity=Decimal(str(markets_cfg.get("min_liquidity_usdc", 1000))),
        min_volume_24h=Decimal(str(markets_cfg.get("min_volume_24h_usdc", 500))),
    )

    runner = TradingRunner(
        broker=broker,
        agent=agent,
        market_data=market_data,
        sentiment_provider=sentiment_provider,
        risk_manager=risk_manager,
        metrics_logger=metrics,
        config=runner_config,
    )

    # Connect all components
    console.print("Connecting to services...")
    try:
        await broker.connect()
        await agent.connect()
        await market_data.connect()
        if sentiment_provider:
            await sentiment_provider.connect()
        await metrics.connect()

        # Show initial state
        balance = await broker.get_balance()
        positions = await broker.get_positions()
        console.print(f"Balance: [green]${balance:.2f}[/green] USDC")
        console.print(f"Open positions: {len(positions)}")

        # Start trading loop
        await runner.start()

    finally:
        console.print("Disconnecting...")
        await metrics.disconnect()
        if sentiment_provider:
            await sentiment_provider.disconnect()
        await market_data.disconnect()
        await agent.disconnect()
        await broker.disconnect()


@app.command()
def status(
    config_file: str = typer.Option("config.yaml", "--config", "-c"),
):
    """Show current trading status."""
    load_dotenv()

    if not os.getenv("WALLET_PRIVATE_KEY"):
        console.print("[red]WALLET_PRIVATE_KEY not set[/red]")
        raise typer.Exit(1)

    asyncio.run(_show_status())


async def _show_status():
    """Show account status."""
    broker = PolymarketBroker(
        private_key=os.environ["WALLET_PRIVATE_KEY"],
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
    )

    await broker.connect()

    try:
        balance = await broker.get_balance()
        positions = await broker.get_positions()
        orders = await broker.get_open_orders()

        console.print("\n[bold]Account Status[/bold]")
        console.print(f"Balance: [green]${balance:.2f}[/green] USDC")

        if positions:
            table = Table(title="Open Positions")
            table.add_column("Market")
            table.add_column("Side")
            table.add_column("Size")
            table.add_column("Entry")
            table.add_column("Current")
            table.add_column("PnL")

            for p in positions:
                pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
                table.add_row(
                    p.market_id[:12] + "...",
                    p.outcome,
                    f"{p.size:.2f}",
                    f"${p.avg_entry_price:.3f}",
                    f"${p.current_price:.3f}",
                    f"[{pnl_color}]${p.unrealized_pnl:.2f}[/{pnl_color}]",
                )
            console.print(table)
        else:
            console.print("No open positions")

        if orders:
            console.print(f"\nOpen orders: {len(orders)}")

    finally:
        await broker.disconnect()


@app.command()
def markets(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of markets to show"),
    category: str = typer.Option(None, "--category", "-cat", help="Filter by category"),
):
    """List active Polymarket markets."""
    asyncio.run(_list_markets(limit, category))


async def _list_markets(limit: int, category: str | None):
    """List markets."""
    market_data = MarketData()
    await market_data.connect()

    try:
        categories = [category] if category else None
        markets = await market_data.get_markets(
            active_only=True,
            categories=categories,
            limit=limit,
        )

        table = Table(title=f"Active Markets ({len(markets)})")
        table.add_column("Question", max_width=50)
        table.add_column("Category")
        table.add_column("Volume 24h")
        table.add_column("Liquidity")
        table.add_column("YES Price")

        for m in markets[:limit]:
            yes_price = m.yes_token.price if m.yes_token else Decimal("0")
            table.add_row(
                m.question[:50] + ("..." if len(m.question) > 50 else ""),
                m.category,
                f"${m.volume_24h:,.0f}",
                f"${m.liquidity:,.0f}",
                f"{yes_price:.2f}",
            )

        console.print(table)

    finally:
        await market_data.disconnect()


@app.command()
def compare_models(
    config_file: str = typer.Option("config.yaml", "--config", "-c"),
):
    """Compare different LLM models on sample data."""
    load_dotenv()
    config = load_config(config_file)

    api_keys = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
    }

    if not api_keys["anthropic"] and not api_keys["openai"]:
        console.print("[red]No API keys configured[/red]")
        raise typer.Exit(1)

    asyncio.run(_compare_models(api_keys, config))


async def _compare_models(api_keys: dict, config: dict):
    """Run model comparison."""
    from datetime import datetime
    from src.agent.base import AgentObservation
    from src.broker.base import Position

    selector = AgentSelector(api_keys)

    # Add models to compare
    if api_keys.get("anthropic"):
        selector.add_model("claude-sonnet", "anthropic", "claude-sonnet-4-20250514")
        selector.add_model("claude-haiku", "anthropic", "claude-3-5-haiku-20241022")

    if api_keys.get("openai"):
        selector.add_model("gpt-4o", "openai", "gpt-4o")
        selector.add_model("gpt-4o-mini", "openai", "gpt-4o-mini")

    await selector.connect()

    try:
        # Fetch real market data for comparison
        market_data = MarketData()
        await market_data.connect()

        markets = await market_data.get_markets(active_only=True, limit=10)
        await market_data.disconnect()

        # Create mock observation with real data
        observation = AgentObservation(
            timestamp=datetime.utcnow(),
            balance=Decimal("1000"),
            positions=[],
            markets=markets,
            sentiment={},
            open_orders=[],
        )

        console.print("\n[bold]Running model comparison...[/bold]\n")
        comparison = await selector.compare_models(observation)

        # Display results
        table = Table(title="Model Comparison Results")
        table.add_column("Model")
        table.add_column("Latency")
        table.add_column("Tokens")
        table.add_column("Cost")
        table.add_column("Signals")
        table.add_column("Avg Conf")
        table.add_column("Reasoning")

        for model_name, (decision, metrics) in comparison.results.items():
            table.add_row(
                model_name,
                f"{metrics.latency_ms}ms",
                str(metrics.tokens_used),
                f"${metrics.cost_estimate:.4f}",
                str(metrics.signal_count),
                f"{metrics.avg_confidence:.2f}",
                f"{metrics.reasoning_quality}/5",
            )

        console.print(table)
        console.print(f"\n[bold green]Recommended:[/bold green] {comparison.recommended_model}")
        console.print(f"Reason: {comparison.recommendation_reasoning}")

    finally:
        await selector.disconnect()


@app.command()
def wallet():
    """Show wallet information."""
    load_dotenv()

    if not os.getenv("WALLET_PRIVATE_KEY"):
        console.print("[red]WALLET_PRIVATE_KEY not set[/red]")
        raise typer.Exit(1)

    asyncio.run(_show_wallet())


async def _show_wallet():
    """Show wallet info."""
    wallet = PolygonWallet(
        private_key=os.environ["WALLET_PRIVATE_KEY"],
    )

    await wallet.connect()

    try:
        info = await wallet.get_info()

        console.print("\n[bold]Wallet Information[/bold]")
        console.print(f"Address: {info.address}")
        console.print(f"USDC Balance: [green]${info.usdc_balance:.2f}[/green]")
        console.print(f"MATIC Balance: {info.matic_balance:.4f}")
        console.print(f"USDC Allowance (CTF): ${info.usdc_allowance:.2f}")

        if info.matic_balance < Decimal("0.01"):
            console.print("[yellow]Warning: Low MATIC for gas[/yellow]")

    finally:
        await wallet.disconnect()


@app.command()
def stats():
    """Show trading statistics from logs."""
    asyncio.run(_show_stats())


async def _show_stats():
    """Show statistics."""
    metrics = MetricsLogger()
    await metrics.connect()

    try:
        stats = await metrics.get_statistics()

        if not stats:
            console.print("No trading data yet")
            return

        console.print("\n[bold]Trading Statistics[/bold]")
        console.print(f"Total decisions: {stats.get('total_decisions', 0)}")
        console.print(f"Total trades: {stats.get('total_trades', 0)}")
        console.print(f"Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms")
        console.print(f"Total tokens: {stats.get('total_tokens', 0):,}")

        if stats.get("current_equity"):
            console.print(f"\nCurrent equity: ${stats['current_equity']:.2f}")
            console.print(f"Realized PnL: ${stats.get('realized_pnl', 0):.2f}")
            console.print(f"Unrealized PnL: ${stats.get('unrealized_pnl', 0):.2f}")
            console.print(f"Max drawdown: {stats.get('max_drawdown_pct', 0):.1%}")

    finally:
        await metrics.disconnect()


@app.command()
def run_enhanced(
    config_file: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without executing trades"),
    max_iterations: int = typer.Option(None, "--max-iter", "-n", help="Max iterations"),
    interval: int = typer.Option(60, "--interval", "-i", help="Loop interval seconds"),
):
    """Start the enhanced trading loop with multi-agent coordination."""
    load_dotenv()
    config = load_config(config_file)

    console.print("[bold cyan]Enhanced Multi-Agent Trading Harness[/bold cyan]")
    console.print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # Validate required env vars
    required_vars = ["WALLET_PRIVATE_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        console.print(f"[red]Missing environment variables: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    asyncio.run(_run_enhanced_trading(config, dry_run, max_iterations, interval))


async def _run_enhanced_trading(
    config: dict,
    dry_run: bool,
    max_iterations: int | None,
    interval: int,
):
    """Run enhanced trading with multi-agent coordination."""
    from src.runner.enhanced import EnhancedTradingRunner, EnhancedRunnerConfig, create_enhanced_runner
    from src.core.config import LLMProviderConfig

    # Initialize broker
    broker = PolymarketBroker(
        private_key=os.environ["WALLET_PRIVATE_KEY"],
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
    )

    market_data = MarketData(cache_ttl_seconds=config.get("data", {}).get("polymarket_refresh_seconds", 30))

    # LLM configs
    llm_configs = {}
    if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
        llm_configs["anthropic"] = LLMProviderConfig(
            provider="anthropic",
            api_key=anthropic_key,
            models=["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022"],
            default_model="claude-sonnet-4-20250514",
        )

    if openai_key := os.getenv("OPENAI_API_KEY"):
        llm_configs["openai"] = LLMProviderConfig(
            provider="openai",
            api_key=openai_key,
            models=["gpt-4o", "gpt-4o-mini"],
            default_model="gpt-4o",
        )

    if xai_key := os.getenv("XAI_API_KEY"):
        llm_configs["xai"] = LLMProviderConfig(
            provider="xai",
            api_key=xai_key,
            models=["grok-2-latest", "grok-3-latest"],
            default_model="grok-2-latest",
        )

    # Enhanced runner config
    runner_config = EnhancedRunnerConfig(
        loop_interval_seconds=interval,
        max_iterations=max_iterations,
        enable_indicators=True,
        enable_arbitrage=True,
        enable_reflection=True,
        enable_memory=True,
    )

    if dry_run:
        console.print("[yellow]Dry run mode - no trades will be executed[/yellow]")

    try:
        await broker.connect()
        await market_data.connect()

        # Create enhanced runner
        runner = await create_enhanced_runner(
            broker=broker,
            market_data=market_data,
            llm_config=llm_configs,
            config=runner_config,
        )

        # Show initial state
        balance = await broker.get_balance()
        console.print(f"Balance: [green]${balance:.2f}[/green] USDC")

        # Start enhanced trading loop
        await runner.start()

    finally:
        await market_data.disconnect()
        await broker.disconnect()


@app.command()
def backtest(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to backtest"),
    capital: float = typer.Option(10000, "--capital", "-c", help="Starting capital"),
    buy_threshold: float = typer.Option(0.40, "--buy", help="Buy threshold"),
    sell_threshold: float = typer.Option(0.60, "--sell", help="Sell threshold"),
):
    """Run backtesting on synthetic or historical data."""
    console.print("[bold cyan]Backtesting Framework[/bold cyan]")
    asyncio.run(_run_backtest(days, Decimal(str(capital)), Decimal(str(buy_threshold)), Decimal(str(sell_threshold))))


async def _run_backtest(days: int, capital: Decimal, buy_threshold: Decimal, sell_threshold: Decimal):
    """Run backtest."""
    from src.backtest.engine import BacktestEngine, BacktestConfig, SimpleStrategy
    from src.backtest.data import HistoricalDataLoader

    console.print(f"Generating {days} days of synthetic data...")

    loader = HistoricalDataLoader()
    market = loader.generate_synthetic(
        market_id="synthetic_test",
        question="Will the synthetic event occur?",
        days=days,
        start_price=Decimal("0.50"),
        volatility=Decimal("0.02"),
        outcome="yes",
    )

    console.print(f"Data points: {len(market.snapshots)}")

    # Create strategy
    strategy = SimpleStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        position_size=capital * Decimal("0.1"),  # 10% per trade
    )

    # Create engine
    config = BacktestConfig(
        starting_capital=capital,
        fee_rate=Decimal("0.001"),
        slippage_pct=Decimal("0.001"),
    )
    engine = BacktestEngine(config)

    console.print("Running backtest...")

    def on_progress(current, total):
        if current % 500 == 0:
            pct = current / total * 100
            console.print(f"  Progress: {pct:.1f}%")

    result = await engine.run(strategy, [market], progress_callback=on_progress)

    # Show results
    console.print(result.summary())


@app.command()
def arbitrage():
    """Scan for cross-platform arbitrage opportunities."""
    load_dotenv()
    console.print("[bold cyan]Arbitrage Scanner[/bold cyan]")
    asyncio.run(_scan_arbitrage())


async def _scan_arbitrage():
    """Scan for arbitrage."""
    from src.arbitrage.detector import ArbitrageDetector
    from src.arbitrage.platforms import PolymarketPlatform

    platforms = [PolymarketPlatform()]

    detector = ArbitrageDetector(platforms)

    # Connect platforms
    console.print("Connecting to platforms...")
    for platform in platforms:
        await platform.connect()

    try:
        console.print("Fetching markets...")
        await detector.refresh_markets()

        console.print("Scanning for arbitrage opportunities...")
        opportunities = await detector.detect_all()

        if not opportunities:
            console.print("[yellow]No arbitrage opportunities found[/yellow]")
            return

        table = Table(title=f"Arbitrage Opportunities ({len(opportunities)})")
        table.add_column("Type")
        table.add_column("Market")
        table.add_column("Gross Profit")
        table.add_column("Net Profit")
        table.add_column("Risk")

        for opp in opportunities[:20]:
            if opp.net_profit_pct > 0:
                profit_color = "green"
            else:
                profit_color = "red"

            market_name = opp.market_a.question[:30] + "..." if len(opp.market_a.question) > 30 else opp.market_a.question

            table.add_row(
                opp.type,
                market_name,
                f"{opp.gross_profit_pct:.2f}%",
                f"[{profit_color}]{opp.net_profit_pct:.2f}%[/{profit_color}]",
                opp.execution_risk,
            )

        console.print(table)

        profitable = [o for o in opportunities if o.is_profitable()]
        console.print(f"\n[green]Profitable opportunities: {len(profitable)}[/green]")

    finally:
        for platform in platforms:
            await platform.disconnect()


@app.command()
def indicators(
    market_id: str = typer.Argument(None, help="Market ID to analyze"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of markets to show"),
):
    """Show technical indicators for markets."""
    console.print("[bold cyan]Technical Indicators[/bold cyan]")
    asyncio.run(_show_indicators(market_id, limit))


async def _show_indicators(market_id: str | None, limit: int):
    """Show technical indicators."""
    from src.core.config import IndicatorConfig
    from src.indicators.calculator import IndicatorCalculator

    market_data = MarketData()
    await market_data.connect()

    try:
        markets = await market_data.get_markets(active_only=True, limit=limit)

        if market_id:
            markets = [m for m in markets if m.condition_id == market_id]

        if not markets:
            console.print("[yellow]No markets found[/yellow]")
            return

        # Create indicator calculator
        config = IndicatorConfig()
        calculator = IndicatorCalculator(config)

        table = Table(title="Market Indicators (simulated - need historical data)")
        table.add_column("Market", max_width=35)
        table.add_column("YES Price")
        table.add_column("Volume 24h")
        table.add_column("Liquidity")

        for m in markets:
            yes_price = m.yes_token.price if m.yes_token else Decimal("0")
            table.add_row(
                m.question[:35] + ("..." if len(m.question) > 35 else ""),
                f"{yes_price:.3f}",
                f"${m.volume_24h:,.0f}",
                f"${m.liquidity:,.0f}",
            )

        console.print(table)
        console.print("\n[dim]Note: Full indicators require historical price data[/dim]")

    finally:
        await market_data.disconnect()


@app.command()
def providers():
    """List configured LLM providers."""
    load_dotenv()

    console.print("[bold cyan]LLM Providers[/bold cyan]\n")

    providers_info = [
        ("Anthropic", "ANTHROPIC_API_KEY", ["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022", "claude-opus-4-20250514"]),
        ("OpenAI", "OPENAI_API_KEY", ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]),
        ("xAI (Grok)", "XAI_API_KEY", ["grok-2-latest", "grok-3-latest"]),
        ("Local (Ollama)", None, ["deepseek-v3", "qwen2.5", "llama3.3"]),
    ]

    table = Table(title="Available LLM Providers")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Models")

    for name, env_var, models in providers_info:
        if env_var:
            status = "[green]Configured[/green]" if os.getenv(env_var) else "[red]Not configured[/red]"
        else:
            status = "[yellow]Local[/yellow]"

        table.add_row(
            name,
            status,
            ", ".join(models),
        )

    console.print(table)

    console.print("\n[dim]Configure providers by setting API keys in .env file[/dim]")


@app.command()
def analyze_wallet(
    address: str = typer.Argument(..., help="Wallet address to analyze"),
    output: str = typer.Option("cli", "--output", "-o", help="Output format: cli, json, html"),
    save: str = typer.Option(None, "--save", "-s", help="Save output to file"),
):
    """Analyze a Polymarket wallet's trading history and strategy."""
    console.print("[bold cyan]Polymarket Wallet Analysis Dashboard[/bold cyan]")
    console.print(f"Analyzing wallet: {address}\n")
    asyncio.run(_analyze_wallet(address, output, save))


async def _analyze_wallet(address: str, output_format: str, save_path: str | None):
    """Run wallet analysis."""
    from src.analysis.dashboard import WalletDashboard

    dashboard = WalletDashboard()

    try:
        console.print("Fetching wallet history...")
        data = await dashboard.analyze(address)

        console.print(f"Found {data.summary.total_transactions} transactions")
        console.print(f"Analyzing {len(data.summary.positions)} positions...\n")

        # Render output
        if output_format == "json":
            result = dashboard.render_json(data)
        elif output_format == "html":
            result = dashboard.render_html(data)
        else:
            result = dashboard.render_cli(data)

        # Display or save
        if save_path:
            with open(save_path, "w") as f:
                f.write(result)
            console.print(f"[green]Saved to {save_path}[/green]")
        else:
            if output_format == "cli":
                console.print(result)
            else:
                print(result)

    finally:
        await dashboard.close()


@app.command()
def compare_wallets(
    addresses: str = typer.Argument(..., help="Comma-separated wallet addresses"),
    output: str = typer.Option("cli", "--output", "-o", help="Output format: cli, json"),
):
    """Compare multiple Polymarket wallets."""
    addr_list = [a.strip() for a in addresses.split(",")]
    console.print(f"[bold cyan]Comparing {len(addr_list)} wallets[/bold cyan]\n")
    asyncio.run(_compare_wallets(addr_list, output))


async def _compare_wallets(addresses: list[str], output_format: str):
    """Compare multiple wallets."""
    from src.analysis.wallet import WalletAnalyzer
    from src.analysis.strategy import StrategyDetector
    from src.analysis.performance import PerformanceAnalyzer

    analyzer = WalletAnalyzer()
    strategy_detector = StrategyDetector()
    performance_analyzer = PerformanceAnalyzer()

    try:
        results = await analyzer.compare_wallets(addresses)

        if output_format == "json":
            import json
            output = {}
            for addr, summary in results.items():
                strategy = strategy_detector.detect_strategy(summary)
                metrics = performance_analyzer.analyze(summary)
                output[addr] = {
                    "total_pnl": float(summary.total_realized_pnl),
                    "win_rate": float(summary.win_rate),
                    "trades": summary.total_transactions,
                    "strategy": strategy.primary_strategy.value,
                    "risk_profile": strategy.risk_profile.value,
                    "sharpe": float(metrics.sharpe_ratio),
                }
            print(json.dumps(output, indent=2))
        else:
            table = Table(title="Wallet Comparison")
            table.add_column("Address")
            table.add_column("P&L")
            table.add_column("Win Rate")
            table.add_column("Trades")
            table.add_column("Strategy")
            table.add_column("Risk")
            table.add_column("Sharpe")

            for addr, summary in results.items():
                strategy = strategy_detector.detect_strategy(summary)
                metrics = performance_analyzer.analyze(summary)

                pnl_color = "green" if summary.total_realized_pnl >= 0 else "red"

                table.add_row(
                    addr[:10] + "...",
                    f"[{pnl_color}]${summary.total_realized_pnl:+,.2f}[/{pnl_color}]",
                    f"{summary.win_rate:.1f}%",
                    str(summary.total_transactions),
                    strategy.primary_strategy.value,
                    strategy.risk_profile.value,
                    f"{metrics.sharpe_ratio:.2f}",
                )

            console.print(table)

    finally:
        await analyzer.close()


@app.command()
def wallet_report(
    address: str = typer.Argument(..., help="Wallet address"),
    output_file: str = typer.Option("wallet_report.html", "--output", "-o", help="Output file"),
):
    """Generate a detailed HTML report for a wallet."""
    console.print("[bold cyan]Generating Wallet Report[/bold cyan]")
    asyncio.run(_generate_report(address, output_file))


async def _generate_report(address: str, output_file: str):
    """Generate HTML report."""
    from src.analysis.dashboard import WalletDashboard

    dashboard = WalletDashboard()

    try:
        console.print(f"Analyzing {address}...")
        data = await dashboard.analyze(address)

        console.print("Generating report...")
        html = dashboard.render_html(data)

        with open(output_file, "w") as f:
            f.write(html)

        console.print(f"[green]Report saved to {output_file}[/green]")
        console.print(f"Open in browser: file://{Path(output_file).absolute()}")

    finally:
        await dashboard.close()


@app.command()
def leaderboard(
    addresses_file: str = typer.Argument(None, help="File with wallet addresses (one per line)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of wallets to show"),
):
    """Show wallet leaderboard by P&L (requires address list)."""
    if addresses_file:
        with open(addresses_file) as f:
            addresses = [line.strip() for line in f if line.strip()]
    else:
        console.print("[yellow]Provide a file with wallet addresses to analyze[/yellow]")
        console.print("Usage: python cli.py leaderboard wallets.txt")
        return

    console.print(f"[bold cyan]Wallet Leaderboard[/bold cyan]")
    console.print(f"Analyzing {len(addresses)} wallets...\n")
    asyncio.run(_show_leaderboard(addresses, limit))


async def _show_leaderboard(addresses: list[str], limit: int):
    """Show wallet leaderboard."""
    from src.analysis.wallet import WalletAnalyzer
    from src.analysis.strategy import StrategyDetector

    analyzer = WalletAnalyzer()
    detector = StrategyDetector()

    try:
        results = await analyzer.compare_wallets(addresses)

        # Sort by P&L
        sorted_wallets = sorted(
            results.items(),
            key=lambda x: x[1].total_realized_pnl,
            reverse=True
        )[:limit]

        table = Table(title=f"Top {limit} Wallets by P&L")
        table.add_column("Rank")
        table.add_column("Address")
        table.add_column("Total P&L")
        table.add_column("Win Rate")
        table.add_column("Trades")
        table.add_column("Volume")
        table.add_column("Strategy")

        for i, (addr, summary) in enumerate(sorted_wallets, 1):
            strategy = detector.detect_strategy(summary)
            pnl_color = "green" if summary.total_realized_pnl >= 0 else "red"

            table.add_row(
                f"#{i}",
                addr[:12] + "...",
                f"[{pnl_color}]${summary.total_realized_pnl:+,.2f}[/{pnl_color}]",
                f"{summary.win_rate:.1f}%",
                str(summary.total_transactions),
                f"${summary.total_volume:,.0f}",
                strategy.primary_strategy.value,
            )

        console.print(table)

    finally:
        await analyzer.close()


if __name__ == "__main__":
    app()
