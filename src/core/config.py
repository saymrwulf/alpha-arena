"""Configuration management for the trading harness."""

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    enabled: bool = True
    models: list[str] = []
    default_model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key_env: str = ""  # Environment variable name
    base_url: str | None = None  # For local models


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size_usdc: Decimal = Decimal("100")
    max_single_trade_usdc: Decimal = Decimal("25")
    max_portfolio_risk_pct: Decimal = Decimal("20")  # Max % of portfolio at risk
    max_open_positions: int = 5
    max_correlation_exposure: Decimal = Decimal("0.5")  # Max correlated positions
    daily_loss_limit_usdc: Decimal = Decimal("50")
    max_drawdown_pct: Decimal = Decimal("15")
    max_orders_per_minute: int = 10
    kelly_fraction: Decimal = Decimal("0.25")  # Use quarter Kelly
    min_edge_threshold: Decimal = Decimal("0.03")  # 3% minimum edge
    min_confidence: Decimal = Decimal("0.6")
    kill_switch: bool = False


class AgentConfig(BaseModel):
    """Configuration for multi-agent system."""
    research_agent_model: str = "claude-sonnet-4-20250514"
    risk_agent_model: str = "claude-sonnet-4-20250514"
    execution_agent_model: str = "gpt-4o"
    reflection_agent_model: str = "claude-sonnet-4-20250514"
    sentiment_agent_model: str = "grok-3"  # Best for X sentiment

    enable_multi_agent_debate: bool = True
    debate_rounds: int = 2
    consensus_threshold: Decimal = Decimal("0.7")  # Agreement needed

    enable_reflection: bool = True
    reflection_interval_trades: int = 10  # Reflect every N trades

    loop_interval_seconds: int = 60
    max_iterations: int | None = None


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    short_term_window: int = 50  # Recent observations
    working_memory_size: int = 20  # Current context
    long_term_capacity: int = 10000  # Historical patterns

    enable_episodic_memory: bool = True
    enable_semantic_memory: bool = True
    enable_reflection_memory: bool = True

    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: Decimal = Decimal("0.8")


class DataConfig(BaseModel):
    """Data source configuration."""
    polymarket_refresh_seconds: int = 30
    orderbook_refresh_seconds: int = 5
    sentiment_refresh_seconds: int = 120
    news_refresh_seconds: int = 60

    # Platforms for arbitrage
    enable_kalshi: bool = False
    enable_predictit: bool = False

    # Sentiment sources
    enable_twitter: bool = True
    enable_news: bool = True
    enable_reddit: bool = False


class IndicatorConfig(BaseModel):
    """Technical indicator configuration."""
    ema_periods: list[int] = [9, 21, 50]
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_ma_period: int = 20
    atr_period: int = 14


class Config(BaseModel):
    """Main configuration container."""
    # Mode
    mode: str = "live"  # live, paper, backtest

    # LLM Providers
    llm_providers: dict[str, LLMProviderConfig] = Field(default_factory=lambda: {
        "anthropic": LLMProviderConfig(
            models=["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
            default_model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "openai": LLMProviderConfig(
            models=["gpt-4o", "gpt-4o-mini", "o1-preview"],
            default_model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
        ),
        "xai": LLMProviderConfig(
            models=["grok-3", "grok-3-mini"],
            default_model="grok-3",
            api_key_env="XAI_API_KEY",
        ),
        "google": LLMProviderConfig(
            models=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            default_model="gemini-2.0-flash",
            api_key_env="GOOGLE_API_KEY",
        ),
        "local": LLMProviderConfig(
            enabled=False,
            models=["deepseek-r1:70b", "qwen2.5:72b", "llama3.3:70b", "kimi-k2"],
            default_model="deepseek-r1:70b",
            base_url="http://localhost:11434/v1",  # Ollama default, change to :1234 for LM Studio
        ),
    })

    # Subsystem configs
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)

    # Market filters
    market_categories: list[str] = []
    min_liquidity_usdc: Decimal = Decimal("5000")
    min_volume_24h_usdc: Decimal = Decimal("1000")

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    metrics_db: str = "logs/metrics.db"
    decisions_jsonl: str = "logs/decisions.jsonl"
    trades_jsonl: str = "logs/trades.jsonl"
    reflections_jsonl: str = "logs/reflections.jsonl"


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_path = Path(path)

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)

    return Config()


def save_config(config: Config, path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
