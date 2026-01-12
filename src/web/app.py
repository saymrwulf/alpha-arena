"""
Alpha Arena Master Control - FastAPI Web Application

Central control hub for all trading operations, monitoring, and analysis.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application state
class AppState:
    """Global application state."""
    def __init__(self):
        self.trading_active = False
        self.trading_mode = "simulation"  # simulation or live
        self.current_runner = None
        self.connected_websockets: list[WebSocket] = []
        self.config: dict = {}
        self.metrics: dict = {}
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_count = 0
        self.daily_pnl = Decimal("0")
        self.total_pnl = Decimal("0")
        self.alerts: list[dict] = []
        self.positions: list[dict] = []
        self.orders: list[dict] = []
        self.balance = Decimal("0")
        self.agent_status: dict = {}
        self.provider_registry = None  # Will be initialized on startup

state = AppState()

# Create FastAPI app
app = FastAPI(
    title="Alpha Arena Master Control",
    description="Polymarket Trading Harness Control Center",
    version="1.0.0"
)

# Templates directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Create directories if they don't exist
TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# Pydantic Models
# ============================================================================

class TradingConfig(BaseModel):
    mode: str = "simulation"
    loop_interval: int = 60
    max_iterations: Optional[int] = None
    enable_agents: bool = True
    enable_indicators: bool = True
    enable_arbitrage: bool = True

class RiskSettings(BaseModel):
    max_position_size: float = 100
    daily_loss_limit: float = 50
    max_open_positions: int = 5
    max_single_trade: float = 25
    kill_switch: bool = False

class WalletAnalysisRequest(BaseModel):
    address: str

class MarketFilter(BaseModel):
    categories: list[str] = []
    min_liquidity: float = 1000
    min_volume: float = 500
    limit: int = 50


# ============================================================================
# Helper Functions
# ============================================================================

def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

def save_config(config: dict):
    """Save configuration to config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

async def broadcast_update(event_type: str, data: dict):
    """Broadcast update to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now().isoformat()})
    disconnected = []
    for ws in state.connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.connected_websockets.remove(ws)

def format_decimal(value) -> str:
    """Format decimal for display."""
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    return f"{float(value):,.2f}"


# ============================================================================
# Template Filters
# ============================================================================

templates.env.filters["format_decimal"] = format_decimal
templates.env.globals["now"] = datetime.now


# ============================================================================
# WebSocket Handler
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates."""
    await websocket.accept()
    state.connected_websockets.append(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "data": {
                "trading_active": state.trading_active,
                "mode": state.trading_mode,
                "balance": str(state.balance),
                "daily_pnl": str(state.daily_pnl),
                "cycle_count": state.cycle_count,
            }
        })

        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        if websocket in state.connected_websockets:
            state.connected_websockets.remove(websocket)


# ============================================================================
# Main Pages
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    state.config = load_config()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "state": state,
        "config": state.config,
        "page": "dashboard"
    })

@app.get("/trading", response_class=HTMLResponse)
async def trading_control(request: Request):
    """Trading control panel."""
    return templates.TemplateResponse("trading.html", {
        "request": request,
        "state": state,
        "config": load_config(),
        "page": "trading"
    })

@app.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request):
    """Positions and orders management."""
    return templates.TemplateResponse("positions.html", {
        "request": request,
        "state": state,
        "page": "positions"
    })

@app.get("/markets", response_class=HTMLResponse)
async def markets_page(request: Request):
    """Market analysis interface."""
    return templates.TemplateResponse("markets.html", {
        "request": request,
        "state": state,
        "page": "markets"
    })

@app.get("/wallet-analysis", response_class=HTMLResponse)
async def wallet_analysis_page(request: Request):
    """Wallet analysis dashboard."""
    return templates.TemplateResponse("wallet_analysis.html", {
        "request": request,
        "state": state,
        "page": "wallet_analysis"
    })

@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Agent monitoring panel."""
    return templates.TemplateResponse("agents.html", {
        "request": request,
        "state": state,
        "page": "agents"
    })

@app.get("/risk", response_class=HTMLResponse)
async def risk_page(request: Request):
    """Risk management dashboard."""
    return templates.TemplateResponse("risk.html", {
        "request": request,
        "state": state,
        "config": load_config(),
        "page": "risk"
    })

@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Configuration management."""
    return templates.TemplateResponse("config.html", {
        "request": request,
        "state": state,
        "config": load_config(),
        "page": "config"
    })

@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Logs and metrics viewer."""
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "state": state,
        "page": "logs"
    })

@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Help and documentation."""
    return templates.TemplateResponse("help.html", {
        "request": request,
        "state": state,
        "page": "help"
    })


# ============================================================================
# Trading Control API
# ============================================================================

@app.post("/api/trading/start")
async def start_trading(config: TradingConfig, background_tasks: BackgroundTasks):
    """Start the trading loop."""
    if state.trading_active:
        raise HTTPException(400, "Trading already active")

    state.trading_active = True
    state.trading_mode = config.mode

    await broadcast_update("trading_started", {
        "mode": config.mode,
        "interval": config.loop_interval
    })

    # In production, this would start the actual trading runner
    # background_tasks.add_task(run_trading_loop, config)

    return {"status": "started", "mode": config.mode}

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the trading loop."""
    if not state.trading_active:
        raise HTTPException(400, "Trading not active")

    state.trading_active = False

    await broadcast_update("trading_stopped", {})

    return {"status": "stopped"}

@app.get("/api/trading/status")
async def trading_status():
    """Get current trading status."""
    return {
        "active": state.trading_active,
        "mode": state.trading_mode,
        "cycle_count": state.cycle_count,
        "last_cycle": state.last_cycle_time.isoformat() if state.last_cycle_time else None,
        "balance": str(state.balance),
        "daily_pnl": str(state.daily_pnl),
        "total_pnl": str(state.total_pnl),
    }

@app.post("/api/trading/kill-switch")
async def toggle_kill_switch(enabled: bool = Form(...)):
    """Enable/disable the kill switch."""
    config = load_config()
    config.setdefault("risk", {})["kill_switch"] = enabled
    save_config(config)

    await broadcast_update("kill_switch", {"enabled": enabled})

    return {"kill_switch": enabled}


# ============================================================================
# Positions & Orders API
# ============================================================================

@app.get("/api/positions")
async def get_positions():
    """Get current positions."""
    # In production, this would fetch from the broker
    return {"positions": state.positions}

@app.get("/api/orders")
async def get_orders():
    """Get open orders."""
    return {"orders": state.orders}

@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an open order."""
    # In production, this would cancel via the broker
    return {"status": "cancelled", "order_id": order_id}

@app.post("/api/positions/{position_id}/close")
async def close_position(position_id: str):
    """Close a position."""
    # In production, this would close via the broker
    return {"status": "closing", "position_id": position_id}


# ============================================================================
# Markets API
# ============================================================================

@app.get("/api/markets")
async def get_markets(
    category: Optional[str] = None,
    min_liquidity: float = 1000,
    limit: int = 50
):
    """Get active markets."""
    # In production, this would fetch from MarketData
    # For now, return mock data structure
    return {"markets": [], "total": 0}

@app.get("/api/markets/{market_id}")
async def get_market_detail(market_id: str):
    """Get detailed market information."""
    return {"market": None}

@app.get("/api/markets/{market_id}/indicators")
async def get_market_indicators(market_id: str):
    """Get technical indicators for a market."""
    return {"indicators": {}}

@app.get("/api/arbitrage")
async def get_arbitrage_opportunities():
    """Get current arbitrage opportunities."""
    return {"opportunities": []}


# ============================================================================
# Wallet Analysis API
# ============================================================================

@app.post("/api/wallet/analyze")
async def analyze_wallet(request: WalletAnalysisRequest):
    """Analyze a wallet address."""
    try:
        from src.analysis.dashboard import WalletDashboard

        dashboard = WalletDashboard()
        try:
            data = await dashboard.analyze(request.address)
            return {
                "success": True,
                "data": {
                    "address": data.address,
                    "summary": {
                        "total_transactions": data.summary.total_transactions,
                        "total_volume": float(data.summary.total_volume),
                        "total_pnl": float(data.summary.total_realized_pnl),
                        "win_rate": float(data.summary.win_rate),
                        "unique_markets": data.summary.unique_markets,
                    },
                    "strategy": {
                        "primary": data.strategy.primary_strategy.value,
                        "secondary": [s.value for s in data.strategy.secondary_strategies],
                        "confidence": float(data.strategy.confidence),
                        "risk_profile": data.strategy.risk_profile.value,
                        "summary": data.strategy.summary,
                    },
                    "metrics": {
                        "sharpe_ratio": float(data.metrics.sharpe_ratio),
                        "sortino_ratio": float(data.metrics.sortino_ratio),
                        "max_drawdown": float(data.metrics.max_drawdown),
                        "profit_factor": float(data.metrics.profit_factor),
                        "expectancy": float(data.metrics.expectancy),
                    },
                    "patterns": [
                        {"type": p.pattern_type, "confidence": float(p.confidence), "description": p.description}
                        for p in data.strategy.patterns
                    ],
                    "strengths": data.strategy.strengths,
                    "weaknesses": data.strategy.weaknesses,
                    "recommendations": data.strategy.recommendations,
                }
            }
        finally:
            await dashboard.close()
    except Exception as e:
        logger.error(f"Wallet analysis failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/wallet/{address}/html")
async def get_wallet_html(address: str):
    """Get wallet analysis as HTML."""
    try:
        from src.analysis.dashboard import WalletDashboard

        dashboard = WalletDashboard()
        try:
            data = await dashboard.analyze(address)
            html = dashboard.render_html(data)
            return HTMLResponse(content=html)
        finally:
            await dashboard.close()
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================
# Agent API
# ============================================================================

@app.get("/api/agents")
async def get_agents():
    """Get agent status."""
    return {
        "agents": [
            {"name": "research", "status": "idle", "last_run": None, "model": "claude-sonnet-4-20250514"},
            {"name": "risk", "status": "idle", "last_run": None, "model": "claude-sonnet-4-20250514"},
            {"name": "execution", "status": "idle", "last_run": None, "model": "claude-haiku-3-5-20241022"},
            {"name": "reflection", "status": "idle", "last_run": None, "model": "claude-sonnet-4-20250514"},
        ],
        "debate_enabled": True,
        "reflection_enabled": True,
    }

@app.get("/api/agents/{agent_name}/history")
async def get_agent_history(agent_name: str, limit: int = 50):
    """Get agent decision history."""
    return {"history": []}


# ============================================================================
# Risk API
# ============================================================================

@app.get("/api/risk/status")
async def get_risk_status():
    """Get current risk status."""
    config = load_config()
    risk_config = config.get("risk", {})

    return {
        "daily_pnl": float(state.daily_pnl),
        "daily_limit": risk_config.get("daily_loss_limit_usdc", 50),
        "daily_pnl_pct": abs(float(state.daily_pnl)) / risk_config.get("daily_loss_limit_usdc", 50) * 100,
        "open_positions": len(state.positions),
        "max_positions": risk_config.get("max_open_positions", 5),
        "total_exposure": sum(p.get("size", 0) for p in state.positions),
        "max_exposure": risk_config.get("max_position_size_usdc", 100) * risk_config.get("max_open_positions", 5),
        "kill_switch": risk_config.get("kill_switch", False),
        "alerts": state.alerts,
    }

@app.post("/api/risk/settings")
async def update_risk_settings(settings: RiskSettings):
    """Update risk settings."""
    config = load_config()
    config.setdefault("risk", {}).update({
        "max_position_size_usdc": settings.max_position_size,
        "daily_loss_limit_usdc": settings.daily_loss_limit,
        "max_open_positions": settings.max_open_positions,
        "max_single_trade_usdc": settings.max_single_trade,
        "kill_switch": settings.kill_switch,
    })
    save_config(config)

    await broadcast_update("risk_settings_updated", settings.dict())

    return {"status": "updated"}


# ============================================================================
# Configuration API
# ============================================================================

@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return load_config()

@app.post("/api/config")
async def update_config(config: dict):
    """Update configuration."""
    save_config(config)
    state.config = config

    await broadcast_update("config_updated", {})

    return {"status": "updated"}

@app.get("/api/config/llm")
async def get_llm_config():
    """Get LLM provider configuration."""
    config = load_config()
    return config.get("llm", {})

@app.post("/api/config/llm")
async def update_llm_config(llm_config: dict):
    """Update LLM configuration."""
    config = load_config()
    config["llm"] = llm_config
    save_config(config)

    return {"status": "updated"}


# ============================================================================
# LLM Provider Management API
# ============================================================================

class ProviderSwitchRequest(BaseModel):
    provider: str

class FallbackOrderRequest(BaseModel):
    order: list[str]

@app.get("/api/llm/providers")
async def get_llm_providers():
    """Get all LLM providers with status and health."""
    if state.provider_registry is None:
        # Return basic provider availability from environment
        return {
            "providers": {
                "anthropic": {
                    "enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
                    "connected": False,
                    "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
                },
                "openai": {
                    "enabled": bool(os.getenv("OPENAI_API_KEY")),
                    "connected": False,
                    "models": ["gpt-4o", "gpt-4o-mini", "o1-preview"],
                },
                "xai": {
                    "enabled": bool(os.getenv("XAI_API_KEY")),
                    "connected": False,
                    "models": ["grok-3", "grok-3-mini"],
                },
                "google": {
                    "enabled": bool(os.getenv("GOOGLE_API_KEY")),
                    "connected": False,
                    "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                },
                "local": {
                    "enabled": True,  # Local always available
                    "connected": False,
                    "models": ["deepseek-r1:70b", "qwen2.5:72b", "llama3.3:70b", "kimi-k2"],
                },
            },
            "active_provider": None,
            "fallback_order": ["anthropic", "openai", "google", "xai", "local"],
            "registry_initialized": False,
        }

    return {
        **state.provider_registry.get_provider_info(),
        "registry_initialized": True,
    }

@app.get("/api/llm/active")
async def get_active_provider():
    """Get the currently active LLM provider."""
    if state.provider_registry is None:
        return {"active_provider": None, "error": "Registry not initialized"}

    return {
        "active_provider": state.provider_registry.active_provider,
        "available_providers": state.provider_registry.available_providers,
    }

@app.post("/api/llm/active")
async def set_active_provider(request: ProviderSwitchRequest):
    """Switch the active LLM provider."""
    if state.provider_registry is None:
        raise HTTPException(503, "Provider registry not initialized")

    success = state.provider_registry.set_active_provider(request.provider)
    if not success:
        raise HTTPException(400, f"Provider '{request.provider}' not available")

    await broadcast_update("provider_changed", {
        "active_provider": request.provider,
    })

    return {
        "status": "switched",
        "active_provider": request.provider,
    }

@app.get("/api/llm/fallback-order")
async def get_fallback_order():
    """Get the current fallback order."""
    if state.provider_registry is None:
        return {"fallback_order": ["anthropic", "openai", "google", "xai", "local"]}

    return {"fallback_order": state.provider_registry.fallback_order}

@app.post("/api/llm/fallback-order")
async def set_fallback_order(request: FallbackOrderRequest):
    """Set the fallback order for providers."""
    if state.provider_registry is None:
        raise HTTPException(503, "Provider registry not initialized")

    state.provider_registry.set_fallback_order(request.order)

    await broadcast_update("fallback_order_changed", {
        "fallback_order": request.order,
    })

    return {"status": "updated", "fallback_order": request.order}

@app.get("/api/llm/health")
async def get_provider_health():
    """Get health status for all providers."""
    if state.provider_registry is None:
        return {"health": {}, "error": "Registry not initialized"}

    return {"health": state.provider_registry.get_health_status()}

@app.post("/api/llm/health/check")
async def check_provider_health(provider: Optional[str] = None):
    """Run health check on providers."""
    if state.provider_registry is None:
        raise HTTPException(503, "Provider registry not initialized")

    await state.provider_registry.check_health(provider)

    return {"status": "checked", "health": state.provider_registry.get_health_status()}

@app.post("/api/llm/initialize")
async def initialize_providers():
    """Initialize the provider registry."""
    try:
        from src.core.config import load_config as load_typed_config
        from src.llm.registry import ProviderRegistry

        config = load_typed_config()
        registry = ProviderRegistry(config)
        await registry.initialize()

        state.provider_registry = registry

        await broadcast_update("providers_initialized", {
            "active_provider": registry.active_provider,
            "connected_providers": registry.available_providers,
        })

        return {
            "status": "initialized",
            "active_provider": registry.active_provider,
            "connected_providers": registry.available_providers,
            "health": registry.get_health_status(),
        }
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        raise HTTPException(500, f"Initialization failed: {str(e)}")


# ============================================================================
# Logs & Metrics API
# ============================================================================

@app.get("/api/logs/decisions")
async def get_decision_logs(limit: int = 100, offset: int = 0):
    """Get decision logs."""
    log_path = Path(__file__).parent.parent.parent / "logs" / "decisions.jsonl"

    logs = []
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {"logs": logs, "total": len(logs)}

@app.get("/api/logs/trades")
async def get_trade_logs(limit: int = 100):
    """Get trade history."""
    return {"trades": []}

@app.get("/api/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics."""
    return {
        "total_pnl": float(state.total_pnl),
        "daily_pnl": float(state.daily_pnl),
        "weekly_pnl": 0,
        "monthly_pnl": 0,
        "sharpe_ratio": 0,
        "win_rate": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
    }

@app.get("/api/metrics/equity-curve")
async def get_equity_curve():
    """Get equity curve data."""
    return {"data": []}


# ============================================================================
# System API
# ============================================================================

@app.get("/api/system/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "trading_active": state.trading_active,
        "websocket_clients": len(state.connected_websockets),
        "uptime": "unknown",  # Would track actual uptime
    }

@app.get("/api/system/providers")
async def get_providers():
    """Get configured providers status."""
    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "xai": bool(os.getenv("XAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "local": True,  # Local models always available if server is running
        "polymarket": bool(os.getenv("POLYMARKET_API_KEY")),
        "wallet": bool(os.getenv("WALLET_PRIVATE_KEY")),
    }


# ============================================================================
# HTMX Partials (for dynamic updates)
# ============================================================================

@app.get("/partials/positions-table", response_class=HTMLResponse)
async def positions_table_partial(request: Request):
    """Partial for positions table."""
    return templates.TemplateResponse("partials/positions_table.html", {
        "request": request,
        "positions": state.positions
    })

@app.get("/partials/orders-table", response_class=HTMLResponse)
async def orders_table_partial(request: Request):
    """Partial for orders table."""
    return templates.TemplateResponse("partials/orders_table.html", {
        "request": request,
        "orders": state.orders
    })

@app.get("/partials/alerts", response_class=HTMLResponse)
async def alerts_partial(request: Request):
    """Partial for alerts list."""
    return templates.TemplateResponse("partials/alerts.html", {
        "request": request,
        "alerts": state.alerts
    })

@app.get("/partials/metrics-cards", response_class=HTMLResponse)
async def metrics_cards_partial(request: Request):
    """Partial for metrics cards."""
    return templates.TemplateResponse("partials/metrics_cards.html", {
        "request": request,
        "state": state
    })


# ============================================================================
# App Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load initial config
    state.config = load_config()

    return app


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
