"""Multi-agent trading system with specialized roles."""

from .base import BaseAgent, AgentRole, AgentResponse
from .research import ResearchAgent
from .risk import RiskAgent
from .execution import ExecutionAgent
from .reflection import ReflectionAgent
from .coordinator import AgentCoordinator
from .debate import (
    DebatePersona,
    DebatePosition,
    DebateRound,
    DebateResult,
    DebateAgent,
    DebateOrchestrator,
    ConfidenceCalibrator,
    get_confidence_calibrator,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentRole",
    "AgentResponse",
    # Specialized agents
    "ResearchAgent",
    "RiskAgent",
    "ExecutionAgent",
    "ReflectionAgent",
    # Coordination
    "AgentCoordinator",
    # Debate system
    "DebatePersona",
    "DebatePosition",
    "DebateRound",
    "DebateResult",
    "DebateAgent",
    "DebateOrchestrator",
    "ConfidenceCalibrator",
    "get_confidence_calibrator",
]
