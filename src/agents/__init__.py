"""Multi-agent trading system with specialized roles."""

from .base import BaseAgent, AgentRole, AgentResponse
from .research import ResearchAgent
from .risk import RiskAgent
from .execution import ExecutionAgent
from .reflection import ReflectionAgent
from .coordinator import AgentCoordinator

__all__ = [
    "BaseAgent",
    "AgentRole",
    "AgentResponse",
    "ResearchAgent",
    "RiskAgent",
    "ExecutionAgent",
    "ReflectionAgent",
    "AgentCoordinator",
]
