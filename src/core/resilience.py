"""Resilience utilities for production hardening.

Provides circuit breaker, retry logic, and timeout protection
for external API calls and critical operations.
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected while OPEN
    last_failure_time: float | None = None
    last_success_time: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for API/logging."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": round(self.failure_rate, 2),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, fail fast without calling
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker("polymarket-api")

        @breaker
        async def fetch_markets():
            return await api.get_markets()

        # Or manually:
        if breaker.allow_request():
            try:
                result = await api.call()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure(e)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for logging/monitoring
            failure_threshold: Consecutive failures before opening
            recovery_timeout: Seconds to wait before half-open
            half_open_max_calls: Max calls in half-open state
            success_threshold: Successes needed to close from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_failure_time: float = 0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout recovery."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state with logging."""
        old_state = self._state
        self._state = new_state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        logger.info(
            f"Circuit breaker [{self.name}] state change: {old_state.value} -> {new_state.value}"
        )

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        current_state = self.state  # Triggers timeout check

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            return False

        # HALF_OPEN: allow limited calls
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True

        return False

    def record_success(self) -> None:
        """Record successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Exception | None = None) -> None:
        """Record failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1

        error_msg = str(error) if error else "Unknown error"
        logger.warning(
            f"Circuit breaker [{self.name}] recorded failure: {error_msg} "
            f"(consecutive: {self._stats.consecutive_failures})"
        )

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_failure_time = 0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker [{self.name}] reset")

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator for protecting async functions."""

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker [{self.name}] is OPEN - failing fast"
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def to_dict(self) -> dict:
        """Get full status for API/monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": self._stats.to_dict(),
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "success_threshold": self.success_threshold,
            },
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


async def with_timeout(
    coro,
    timeout_seconds: float,
    timeout_message: str = "Operation timed out",
) -> Any:
    """
    Execute coroutine with timeout protection.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum time to wait
        timeout_message: Message for timeout error

    Raises:
        asyncio.TimeoutError: If operation exceeds timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Timeout after {timeout_seconds}s: {timeout_message}")
        raise asyncio.TimeoutError(timeout_message)


async def with_retry(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> T:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to call (no arguments)
        max_attempts: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        exponential_base: Multiplier for exponential backoff
        retryable_exceptions: Exception types to retry on

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_attempts:
                logger.error(
                    f"All {max_attempts} retry attempts failed. Last error: {e}"
                )
                raise

            delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    raise last_exception  # Should never reach here


@dataclass
class HealthStatus:
    """Health status for a component."""

    name: str
    healthy: bool
    message: str = ""
    latency_ms: int | None = None
    last_check: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
        }


class HealthChecker:
    """
    Aggregates health checks for multiple components.

    Usage:
        checker = HealthChecker()
        checker.register("broker", check_broker_health)
        checker.register("llm", check_llm_health)

        status = await checker.check_all()
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], HealthStatus]] = {}

    def register(
        self,
        name: str,
        check_func: Callable[[], HealthStatus],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func

    async def check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthStatus(name=name, healthy=False, message="Check not found")

        try:
            start = time.time()
            check_func = self._checks[name]

            # Handle both sync and async checks
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            if result.latency_ms is None:
                result.latency_ms = int((time.time() - start) * 1000)

            return result
        except Exception as e:
            return HealthStatus(
                name=name,
                healthy=False,
                message=f"Check failed: {e}",
            )

    async def check_all(self) -> dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = await self.check(name)
        return results

    def overall_healthy(self, results: dict[str, HealthStatus]) -> bool:
        """Check if all components are healthy."""
        return all(status.healthy for status in results.values())


# Global circuit breakers for common services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all circuit breakers for monitoring."""
    return _circuit_breakers.copy()
