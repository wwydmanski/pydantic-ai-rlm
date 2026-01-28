from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ContextType = str | dict[str, Any] | list[Any]


@dataclass
class RLMConfig:
    """Configuration for RLM behavior."""

    code_timeout: float = 60.0
    """Timeout in seconds for code execution."""

    truncate_output_chars: int = 50_000
    """Maximum characters to return from code execution output."""

    sub_model: str | None = None
    """
    Model to use for llm_query() within the REPL environment.

    If set, a `llm_query(prompt: str) -> str` function becomes available
    in the REPL environment, allowing the main LLM to delegate sub-queries
    to another model. This is useful for processing large contexts in chunks.
    """


@dataclass
class RLMDependencies:
    """
    Dependencies injected into RLM tools via RunContext.

    This holds the context data and configuration that
    the RLM toolset needs to operate.
    """

    context: ContextType
    """The context to analyze (string, dict, or list)."""

    config: RLMConfig = field(default_factory=RLMConfig)
    """RLM configuration options."""

    def __post_init__(self):
        """Validate dependencies after initialization."""
        if self.context is None:
            raise ValueError("context cannot be None")
