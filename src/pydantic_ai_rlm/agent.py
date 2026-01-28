from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, UsageLimits

from .dependencies import ContextType, RLMConfig, RLMDependencies
from .prompts import build_rlm_instructions
from .toolset import create_rlm_toolset


def create_rlm_agent(
    model: str = "openai:gpt-5",
    sub_model: str | None = None,
    code_timeout: float = 60.0,
    include_example_instructions: bool = True,
    custom_instructions: str | None = None,
) -> Agent[RLMDependencies, str]:
    """
    Create a Pydantic AI agent with REPL code execution capabilities.

    Args:
        model: Model to use for the main agent
        sub_model: Model to use for llm_query() within the REPL environment.
            If provided, a `llm_query(prompt: str) -> str` function becomes
            available in the REPL, allowing the agent to delegate sub-queries.
            Example: "openai:gpt-5-mini" or "anthropic:claude-3-haiku-20240307"
        code_timeout: Timeout for code execution in seconds
        include_example_instructions: Include detailed examples in instructions
        custom_instructions: Additional instructions to append

    Returns:
        Configured Agent instance

    Example:
        ```python
        from pydantic_ai_rlm import create_rlm_agent, RLMDependencies, RLMConfig

        # Create agent with sub-model for llm_query
        agent = create_rlm_agent(
            model="openai:gpt-5",
            sub_model="openai:gpt-5-mini",
        )

        deps = RLMDependencies(
            context=very_large_document,
            config=RLMConfig(sub_model="openai:gpt-5-mini"),
        )
        result = await agent.run("What are the main themes?", deps=deps)
        print(result.output)
        ```
    """
    toolset = create_rlm_toolset(code_timeout=code_timeout, sub_model=sub_model)

    instructions = build_rlm_instructions(
        include_llm_query=sub_model is not None,
        custom_suffix=custom_instructions,
    )

    agent: Agent[RLMDependencies, str] = Agent(
        model,
        deps_type=RLMDependencies,
        output_type=str,
        toolsets=[toolset],
        instructions=instructions,
    )

    return agent


async def run_rlm_analysis(
    context: ContextType,
    query: str,
    model: str = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    **agent_kwargs: Any,
) -> str:
    """
    Convenience function to run RLM analysis on a context.

    Args:
        context: The large context to analyze (string, dict, or list)
        query: The question to answer about the context
        model: Model to use for the main agent
        sub_model: Model to use for llm_query() within the REPL environment.
            If provided, a `llm_query(prompt: str) -> str` function becomes
            available in the REPL, allowing the agent to delegate sub-queries.
        config: Optional RLMConfig for customization
        max_tool_calls: Maximum tool calls allowed
        **agent_kwargs: Additional arguments passed to create_rlm_agent()

    Returns:
        The agent's final answer as a string

    Example:
        ```python
        from pydantic_ai_rlm import run_rlm_analysis

        # With sub-model for llm_query
        answer = await run_rlm_analysis(
            context=huge_document,
            query="Find the magic number hidden in the text",
            sub_model="openai:gpt-5-mini",
        )
        print(answer)
        ```
    """
    agent = create_rlm_agent(model=model, sub_model=sub_model, **agent_kwargs)

    effective_config = config or RLMConfig()
    if sub_model and not effective_config.sub_model:
        effective_config.sub_model = sub_model

    deps = RLMDependencies(
        context=context,
        config=effective_config,
    )

    result = await agent.run(
        query,
        deps=deps,
        usage_limits=UsageLimits(tool_calls_limit=max_tool_calls),
    )

    return result.output


def run_rlm_analysis_sync(
    context: ContextType,
    query: str,
    model: str = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    **agent_kwargs: Any,
) -> str:
    """
    Synchronous version of run_rlm_analysis.

    See run_rlm_analysis() for full documentation.
    """
    agent = create_rlm_agent(model=model, sub_model=sub_model, **agent_kwargs)

    effective_config = config or RLMConfig()
    if sub_model and not effective_config.sub_model:
        effective_config.sub_model = sub_model

    deps = RLMDependencies(
        context=context,
        config=effective_config,
    )

    result = agent.run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(tool_calls_limit=max_tool_calls),
    )

    return result.output
