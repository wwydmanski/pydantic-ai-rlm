from __future__ import annotations

from .repl import REPLResult


def format_repl_result(result: REPLResult, max_var_display: int = 200) -> str:
    """
    Format a REPL execution result for display to the LLM.

    Args:
        result: The REPLResult from code execution
        max_var_display: Maximum characters to show per variable value

    Returns:
        Formatted string suitable for LLM consumption
    """
    parts = []

    if result.stdout.strip():
        parts.append(f"Output:\n{result.stdout}")

    if result.stderr.strip():
        parts.append(f"Errors:\n{result.stderr}")

    # Show created/modified variables (excluding internal ones)
    user_vars = {k: v for k, v in result.locals.items() if not k.startswith("_") and k not in ("context", "json", "re", "os")}

    if user_vars:
        var_summaries = []
        for name, value in user_vars.items():
            try:
                value_str = repr(value)
                if len(value_str) > max_var_display:
                    value_str = value_str[:max_var_display] + "..."
                var_summaries.append(f"  {name} = {value_str}")
            except Exception:
                var_summaries.append(f"  {name} = <{type(value).__name__}>")

        if var_summaries:
            parts.append("Variables:\n" + "\n".join(var_summaries))

    parts.append(f"Execution time: {result.execution_time:.3f}s")

    if not parts:
        return "Code executed successfully (no output)"

    return "\n\n".join(parts)
