"""Utility functions for internal use."""

import os
import warnings
from typing import Optional, Tuple


def handle_deprecated_prompt_format(prompt_format: Optional[str]) -> None:
    """
    Warn if a caller still passes the retired ``prompt_format`` argument.

    Prompts used to be rendered as either a combined string or a system/user pair,
    chosen before the LLM wrapper was reached. Prompts now carry every rendering
    and the wrapper selects one at call time, so the argument no longer does
    anything.

    Parameters
    ----------
    prompt_format : str, optional
        The value the caller passed. ``None`` means the caller did not pass one.
    """
    if prompt_format is not None:
        warnings.warn(
            "prompt_format is deprecated and ignored, and will be removed in v2.0. "
            "Prompts now carry every rendering and the LLM wrapper selects one at "
            "call time.",
            DeprecationWarning,
            stacklevel=3,
        )


def handle_verbose_params(
    verbose: Optional[bool] = None,
    verbose_legacy: Optional[bool] = None,
    show_progress_bar: Optional[bool] = None,
    show_progress_bars: Optional[bool] = None,
    default_verbose: bool = True,
) -> Tuple[bool, bool]:
    """
    Handle the transition from verbose/show_progress_bar to unified verbose parameter.

    Parameters
    ----------
    verbose : bool, optional
        New unified parameter. If True, shows both progress bars and verbose output.
        If False, suppresses all output. Takes precedence over legacy parameters.
    verbose_legacy : bool, optional
        Legacy parameter for verbose output (deprecated).
    show_progress_bar : bool, optional
        Legacy parameter for progress bar display.
    show_progress_bars : bool, optional
        Legacy parameter for progress bar display (used in Toponymy class).
    default_verbose : bool, default=True
        Default value to use when no parameters are provided.

    Returns
    -------
    tuple of (bool, bool)
        Returns (show_progress_bar, verbose) for internal use.
    """
    # If new verbose parameter is provided, use it
    if verbose is not None:
        return verbose, verbose

    # Handle legacy parameters
    legacy_params_used = []
    if verbose_legacy is not None:
        legacy_params_used.append("verbose")
    if show_progress_bar is not None:
        legacy_params_used.append("show_progress_bar")
    if show_progress_bars is not None:
        legacy_params_used.append("show_progress_bars")

    # Issue deprecation warning if legacy parameters are used
    if legacy_params_used:
        params_str = ", ".join(legacy_params_used)
        warnings.warn(
            f"Parameters {params_str} are deprecated and will be removed in v2.0. "
            f"Use 'verbose' parameter instead. "
            f"Set verbose=True to show all output, verbose=False to suppress all output.",
            DeprecationWarning,
            stacklevel=3,
        )

    # Determine values from legacy parameters
    # show_progress_bars takes precedence over show_progress_bar for backward compatibility
    progress_bar_value = (
        show_progress_bars if show_progress_bars is not None else show_progress_bar
    )

    # If only verbose_legacy is set to True, we should show progress bars too (expected behavior)
    if verbose_legacy is True and progress_bar_value is None:
        progress_bar_value = True

    # Use default if no legacy parameters provided
    if verbose_legacy is None and progress_bar_value is None:
        return default_verbose, default_verbose

    # Return the resolved values
    return (
        progress_bar_value if progress_bar_value is not None else default_verbose,
        verbose_legacy if verbose_legacy is not None else default_verbose,
    )


def resolve_api_key(
    api_key: str | None,
    env_new: str | None,
    env_legacy: str | None = None,
    required: bool = True,
) -> str | None:
    """
    Resolve API key from explicit parameter or environment variables.

    Parameters
    ----------
    api_key : str | None
        Explicitly provided API key (takes precedence over environment variables).
    env_new : str | None
        Primary environment variable name to check (e.g., "COHERE_API_KEY").
    env_legacy : str | None, optional
        Deprecated environment variable name for backwards compatibility.
        If found, a deprecation warning is issued.
    required : bool, default=True
        If True, raises ValueError when no API key is found.
        If False, returns None when no API key is found.

    Returns
    -------
    str | None
        The resolved API key, or None if not found and not required.

    Raises
    ------
    ValueError
        If required=True and no API key is found.
    """
    # Normalize: treat empty/whitespace strings as None for consistent behavior
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            api_key = None

    if api_key is not None:
        return api_key

    # Get environment variables and normalize them too
    new_key = os.getenv(env_new) if env_new else None
    if new_key:
        new_key = new_key.strip()
        if new_key:
            return new_key

    legacy_key = os.getenv(env_legacy) if env_legacy else None
    if legacy_key:
        legacy_key = legacy_key.strip()
        if not legacy_key:
            legacy_key = None

    if legacy_key:
        warnings.warn(
            f"{env_legacy} is deprecated. Use {env_new} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy_key

    if required:
        # Extract readable provider name from env variable (e.g., "COHERE_API_KEY" -> "Cohere")
        provider_name = (
            env_new.replace("_API_KEY", "").replace("_", " ").title()
            if env_new
            else "API"
        )
        raise ValueError(
            f"No {provider_name} API key provided. Set {env_new} environment variable "
            f"or pass api_key parameter."
        )

    return None
