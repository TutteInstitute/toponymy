"""Utility functions for internal use."""

import warnings
from typing import Optional, Tuple


def handle_verbosity_params(
    verbosity: Optional[bool] = None,
    verbose: Optional[bool] = None,
    show_progress_bar: Optional[bool] = None,
    show_progress_bars: Optional[bool] = None,
    default_verbosity: bool = True,
) -> Tuple[bool, bool]:
    """
    Handle the transition from verbose/show_progress_bar to unified verbosity parameter.
    
    Parameters
    ----------
    verbosity : bool, optional
        New unified parameter. If True, shows both progress bars and verbose output.
        If False, suppresses all output. Takes precedence over legacy parameters.
    verbose : bool, optional
        Legacy parameter for verbose output.
    show_progress_bar : bool, optional
        Legacy parameter for progress bar display.
    show_progress_bars : bool, optional
        Legacy parameter for progress bar display (used in Toponymy class).
    default_verbosity : bool, default=True
        Default value to use when no parameters are provided.
    
    Returns
    -------
    tuple of (bool, bool)
        Returns (show_progress_bar, verbose) for internal use.
    """
    # If new verbosity parameter is provided, use it
    if verbosity is not None:
        return verbosity, verbosity
    
    # Handle legacy parameters
    legacy_params_used = []
    if verbose is not None:
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
            f"Use 'verbosity' parameter instead. "
            f"Set verbosity=True to show all output, verbosity=False to suppress all output.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # Determine values from legacy parameters
    # show_progress_bars takes precedence over show_progress_bar for backward compatibility
    progress_bar_value = show_progress_bars if show_progress_bars is not None else show_progress_bar
    
    # If only verbose is set to True, we should show progress bars too (expected behavior)
    if verbose is True and progress_bar_value is None:
        progress_bar_value = True
    
    # Use default if no legacy parameters provided
    if verbose is None and progress_bar_value is None:
        return default_verbosity, default_verbosity
    
    # Return the resolved values
    return (
        progress_bar_value if progress_bar_value is not None else default_verbosity,
        verbose if verbose is not None else default_verbosity
    )