"""Evaluation visualization and trade-off plotting module."""

from typing import Any, Optional
import pandas as pd


def plot_accuracy_vs_cost(
    summary_df: pd.DataFrame,
    accuracy_metric: str = "HOTA",
    cost_metric: str = "fps",
    output_path: Optional[str] = None,
) -> Any:
    """Generate accuracy vs compute cost Pareto trade-off plot.

    Compares Always FAST, Always ROBUST, and Adaptive Perception modes.

    Args:
        summary_df: DataFrame containing method names and their evaluation metrics.
        accuracy_metric: Tracking performance metric column (e.g. 'HOTA', 'IDF1').
        cost_metric: Efficiency metric column (e.g. 'fps', 'detector_calls').
        output_path: Optional path to save rendered plot image or HTML.

    Returns:
        Plotly Figure or Matplotlib Figure object.
    """
    raise NotImplementedError("plot_accuracy_vs_cost is a stub.")


def plot_mode_switching_timeline(
    mode_history_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Any:
    """Generate timeline of dynamic mode transitions overlaid on condition difficulty.

    Args:
        mode_history_df: DataFrame containing frame-by-frame mode selections and features.
        output_path: Optional path to save rendered plot.

    Returns:
        Figure object displaying timeline.
    """
    raise NotImplementedError("plot_mode_switching_timeline is a stub.")

