"""Evaluation module: benchmarking, compute cost measurement, and Pareto trade-off plotting."""

from typing import Any, Dict, List, Optional
import pandas as pd


def run_trackeval(
    tracker_results_dir: str,
    ground_truth_dir: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run TrackEval evaluation on predicted tracks against ground truth.

    Args:
        tracker_results_dir: Directory containing tracker output text/CSV files.
        ground_truth_dir: Directory containing ground truth annotations.
        metrics: List of metrics to compute (e.g., ['HOTA', 'IDF1', 'MOTA', 'IDSW']).

    Returns:
        Dictionary of evaluated metric values per sequence and overall summary.
    """
    raise NotImplementedError("run_trackeval is a stub.")


def compute_pipeline_cost(
    frame_count: int,
    total_time_seconds: float,
    detector_calls: int,
    gpu_memory_mb: float = 0.0,
) -> Dict[str, float]:
    """Compute FPS, detector invocation ratio, latency per frame, and GPU utilization.

    Args:
        frame_count: Total processed video frames.
        total_time_seconds: Total wall-clock time elapsed in seconds.
        detector_calls: Number of times heavy YOLO detector was invoked.
        gpu_memory_mb: Peak GPU memory footprint observed.

    Returns:
        Dictionary containing 'fps', 'avg_latency_ms', 'detector_call_ratio', and 'peak_gpu_mb'.
    """
    raise NotImplementedError("compute_pipeline_cost is a stub.")


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

