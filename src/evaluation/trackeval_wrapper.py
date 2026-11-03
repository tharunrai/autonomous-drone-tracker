"""TrackEval benchmarking wrapper for UAV tracking evaluation."""

from typing import Any, Dict, List, Optional


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

