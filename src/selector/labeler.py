"""Label generation module for offline training of the mode selector."""

from typing import Any, Dict
import pandas as pd


def generate_mode_labels(
    ground_truth_tracks: Any,
    fast_tracks: Any,
    robust_tracks: Any,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate optimal mode labels (FAST vs ROBUST) per frame based on tracking success.

    If FAST mode maintains sufficient IoU / tracking accuracy, it is preferred to save compute;
    otherwise ROBUST is assigned.

    Args:
        ground_truth_tracks: Ground truth track annotations.
        fast_tracks: Tracking results using Always FAST mode.
        robust_tracks: Tracking results using Always ROBUST mode.
        iou_threshold: Tracking success IoU threshold against ground truth.

    Returns:
        DataFrame containing frame IDs, extracted features, and target mode labels.
    """
    raise NotImplementedError("generate_mode_labels is a stub.")

