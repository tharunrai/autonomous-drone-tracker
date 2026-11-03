"""Input/output helper utilities for video frames and tracking outputs."""

from pathlib import Path
from typing import Any, List, Union
import pandas as pd


def save_tracking_results(tracks: Any, output_path: Union[str, Path]) -> None:
    """Save tracking output in MOT challenge format (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z).

    Args:
        tracks: Tracking output data structure or DataFrame.
        output_path: Path to target output file (.txt or .csv).
    """
    raise NotImplementedError("save_tracking_results is a stub.")


def load_tracking_results(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load tracking results from text or CSV file into a pandas DataFrame.

    Args:
        file_path: Path to tracking results file.

    Returns:
        DataFrame containing parsed track records.
    """
    raise NotImplementedError("load_tracking_results is a stub.")

