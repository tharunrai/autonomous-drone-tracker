"""Brightness and contrast statistics module."""

from typing import Dict
import numpy as np


def compute_brightness_stats(frame: np.ndarray) -> Dict[str, float]:
    """Compute mean brightness, standard deviation (contrast), and dynamic range.

    Args:
        frame: Input video frame as a NumPy array.

    Returns:
        Dictionary containing 'mean_brightness', 'contrast', and 'under_exposure_ratio'.
    """
    raise NotImplementedError("compute_brightness_stats is a stub.")

