"""Blur estimation module using Laplacian variance."""

import numpy as np


def estimate_blur(frame: np.ndarray) -> float:
    """Compute the blur metric of a frame via the variance of the Laplacian.

    Args:
        frame: Input video frame as a NumPy array (grayscale or BGR).

    Returns:
        Variance of the Laplacian representing image sharpness/blur.
    """
    raise NotImplementedError("estimate_blur is a stub.")

