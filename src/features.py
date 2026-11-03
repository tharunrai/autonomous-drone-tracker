"""Condition feature extraction module: blur, illumination, camera motion, and bbox geometry."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def estimate_blur(frame: np.ndarray) -> float:
    """Compute the blur metric of a frame via the variance of the Laplacian.

    Args:
        frame: Input video frame as a NumPy array (grayscale or BGR).

    Returns:
        Variance of the Laplacian representing image sharpness/blur.
    """
    raise NotImplementedError("estimate_blur is a stub.")


def compute_brightness_stats(frame: np.ndarray) -> Dict[str, float]:
    """Compute mean brightness, standard deviation (contrast), and dynamic range.

    Args:
        frame: Input video frame as a NumPy array.

    Returns:
        Dictionary containing 'mean_brightness', 'contrast', and 'under_exposure_ratio'.
    """
    raise NotImplementedError("compute_brightness_stats is a stub.")


def estimate_camera_motion(
    prev_frame: np.ndarray, curr_frame: np.ndarray, max_corners: int = 200
) -> Dict[str, float]:
    """Estimate camera motion magnitude and angle using Lucas-Kanade optical flow.

    Args:
        prev_frame: Previous video frame as a NumPy array.
        curr_frame: Current video frame as a NumPy array.
        max_corners: Maximum number of corners to track for optical flow.

    Returns:
        Dictionary containing 'optical_flow_magnitude', 'flow_mean_dx', and 'flow_mean_dy'.
    """
    raise NotImplementedError("estimate_camera_motion is a stub.")


def extract_box_geometry(
    bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int]
) -> Dict[str, float]:
    """Compute normalized geometric features for a target bounding box.

    Args:
        bbox: Target box in (x1, y1, x2, y2) format.
        frame_shape: Height and width of the frame (H, W).

    Returns:
        Dictionary containing 'normalized_area', 'aspect_ratio', and 'center_distance_from_edge'.
    """
    raise NotImplementedError("extract_box_geometry is a stub.")


class ConditionFeatureExtractor:
    """Extracts frame-level and target-level environmental features for mode selection."""

    def __init__(self, window_size: int = 5, feature_names: Optional[List[str]] = None) -> None:
        """Initialize feature extractor with sliding window size.

        Args:
            window_size: Number of frames over which to aggregate temporal features.
            feature_names: List of specific feature keys to return.
        """
        self.window_size = window_size
        self.feature_names = feature_names or []
        self._history: List[Dict[str, float]] = []

    def extract_frame_features(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        bboxes: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """Extract multi-modal condition features from current frame and context.

        Args:
            frame: Current video frame.
            prev_frame: Optional preceding video frame for motion extraction.
            bboxes: Optional detected or tracked bounding boxes.

        Returns:
            Dictionary of extracted scalar features.
        """
        raise NotImplementedError("ConditionFeatureExtractor.extract_frame_features is a stub.")

    def reset(self) -> None:
        """Reset temporal history buffer."""
        self._history.clear()
