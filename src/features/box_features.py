"""Bounding box geometric and temporal feature extraction module."""

from typing import Dict, Tuple


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

