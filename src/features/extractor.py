"""Unified condition and feature extractor orchestrator."""

from typing import Any, Dict, List, Optional
import numpy as np


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

