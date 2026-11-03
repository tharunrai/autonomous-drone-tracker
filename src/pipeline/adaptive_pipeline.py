"""Adaptive perception pipeline orchestrator dynamically switching between modes."""

from typing import Any, Dict, List, Optional
import numpy as np
from src.pipeline.modes import Mode


class AdaptivePerceptionPipeline:
    """Adaptive UAV tracking pipeline with dynamic condition-aware mode selection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize pipeline components with project configuration.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config
        self.current_mode = Mode.FAST
        # Stub: will initialize detectors, tracker, extractor, predictor, smoother

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Dict[str, Any]:
        """Process a single video frame with adaptive mode selection and tracking.

        Args:
            frame: Input video frame as a NumPy array.
            frame_idx: Index of the current frame in the video sequence.

        Returns:
            Dictionary with tracks, selected_mode, inference_time_ms, and condition_features.
        """
        raise NotImplementedError("AdaptivePerceptionPipeline.process_frame is a stub.")

    def run_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process an entire video sequence and record tracking metrics.

        Args:
            video_path: Path to input UAV video file.
            output_path: Optional path to save visual tracking result video.

        Returns:
            Dictionary summarizing sequence tracking results and compute cost.
        """
        raise NotImplementedError("AdaptivePerceptionPipeline.run_video is a stub.")

