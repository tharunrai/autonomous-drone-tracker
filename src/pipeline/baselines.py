"""Fixed-mode baseline pipelines (Always FAST and Always ROBUST)."""

from typing import Any, Dict, Optional
import numpy as np
from src.pipeline.modes import Mode


class FixedModeBaselinePipeline:
    """Baseline tracking pipeline operating strictly in either FAST or ROBUST mode."""

    def __init__(self, mode: Mode, config: Dict[str, Any]) -> None:
        """Initialize fixed baseline pipeline.

        Args:
            mode: Fixed Mode (Mode.FAST or Mode.ROBUST).
            config: Full configuration dictionary.
        """
        self.mode = mode
        self.config = config
        # Stub: will initialize fixed detector and tracker

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Process a single frame using the fixed perception configuration.

        Args:
            frame: Input video frame.
            frame_idx: Index of the current frame.

        Returns:
            Dictionary with tracks and inference timing.
        """
        raise NotImplementedError("FixedModeBaselinePipeline.process_frame is a stub.")

    def run_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process an entire video sequence under fixed baseline mode.

        Args:
            video_path: Path to input UAV video file.
            output_path: Optional path to save visual tracking result video.

        Returns:
            Dictionary summarizing sequence tracking results and compute cost.
        """
        raise NotImplementedError("FixedModeBaselinePipeline.run_video is a stub.")

