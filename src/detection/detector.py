"""YOLO detector wrapper module for UAV detection."""

from typing import Any, Dict, List, Optional
import numpy as np


class YOLODetector:
    """Wrapper around YOLO object detector for UAV detection."""

    def __init__(self, model_weights: str, imgsz: int = 640, conf_thres: float = 0.25) -> None:
        """Initialize detector with model weights and inference resolution.

        Args:
            model_weights: Path to YOLO model weights file.
            imgsz: Input image resolution for inference.
            conf_thres: Detection confidence threshold.
        """
        self.model_weights = model_weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        # Stub: will initialize YOLO model instance

    def detect(self, frame: np.ndarray, roi: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Run object detection on a single frame.

        Args:
            frame: Input video frame as a NumPy array (BGR/RGB).
            roi: Optional region of interest [x1, y1, x2, y2] to crop before detection.

        Returns:
            List of detected bounding boxes with confidence and class id.
        """
        raise NotImplementedError("YOLODetector.detect is a stub.")

