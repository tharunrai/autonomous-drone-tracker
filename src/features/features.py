"""Condition feature extraction: blur, brightness, bounding box area, confidence, and camera motion."""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class ConditionFeatureExtractor:
    """Extracts lightweight visual and motion condition features from video frames."""

    def __init__(self, downscale_flow_width: int = 320) -> None:
        """Initialize feature extractor.

        Args:
            downscale_flow_width: Width to downscale frames for optical flow motion estimation.
        """
        self.downscale_width = downscale_flow_width
        self.prev_gray_small: Optional[np.ndarray] = None
        self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.max_gap = 60  # cap for frames_since_target
        self.frames_since_target = self.max_gap
        self.seen_target = False

    def extract(
        self,
        frame: np.ndarray,
        tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute condition features for the current frame.

        Args:
            frame: Input video frame (BGR format).
            tracks: Optional list of active track dictionaries with 'box' and 'conf'.

        Returns:
            Dictionary containing 'blur', 'mean_brightness', 'box_area_ratio',
            'mean_confidence', and 'camera_motion'.
        """
        h, w = frame.shape[:2]
        frame_area = float(h * w) if h > 0 and w > 0 else 1.0

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur metric: Variance of Laplacian
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Mean brightness (0-255 scale)
        mean_brightness = float(np.mean(gray))

        # 3. Box area ratio & 4. Mean confidence
        box_area_ratio = 0.0
        mean_confidence = 0.0
        if tracks and len(tracks) > 0:
            areas = []
            confs = []
            for t in tracks:
                box = t.get("box")
                if box:
                    bw = max(0.0, box[2] - box[0])
                    bh = max(0.0, box[3] - box[1])
                    areas.append(bw * bh)
                if "conf" in t:
                    confs.append(float(t["conf"]))

            if areas:
                box_area_ratio = float(np.mean(areas) / frame_area)
            if confs:
                mean_confidence = float(np.mean(confs))

        # 5. Camera motion: Optical flow magnitude on downscaled grayscale
        camera_motion = 0.0
        target_w = self.downscale_width
        target_h = max(16, int(h * (target_w / float(w))))
        curr_gray_small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if self.prev_gray_small is not None:
            flow = self.dis_flow.calc(self.prev_gray_small, curr_gray_small, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            camera_motion = float(np.mean(mag))

        self.prev_gray_small = curr_gray_small

        # 6. Track state: how many targets are tracked, and for how long nothing has been.
        # A target that was there and is suddenly gone is the clearest sign the running
        # detector is missing it, so the selector can react to it directly.
        num_tracks = len(tracks) if tracks else 0
        if num_tracks:
            self.frames_since_target = 0
            self.seen_target = True
        else:
            self.frames_since_target = min(self.frames_since_target + 1, self.max_gap)

        return {
            "blur": round(laplacian_var, 2),
            "mean_brightness": round(mean_brightness, 2),
            "box_area_ratio": round(box_area_ratio, 6),
            "mean_confidence": round(mean_confidence, 4),
            "camera_motion": round(camera_motion, 4),
            "num_tracks": float(num_tracks),
            # Frames since a target was last tracked (capped). Before any target is seen it
            # stays at the cap, so "never seen" and "lost long ago" look the same.
            "frames_since_target": float(self.frames_since_target if self.seen_target else self.max_gap),
        }

    def reset(self) -> None:
        """Reset historical optical flow and track-gap state."""
        self.prev_gray_small = None
        self.frames_since_target = self.max_gap
        self.seen_target = False

