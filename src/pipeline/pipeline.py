"""Tracking pipeline orchestrating detection, tracking, condition extraction, and mode switching."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.detection.detector import YOLODetector
from src.features.features import ConditionFeatureExtractor
from src.selector.selector import HysteresisSmoother, RuleBasedSelector
from src.tracking.tracker import ByteTrackerWrapper
from src.utils.config import load_config
from src.utils.video_io import VideoWriterWrapper, get_video_properties, read_video_frames


class Mode(str, Enum):
    """Operational perception modes."""

    FAST = "FAST"
    ROBUST = "ROBUST"
    ADAPTIVE = "ADAPTIVE"


@dataclass
class ModeConfig:
    """Configuration parameters for a perception mode."""

    weights: str
    imgsz: int
    conf: float
    detect_every_n_frames: int


def get_mode_configs(config_dict: Dict[str, Any]) -> Tuple[ModeConfig, ModeConfig]:
    """Parse FAST and ROBUST ModeConfigs from configuration dictionary.

    Args:
        config_dict: Full parsed configuration dictionary.

    Returns:
        Tuple of (fast_mode_config, robust_mode_config).
    """
    modes_section = config_dict.get("modes", {})
    fast_data = modes_section.get("fast", {})
    robust_data = modes_section.get("robust", {})

    fast_cfg = ModeConfig(
        weights=fast_data.get("weights", "weights/yolov8n.pt"),
        imgsz=int(fast_data.get("imgsz", 640)),
        conf=float(fast_data.get("conf", 0.25)),
        detect_every_n_frames=int(fast_data.get("detect_every_n_frames", 3)),
    )
    robust_cfg = ModeConfig(
        weights=robust_data.get("weights", "weights/yolov8s.pt"),
        imgsz=int(robust_data.get("imgsz", 1280)),
        conf=float(robust_data.get("conf", 0.20)),
        detect_every_n_frames=int(robust_data.get("detect_every_n_frames", 1)),
    )
    return fast_cfg, robust_cfg


class TrackingPipeline:
    """End-to-end UAV tracking pipeline with FAST, ROBUST, and ADAPTIVE operational modes."""

    def __init__(
        self,
        mode: Union[Mode, str] = Mode.ADAPTIVE,
        config_path: Union[str, Path] = "config.yaml",
        custom_conf: Optional[float] = None,
    ) -> None:
        """Initialize pipeline components according to target operational mode.

        Args:
            mode: Pipeline mode ('FAST', 'ROBUST', or 'ADAPTIVE').
            config_path: Path to YAML configuration file.
            custom_conf: Optional override for detector confidence threshold.
        """
        self.mode = Mode(mode) if isinstance(mode, str) else mode
        self.config = load_config(config_path)

        self.fast_cfg, self.robust_cfg = get_mode_configs(self.config)
        if custom_conf is not None:
            self.fast_cfg.conf = custom_conf
            self.robust_cfg.conf = custom_conf

        classes_filter = self.config.get("classes") or None

        # Instantiate required detectors
        self.fast_detector: Optional[YOLODetector] = None
        self.robust_detector: Optional[YOLODetector] = None

        if self.mode in (Mode.FAST, Mode.ADAPTIVE):
            self.fast_detector = YOLODetector(
                weights=self.fast_cfg.weights,
                imgsz=self.fast_cfg.imgsz,
                conf=self.fast_cfg.conf,
                classes=classes_filter,
            )
        if self.mode in (Mode.ROBUST, Mode.ADAPTIVE):
            self.robust_detector = YOLODetector(
                weights=self.robust_cfg.weights,
                imgsz=self.robust_cfg.imgsz,
                conf=self.robust_cfg.conf,
                classes=classes_filter,
            )

        # Initialize tracking components
        trk_cfg = self.config.get("tracking", {})
        self.tracker = ByteTrackerWrapper(
            config_path=trk_cfg.get("tracker_type", "bytetrack.yaml"),
            trail_length=int(trk_cfg.get("trail_length", 30)),
        )

        # Initialize condition extractor
        feat_cfg = self.config.get("features", {})
        self.extractor = ConditionFeatureExtractor(
            downscale_flow_width=int(feat_cfg.get("downscale_flow_width", 320))
        )

        # Initialize mode selector and smoother
        sel_cfg = self.config.get("selector", {})
        self.selector = RuleBasedSelector(
            thresholds=sel_cfg.get("rule_thresholds", {})
        )

        hyst_cfg = self.config.get("hysteresis", {})
        self.smoother = HysteresisSmoother(
            min_mode_duration=int(hyst_cfg.get("min_mode_duration", 15)),
            persistence_frames=int(hyst_cfg.get("persistence_frames", 5)),
            initial_mode="FAST",
        )

    def run_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end tracking pipeline on a video file.

        Args:
            input_path: Path to source video file.
            output_path: Destination path for annotated MP4 video.
            max_frames: Optional cap on frames to process.
            progress_callback: Optional callable(frame_idx, total_frames, fps).

        Returns:
            Dictionary containing metrics summary and DataFrame of per-frame records.
        """
        input_file = Path(input_path).resolve()
        if not input_file.is_file():
            raise FileNotFoundError(f"Video file not found: {input_file}")

        props = get_video_properties(input_file)
        fps = props["fps"]
        width = int(props["width"])
        height = int(props["height"])
        total_frames = int(props["total_frames"])
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        # Determine output video path
        if output_path is None:
            videos_dir = Path(self.config.get("paths", {}).get("videos_dir", "outputs/videos"))
            videos_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = videos_dir / f"{input_file.stem}_{self.mode.value.lower()}_annotated.mp4"
        else:
            output_video_path = Path(output_path).resolve()

        writer = VideoWriterWrapper(output_video_path, fps=fps, width=width, height=height)

        # Reset state
        self.tracker.reset()
        self.extractor.reset()
        self.smoother.reset(initial_mode="FAST")

        frame_records: List[Dict[str, Any]] = []
        detector_calls = 0
        mode_switches = 0
        prev_active_mode = "FAST"
        previous_tracks: List[Dict[str, Any]] = []

        total_start_time = time.perf_counter()

        for frame_idx, frame in read_video_frames(input_file, max_frames=max_frames):
            frame_t0 = time.perf_counter()

            # 1. Extract condition features based on previous tracks & current frame
            features = self.extractor.extract(frame, tracks=previous_tracks)

            # 2. Select operational mode for this frame
            if self.mode == Mode.FAST:
                current_mode_name = "FAST"
                active_cfg = self.fast_cfg
                detector = self.fast_detector
            elif self.mode == Mode.ROBUST:
                current_mode_name = "ROBUST"
                active_cfg = self.robust_cfg
                detector = self.robust_detector
            else:  # ADAPTIVE
                raw_mode = self.selector.predict(features)
                current_mode_name = self.smoother.update(raw_mode)
                if current_mode_name == "FAST":
                    active_cfg = self.fast_cfg
                    detector = self.fast_detector
                else:
                    active_cfg = self.robust_cfg
                    detector = self.robust_detector

            if frame_idx > 0 and current_mode_name != prev_active_mode:
                mode_switches += 1
            prev_active_mode = current_mode_name

            # 3. Detection & Tracking
            should_detect = (frame_idx % active_cfg.detect_every_n_frames) == 0
            if should_detect and detector is not None:
                detector_calls += 1
                boxes = detector.detect(frame)
                tracks = self.tracker.update(boxes, frame)
            else:
                # Detector skipped: advance tracker Kalman prediction
                tracks = self.tracker.update(None, frame)

            previous_tracks = tracks
            frame_dt = time.perf_counter() - frame_t0
            instant_fps = 1.0 / (frame_dt + 1e-6)

            # 4. Render visual overlays
            annotated_frame = self._render_frame(
                frame.copy(),
                tracks=tracks,
                mode_name=current_mode_name,
                fps=instant_fps,
                frame_idx=frame_idx,
            )
            writer.write_frame(annotated_frame)

            # 5. Log per-frame telemetry
            record = {
                "frame": frame_idx,
                "mode": current_mode_name,
                "fps": round(instant_fps, 1),
                "num_tracks": len(tracks),
                "detector_invoked": should_detect,
                **features,
            }
            frame_records.append(record)

            if progress_callback:
                progress_callback(frame_idx + 1, total_frames, instant_fps)

        final_video_path = writer.release()
        total_time = time.perf_counter() - total_start_time
        processed_count = len(frame_records)
        avg_fps = float(processed_count / (total_time + 1e-6))

        # Save metrics CSV
        metrics_df = pd.DataFrame(frame_records)
        metrics_dir = Path(self.config.get("paths", {}).get("metrics_dir", "outputs/metrics"))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f"{input_file.stem}_{self.mode.value.lower()}_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)

        summary = {
            "total_frames": processed_count,
            "total_time_seconds": round(total_time, 2),
            "avg_fps": round(avg_fps, 1),
            "detector_calls": detector_calls,
            "detector_call_ratio": round(detector_calls / max(1, processed_count), 3),
            "mode_switches": mode_switches,
            "output_video": str(final_video_path),
            "metrics_csv": str(metrics_file),
            "records_df": metrics_df,
        }
        return summary

    def _render_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        mode_name: str,
        fps: float,
        frame_idx: int,
    ) -> np.ndarray:
        """Render target bounding boxes, ID banners, motion trails, and HUD telemetry overlay."""
        # Color palette: Emerald for FAST, Amber/Crimson for ROBUST
        mode_color = (0, 200, 50) if mode_name == "FAST" else (30, 80, 240)

        # Draw motion trails and bounding boxes
        for track in tracks:
            track_id = track["id"]
            box = track["box"]
            conf = track["conf"]
            trail = track.get("trail", [])

            # Draw motion trail
            if len(trail) > 1:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), mode_color, 2)

            # Draw label banner
            label = f"ID #{track_id} ({conf:.2f})"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), mode_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw top HUD banner
        hud_text = f"Frame: {frame_idx:04d} | Mode: {mode_name} | FPS: {fps:.1f} | Active Tracks: {len(tracks)}"
        (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (10, 10), (20 + hw, 20 + hh + 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (20 + hw, 20 + hh + 10), mode_color, 2)
        cv2.putText(
            frame,
            hud_text,
            (15, 18 + hh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame
