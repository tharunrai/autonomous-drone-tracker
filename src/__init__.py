"""Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking source package."""

from src.detect_track import (
    AdaptivePerceptionPipeline,
    ByteTrackerWrapper,
    FixedModeBaselinePipeline,
    Mode,
    ModeConfig,
    YOLODetector,
    load_config,
    load_mode_config,
    load_tracking_results,
    save_tracking_results,
)
from src.evaluate import (
    compute_pipeline_cost,
    plot_accuracy_vs_cost,
    plot_mode_switching_timeline,
    run_trackeval,
)
from src.features import (
    ConditionFeatureExtractor,
    compute_brightness_stats,
    estimate_blur,
    estimate_camera_motion,
    extract_box_geometry,
)
from src.selector import (
    HysteresisSmoother,
    ModePredictor,
    generate_mode_labels,
    train_mode_selector,
)

__all__ = [
    "AdaptivePerceptionPipeline",
    "ByteTrackerWrapper",
    "ConditionFeatureExtractor",
    "FixedModeBaselinePipeline",
    "HysteresisSmoother",
    "Mode",
    "ModeConfig",
    "ModePredictor",
    "YOLODetector",
    "compute_brightness_stats",
    "compute_pipeline_cost",
    "estimate_blur",
    "estimate_camera_motion",
    "extract_box_geometry",
    "generate_mode_labels",
    "load_config",
    "load_mode_config",
    "load_tracking_results",
    "plot_accuracy_vs_cost",
    "plot_mode_switching_timeline",
    "run_trackeval",
    "save_tracking_results",
    "train_mode_selector",
]
