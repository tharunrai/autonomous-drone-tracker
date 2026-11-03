"""Perception mode definitions and configurations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class Mode(str, Enum):
    """Operational perception modes."""

    FAST = "FAST"
    ROBUST = "ROBUST"


@dataclass
class ModeConfig:
    """Configuration parameters for a perception mode."""

    model_weights: str
    imgsz: int
    detect_every_n_frames: int
    use_roi_crop: bool


def load_mode_config(config_dict: Dict[str, Any], mode: Mode) -> ModeConfig:
    """Load mode configuration parameters from a configuration dictionary.

    Args:
        config_dict: Full parsed configuration dictionary.
        mode: Target Mode enum (FAST or ROBUST).

    Returns:
        ModeConfig populated with settings from the configuration.
    """
    section_name = "fast_mode" if mode == Mode.FAST else "robust_mode"
    section = config_dict.get(section_name, {})
    return ModeConfig(
        model_weights=section.get("model_weights", ""),
        imgsz=section.get("imgsz", 640),
        detect_every_n_frames=section.get("detect_every_n_frames", 1),
        use_roi_crop=section.get("use_roi_crop", False),
    )

