"""Perception mode selectors (rule-based and ML placeholder) and hysteresis smoothing."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import pandas as pd


class RuleBasedSelector:
    """Heuristic mode selector deciding between FAST and ROBUST perception modes."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        """Initialize selector with configurable decision thresholds.

        Args:
            thresholds: Dictionary of feature thresholds from config.yaml.
        """
        defaults = {
            "blur_threshold": 100.0,
            "area_ratio_threshold": 0.003,
            "conf_threshold": 0.45,
            "camera_motion_threshold": 3.5,
        }
        self.thresholds = {**defaults, **(thresholds or {})}

    def predict(self, features: Dict[str, float]) -> str:
        """Determine whether current frame conditions warrant ROBUST mode or FAST mode.

        Args:
            features: Dictionary of extracted condition features.

        Returns:
            'ROBUST' if adverse environmental conditions are detected, else 'FAST'.
        """
        blur = features.get("blur", 200.0)
        box_area = features.get("box_area_ratio", 0.01)
        conf = features.get("mean_confidence", 0.8)
        motion = features.get("camera_motion", 0.0)

        # Condition checks pushing toward ROBUST:
        is_blurry = blur < self.thresholds["blur_threshold"]
        is_tiny_target = 0.0 < box_area < self.thresholds["area_ratio_threshold"]
        is_low_conf = 0.0 < conf < self.thresholds["conf_threshold"]
        is_fast_motion = motion > self.thresholds["camera_motion_threshold"]

        if is_blurry or is_tiny_target or is_low_conf or is_fast_motion:
            return "ROBUST"
        return "FAST"


class HysteresisSmoother:
    """Smoother preventing frequent rapid oscillations between perception modes."""

    def __init__(
        self,
        min_mode_duration: int = 15,
        persistence_frames: int = 5,
        initial_mode: str = "FAST",
    ) -> None:
        """Initialize smoother with stability duration constraints.

        Args:
            min_mode_duration: Minimum frames to stay in a mode before switching.
            persistence_frames: Consecutive frames candidate mode must persist before trigger.
            initial_mode: Default starting mode.
        """
        self.min_mode_duration = min_mode_duration
        self.persistence_frames = persistence_frames
        self.current_mode = initial_mode
        self.frames_in_current_mode = 0
        self.candidate_mode: Optional[str] = None
        self.candidate_count = 0

    def update(self, proposed_mode: str) -> str:
        """Filter proposed mode through hysteresis duration rules.

        Args:
            proposed_mode: Mode suggested by selector ('FAST' or 'ROBUST').

        Returns:
            Smoothed operational mode for current frame.
        """
        self.frames_in_current_mode += 1

        if proposed_mode != self.current_mode:
            if proposed_mode == self.candidate_mode:
                self.candidate_count += 1
            else:
                self.candidate_mode = proposed_mode
                self.candidate_count = 1

            if (
                self.frames_in_current_mode >= self.min_mode_duration
                and self.candidate_count >= self.persistence_frames
            ):
                self.current_mode = proposed_mode
                self.frames_in_current_mode = 0
                self.candidate_mode = None
                self.candidate_count = 0
        else:
            self.candidate_mode = None
            self.candidate_count = 0

        return self.current_mode

    def reset(self, initial_mode: str = "FAST") -> None:
        """Reset internal counters to initial mode state."""
        self.current_mode = initial_mode
        self.frames_in_current_mode = 0
        self.candidate_mode = None
        self.candidate_count = 0


class MLSelector:
    """Machine learning mode selector using a serialized Random Forest classifier (placeholder)."""

    def __init__(self, model_path: Union[str, Path] = "weights/selector_rf.joblib") -> None:
        """Initialize ML selector, loading weights if available.

        Args:
            model_path: Path to trained joblib model artifact.
        """
        self.model_path = Path(model_path)
        self.model: Optional[Any] = None
        if self.model_path.is_file():
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def predict(self, features: Dict[str, float]) -> str:
        """Predict mode using Random Forest classifier.

        Args:
            features: Dictionary of extracted frame features.

        Returns:
            Predicted mode name ('FAST' or 'ROBUST').
        """
        if self.model is None:
            raise NotImplementedError(
                f"MLSelector: No trained model found at '{self.model_path}'. "
                "TODO: Execute src/training/train_selector.py to train and save the Random Forest model."
            )

        df = pd.DataFrame([features])
        pred = self.model.predict(df)[0]
        return str(pred)
