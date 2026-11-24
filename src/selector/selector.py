"""Perception mode selectors and hysteresis smoothing.

Three selectors decide, frame by frame, whether the cheap FAST setup is enough or the
expensive ROBUST setup is needed:

* ``rules``  - RuleBasedSelector, hand-set thresholds. Kept as the baseline to beat.
* ``tree``   - a trained DecisionTreeClassifier (MLSelector).
* ``forest`` - a trained RandomForestClassifier (MLSelector).

The learned selectors are trained by ``python -m src.training.train_selector``.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
import joblib
import numpy as np

# Per-frame features produced by ConditionFeatureExtractor that the learned selectors use.
BASE_FEATURES = ["blur", "mean_brightness", "box_area_ratio", "mean_confidence", "camera_motion"]
ROLLING_WINDOW = 15  # frames averaged into the "_avg" features

SELECTOR_LABELS = {"rules": "Rules", "tree": "Decision tree", "forest": "Random forest"}
_ALIASES = {"rule_based": "rules", "rule": "rules", "ml": "forest", "rf": "forest",
            "random_forest": "forest", "decision_tree": "tree"}


class RollingFeatures:
    """Turns per-frame features into the vector a learned selector sees.

    The vector is the current value of each base feature plus its average over the last
    ``window`` frames (causal: only past frames). Training and inference both build their
    inputs with this class, so the model always sees exactly the same columns.
    """

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self.window = window
        self.history: deque = deque(maxlen=window)

    @staticmethod
    def names() -> List[str]:
        return BASE_FEATURES + [f"{n}_avg" for n in BASE_FEATURES]

    def update(self, features: Dict[str, float]) -> List[float]:
        row = [float(features.get(n, 0.0)) for n in BASE_FEATURES]
        self.history.append(row)
        return row + np.mean(self.history, axis=0).tolist()

    def reset(self) -> None:
        self.history.clear()


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

    def reset(self) -> None:
        """Rules keep no state between frames."""


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
    """Learned mode selector (decision tree or random forest) loaded from a joblib bundle.

    The bundle is written by ``src.training.train_selector`` and holds the fitted sklearn
    model, the feature names, the decision threshold and the training report.
    """

    def __init__(self, model_path: Union[str, Path], decide_every: Optional[int] = None,
                 threshold: Optional[float] = None) -> None:
        """Load a trained selector bundle.

        Args:
            model_path: Path to the joblib bundle (e.g. weights/selector_forest.joblib).
            decide_every: Ask the model every N frames and reuse the answer in between.
                Rolling features still update every frame. Defaults to the bundle's value.
            threshold: Ask for ROBUST when the model's probability is at least this. Higher
                saves more compute and catches fewer targets. Defaults to the bundle's 0.5.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No trained selector at '{self.model_path}'. "
                "Train one with: python -m src.training.train_selector"
            )
        bundle: Dict[str, Any] = joblib.load(self.model_path)
        self.bundle = bundle
        self.kind: str = bundle.get("kind", "forest")
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1  # one row at a time: thread start-up would cost more than it saves
        self.feature_names: List[str] = bundle["features"]
        self.threshold: float = float(threshold if threshold is not None else bundle.get("threshold", 0.5))
        # Raw sklearn trees: calling them directly skips predict_proba's per-call input checks,
        # which cost ~30 ms per row for a 100-tree forest (vs well under 1 ms this way).
        self._trees = [est.tree_ for est in getattr(self.model, "estimators_", [self.model])]
        self.decide_every = int(decide_every or bundle.get("decide_every", 3))
        self.rolling = RollingFeatures(int(bundle.get("window", ROLLING_WINDOW)))
        self._robust_col = list(self.model.classes_).index("ROBUST")
        self._frame = 0
        self._last_mode = "FAST"
        self.last_robust_prob = 0.0

    def predict(self, features: Dict[str, float]) -> str:
        """Return 'FAST' or 'ROBUST' for the current frame."""
        vec = self.rolling.update(features)
        if self._frame % self.decide_every == 0:
            self.last_robust_prob = float(self.robust_probability(vec))
            self._last_mode = "ROBUST" if self.last_robust_prob >= self.threshold else "FAST"
        self._frame += 1
        return self._last_mode

    def robust_probability(self, vec: List[float]) -> float:
        """Same value as model.predict_proba(...)[ROBUST], computed straight from the trees."""
        x = np.asarray([vec], dtype=np.float32)
        total = 0.0
        for tree in self._trees:
            counts = np.ravel(tree.predict(x)[0])  # class weights/fractions in the leaf this row lands in
            total += counts[self._robust_col] / max(counts.sum(), 1e-12)
        return total / len(self._trees)

    def reset(self) -> None:
        self.rolling.reset()
        self._frame = 0
        self._last_mode = "FAST"
        self.last_robust_prob = 0.0


def normalize_selector_kind(kind: Optional[str]) -> str:
    kind = (kind or "rules").strip().lower()
    kind = _ALIASES.get(kind, kind)
    if kind not in SELECTOR_LABELS:
        raise ValueError(f"Unknown selector '{kind}'. Use one of: {', '.join(SELECTOR_LABELS)}")
    return kind


def selector_model_path(kind: str, config: Dict[str, Any]) -> Path:
    models = (config.get("selector") or {}).get("models") or {}
    return Path(models.get(kind, f"weights/selector_{kind}.joblib"))


def available_selectors(config: Dict[str, Any]) -> List[str]:
    """Selectors that can run right now: rules, plus every learned model that has been trained."""
    return ["rules"] + [k for k in ("tree", "forest") if selector_model_path(k, config).is_file()]


def make_selector(kind: Optional[str], config: Dict[str, Any], threshold: Optional[float] = None):
    """Build the requested selector. Returns (selector, kind actually used).

    ``threshold`` (learned selectors only) overrides ``selector.threshold`` from config.

    A learned selector whose model file is missing falls back to rules with a warning, and
    the returned kind says so, so callers can show the truth to the user.
    """
    kind = normalize_selector_kind(kind)
    sel_cfg = config.get("selector") or {}
    if kind == "rules":
        return RuleBasedSelector(thresholds=sel_cfg.get("rule_thresholds", {})), "rules"
    path = selector_model_path(kind, config)
    if not path.is_file():
        warnings.warn(f"{SELECTOR_LABELS[kind]} selector not trained yet ({path} missing); "
                      "using rules. Train with: python -m src.training.train_selector")
        return RuleBasedSelector(thresholds=sel_cfg.get("rule_thresholds", {})), "rules"
    if threshold is None:
        threshold = sel_cfg.get("threshold")
    return MLSelector(path, decide_every=sel_cfg.get("decide_every"), threshold=threshold), kind

