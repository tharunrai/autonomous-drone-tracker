"""Runtime mode predictor module."""

from typing import Any, Dict


class ModePredictor:
    """Predicts perception mode based on extracted frame condition features."""

    def __init__(self, model_path: str) -> None:
        """Load trained mode selector model.

        Args:
            model_path: File path to serialized trained Random Forest model.
        """
        self.model_path = model_path
        self.model: Any = None
        # Stub: will load model via joblib

    def predict_mode(self, features: Dict[str, float]) -> str:
        """Predict whether the frame requires FAST or ROBUST mode.

        Args:
            features: Dictionary of current frame condition features.

        Returns:
            Selected mode name ('FAST' or 'ROBUST').
        """
        raise NotImplementedError("ModePredictor.predict_mode is a stub.")

