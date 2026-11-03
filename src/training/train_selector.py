"""Training pipeline for Random Forest perception mode selector (Placeholder)."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


def build_labels(
    sequences_dir: Union[str, Path] = "data/raw",
    ground_truth_dir: Union[str, Path] = "data/raw",
    output_path: Union[str, Path] = "data/processed/training_labels.csv",
    window_size: int = 15,
) -> pd.DataFrame:
    """Construct mode selection training labels by comparing baseline tracking accuracy.

    Intended Flow:
      1. Run both Always FAST and Always ROBUST pipelines on labelled video sequences.
      2. For each temporal window, evaluate tracking quality (IoU, HOTA) vs compute cost.
      3. If FAST mode preserves tracking fidelity within a tolerance margin, label window as 'FAST';
         otherwise, assign 'ROBUST'.
      4. Export feature vectors paired with optimal target mode labels.

    Args:
        sequences_dir: Path to raw video sequences.
        ground_truth_dir: Path to ground truth annotations.
        output_path: Destination path for labeled dataset CSV.
        window_size: Number of consecutive frames to aggregate for window labeling.

    Raises:
        NotImplementedError: Placeholder for selector training workflow.
    """
    # TODO: Implement offline sequence runner and Pareto label assignment
    raise NotImplementedError(
        "build_labels is a placeholder. "
        "TODO: Run FAST and ROBUST pipelines on annotated videos to generate training labels."
    )


def train_selector(
    training_data_path: Union[str, Path] = "data/processed/training_labels.csv",
    model_output_path: Union[str, Path] = "weights/selector_rf.joblib",
    n_estimators: int = 100,
    random_state: int = 42,
) -> Any:
    """Train a Random Forest classifier to select between FAST and ROBUST perception modes.

    Intended Flow:
      1. Load extracted condition feature matrix and target mode labels.
      2. Perform train/validation split by video sequence to prevent temporal data leakage.
      3. Fit sklearn.ensemble.RandomForestClassifier on condition features.
      4. Serialize trained classifier to weights/selector_rf.joblib.

    Args:
        training_data_path: Path to CSV dataset containing features and labels.
        model_output_path: Destination path for serialized joblib model.
        n_estimators: Number of trees in the Random Forest.
        random_state: Seed for reproducible training.

    Raises:
        NotImplementedError: Placeholder for selector model fitting.
    """
    # TODO: Implement sequence-split cross validation, RandomForest fitting, and joblib dump
    raise NotImplementedError(
        "train_selector is a placeholder. "
        "TODO: Fit sklearn.ensemble.RandomForestClassifier and serialize to weights/selector_rf.joblib."
    )
