"""Script to train and serialize the Random Forest mode selector."""

import argparse


def train_selector(
    training_data_path: str = "data/processed/training_set.csv",
    model_output_path: str = "weights/selector_rf.joblib",
    config_path: str = "configs/default.yaml",
) -> None:
    """Train Random Forest classifier on condition features and save model artifact.

    Args:
        training_data_path: Path to CSV with features and target mode labels.
        model_output_path: Destination path for trained model artifact.
        config_path: Path to configuration file.
    """
    raise NotImplementedError("train_selector is a stub.")


def main() -> None:
    """Main CLI entrypoint for training the mode selector."""
    parser = argparse.ArgumentParser(description="Train Random Forest mode selector.")
    parser.add_argument("--data", type=str, default="data/processed/training_set.csv", help="Path to training CSV")
    parser.add_argument("--output", type=str, default="weights/selector_rf.joblib", help="Output model path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    train_selector(args.data, args.output, args.config)


if __name__ == "__main__":
    main()

