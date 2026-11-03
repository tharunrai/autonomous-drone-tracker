"""Script to construct training labels for the mode selector."""

import argparse


def build_training_labels(
    features_path: str = "data/processed/features.csv",
    gt_path: str = "data/raw",
    output_path: str = "data/processed/training_set.csv",
    config_path: str = "configs/default.yaml",
) -> None:
    """Compare tracking accuracy across baseline modes to label frames FAST or ROBUST.

    Args:
        features_path: Path to extracted features dataset.
        gt_path: Path to ground truth annotations directory.
        output_path: Destination path for labeled dataset CSV.
        config_path: Path to configuration file.
    """
    raise NotImplementedError("build_training_labels is a stub.")


def main() -> None:
    """Main CLI entrypoint for label generation."""
    parser = argparse.ArgumentParser(description="Build mode selector training labels.")
    parser.add_argument("--features", type=str, default="data/processed/features.csv", help="Extracted features CSV")
    parser.add_argument("--gt", type=str, default="data/raw", help="Ground truth annotations directory")
    parser.add_argument("--output", type=str, default="data/processed/training_set.csv", help="Labeled output path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    build_training_labels(args.features, args.gt, args.output, args.config)


if __name__ == "__main__":
    main()

