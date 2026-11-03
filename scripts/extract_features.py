"""Script to extract condition features from raw video sequences."""

import argparse
from pathlib import Path


def run_feature_extraction(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    config_path: str = "configs/default.yaml",
) -> None:
    """Extract condition features for all video sequences in data directory.

    Args:
        data_dir: Directory containing input video datasets.
        output_dir: Output directory to store extracted feature CSVs.
        config_path: Path to configuration file.
    """
    raise NotImplementedError("run_feature_extraction is a stub.")


def main() -> None:
    """Main CLI entrypoint for condition feature extraction."""
    parser = argparse.ArgumentParser(description="Extract environmental condition features.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw video sequences")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to store extracted features")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    run_feature_extraction(args.data_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()

