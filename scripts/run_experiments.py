"""Script to evaluate Always FAST, Always ROBUST, and Adaptive systems."""

import argparse


def run_experiments(
    config_path: str = "configs/default.yaml",
    output_dir: str = "outputs",
) -> None:
    """Execute end-to-end evaluation pipeline comparing the 3 perception modes.

    Args:
        config_path: Path to YAML configuration file.
        output_dir: Directory where metric reports and visualizations will be saved.
    """
    raise NotImplementedError("run_experiments is a stub.")


def main() -> None:
    """Main CLI entrypoint for running evaluation experiments."""
    parser = argparse.ArgumentParser(description="Run benchmarking experiments across all three modes.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for results")
    args = parser.parse_args()
    run_experiments(args.config, args.output_dir)


if __name__ == "__main__":
    main()

