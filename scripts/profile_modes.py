"""Script to profile Always FAST vs Always ROBUST perception modes."""

import argparse
from typing import Dict, Any


def profile_modes(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Profile latency, detector calls, and memory consumption for FAST vs ROBUST modes.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary of profiling measurements per mode.
    """
    raise NotImplementedError("profile_modes is a stub.")


def main() -> None:
    """Main CLI entrypoint for profiling perception modes."""
    parser = argparse.ArgumentParser(description="Profile FAST and ROBUST perception modes.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    profile_modes(args.config)


if __name__ == "__main__":
    main()

