"""Unified CLI entrypoint for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking."""

import argparse
import sys
from pathlib import Path
import numpy as np


def check_setup() -> bool:
    """Run verification checks for the project environment."""
    all_passed = True
    print("=" * 60)
    print("Condition-Aware UAV Tracking: Environment Setup Verification")
    print("=" * 60)

    dependencies = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("streamlit", "streamlit"),
        ("supervision", "supervision"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("joblib", "joblib"),
        ("pytest", "pytest"),
    ]

    print("\n[1] Checking Dependencies:")
    for pkg_name, import_name in dependencies:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [PASS] {pkg_name} (imported as {import_name}) - version: {version}")
        except Exception as e:
            print(f"  [FAIL] {pkg_name} could not be imported: {e}")
            all_passed = False

    print("\n[2] Checking CUDA / Hardware Acceleration:")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  [PASS] CUDA is available! Device 0: {device_name} (CUDA {cuda_version})")
        else:
            print("  [INFO] CUDA is not available. PyTorch running in CPU-only mode.")
    except Exception as e:
        print(f"  [FAIL] Failed to inspect torch CUDA status: {e}")
        all_passed = False

    print("\n[3] Checking YOLO Weights and Test Inference:")
    yolo_weights_path = Path("weights/yolov8n.pt")
    if not yolo_weights_path.is_file():
        print(f"  [FAIL] YOLO weights file missing: {yolo_weights_path.resolve()}")
        all_passed = False
    else:
        try:
            from ultralytics import YOLO

            model = YOLO(str(yolo_weights_path))
            blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model.predict(source=blank_image, verbose=False)
            print(f"  [PASS] Successfully loaded {yolo_weights_path.name} and ran inference (boxes detected: {len(results[0].boxes)})")
        except Exception as e:
            print(f"  [FAIL] Failed YOLO inference test: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    print("OVERALL RESULT:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 60)
    return all_passed


def profile_modes(config_path: str = "config.yaml") -> None:
    """Profile latency, detector calls, and memory consumption for FAST vs ROBUST modes."""
    raise NotImplementedError("profile_modes is a stub.")


def extract_features(data_dir: str = "data/raw", output_dir: str = "data/processed", config_path: str = "config.yaml") -> None:
    """Extract condition features from video sequences."""
    raise NotImplementedError("extract_features is a stub.")


def build_labels(features_path: str = "data/processed/features.csv", gt_path: str = "data/raw", output_path: str = "data/processed/training_set.csv", config_path: str = "config.yaml") -> None:
    """Generate mode selection ground truth labels based on tracking accuracy."""
    raise NotImplementedError("build_labels is a stub.")


def train_selector(data_path: str = "data/processed/training_set.csv", model_output: str = "weights/selector_rf.joblib", config_path: str = "config.yaml") -> None:
    """Train Random Forest classifier for mode selection."""
    raise NotImplementedError("train_selector is a stub.")


def run_evaluation(config_path: str = "config.yaml", output_dir: str = "outputs") -> None:
    """Benchmark Always FAST, Always ROBUST, and Adaptive Perception modes."""
    raise NotImplementedError("run_evaluation is a stub.")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="UAV Tracking Pipeline CLI Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--check-setup", action="store_true", help="Verify dependencies, GPU, and YOLO weights")
    parser.add_argument("--profile", action="store_true", help="Profile baseline FAST and ROBUST modes")
    parser.add_argument("--extract-features", action="store_true", help="Extract frame-by-frame condition features")
    parser.add_argument("--build-labels", action="store_true", help="Build training labels from baseline tracks")
    parser.add_argument("--train-selector", action="store_true", help="Train Random Forest mode selector")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation comparing all 3 modes")

    args = parser.parse_args()

    if args.check_setup:
        success = check_setup()
        sys.exit(0 if success else 1)
    elif args.profile:
        profile_modes(args.config)
    elif args.extract_features:
        extract_features(config_path=args.config)
    elif args.build_labels:
        build_labels(config_path=args.config)
    elif args.train_selector:
        train_selector(config_path=args.config)
    elif args.evaluate:
        run_evaluation(config_path=args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

