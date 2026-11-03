"""Verification script to test environment, dependencies, GPU/CUDA, YOLO weights, and TrackEval."""

import sys
from pathlib import Path
import numpy as np


def check_setup() -> bool:
    """Run verification checks for the project environment.

    Returns:
        True if all checks pass, False otherwise.
    """
    all_passed = True
    print("=" * 60)
    print("Condition-Aware UAV Tracking: Environment Setup Verification")
    print("=" * 60)

    # 1. Dependency import and version checks
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

    # 2. CUDA / GPU availability
    print("\n[2] Checking CUDA / Hardware Acceleration:")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            print(f"  [PASS] CUDA is available! Device 0: {device_name} ({device_count} GPU(s), CUDA {cuda_version})")
        else:
            print("  [INFO] CUDA is not available. PyTorch running in CPU-only mode.")
    except Exception as e:
        print(f"  [FAIL] Failed to inspect torch CUDA status: {e}")
        all_passed = False

    # 3. Third-party TrackEval repository check
    print("\n[3] Checking TrackEval Repository:")
    trackeval_path = Path("third_party/TrackEval")
    if trackeval_path.is_dir() and (trackeval_path / "trackeval").is_dir():
        print(f"  [PASS] third_party/TrackEval found at: {trackeval_path.resolve()}")
    else:
        print(f"  [FAIL] third_party/TrackEval directory missing or incomplete: {trackeval_path.resolve()}")
        all_passed = False

    # 4. YOLO model weights and inference check
    print("\n[4] Checking YOLO Weights and Test Inference:")
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
    if all_passed:
        print("OVERALL RESULT: ALL CHECKS PASSED")
    else:
        print("OVERALL RESULT: SOME CHECKS FAILED")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)

