"""CLI runner for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking."""

import argparse
from pathlib import Path
import sys
import numpy as np


def check_setup() -> bool:
    """Verify dependencies, CUDA availability, and YOLO weights."""
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
        ("matplotlib", "matplotlib"),
        ("streamlit", "streamlit"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
        ("joblib", "joblib"),
    ]

    print("\n[1] Checking Dependencies:")
    for pkg_name, import_name in dependencies:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "ok")
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
            print(f"  [PASS] CUDA is available! Device: {device_name} (CUDA {cuda_version})")
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
            print(f"  [PASS] Successfully loaded {yolo_weights_path.name} and ran inference.")
        except Exception as e:
            print(f"  [FAIL] Failed YOLO inference test: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    print("OVERALL RESULT:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 60)
    return all_passed


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="UAV Tracking Pipeline CLI Runner")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument(
        "--mode",
        type=str,
        default="ADAPTIVE",
        choices=["FAST", "ROBUST", "ADAPTIVE"],
        help="Perception mode (FAST, ROBUST, ADAPTIVE)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--conf", type=float, default=None, help="Override detector confidence threshold")
    parser.add_argument("--check-setup", action="store_true", help="Verify dependencies, GPU, and YOLO weights")

    args = parser.parse_args()

    if args.check_setup:
        success = check_setup()
        sys.exit(0 if success else 1)

    if not args.video:
        parser.print_help()
        print("\n[ERROR] Please provide --video PATH or use --check-setup.")
        sys.exit(1)

    # Lazy import pipeline to ensure fast CLI response for --help or --check-setup
    from src.pipeline.pipeline import Mode, TrackingPipeline

    print(f"\n[INFO] Starting tracking pipeline on '{args.video}' with mode '{args.mode}'...")
    pipeline = TrackingPipeline(mode=Mode(args.mode), config_path=args.config, custom_conf=args.conf)

    summary = pipeline.run_video(
        input_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    print(f"  Mode:                {args.mode}")
    print(f"  Processed Frames:    {summary['total_frames']}")
    print(f"  Total Time:          {summary['total_time_seconds']}s")
    print(f"  Average FPS:         {summary['avg_fps']}")
    print(f"  Detector Calls:      {summary['detector_calls']} ({summary['detector_call_ratio'] * 100:.1f}%)")
    print(f"  Mode Switches:       {summary['mode_switches']}")
    print(f"  Unique Targets:      {summary.get('total_unique_targets', 0)}")
    print(f"  Annotated Video:     {summary['output_video']}")
    print(f"  Metrics CSV:         {summary['metrics_csv']}")
    print(f"  Positions CSV:       {summary['tracks_csv']}")
    print("=" * 50)

    tracks_df = summary.get("tracks_df")
    if tracks_df is not None and not tracks_df.empty:
        print("\n[SAMPLE OBJECT POSITIONS]")
        sample = tracks_df[["frame", "track_id", "center_x", "center_y", "bbox_width", "bbox_height", "confidence"]].head(10)
        print(sample.to_string(index=False))
        print("...")


if __name__ == "__main__":
    main()
