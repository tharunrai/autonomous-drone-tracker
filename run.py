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


METRIC_COLUMNS = [
    ("run", "Run"), ("frames", "Frames"), ("seconds", "Time s"), ("fps", "FPS"),
    ("detector_calls", "Det calls"), ("detector_pct", "Det %"), ("robust_pct", "Robust %"),
    ("switches", "Switches"), ("targets", "Targets"), ("frames_with_target_pct", "Frames w/ target %"),
    ("agreement_pct", "Agreement %"),
]


def metrics_table(rows):
    """Format a list of run_metrics() dicts (each with a 'run' name) as an aligned text table."""
    import pandas as pd

    df = pd.DataFrame(rows)
    cols = [(k, label) for k, label in METRIC_COLUMNS if k in df.columns]
    return df[[k for k, _ in cols]].rename(columns=dict(cols)).to_string(index=False)


def run_comparison(args) -> None:
    """Run FAST, ROBUST and ADAPTIVE (every available selector) on one video and print the metrics."""
    import warnings

    import numpy as np
    import pandas as pd

    from src.evaluation.metrics import run_metrics
    from src.pipeline.pipeline import Mode, TrackingPipeline
    from src.selector.selector import SELECTOR_LABELS, available_selectors
    from src.utils.config import load_config

    config = load_config(args.config)
    selectors = [args.selector] if args.selector else list(reversed(available_selectors(config)))
    plan = [("Fast", "FAST", None), ("Robust", "ROBUST", None)]
    plan += [(f"Adaptive + {SELECTOR_LABELS[s]}", "ADAPTIVE", s) for s in selectors]

    video = Path(args.video)
    print(f"\n[INFO] Comparing {len(plan)} runs on '{video.name}' "
          f"(max frames: {args.max_frames or 'all'}). Each run is timed after a warm-up frame.")
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    summaries, rows = {}, []
    for name, mode, selector in plan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = TrackingPipeline(Mode(mode), config_path=args.config, custom_conf=args.conf,
                                        selector_type=selector, selector_threshold=args.threshold)
        for det in (pipeline.fast_detector, pipeline.robust_detector):
            if det is not None:
                det.detect(blank)  # warm-up so the first run isn't unfairly slow
        print(f"  running {name}...", flush=True)
        summaries[name] = pipeline.run_video(video, max_frames=args.max_frames, save_outputs=False)

    reference = summaries["Robust"]
    for name, _, _ in plan:
        rows.append({"run": name, **run_metrics(summaries[name], reference)})

    print("\n" + "=" * 100)
    print(f"METRICS · {video.name}")
    print("=" * 100)
    print(metrics_table(rows))
    print("=" * 100)
    print("Det % = share of frames the detector ran on · Robust % = share of frames in ROBUST mode")
    print("Agreement % = how much of ALWAYS-ROBUST's result the run keeps (share of its confident targets "
          "also found).\nIt is relative to Robust, not ground truth. Detector calls are the compute measure; "
          "on a fast GPU, FPS barely differs.")

    out_dir = Path(load_config(args.config).get("paths", {}).get("metrics_dir", "outputs/metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{video.stem}_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


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
    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        choices=["rules", "tree", "forest"],
        help="Who decides FAST vs ROBUST in ADAPTIVE mode (default: selector.type in config.yaml)",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Tree/forest: switch to ROBUST at this probability or higher (default: config)")
    parser.add_argument("--compare", action="store_true",
                        help="Run Fast, Robust and Adaptive (every selector) and print one metrics table")
    parser.add_argument("--check-setup", action="store_true", help="Verify dependencies, GPU, and YOLO weights")

    args = parser.parse_args()

    if args.check_setup:
        success = check_setup()
        sys.exit(0 if success else 1)

    if not args.video:
        parser.print_help()
        print("\n[ERROR] Please provide --video PATH or use --check-setup.")
        sys.exit(1)

    if args.compare:
        run_comparison(args)
        return

    # Lazy import pipeline to ensure fast CLI response for --help or --check-setup
    from src.evaluation.metrics import run_metrics
    from src.pipeline.pipeline import Mode, TrackingPipeline
    from src.selector.selector import SELECTOR_LABELS

    print(f"\n[INFO] Starting tracking pipeline on '{args.video}' with mode '{args.mode}'...")
    pipeline = TrackingPipeline(mode=Mode(args.mode), config_path=args.config, custom_conf=args.conf,
                                selector_type=args.selector, selector_threshold=args.threshold)

    summary = pipeline.run_video(
        input_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
    )
    m = run_metrics(summary)

    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    print(f"  Mode:                {args.mode}")
    if summary.get("selector"):
        print(f"  Decided by:          {SELECTOR_LABELS[summary['selector']]}")
    print(f"  Processed Frames:    {summary['total_frames']}")
    print(f"  Total Time:          {summary['total_time_seconds']}s")
    print(f"  Average FPS:         {summary['avg_fps']}")
    print(f"  Detector Calls:      {summary['detector_calls']} ({summary['detector_call_ratio'] * 100:.1f}%)")
    print(f"  Time in ROBUST:      {m['robust_pct']}%")
    print(f"  Mode Switches:       {summary['mode_switches']}")
    print(f"  Unique Targets:      {summary.get('total_unique_targets', 0)}")
    print(f"  Frames with target:  {m['frames_with_target_pct']}%  (avg {m['mean_targets_per_frame']} per frame)")
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
