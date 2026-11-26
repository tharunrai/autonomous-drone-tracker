"""Evaluation metrics and benchmarking helpers."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def compute_fps(frame_count: int, elapsed_time_seconds: float) -> float:
    """Compute overall frames per second (FPS).

    Args:
        frame_count: Total processed video frames.
        elapsed_time_seconds: Total wall-clock time elapsed.

    Returns:
        Frames per second as a float.
    """
    if elapsed_time_seconds <= 0:
        return 0.0
    return float(frame_count / elapsed_time_seconds)


def compute_detector_call_ratio(detector_calls: int, total_frames: int) -> float:
    """Compute fraction of frames where the detector model was invoked.

    Args:
        detector_calls: Number of times detector ran.
        total_frames: Total video frames processed.

    Returns:
        Ratio in range [0.0, 1.0].
    """
    if total_frames <= 0:
        return 0.0
    return float(detector_calls / total_frames)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def frame_agreement(tracks: pd.DataFrame, reference_tracks: pd.DataFrame, n_frames: int,
                    reference_conf: float = 0.5, iou_thr: float = 0.5) -> np.ndarray:
    """Per frame: share of the reference's confident targets that ``tracks`` also found.

    A frame where the reference has no confident target counts as 1.0 (nothing to miss).
    With always-ROBUST as the reference this measures how much of the careful setup's
    result a cheaper run keeps. It is not ground-truth accuracy (that needs labelled
    boxes, see run_trackeval).
    """
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    found = {f: g[cols].to_numpy(float) for f, g in tracks.groupby("frame")} if len(tracks) else {}
    ref = reference_tracks[reference_tracks["confidence"] >= reference_conf] if len(reference_tracks) else reference_tracks
    ref_by = {f: g[cols].to_numpy(float) for f, g in ref.groupby("frame")} if len(ref) else {}

    agree = np.ones(n_frames, dtype=float)
    for f, rb in ref_by.items():
        if f >= n_frames:
            continue
        fb = found.get(f)
        if fb is None or len(fb) == 0:
            agree[f] = 0.0
            continue
        iou = _iou(rb, fb)
        matched, free = 0, np.ones(len(fb), dtype=bool)
        for r in np.argsort(-iou.max(axis=1)):  # greedy one-to-one matching, best overlaps first
            cand = np.where(free, iou[r], -1.0)
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                free[j] = False
                matched += 1
        agree[f] = matched / len(rb)
    return agree


# ----------------------------------------------------------------------------
# Ground truth (Anti-UAV style: <video>.json next to <video>.mp4)
# ----------------------------------------------------------------------------
def find_ground_truth(video: Union[str, Path]) -> Optional[Path]:
    """The label file for a video, if there is one (visible.mp4 -> visible.json)."""
    p = Path(video).with_suffix(".json")
    return p if p.is_file() else None


def load_ground_truth(path: Union[str, Path]) -> np.ndarray:
    """Per-frame target box as x1,y1,x2,y2, NaN where the target is absent.

    Reads the Anti-UAV format: {"exist": [0|1, ...], "gt_rect": [[x, y, w, h], ...]}.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rects, exist = data.get("gt_rect", []), data.get("exist", [])
    out = np.full((len(rects), 4), np.nan)
    for i, r in enumerate(rects):
        present = bool(exist[i]) if i < len(exist) else bool(r)
        if present and r and len(r) == 4 and r[2] > 0 and r[3] > 0:
            out[i] = (r[0], r[1], r[0] + r[2], r[1] + r[3])
    return out


def gt_hits(tracks: pd.DataFrame, gt_boxes: np.ndarray, n_frames: int, iou_thr: float = 0.5) -> np.ndarray:
    """Per frame: 1 if a track overlaps the true target (IoU >= iou_thr), 0 if missed,
    NaN if the target is not in the frame."""
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    found = {f: g[cols].to_numpy(float) for f, g in tracks.groupby("frame")} if len(tracks) else {}
    hits = np.full(n_frames, np.nan)
    for f in range(min(n_frames, len(gt_boxes))):
        g = gt_boxes[f]
        if np.isnan(g).any():
            continue
        fb = found.get(f)
        hits[f] = float(fb is not None and len(fb) > 0 and _iou(g[None, :], fb).max() >= iou_thr)
    return hits


def gt_scores(tracks: pd.DataFrame, gt_boxes: np.ndarray, n_frames: int, iou_thr: float = 0.5) -> Dict[str, float]:
    """Recall (share of target frames where it was found) and precision (share of output
    boxes that are the target) against ground truth."""
    hits = gt_hits(tracks, gt_boxes, n_frames, iou_thr)
    present = int(np.sum(~np.isnan(hits)))
    found = int(np.nansum(hits))
    boxes = int((tracks["frame"] < n_frames).sum()) if len(tracks) else 0
    return {
        "gt_frames": present,
        "recall": round(found / present, 4) if present else float("nan"),
        "precision": round(found / boxes, 4) if boxes else float("nan"),
        "false_boxes_per_frame": round((boxes - found) / max(1, n_frames), 3),
    }


def run_metrics(summary: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Headline numbers for one TrackingPipeline.run_video() summary.

    Args:
        summary: Dictionary returned by run_video.
        reference: Summary of the always-ROBUST run on the same video, to score agreement.
    """
    frames, tracks = summary["records_df"], summary["tracks_df"]
    n = max(1, summary["total_frames"])
    out = {
        "frames": summary["total_frames"],
        "seconds": summary["total_time_seconds"],
        "fps": summary["avg_fps"],
        "detector_calls": summary["detector_calls"],
        "detector_pct": round(summary["detector_call_ratio"] * 100, 1),
        "robust_pct": round(float((frames["mode"] == "ROBUST").mean() * 100), 1) if len(frames) else 0.0,
        "switches": summary["mode_switches"],
        "targets": summary.get("total_unique_targets", 0),
        "frames_with_target_pct": round(tracks["frame"].nunique() / n * 100, 1) if len(tracks) else 0.0,
        "mean_targets_per_frame": round(float(frames["num_tracks"].mean()), 2) if len(frames) else 0.0,
    }
    if reference is not None:
        agree = frame_agreement(tracks, reference["tracks_df"], len(reference["records_df"]))
        out["agreement_pct"] = round(float(agree.mean() * 100), 1)
    return out


def summarize_run_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from a pipeline run's per-frame records.

    Args:
        metrics_df: DataFrame of per-frame records generated by TrackingPipeline.

    Returns:
        Dictionary of aggregate statistics.
    """
    if metrics_df.empty:
        return {}

    total_frames = len(metrics_df)
    fast_count = int((metrics_df["mode"] == "FAST").sum())
    robust_count = int((metrics_df["mode"] == "ROBUST").sum())

    detector_calls = int(metrics_df["detector_invoked"].sum()) if "detector_invoked" in metrics_df else total_frames

    return {
        "total_frames": total_frames,
        "avg_fps": round(float(metrics_df["fps"].mean()), 1) if "fps" in metrics_df else 0.0,
        "detector_calls": detector_calls,
        "detector_call_ratio": round(detector_calls / max(1, total_frames), 3),
        "fast_mode_ratio": round(fast_count / max(1, total_frames), 3),
        "robust_mode_ratio": round(robust_count / max(1, total_frames), 3),
        "mean_tracks": round(float(metrics_df["num_tracks"].mean()), 2) if "num_tracks" in metrics_df else 0.0,
    }


def run_trackeval(
    predicted_tracks_dir: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    benchmark: str = "MOT_FLY",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate predicted tracking outputs against MOT-format ground truth annotations.

    PLACEHOLDER: Intended to wrap TrackEval benchmarking for HOTA, IDF1, MOTA, and IDSW.

    TODO:
      1. Ensure MOT-format ground truth annotations exist in data/raw/<benchmark>/
      2. Import TrackEval evaluator: `from third_party.TrackEval import trackeval`
      3. Compute and return standardized HOTA/IDF1/MOTA/IDSW scores per sequence.

    Args:
        predicted_tracks_dir: Directory containing predicted tracker output files.
        ground_truth_dir: Directory containing ground truth annotation text files.
        benchmark: Target benchmark dataset name.
        metrics: List of metrics to compute (defaults to ['HOTA', 'IDF1', 'MOTA', 'IDSW']).

    Raises:
        NotImplementedError: TrackEval integration is a documented placeholder.
    """
    raise NotImplementedError(
        "run_trackeval is a placeholder. "
        "TODO: Download MOT-FLY ground truth and run TrackEval evaluation pipeline."
    )

