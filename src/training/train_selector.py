"""Train the learned mode selectors (decision tree and random forest).

Run:  python -m src.training.train_selector            (labels + training)
      python -m src.training.train_selector --skip-labels   (retrain on existing labels)

How labels are made (no manual annotation needed)
--------------------------------------------------
ROBUST is the teacher. Every clip is run twice, once always-FAST and once always-ROBUST.
For each frame we check how many of the targets ROBUST is confident about were also found
by FAST (box overlap, IoU >= 0.5). A frame is labelled FAST when, over the next ``window``
frames, FAST found at least ``tau`` of them; otherwise ROBUST. So the label answers:
"starting here, is the cheap setup good enough, or is the expensive one worth paying for?"
If ROBUST finds nothing confident either, the label is FAST: extra compute would not help.

Where the variety comes from
----------------------------
Each source video is also re-rendered into a few *condition-varied* copies, where every
~3 s segment gets a different condition (blur, camera shake, tiny targets, low light,
noise, or clean). That gives windows where FAST copes and windows where it does not,
which is what the selector has to learn to tell apart.

Features are built with RollingFeatures, the same class the selector uses at run time.
Rows come from both the FAST and the ROBUST pass, because at run time the features are
computed from whichever setup was running.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from src.evaluation.metrics import find_ground_truth, frame_agreement, gt_hits, load_ground_truth
from src.selector.selector import ROLLING_WINDOW, RollingFeatures, RuleBasedSelector
from src.utils.config import load_config

VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
CONDITIONS = ["clean", "blur", "shake", "tiny", "dark", "noise"]
# Conditions that leave every pixel where it was, so a clip's ground-truth boxes still
# fit the condition-varied copy. "shake" and "tiny" move the target and are skipped then.
GEOMETRY_SAFE = ["clean", "blur", "dark", "noise"]


def clip_id(video: Union[str, Path]) -> str:
    """Unique name for a clip. Datasets that call every file visible.mp4 use the folder name."""
    p = Path(video)
    return p.parent.name if p.stem.lower() in ("visible", "infrared", "video") else p.stem


# ----------------------------------------------------------------------------
# 1. Condition-varied training clips
# ----------------------------------------------------------------------------
def _apply_condition(frame: np.ndarray, condition: str, strength: float, rng: np.random.Generator,
                     state: Dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    if condition == "blur":
        k = int(7 + strength * 18) | 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    if condition == "shake":
        # Smoothly wandering offset, so optical flow sees real camera motion.
        amp = 10 + strength * 30
        state["offset"] = 0.7 * state.get("offset", np.zeros(2)) + rng.normal(0, amp * 0.5, 2)
        dx, dy = np.clip(state["offset"], -amp, amp)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if condition == "tiny":
        # Zoom out: everything gets smaller, like targets far away.
        s = 0.55 - strength * 0.25
        small = cv2.resize(frame, (max(16, int(w * s)), max(16, int(h * s))), interpolation=cv2.INTER_AREA)
        canvas = np.empty_like(frame)
        canvas[:] = frame.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return canvas
    if condition == "dark":
        gamma = 2.0 + strength * 1.5
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        dark = cv2.LUT(frame, lut).astype(np.int16)
        return np.clip(dark + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
    if condition == "noise":
        sigma = 12 + strength * 25
        return np.clip(frame.astype(np.int16) + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)
    return frame


def make_condition_variant(src: Union[str, Path], dst: Union[str, Path], seed: int,
                           segment_frames: int = 90, max_frames: Optional[int] = None,
                           conditions: Optional[List[str]] = None) -> List[str]:
    """Write a copy of ``src`` where each segment has a random condition. Returns the schedule."""
    conditions = conditions or CONDITIONS
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    schedule: List[str] = []
    i, condition, strength, state = 0, "clean", 0.0, {}
    while max_frames is None or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % segment_frames == 0:
            condition, strength, state = str(rng.choice(conditions)), float(rng.uniform()), {}
        schedule.append(condition)
        out.write(_apply_condition(frame, condition, strength, rng, state))
        i += 1
    cap.release()
    out.release()
    return schedule


# ----------------------------------------------------------------------------
# 2. Teacher labels
# ----------------------------------------------------------------------------
def forward_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of values[t : t+window] for every t (shorter at the end)."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(len(values))
    end = np.minimum(idx + window, len(values))
    return (csum[end] - csum[idx]) / (end - idx)


def forward_window_nanmean(values: np.ndarray, window: int):
    """Like forward_window_mean but ignoring NaN. Returns (mean, count of non-NaN values)."""
    valid = ~np.isnan(values)
    s = forward_window_mean(np.where(valid, values, 0.0), window)
    c = forward_window_mean(valid.astype(float), window)
    span = np.minimum(np.arange(len(values)) + window, len(values)) - np.arange(len(values))
    count = np.rint(c * span)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, s / np.maximum(c, 1e-12), np.nan), count


def _rolling_rows(records: pd.DataFrame) -> np.ndarray:
    roll = RollingFeatures(ROLLING_WINDOW)
    return np.asarray([roll.update(r) for r in records.to_dict("records")], dtype=float)


def find_videos(roots: Iterable[Union[str, Path]]) -> List[Path]:
    """Video files under the given files/folders, skipping duplicates (same name and size)."""
    seen, found = set(), []
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if p.suffix.lower() not in VIDEO_TYPES or p.name.startswith("temp_raw_"):
                continue
            key = (clip_id(p), p.stat().st_size)
            if key not in seen:
                seen.add(key)
                found.append(p)
    return found


def build_labels(
    videos: Optional[List[Union[str, Path]]] = None,
    output_path: Union[str, Path] = "data/processed/training_labels.csv",
    variants_dir: Union[str, Path] = "data/processed/variants",
    variants_per_video: int = 3,
    window: int = 15,
    tau: float = 0.8,
    teacher_conf: float = 0.5,
    min_gain: float = 0.2,
    max_frames: Optional[int] = None,
    config_path: Union[str, Path] = "config.yaml",
) -> pd.DataFrame:
    """Make the selector training table: one row per frame per pass, with a FAST/ROBUST label.

    Clips with ground truth (``<video>.json`` next to the video) are labelled against it:
    a frame is ROBUST when, over the next ``window`` frames, FAST finds the real target in
    less than ``tau`` of the frames it is present AND ROBUST finds it in at least
    ``min_gain`` more of them. So ROBUST is asked for only where it actually helps; if
    neither setup finds the target, paying for ROBUST would be wasted and the label is FAST.
    Clips without ground truth fall back to the ROBUST-as-teacher agreement label.
    """
    from src.pipeline.pipeline import Mode, TrackingPipeline  # heavy import, only when labelling

    videos = [Path(v) for v in (videos or find_videos(["data/uploads", "data/raw", "data/sample_test.mp4"]))]
    if not videos:
        raise FileNotFoundError("No videos found. Put flight videos in data/raw/ or data/uploads/.")

    clips: List[tuple] = []  # (clip path, clip name, source name, ground-truth path or None)
    for v in videos:
        gt = find_ground_truth(v)
        cid = clip_id(v)
        clips.append((v, cid, cid, gt))
        for k in range(variants_per_video):
            dst = Path(variants_dir) / f"{cid}__var{k}.mp4"
            if not dst.is_file():
                print(f"  making condition-varied copy {dst.name}")
                make_condition_variant(v, dst, seed=1000 + k, max_frames=max_frames,
                                       conditions=GEOMETRY_SAFE if gt else None)
            clips.append((dst, dst.stem, cid, gt))

    fast = TrackingPipeline(Mode.FAST, config_path=config_path)
    robust = TrackingPipeline(Mode.ROBUST, config_path=config_path)
    names = RollingFeatures.names()
    tables = []
    for n, (clip, cname, source_name, gt_path) in enumerate(clips, 1):
        print(f"[{n}/{len(clips)}] {cname}: FAST pass...", flush=True)
        f_sum = fast.run_video(clip, max_frames=max_frames, save_outputs=False)
        print(f"[{n}/{len(clips)}] {cname}: ROBUST pass...", flush=True)
        r_sum = robust.run_video(clip, max_frames=max_frames, save_outputs=False)

        frames = min(len(f_sum["records_df"]), len(r_sum["records_df"]))
        agree = frame_agreement(f_sum["tracks_df"], r_sum["tracks_df"], frames, reference_conf=teacher_conf)
        if gt_path is not None:
            gt_boxes = load_ground_truth(gt_path)
            fast_rec, present = forward_window_nanmean(gt_hits(f_sum["tracks_df"], gt_boxes, frames), window)
            robust_rec, _ = forward_window_nanmean(gt_hits(r_sum["tracks_df"], gt_boxes, frames), window)
            need = (present > 0) & (np.nan_to_num(fast_rec, nan=1.0) < tau) & \
                   (np.nan_to_num(robust_rec, nan=0.0) - np.nan_to_num(fast_rec, nan=1.0) >= min_gain)
            label = np.where(need, "ROBUST", "FAST")
            future, label_source = fast_rec, "ground_truth"
        else:
            future = forward_window_mean(agree, window)
            robust_rec = np.full(frames, np.nan)
            label = np.where(future >= tau, "FAST", "ROBUST")
            label_source = "robust_teacher"

        for source, summ in (("fast_pass", f_sum), ("robust_pass", r_sum)):
            rows = pd.DataFrame(_rolling_rows(summ["records_df"].iloc[:frames]), columns=names)
            rows["frame"] = np.arange(frames)
            rows["agreement"] = agree.round(3)
            rows["future_agreement"] = np.round(future, 3)
            rows["future_robust_recall"] = np.round(robust_rec, 3)
            rows["label"] = label
            rows["label_source"] = label_source
            rows["clip"] = cname
            rows["source_video"] = source_name
            rows["pass"] = source
            tables.append(rows)
        print(f"    {frames} frames ({label_source}), ROBUST needed on {np.mean(label == 'ROBUST') * 100:.0f}% of them")

    table = pd.concat(tables, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved {len(table)} labelled rows to {output_path}")
    return table


# ----------------------------------------------------------------------------
# 3. Training and evaluation
# ----------------------------------------------------------------------------
def _scores(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_robust": round(float(f1_score(y_true, y_pred, pos_label="ROBUST", zero_division=0)), 4),
        "predicted_robust_share": round(float(np.mean(np.asarray(y_pred) == "ROBUST")), 4),
    }


def _models(random_state: int, n_estimators: int) -> Dict[str, Any]:
    return {
        "tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced",
                                       random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=10,
                                         class_weight="balanced", n_jobs=-1, random_state=random_state),
    }


def train_selector(
    training_data_path: Union[str, Path] = "data/processed/training_labels.csv",
    weights_dir: Union[str, Path] = "weights",
    report_path: Union[str, Path] = "outputs/selector/report.json",
    n_estimators: int = 100,
    random_state: int = 42,
    config_path: Union[str, Path] = "config.yaml",
) -> Dict[str, Any]:
    """Fit a decision tree and a random forest, compare them with the rules, save both."""
    data = pd.read_csv(training_data_path)
    names = RollingFeatures.names()
    # Group by source video, so a clip and its condition-varied copies stay on the same side
    # of every cross-validation split.
    group_col = "source_video" if "source_video" in data else "clip"
    X, y, groups = data[names].to_numpy(float), data["label"].to_numpy(), data[group_col].to_numpy()
    if len(set(y)) < 2:
        raise ValueError(f"Every label is '{y[0]}'. The selector needs both FAST and ROBUST examples: "
                         "add more varied videos or change --tau.")

    # Baseline: the hand-set rules, scored on the same rows.
    cfg = load_config(config_path)
    rules = RuleBasedSelector((cfg.get("selector") or {}).get("rule_thresholds", {}))
    rule_pred = np.array([rules.predict(r) for r in data[names[:5]].to_dict("records")])

    # Cross-validation split by clip, so no clip is in both train and test.
    n_groups = len(set(groups))
    folds = list(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups)) if n_groups >= 2 else []
    report: Dict[str, Any] = {
        "rows": int(len(data)), "clips": int(n_groups),
        "label_share_robust": round(float(np.mean(y == "ROBUST")), 4),
        "features": names, "cv_folds": len(folds),
        "rules_baseline": _scores(y, rule_pred), "models": {},
    }

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    for kind, model in _models(random_state, n_estimators).items():
        cv_pred = np.empty(len(y), dtype=object)
        for train_idx, test_idx in folds:
            fold_model = _models(random_state, n_estimators)[kind].fit(X[train_idx], y[train_idx])
            cv_pred[test_idx] = fold_model.predict(X[test_idx])
        cv_scores = _scores(y, cv_pred) if folds else None

        model.fit(X, y)  # final model on all rows
        importances = dict(sorted(zip(names, model.feature_importances_.round(4).tolist()),
                                  key=lambda kv: -kv[1]))
        entry = {
            "cv": cv_scores,
            "cv_confusion": (confusion_matrix(y, cv_pred, labels=["FAST", "ROBUST"]).tolist()
                             if folds else None),
            "train": _scores(y, model.predict(X)),
            "feature_importance": importances,
        }
        if kind == "tree":
            entry["rules_text"] = export_text(model, feature_names=names, max_depth=5)
        report["models"][kind] = entry

        bundle = {"kind": kind, "model": model, "features": names, "threshold": 0.5,
                  "window": ROLLING_WINDOW, "decide_every": 3, "report": entry,
                  "trained_on": {"rows": int(len(data)), "clips": int(n_groups)}}
        path = weights_dir / f"selector_{kind}.joblib"
        joblib.dump(bundle, path)
        print(f"Saved {kind} -> {path}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {report_path}")
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    print(f"\n{report['rows']} rows from {report['clips']} clips, "
          f"ROBUST label on {report['label_share_robust'] * 100:.0f}% of rows")
    print(f"{'selector':<10}{'accuracy':>10}{'bal. acc':>10}{'F1 robust':>11}{'robust %':>10}   (cross-validated by clip)")
    rows = [("rules", report["rules_baseline"])] + [(k, v["cv"] or v["train"]) for k, v in report["models"].items()]
    for name, s in rows:
        print(f"{name:<10}{s['accuracy']:>10.3f}{s['balanced_accuracy']:>10.3f}{s['f1_robust']:>11.3f}"
              f"{s['predicted_robust_share'] * 100:>9.0f}%")
    top = list(report["models"]["forest"]["feature_importance"].items())[:4]
    print("forest's top features:", ", ".join(f"{k} {v:.2f}" for k, v in top))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--videos", nargs="*", help="video files or folders (default: data/uploads, data/raw, data/sample_test.mp4)")
    p.add_argument("--variants", type=int, default=3, help="condition-varied copies per video")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per clip (quick runs)")
    p.add_argument("--window", type=int, default=15, help="frames ahead the label looks at")
    p.add_argument("--tau", type=float, default=0.8, help="share of ROBUST's targets FAST must find")
    p.add_argument("--min-gain", type=float, default=0.2,
                   help="ground-truth labels: how much more ROBUST must find before it is worth asking for")
    p.add_argument("--videos-file", help="text file with one video path per line")
    p.add_argument("--skip-labels", action="store_true", help="reuse data/processed/training_labels.csv")
    args = p.parse_args()

    if not args.skip_labels:
        roots = list(args.videos or [])
        if args.videos_file:
            roots += [l.strip() for l in Path(args.videos_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        videos = find_videos(roots) if roots else None
        build_labels(videos, variants_per_video=args.variants, window=args.window,
                     tau=args.tau, min_gain=args.min_gain, max_frames=args.max_frames)
    _print_summary(train_selector())


if __name__ == "__main__":
    main()
