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

from src.selector.selector import ROLLING_WINDOW, RollingFeatures, RuleBasedSelector
from src.utils.config import load_config

VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
CONDITIONS = ["clean", "blur", "shake", "tiny", "dark", "noise"]


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
                           segment_frames: int = 90, max_frames: Optional[int] = None) -> List[str]:
    """Write a copy of ``src`` where each segment has a random condition. Returns the schedule."""
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
            condition, strength, state = str(rng.choice(CONDITIONS)), float(rng.uniform()), {}
        schedule.append(condition)
        out.write(_apply_condition(frame, condition, strength, rng, state))
        i += 1
    cap.release()
    out.release()
    return schedule


# ----------------------------------------------------------------------------
# 2. Teacher labels
# ----------------------------------------------------------------------------
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


def frame_agreement(fast_tracks: pd.DataFrame, robust_tracks: pd.DataFrame, n_frames: int,
                    teacher_conf: float = 0.5, iou_thr: float = 0.5) -> np.ndarray:
    """Per frame: share of ROBUST's confident targets that FAST also found (1.0 if none)."""
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    fast_by = {f: g[cols].to_numpy(float) for f, g in fast_tracks.groupby("frame")} if len(fast_tracks) else {}
    teach = robust_tracks[robust_tracks["confidence"] >= teacher_conf] if len(robust_tracks) else robust_tracks
    robust_by = {f: g[cols].to_numpy(float) for f, g in teach.groupby("frame")} if len(teach) else {}

    agree = np.ones(n_frames, dtype=float)
    for f, rb in robust_by.items():
        if f >= n_frames:
            continue
        fb = fast_by.get(f)
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


def forward_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of values[t : t+window] for every t (shorter at the end)."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(len(values))
    end = np.minimum(idx + window, len(values))
    return (csum[end] - csum[idx]) / (end - idx)


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
            key = (p.name, p.stat().st_size)
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
    max_frames: Optional[int] = None,
    config_path: Union[str, Path] = "config.yaml",
) -> pd.DataFrame:
    """Make the selector training table: one row per frame per pass, with a FAST/ROBUST label."""
    from src.pipeline.pipeline import Mode, TrackingPipeline  # heavy import, only when labelling

    videos = [Path(v) for v in (videos or find_videos(["data/uploads", "data/raw", "data/sample_test.mp4"]))]
    if not videos:
        raise FileNotFoundError("No videos found. Put flight videos in data/raw/ or data/uploads/.")

    clips: List[Path] = []
    for v in videos:
        clips.append(v)
        for k in range(variants_per_video):
            dst = Path(variants_dir) / f"{v.stem}__var{k}.mp4"
            if not dst.is_file():
                print(f"  making condition-varied copy {dst.name}")
                make_condition_variant(v, dst, seed=1000 + k, max_frames=max_frames)
            clips.append(dst)

    fast = TrackingPipeline(Mode.FAST, config_path=config_path)
    robust = TrackingPipeline(Mode.ROBUST, config_path=config_path)
    names = RollingFeatures.names()
    tables = []
    for n, clip in enumerate(clips, 1):
        print(f"[{n}/{len(clips)}] {clip.name}: FAST pass...", flush=True)
        f_sum = fast.run_video(clip, max_frames=max_frames, save_outputs=False)
        print(f"[{n}/{len(clips)}] {clip.name}: ROBUST pass...", flush=True)
        r_sum = robust.run_video(clip, max_frames=max_frames, save_outputs=False)

        frames = min(len(f_sum["records_df"]), len(r_sum["records_df"]))
        agree = frame_agreement(f_sum["tracks_df"], r_sum["tracks_df"], frames, teacher_conf)
        future = forward_window_mean(agree, window)
        label = np.where(future >= tau, "FAST", "ROBUST")

        for source, summ in (("fast_pass", f_sum), ("robust_pass", r_sum)):
            rows = pd.DataFrame(_rolling_rows(summ["records_df"].iloc[:frames]), columns=names)
            rows["frame"] = np.arange(frames)
            rows["agreement"] = agree.round(3)
            rows["future_agreement"] = future.round(3)
            rows["label"] = label
            rows["clip"] = clip.stem
            rows["source_video"] = clip.stem.split("__var")[0]
            rows["pass"] = source
            tables.append(rows)
        print(f"    {frames} frames, ROBUST needed on {np.mean(label == 'ROBUST') * 100:.0f}% of them")

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
    X, y, groups = data[names].to_numpy(float), data["label"].to_numpy(), data["clip"].to_numpy()
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
    p.add_argument("--skip-labels", action="store_true", help="reuse data/processed/training_labels.csv")
    args = p.parse_args()

    if not args.skip_labels:
        videos = find_videos(args.videos) if args.videos else None
        build_labels(videos, variants_per_video=args.variants, window=args.window,
                     tau=args.tau, max_frames=args.max_frames)
    _print_summary(train_selector())


if __name__ == "__main__":
    main()
