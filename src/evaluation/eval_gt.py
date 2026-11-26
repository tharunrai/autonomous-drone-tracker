"""Score FAST, ROBUST and ADAPTIVE against ground truth on labelled clips.

Run:  python -m src.evaluation.eval_gt --videos-file data/splits/val_clips.txt

Every clip needs a <video>.json label file next to it (Anti-UAV format). For each run it
reports recall (share of frames with the target where it was found, IoU >= 0.5), precision,
detector calls and time in ROBUST, summed over all clips. The goal for ADAPTIVE is recall
close to ROBUST's at a cost close to FAST's.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import find_ground_truth, gt_hits, load_ground_truth
from src.pipeline.pipeline import Mode, TrackingPipeline
from src.selector.selector import available_selectors, SELECTOR_LABELS
from src.utils.config import load_config


def evaluate(videos: List[Path], selectors: Optional[List[str]] = None, threshold: Optional[float] = None,
             config_path: str = "config.yaml", out_csv: str = "outputs/metrics/gt_eval.csv") -> pd.DataFrame:
    cfg = load_config(config_path)
    selectors = selectors or available_selectors(cfg)
    runs = [("Fast", Mode.FAST, None), ("Robust", Mode.ROBUST, None)] + \
           [(f"Adaptive + {SELECTOR_LABELS[s]}", Mode.ADAPTIVE, s) for s in selectors]
    pipes = {name: TrackingPipeline(mode, config_path=config_path, selector_type=sel, selector_threshold=threshold)
             for name, mode, sel in runs}

    rows = []
    for n, video in enumerate(videos, 1):
        gt_path = find_ground_truth(video)
        if gt_path is None:
            print(f"  skip {video}: no ground truth")
            continue
        gt = load_ground_truth(gt_path)
        print(f"[{n}/{len(videos)}] {video.parent.name}", flush=True)
        for name, _, _ in runs:
            s = pipes[name].run_video(video, save_outputs=False)
            frames = s["total_frames"]
            hits = gt_hits(s["tracks_df"], gt, frames)
            boxes = int(len(s["tracks_df"]))
            rec = s["records_df"]
            rows.append({
                "clip": video.parent.name, "run": name, "frames": frames,
                "gt_frames": int(np.sum(~np.isnan(hits))), "found": int(np.nansum(hits)), "boxes": boxes,
                "detector_calls": s["detector_calls"],
                "robust_frames": int((rec["mode"] == "ROBUST").sum()) if len(rec) else 0,
                "switches": s["mode_switches"], "ids": s["total_unique_targets"],
                "seconds": s["total_time_seconds"],
            })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    t = df.groupby("run", sort=False).sum(numeric_only=True)
    summary = pd.DataFrame({
        "Recall %": (t["found"] / t["gt_frames"] * 100).round(1),
        "Precision %": (t["found"] / t["boxes"].clip(lower=1) * 100).round(1),
        "Det calls": t["detector_calls"],
        "Det %": (t["detector_calls"] / t["frames"] * 100).round(1),
        "Robust %": (t["robust_frames"] / t["frames"] * 100).round(1),
        "Switches": t["switches"],
        "Track IDs": t["ids"],
        "FPS": (t["frames"] / t["seconds"].clip(lower=1e-6)).round(1),
    })
    clips = df["clip"].nunique()
    print("=" * 100)
    print(f"GROUND-TRUTH EVALUATION · {clips} clips, {int(t['frames'].iloc[0])} frames, "
          f"target present in {int(t['gt_frames'].iloc[0])}")
    print("=" * 100)
    print(summary.to_string())
    print(f"Per-clip rows: {out_csv}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--videos", nargs="*", default=[])
    p.add_argument("--videos-file")
    p.add_argument("--selectors", nargs="*", help="rules tree forest (default: every trained one)")
    p.add_argument("--threshold", type=float, help="tree/forest decision threshold (default: config)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="outputs/metrics/gt_eval.csv")
    a = p.parse_args()
    vids = list(a.videos)
    if a.videos_file:
        vids += [l.strip() for l in Path(a.videos_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not vids:
        p.error("give --videos or --videos-file")
    evaluate([Path(v) for v in vids], a.selectors, a.threshold, a.config, a.out)


if __name__ == "__main__":
    main()
