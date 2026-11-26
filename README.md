# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

Team Ctrl Alt Elite, Dept. of CSE (AI & ML), Sahyadri College of Engineering & Management  
**Guide:** Dr. Pushpalatha K

---

## Overview

In air-to-air (A2A) unmanned aerial vehicle (UAV) tracking, edge compute constraints clash with volatile flight environments. Rapid camera maneuvers, target distance changes, illumination shifts, and motion blur degrade tracking fidelity if perception is underpowered. Running high-resolution heavy detectors on every frame drains onboard power and lowers frame rates.

This project introduces a **condition-aware adaptive perception pipeline**:
- Estimates scene difficulty per frame (blur variance, brightness, target scale, confidence, camera optical flow).
- Dynamically selects between:
  - ⚡ **FAST Mode**: Lightweight detector (`yolov8n`), lower resolution (`640x640`), frame skipping (`detect_every_n_frames=3`), ByteTrack association.
  - 🛡️ **ROBUST Mode**: High-capacity detector (`yolov8s`), high resolution (`1280x1280`), dense detection (`detect_every_n_frames=1`), ByteTrack association.
- **A learned selector decides the mode** (ADAPTIVE): a trained **decision tree** or **random forest** reads the
  condition features and predicts when FAST would miss targets that ROBUST catches. Hand-set threshold rules are kept
  only as the baseline to beat.
- **Hysteresis Smoothing**: Prevents rapid mode flipping on transient anomalies.
- Compares **Always FAST**, **Always ROBUST**, and **ADAPTIVE** (rules / tree / forest) on tracking quality vs
  computational cost (detector calls, FPS).

### Results

Printed by `python run.py --video <clip> --compare` (see [Usage](#2-command-line-interface-cli)). Three free stock
clips (1280x720) that the selector was **not** trained on. All runs use the default selector settings (threshold 0.5).
*Agreement* is the share of always-ROBUST's confident targets that the run also found, so it is relative to ROBUST,
not ground truth.

**Small, distant drone** (`mixkit_44641`, 193 frames)

| Run | Det calls | Det % | Robust % | Switches | Frames w/ target | Agreement |
|---|---|---|---|---|---|---|
| Always Fast | 65 | 33.7% | 0% | 0 | 62.2% | 62.7% |
| Always Robust | 193 | 100% | 100% | 0 | 100% | 100% |
| Adaptive + random forest | **164** | 85.0% | 77.2% | 5 | 100% | 100% |
| Adaptive + decision tree | 169 | 87.6% | 80.8% | 1 | 100% | 100% |
| Adaptive + rules | 184 | 95.3% | 92.7% | 1 | 100% | 100% |

**Close-up drone** (`mixkit_44644`, 579 frames)

| Run | Det calls | Det % | Robust % | Switches | Frames w/ target | Agreement |
|---|---|---|---|---|---|---|
| Always Fast | 193 | 33.3% | 0% | 0 | 100% | 99.3% |
| Always Robust | 579 | 100% | 100% | 0 | 99.0% | 100% |
| Adaptive + random forest | **203** | 35.1% | 2.6% | 2 | 99.3% | 99.7% |
| Adaptive + decision tree | 491 | 84.8% | 77.0% | 7 | 99.8% | 100% |
| Adaptive + rules | 570 | 98.4% | 97.6% | 1 | 99.0% | 100% |

**Four fighter jets in formation** (`mixkit_7373`, 549 frames, appears to be rendered footage)

| Run | Det calls | Det % | Robust % | Switches | Frames w/ target | Agreement |
|---|---|---|---|---|---|---|
| Always Fast | 183 | 33.3% | 0% | 0 | 98.4% | 79.2% |
| Always Robust | 549 | 100% | 100% | 0 | 100% | 100% |
| Adaptive + random forest | **245** | 44.6% | 16.9% | 6 | 98.4% | 83.1% |
| Adaptive + decision tree | 536 | 97.6% | 96.5% | 2 | 100% | 99.7% |
| Adaptive + rules | 540 | 98.4% | 97.4% | 1 | 100% | 100% |

**What this shows**

- **Fast alone is not enough.** On the small distant drone it loses the target in about 4 of 10 frames, and on the
  jet formation it finds only 79% of what Robust finds.
- **The learned selectors spend less compute than the rules** on all three clips. The rules stay in ROBUST 93-98% of
  the time, which saves almost nothing.
- **The random forest is the cheapest, but it trades quality for it on hard scenes.** It keeps full quality on both
  drone clips (100% and 99.7% agreement with 15% and 65% fewer detector calls than always-Robust), but on the
  crowded jet formation it saves 55% of detector calls and drops to 83% agreement. The decision tree is safer there
  (99.7%) but spends nearly the full Robust budget. Lowering `--threshold` makes the forest more cautious (see [the threshold test](#choosing-the-forest-threshold)).
- **Tracking identity is fragmented**: the formation has 4 jets but 19-22 track IDs are created, so IDs change when
  jets overlap. ID quality is not measured by these tables.

**Caveats:** one run per setting, three short clips, and a detector with no drone class (see
[Limitations](#limitations)). FPS differs little between modes on a fast GPU, so detector calls are the compute
measure. Treat these as a first check, not a benchmark; ground-truth metrics (HOTA/IDF1) need labelled data.

### Choosing the forest threshold

The forest switches to ROBUST when its estimated chance that FAST would miss targets is at least the threshold
(`--threshold`, default 0.5). **Lower = safer, higher = cheaper.** Tested with `--selector forest` on the
four-jet formation, where the default was too aggressive:

| Threshold | Det calls (Robust needs 549) | Robust % | Agreement |
|---|---|---|---|
| 0.8 | 193 | 2.7% | 80.1% |
| 0.7 | 201 | 4.9% | 80.5% |
| 0.5 (default) | 245 | 16.9% | 83.1% |
| 0.4 | 400 | 59.2% | 95.8% |
| 0.3 | 508 | 88.7% | 99.3% |
| 0.2 | 540 | 97.4% | 100% |

On both drone clips the default 0.5 already kept full quality, and a lower threshold only added compute
(small drone: 184 calls at 0.1-0.3 vs 164 at 0.5). These values were tuned on the same test clips, so treat them as
a guide, not a final setting.

---

## Conclusion

**Which Adaptive selector is best? Adaptive + random forest, with a lower threshold for crowded scenes.**

| | Adaptive + random forest | Adaptive + decision tree | Adaptive + rules |
|---|---|---|---|
| Compute (detector calls vs always-Robust) | **Lowest on all three clips**: 85%, 35% and 45% | 88%, 85% and 98% | 95%, 98% and 98% |
| Quality kept (agreement with Robust) | 100%, 99.7%, **83%** at the default threshold | 100%, 100%, 99.7% | 100%, 100%, 100% |
| Main weakness | Misses targets in crowded scenes unless the threshold is lowered (0.4 gives 95.8%, 0.3 gives 99.3%) | Saves little compute | Saves almost nothing: it sits in ROBUST 93-98% of the time |
| Strength | Biggest savings with full quality on single-target clips | Safest learned option, and its rules can be printed and read | Needs no training |

1. **Random forest is the best overall.** It is the cheapest on every clip and, on the two single-drone clips,
   it matches always-Robust quality (99.7-100% agreement) with 15% and 65% fewer detector calls. It is the
   recommended default for single-target tracking. For crowded multi-target scenes, lower the threshold to about 0.3-0.4.
2. **Decision tree is the safe second choice.** It never dropped below 99.7% agreement, but it saved only 12-15%
   of detector calls on two clips and 2% on the formation. Choose it when you need to explain the decision
   (`outputs/selector/report.json` prints its rules) or cannot risk missing a target.
3. **Hand-set rules are the weakest.** They keep full quality only because they almost never leave ROBUST, so they
   barely save compute. Their value is as the baseline the learned models beat.
4. **None of the three is better than always-Robust on quality; the point is cost.** If compute does not matter,
   always-Robust is simply the most accurate. Always-Fast is cheapest but unreliable on small or crowded targets
   (62-79% agreement on two of the three clips).

**How far to trust this:** three short stock clips, one run per setting, a COCO detector with no drone class, a
selector trained mostly on one indoor video, and agreement measured against Robust rather than labelled boxes. The
ranking (forest cheapest, rules weakest) was consistent across all three clips, but a final claim needs a labelled UAV
dataset, detector fine-tuning, a retrained selector and ground-truth metrics (HOTA/IDF1).

---|---|---|---|---|
| Small, distant drone (193 frames) | Always FAST | 0% | 65 | 62% of frames |
| | Adaptive, rules | 93% | 184 | 100% |
| | Adaptive, random forest | 77% | 164 | 100% |
| Close-up drone (579 frames) | Adaptive, rules | 98% | 570 | 99% |
| | Adaptive, random forest | 3% | 203 | 99% |

These are two short clips and a detector that has no drone class (see [Limitations](#limitations)), so treat them as
a first sanity check, not a benchmark.

---

## Repository Structure

```
.
├── app.py                      # Streamlit entry point: routes Home and Tracker pages
├── app_pages/                  # home.py (landing page), tracker.py (upload, playback, telemetry)
├── run.py                      # CLI runner (--video PATH --mode FAST|ROBUST|ADAPTIVE)
├── config.yaml                 # Central configuration (modes, thresholds, hysteresis)
├── requirements.txt            # Python dependencies
├── start.bat                   # One-click Windows startup script
├── README.md
├── data/                       # Uploads & datasets (gitignored, kept via .gitkeep)
│   ├── uploads/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── weights/                    # yolov8n.pt, yolov8s.pt, selector_tree.joblib, selector_forest.joblib
├── outputs/                    # Annotated videos, tracks, metrics CSVs, and plots
└── src/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py           # load_config()
    │   └── video_io.py         # Frame reading, OpenCV video writer, H.264 re-encoding
    ├── detection/
    │   ├── __init__.py
    │   └── detector.py         # YOLODetector (swappable weights, resolution, conf, class filters)
    ├── tracking/
    │   ├── __init__.py
    │   └── tracker.py          # Ultralytics ByteTrack wrapper with motion trail history
    ├── features/
    │   ├── __init__.py
    │   └── features.py         # Blur, brightness, bbox area ratio, confidence, optical flow
    ├── selector/
    │   ├── __init__.py
    │   └── selector.py         # MLSelector (decision tree / random forest), RuleBasedSelector baseline, HysteresisSmoother
    ├── pipeline/
    │   ├── __init__.py
    │   └── pipeline.py         # Mode enum, ModeConfig, TrackingPipeline with visual HUD
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py          # FPS, detector call ratio, summary stats, TrackEval placeholder
    └── training/
        ├── __init__.py
        ├── train_selector.py   # Teacher labels (ROBUST vs FAST) + train decision tree & random forest selectors
        └── finetune_detector.py# PLACEHOLDER: fine-tune YOLO on custom drone tracking dataset
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Model weights
# yolov8n.pt and yolov8s.pt download automatically into weights/ on first run.

# 4. Learned selector (weights/ is gitignored, so train it once after cloning)
python -m src.training.train_selector
# Without it, ADAPTIVE mode falls back to the rule-based selector with a warning.
```

Training needs videos with real objects in `data/raw/` or `data/uploads/` (see [Test videos](#test-videos)).

---

## Usage

### 1. Streamlit Web Dashboard
Run the interactive application:
```bash
streamlit run app.py
```
*(Or double-click `start.bat` on Windows)*

**Features:**
- Clean white UI with the upload, mode and run buttons on the main page; settings and saved runs in the sidebar.
- Upload your own video (`.mp4`, `.avi`, `.mov`, `.mkv`).
- Choose **ADAPTIVE**, **FAST**, or **ROBUST** mode. For ADAPTIVE, pick who decides: **Random forest**,
  **Decision tree** or **Rules**, and set the threshold for switching to ROBUST (lower is safer and uses more ROBUST, higher saves compute and misses more).
- **Compare all modes**: one click runs Fast, Robust and Adaptive with every available selector, then opens the comparison.
- Tune confidence threshold and test with max frames limit.
- Play original vs annotated video side-by-side with boxes, target positions, IDs and trajectory trails.
- Live preview while processing, plus speed and target-count metrics (FPS, detector calls, mode switches).
- **Targets & paths**: flight paths and last known position of every tracked target.
- **Conditions**: timeline of mode selections, scene features, and for learned selectors a
  *Why it switched* view (the model's per-frame probability and its feature importance).
- Download the annotated video and the target-position and per-frame CSVs.

### 2. Command Line Interface (CLI)
```bash
# Verify environment, GPU, and model weights
python run.py --check-setup

# Run ADAPTIVE tracking on a video (the selector comes from `selector.type` in config.yaml)
python run.py --video path/to/drone_flight.mp4 --mode ADAPTIVE

# Run FAST or ROBUST baseline with frame cap
python run.py --video path/to/drone_flight.mp4 --mode FAST --max-frames 150

# Pick who decides the mode in ADAPTIVE: rules, tree or forest (and the switch threshold)
python run.py --video path/to/drone_flight.mp4 --mode ADAPTIVE --selector forest --threshold 0.7

# Print the metrics table for every mode on one video (Fast, Robust, Adaptive + each selector)
python run.py --video path/to/drone_flight.mp4 --compare
```

`--compare` prints one table per video and saves it to `outputs/metrics/<video>_comparison.csv`:

| Column | Meaning |
|---|---|
| Frames, Time s, FPS | Frames processed, processing time (without video re-encoding), average speed |
| Det calls, Det % | Times the detector ran, and the share of frames that is |
| Robust % | Share of frames spent in ROBUST mode |
| Switches | Number of FAST/ROBUST mode changes |
| Targets | Different targets tracked (unique track IDs) |
| Frames w/ target % | Share of frames where at least one target was tracked |
| Agreement % | How much of always-ROBUST's result the run keeps: the share of ROBUST's confident targets (IoU ≥ 0.5) that the run also found. Relative to ROBUST, not ground truth |

---

## Test videos

Videos are not stored in the repo (`data/` is gitignored). Use clips with real, recognisable objects:

- `data/sample_test.mp4` is a **synthetic smoke test** (plain rectangles on a blue background). The app runs on it
  but the detector finds nothing, so no boxes are drawn. That is expected.
- Free stock footage works well for demos, for example drones and fighter jets on
  [Mixkit](https://mixkit.co/free-stock-video/drone/) (free under the Mixkit License). Public-domain US military
  footage is available on [Wikimedia Commons](https://commons.wikimedia.org/) and DVIDS.
- For training and evaluation, use labelled drone datasets such as
  [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) or DUT Anti-UAV.

`train_selector` scans `data/raw/` by default, so keep any clip you want as a fair held-out test outside that folder.

---

## Limitations

- **The detectors are generic COCO models.** COCO has no "drone" class, so drones are usually labelled "airplane",
  and small or distant drones are often missed. Fine-tuning YOLO on a UAV dataset is the biggest open improvement.
- **The selector's labels come from the detectors**, so it should be retrained (`python -m src.training.train_selector`)
  whenever the detector weights change.
- **Training data is small.** The first selector was trained on one real video plus condition-varied copies, so its
  cross-validated scores are optimistic. Add more varied footage and retrain before drawing conclusions.
- On a fast GPU, FAST and ROBUST run at similar fps, so **detector calls** are the more meaningful compute measure.

---

## Where to Extend

1. **Swap Detector Weights**: Change model weights in `config.yaml` (`modes.fast.weights` or `modes.robust.weights`) to custom fine-tuned drone weights (e.g. YOLOv8 fine-tuned on drone datasets).
2. **Add Custom Features**: Add spatial-temporal features (e.g. target velocity variance, contrast entropy) in `src/features/features.py`.
3. **Train the learned selector** (decides FAST vs ROBUST in ADAPTIVE mode):
   - Put flight videos in `data/raw/` or `data/uploads/`, then run `python -m src.training.train_selector`.
   - Labels need no manual annotation: every clip is run always-FAST and always-ROBUST, and a frame is
     labelled FAST when FAST finds at least 80% of ROBUST's confident targets over the next 15 frames.
     Condition-varied copies (blur, shake, tiny targets, low light, noise) are generated for variety.
   - A `DecisionTreeClassifier` and a `RandomForestClassifier` are trained, cross-validated by clip and
     compared with the rule-based baseline; they are saved to `weights/selector_{tree,forest}.joblib`
     and the report to `outputs/selector/report.json`.
   - `selector.type` in `config.yaml` picks `forest`, `tree` or `rules`; the app also lets you pick per run.
4. **Fine-tune YOLO Detector**: Implement training loop in `src/training/finetune_detector.py` using your labeled UAV dataset.
5. **Benchmark with TrackEval**: Implement `run_trackeval()` in `src/evaluation/metrics.py` to evaluate HOTA/IDF1 against MOT-FLY ground truth.

---

## Team

- **Adhithi Gatty** (4SF24CI007)
- **Chaitrika** (4SF24CI042)
- **Tharun Rai** (4SF24CI176)
- **Abdul Wahid**
