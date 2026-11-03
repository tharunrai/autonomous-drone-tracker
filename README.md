# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

Team Ctrl Alt Elite, Dept. of CSE (AI & ML), Sahyadri College of Engineering & Management
Guide: Dr. Pushpalatha K

## What this project does

In air-to-air UAV tracking, both the camera UAV and the target UAV move. The target can become tiny, blurred, occluded or confused with the background. A fixed tracking setup either wastes compute when the scene is easy or struggles when it is hard.

This project estimates how difficult the current situation is and switches between two modes:

- **FAST**: lightweight settings for easy, stable scenes
- **ROBUST**: heavier settings for difficult scenes

We compare three systems: **Always FAST**, **Always ROBUST** and **Adaptive**, on tracking quality versus compute cost.

We do not claim a new detector or tracker. The contribution is the condition-aware policy that decides when extra computation is worth it.

## Pipeline

```
Video frames
   -> YOLO detector
   -> ByteTrack (tracks + IDs)
   -> Feature extraction (size, motion, blur, confidence, track health)
   -> Selector (rules first, Random Forest later) + hysteresis
   -> FAST or ROBUST mode for the next frames
   -> Annotated output + metrics
```

## Modes

| | FAST | ROBUST |
|---|---|---|
| Model | yolov8n | yolov8s |
| Image size | 640 | 1280 (lower to 960 on CPU) |
| Detection | every 3rd frame, tracker predicts between | every frame |

Values are set in `config.yaml`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── config.yaml        # mode settings, thresholds, paths
├── data/              # datasets (not committed)
├── weights/           # model weights (not committed)
├── src/
│   ├── detect_track.py   # YOLO + ByteTrack, FAST/ROBUST settings
│   ├── features.py       # blur, size, motion, confidence, track health
│   ├── selector.py       # rule-based selector, Random Forest later, hysteresis
│   └── evaluate.py       # metrics, FPS, detector calls, plots
├── run.py             # runs FAST, ROBUST and ADAPTIVE on a video
├── app.py             # Streamlit demo
└── outputs/           # annotated videos, metrics, plots
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put detector weights in `weights/` and set the path in `config.yaml`. Community drone-detection YOLO weights work best. Plain `yolov8n.pt` is the fallback, with detections filtered to airplane, bird and kite classes at a low confidence threshold.

## Run

```bash
# compare all three systems on one video
python run.py --video data/sample.mp4

# demo dashboard
streamlit run app.py
```

Outputs (annotated video, per-frame mode log, metrics CSV) are saved in `outputs/`.

## Evaluation

- **Compute cost** (no labels needed): FPS, latency, detector calls
- **Tracking quality** (needs ground truth): HOTA, IDF1, MOTA, ID switches (MOT-format data such as MOT-FLY)
- Train, validation and test splits are by **video sequence**, never by frame, to avoid leakage.

## Datasets

- MOT-FLY (primary candidate, download and labels to be verified)
- A2A-UAV-Tracking (secondary or backup)

## Status

- [ ] Detector + ByteTrack running on one video
- [ ] FAST and ROBUST modes
- [ ] Rule-based selector with hysteresis
- [ ] Streamlit demo
- [ ] Dataset verified and metrics running
- [ ] Profile FAST vs ROBUST across conditions
- [ ] Random Forest selector vs rule-based
- [ ] Final comparison: Always FAST vs Always ROBUST vs Adaptive

## References

1. Chu et al., An Experimental Evaluation Based on New Air-to-Air Multi-UAV Tracking Dataset, IEEE ICUS, 2023.
2. Luo et al., A2A-UAV-Tracking: A Benchmark for Air-to-Air UAV Tracking, Int. J. Antennas Propag., 2026.
3. Kang et al., ATA: A Benchmark for Vision-Language Tracking in Air-to-Air Counter-UAV of Tiny Drones, Drones 10(6), 2026.
4. Qi et al., OATrack: A Quality-Gated Progressive Association Framework with a YOLO Detection Cache for UAV Small-Object MOT, Sensors 26(16), 2026.

## Team

Adhithi Gatty (4SF24CI007), Chaitrika (4SF24CI042), Tharun Rai (4SF24CI176), Abdul Wahid 