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
- **Hysteresis Smoothing**: Prevents rapid mode flipping on transient anomalies.
- Compares **Always FAST**, **Always ROBUST**, and **ADAPTIVE** on tracking quality vs computational cost (FPS, detector calls).

---

## Repository Structure

```
.
├── app.py                      # Interactive Streamlit UI (video upload, playback, telemetry)
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
├── weights/                    # Model weights (yolov8n.pt, yolov8s.pt, selector_rf.joblib)
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
    │   └── selector.py         # RuleBasedSelector, MLSelector (placeholder), HysteresisSmoother
    ├── pipeline/
    │   ├── __init__.py
    │   └── pipeline.py         # Mode enum, ModeConfig, TrackingPipeline with visual HUD
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py          # FPS, detector call ratio, summary stats, TrackEval placeholder
    └── training/
        ├── __init__.py
        ├── train_selector.py   # PLACEHOLDER: build labels + train Random Forest selector
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
```

---

## Usage

### 1. Streamlit Web Dashboard
Run the interactive application:
```bash
streamlit run app.py
```
*(Or double-click `start.bat` on Windows)*

**Features:**
- Upload your own video (`.mp4`, `.avi`, `.mov`).
- Choose **ADAPTIVE**, **FAST**, or **ROBUST** mode.
- Tune confidence threshold and test with max frames limit.
- Play original vs annotated video side-by-side with trajectory trails and live HUD.
- Inspect real-time throughput metrics (FPS, detector calls, mode switches).
- View timeline plot of mode selections against condition features.
- Download the browser-playable H.264 annotated video.

### 2. Command Line Interface (CLI)
```bash
# Verify environment, GPU, and model weights
python run.py --check-setup

# Run ADAPTIVE tracking on a video
python run.py --video path/to/drone_flight.mp4 --mode ADAPTIVE

# Run FAST or ROBUST baseline with frame cap
python run.py --video path/to/drone_flight.mp4 --mode FAST --max-frames 150
```

---

## Where to Extend

1. **Swap Detector Weights**: Change model weights in `config.yaml` (`modes.fast.weights` or `modes.robust.weights`) to custom fine-tuned drone weights (e.g. YOLOv8 fine-tuned on drone datasets).
2. **Add Custom Features**: Add spatial-temporal features (e.g. target velocity variance, contrast entropy) in `src/features/features.py`.
3. **Train & Switch to MLSelector**:
   - Implement labeling logic in `src/training/train_selector.py` to produce `data/processed/training_labels.csv`.
   - Train and serialize a `RandomForestClassifier` into `weights/selector_rf.joblib`.
   - Enable `selector.type: "ml"` in `config.yaml` to deploy `MLSelector`.
4. **Fine-tune YOLO Detector**: Implement training loop in `src/training/finetune_detector.py` using your labeled UAV dataset.
5. **Benchmark with TrackEval**: Implement `run_trackeval()` in `src/evaluation/metrics.py` to evaluate HOTA/IDF1 against MOT-FLY ground truth.

---

## Team

- **Adhithi Gatty** (4SF24CI007)
- **Chaitrika** (4SF24CI042)
- **Tharun Rai** (4SF24CI176)
- **Abdul Wahid**
