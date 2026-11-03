# Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking

A condition-aware adaptive perception system for Air-to-Air (A2A) UAV tracking that dynamically balances tracking quality (HOTA, IDF1, MOTA, IDSW) and computational resource consumption (FPS, detector invocations) on resource-constrained aerial platforms. Operating on continuous video feeds, the system extracts lightweight environmental and target condition metrics (blur, illumination, optical flow, geometric scale) and employs a Random Forest selector with hysteresis smoothing to dynamically toggle between a high-throughput **FAST** mode and a high-accuracy **ROBUST** mode.

```
                                  +---------------------------------------+
                                  |         Air-to-Air UAV Video          |
                                  +---------------------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |  Condition Feature        |
                                        |  Extractor (Blur, Flow,   |
                                        |  Brightness, Box Scale)   |
                                        +---------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |   Random Forest Selector  |
                                        |   + Hysteresis Smoother   |
                                        +---------------------------+
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                                     v                                 v
                        [ FAST Mode: YOLOv8n ]             [ ROBUST Mode: YOLOv8s ]
                        - 640x640 resolution               - 1280x1280 resolution
                        - Detect every 3 frames            - Detect every frame
                        - ByteTrack association            - RoI crop + ByteTrack
                                     |                                 |
                                     +----------------+----------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |  Continuous UAV Tracks    |
                                        |  (ID, Bounding Box, Conf) |
                                        +---------------------------+
```

---

## Folder Layout

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── run.py                # Unified CLI runner (check-setup, profile, features, train, evaluate)
├── app.py                # Streamlit web app (overview, live demo simulation, results)
├── src/
│   ├── __init__.py
│   ├── detect_track.py   # YOLO detector, ByteTrack, FAST/ROBUST modes, adaptive pipeline
│   ├── features.py       # Blur, illumination, optical flow, and bbox geometry extractors
│   ├── selector.py       # Label generator, RF selector training, inference, hysteresis
│   └── evaluate.py       # Evaluation metrics, TrackEval wrapper, compute cost, plots
├── data/
│   ├── raw/
│   │   ├── mot_fly/
│   │   └── a2a_uav_tracking/
│   ├── processed/
│   └── splits/
├── weights/
│   ├── yolov8n.pt
│   └── yolov8s.pt
└── outputs/
    ├── videos/
    ├── tracks/
    ├── metrics/
    └── plots/
```

---

## Setup & Quickstart

### 1. Environment & Dependencies
```bash
# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Environment & GPU
Run the verification check:
```bash
python run.py --check-setup
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view the Project Overview, test the interactive Live Demo Simulation, and explore benchmark results.

### 4. CLI Workflow
```bash
# Profile baseline FAST vs ROBUST modes
python run.py --profile

# Extract frame condition features
python run.py --extract-features

# Build training labels
python run.py --build-labels

# Train Random Forest mode selector
python run.py --train-selector

# Run comparative evaluation across all 3 modes
python run.py --evaluate
```
