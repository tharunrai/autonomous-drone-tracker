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
├── requirements.lock.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   ├── mot_fly/
│   │   └── a2a_uav_tracking/
│   ├── processed/
│   └── splits/
├── weights/
│   ├── yolov8n.pt
│   └── yolov8s.pt
├── src/
│   ├── __init__.py
│   ├── detection/        # YOLO detector module (detector.py)
│   ├── tracking/         # ByteTrack association module (tracker.py)
│   ├── features/         # Blur, illumination, motion, scale extractors
│   ├── selector/         # Labeling, training, inference, and hysteresis smoothing
│   ├── pipeline/         # FAST, ROBUST, and Adaptive pipeline orchestrators
│   ├── evaluation/       # TrackEval wrapper, compute profiling, plotting
│   └── utils/            # Config parser, I/O helpers, logging setup
├── scripts/
│   ├── check_setup.py
│   ├── profile_modes.py
│   ├── extract_features.py
│   ├── build_labels.py
│   ├── train_selector.py
│   └── run_experiments.py
├── app/
│   ├── streamlit_app.py  # Landing page
│   └── pages/
│       ├── 1_Demo.py     # Interactive video simulation & mode telemetry
│       └── 2_Results.py  # Tracking quality vs compute trade-off dashboard
├── outputs/
│   ├── videos/
│   ├── tracks/
│   ├── metrics/
│   └── plots/
├── notebooks/
├── tests/
└── third_party/
    └── TrackEval/        # Official TrackEval evaluation repository
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or 3.11 (Python 3.11 recommended)
- Git
- NVIDIA GPU with CUDA drivers (optional, falls back to CPU if unavailable)

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# If CUDA GPU is available (example for CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install base requirements
pip install -r requirements.txt
```

### 4. Clone TrackEval
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
pip install -r third_party/TrackEval/requirements.txt
```

### 5. Download Model Weights
Download the baseline model checkpoints into `weights/`:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
# Move weights to weights/ directory if not downloaded directly
```

### 6. Verify Environment Setup
Run the automated pre-flight setup check:
```bash
python scripts/check_setup.py
```

---

## Running the Streamlit Application

Launch the interactive Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to access the landing page, live interactive mode simulation demo, and benchmark evaluation results.

---

## Development Roadmap & Checklist

- [ ] **Baseline Pipeline**: Implement YOLODetector (`src/detection/detector.py`) and ByteTrack association (`src/tracking/tracker.py`)
- [ ] **Profiling**: Benchmark Always FAST and Always ROBUST baseline configurations across target videos (`scripts/profile_modes.py`)
- [ ] **Feature Extraction**: Implement modular image quality, motion, and bbox scale extractors (`src/features/`)
- [ ] **Selector**: Build label generator (`src/selector/labeler.py`) and train Random Forest mode classifier (`src/selector/train.py`)
- [ ] **Adaptive Pipeline**: Wire condition extractor, selector, hysteresis smoother, and dynamic perception switches (`src/pipeline/adaptive_pipeline.py`)
- [ ] **Evaluation**: Benchmark Always FAST vs Always ROBUST vs Adaptive using TrackEval (`scripts/run_experiments.py`)
- [ ] **Demo UI**: Connect real video feeds, live telemetry charts, and model inference into Streamlit (`app/`)

