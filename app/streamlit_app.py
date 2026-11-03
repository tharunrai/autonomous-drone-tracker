"""Streamlit landing page for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking."""

import streamlit as st

st.set_page_config(
    page_title="Condition-Aware Adaptive Perception",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking")

st.markdown(
    """
### Project Overview
In air-to-air (A2A) unmanned aerial vehicle (UAV) tracking, edge compute constraints clash with dynamic, 
unpredictable aerial environments. Rapid maneuvers, motion blur, varying lighting, and extreme range 
degrade tracking fidelity if perception is underpowered. Conversely, running maximal-resolution heavy 
detectors on every frame causes thermal throttling and unacceptable latency.

**This project introduces a condition-aware adaptive perception pipeline:**
- **Dynamic Mode Selection:** An online Random Forest selector predicts environmental difficulty per frame.
- **Perception Modes:**
  - ⚡ **FAST Mode:** Lightweight detector (`yolov8n`), lower resolution (`640x640`), frame skipping (`detect_every_n_frames=3`), ByteTrack association.
  - 🛡️ **ROBUST Mode:** High-capacity detector (`yolov8s`), high resolution (`1280x1280`), dense detection (`detect_every_n_frames=1`), target RoI cropping.
- **Hysteresis Smoothing:** Stabilizes mode toggling across consecutive frames to avoid rapid switching jitter.
- **Evaluation:** Benchmarked on tracking accuracy (HOTA, IDF1, MOTA, IDSW) vs compute expenditure (FPS, detector invocations).

---
👈 **Navigate using the sidebar:**
- **1_Demo:** Interactive sequence runner with real-time condition, mode, and FPS indicators.
- **2_Results:** Benchmark comparison tables and accuracy vs compute Pareto trade-off charts.
"""
)

st.info("💡 Select a page from the sidebar to test demo sequences or review benchmarking results.")

