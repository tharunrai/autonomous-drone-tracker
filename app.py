"""Streamlit web application for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Condition-Aware Adaptive UAV Tracking",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking")

# Navigation via sidebar
view_selection = st.sidebar.radio(
    "Navigation",
    options=["📖 Project Overview", "🎥 Live Demo Simulation", "📊 Benchmark Results"],
    index=0,
)

if view_selection == "📖 Project Overview":
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
💡 **Use the sidebar to navigate to the Live Demo Simulation or Benchmark Results.**
"""
    )

elif view_selection == "🎥 Live Demo Simulation":
    st.header("🎥 Live Demo & Perception Mode Switching")
    st.markdown("Run UAV tracking with real-time condition estimation and dynamic mode adaptation.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Controls")
    sequence = st.sidebar.selectbox(
        "Select UAV Sequence",
        options=["Sequence 01 - High Altitude Clear", "Sequence 02 - Fast Turn & Blur", "Sequence 03 - Low-Light Cloud Crossing"],
        index=0,
    )
    pipeline_mode = st.sidebar.radio(
        "Pipeline Operational Mode",
        options=["Adaptive (Condition-Aware)", "Baseline Always FAST", "Baseline Always ROBUST"],
        index=0,
    )
    run_button = st.sidebar.button("Run Sequence Simulation", type="primary")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Video Playback: {sequence}")
        video_placeholder = st.empty()
        video_placeholder.info("📹 Video playback placeholder: Real-time annotated tracking video will render here.")

    with col2:
        st.subheader("Real-Time Telemetry")
        condition_metric = st.empty()
        mode_metric = st.empty()
        fps_metric = st.empty()

        condition_metric.metric(label="Condition Difficulty", value="Moderate Blur", delta="-0.12 vs Clear")
        mode_metric.metric(label="Active Perception Mode", value="⚡ FAST", delta=None)
        fps_metric.metric(label="Inference FPS", value="42.5 FPS", delta="+18.2 vs ROBUST")

        st.markdown("---")
        st.subheader("Extracted Frame Features (Mock)")
        st.json({
            "laplacian_variance_blur": 142.8,
            "mean_brightness": 128.4,
            "optical_flow_magnitude": 3.75,
            "target_bbox_area_norm": 0.012,
            "mode_persistence_count": 8,
        })

    if run_button:
        st.toast(f"Running simulation on {sequence} ({pipeline_mode})")

elif view_selection == "📊 Benchmark Results":
    st.header("📊 Benchmark Results & Compute Trade-Offs")
    st.markdown(
        "Comparative evaluation of **Always FAST**, **Always ROBUST**, and **Condition-Aware Adaptive** pipelines "
        "across tracking quality metrics (HOTA, IDF1, MOTA, IDSW) and compute resource footprint."
    )

    dummy_results = pd.DataFrame(
        [
            {"Pipeline System": "Always FAST", "HOTA (%)": 56.4, "IDF1 (%)": 63.8, "MOTA (%)": 68.2, "IDSW": 42, "Avg FPS": 48.6, "Detector Calls (%)": 33.3},
            {"Pipeline System": "Always ROBUST", "HOTA (%)": 71.2, "IDF1 (%)": 78.5, "MOTA (%)": 79.1, "IDSW": 14, "Avg FPS": 18.2, "Detector Calls (%)": 100.0},
            {"Pipeline System": "Adaptive (Ours)", "HOTA (%)": 69.8, "IDF1 (%)": 77.1, "MOTA (%)": 77.9, "IDSW": 18, "Avg FPS": 38.4, "Detector Calls (%)": 52.1},
        ]
    )

    st.subheader("Tracking Quality vs Compute Efficiency Summary")
    st.dataframe(dummy_results, use_container_width=True)

    st.markdown("---")
    st.subheader("Accuracy vs Compute Cost Trade-Off")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("##### HOTA vs Inference FPS (Dummy Plot)")
        chart_data = pd.DataFrame(
            {
                "Method": ["Always FAST", "Always ROBUST", "Adaptive (Ours)"],
                "FPS": [48.6, 18.2, 38.4],
                "HOTA": [56.4, 71.2, 69.8],
            }
        ).set_index("Method")
        st.scatter_chart(chart_data, x="FPS", y="HOTA", size=[150, 150, 200])

    with plot_col2:
        st.markdown("##### Detector Invocations vs Identity Switches")
        bar_data = pd.DataFrame(
            {
                "Method": ["Always FAST", "Always ROBUST", "Adaptive (Ours)"],
                "Detector Call %": [33.3, 100.0, 52.1],
                "ID Switches": [42, 14, 18],
            }
        ).set_index("Method")
        st.bar_chart(bar_data)

