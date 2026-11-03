"""Interactive sequence demonstration page for adaptive UAV tracking."""

import streamlit as st

st.set_page_config(page_title="Live Demo - Adaptive UAV Tracking", page_icon="🎥", layout="wide")

st.title("🎥 Live Demo & Perception Mode Switching")
st.markdown("Run UAV tracking with real-time condition estimation and dynamic mode adaptation.")

# Sidebar controls
st.sidebar.header("Demo Configuration")
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

    # Initial placeholder indicators
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

