"""Streamlit interactive UI for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pipeline.pipeline import Mode, TrackingPipeline

st.set_page_config(
    page_title="Condition-Aware Adaptive UAV Tracking",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking")
st.markdown(
    "Dynamically adjust perception workload between **FAST** (lightweight, high FPS) "
    "and **ROBUST** (dense detection, high resolution) based on real-time environmental conditions."
)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.header("🕹️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Video File",
    type=["mp4", "avi", "mov", "mkv"],
    help="Select a video file to run tracking on.",
)

selected_mode_str = st.sidebar.selectbox(
    "Operational Mode",
    options=["ADAPTIVE", "FAST", "ROBUST"],
    index=0,
    help="ADAPTIVE toggles between FAST and ROBUST per frame based on scene difficulty.",
)

conf_threshold = st.sidebar.slider(
    "Detector Confidence Threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Minimum confidence score required for YOLO detection.",
)

max_frames_input = st.sidebar.number_input(
    "Max Frames Limit (0 for full video)",
    min_value=0,
    max_value=20000,
    value=0,
    step=50,
    help="Process only the first N frames for rapid benchmarking.",
)
max_frames = int(max_frames_input) if max_frames_input > 0 else None

run_btn = st.sidebar.button("🚀 Run Tracking", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Video Processing Execution
# ---------------------------------------------------------
if uploaded_file is not None:
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_video_path = uploads_dir / uploaded_file.name

    # Save uploaded file to disk
    with open(saved_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({saved_video_path.stat().st_size / (1024*1024):.1f} MB)")

    if run_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def on_progress(current: int, total: int, instant_fps: float) -> None:
            pct = min(1.0, current / max(1, total))
            progress_bar.progress(pct)
            status_box.markdown(f"**Processing frame {current} / {total}** — Current Speed: `{instant_fps:.1f} FPS`")

        try:
            pipeline = TrackingPipeline(
                mode=Mode(selected_mode_str),
                custom_conf=conf_threshold,
            )
            with st.spinner("Executing perception pipeline and re-encoding video..."):
                summary = pipeline.run_video(
                    input_path=saved_video_path,
                    max_frames=max_frames,
                    progress_callback=on_progress,
                )
            progress_bar.progress(1.0)
            status_box.success("✅ Tracking and video generation completed successfully!")
            st.session_state["last_summary"] = summary
            st.session_state["last_input_video"] = str(saved_video_path)
            st.session_state["last_mode"] = selected_mode_str
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            raise e

# ---------------------------------------------------------
# Display Results Area
# ---------------------------------------------------------
if "last_summary" in st.session_state:
    summary = st.session_state["last_summary"]
    input_video = st.session_state["last_input_video"]
    active_mode = st.session_state["last_mode"]
    records_df: pd.DataFrame = summary["records_df"]

    st.markdown("---")
    st.header("📊 Tracking Output & Real-Time Telemetry")

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Average Throughput", value=f"{summary['avg_fps']} FPS")
    with col2:
        call_pct = summary["detector_call_ratio"] * 100
        st.metric(label="Detector Invocations", value=f"{summary['detector_calls']} ({call_pct:.1f}%)")
    with col3:
        st.metric(label="Mode Switches", value=summary["mode_switches"])
    with col4:
        max_tracks = int(records_df["num_tracks"].max()) if not records_df.empty else 0
        st.metric(label="Peak Tracked Objects", value=max_tracks)

    # Side-by-side Video Playback
    st.markdown("### 🎥 Side-by-Side Video Comparison")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Original Input Video")
        if Path(input_video).is_file():
            st.video(input_video)

    with col_right:
        st.subheader(f"Annotated Playback ({active_mode} Mode)")
        annotated_path = summary["output_video"]
        if Path(annotated_path).is_file():
            st.video(annotated_path)

    # Download Button
    if Path(summary["output_video"]).is_file():
        with open(summary["output_video"], "rb") as video_bytes:
            st.download_button(
                label="⬇️ Download Annotated Video (H.264 MP4)",
                data=video_bytes,
                file_name=Path(summary["output_video"]).name,
                mime="video/mp4",
                use_container_width=True,
            )

    # Telemetry Line Chart for ADAPTIVE mode
    if not records_df.empty:
        st.markdown("### 📈 Environmental Conditions & Mode Switching Over Time")

        fig, (ax_mode, ax_features) = plt.subplots(
            2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )

        frames = records_df["frame"]

        # 1. Mode plot (Step chart)
        mode_numeric = records_df["mode"].map({"FAST": 0, "ROBUST": 1})
        ax_mode.step(frames, mode_numeric, where="mid", color="#00aa44", linewidth=2)
        ax_mode.set_yticks([0, 1])
        ax_mode.set_yticklabels(["FAST", "ROBUST"])
        ax_mode.set_ylabel("Active Mode")
        ax_mode.set_title("Operational Mode Timeline")
        ax_mode.grid(True, linestyle="--", alpha=0.5)

        # 2. Condition Features plot
        if "blur" in records_df:
            # Normalize blur for visual comparison
            norm_blur = records_df["blur"] / max(1.0, records_df["blur"].max())
            ax_features.plot(frames, norm_blur, label="Blur (Norm)", color="#3388ff", alpha=0.8)

        if "camera_motion" in records_df:
            norm_motion = records_df["camera_motion"] / max(1.0, records_df["camera_motion"].max())
            ax_features.plot(frames, norm_motion, label="Camera Motion (Norm)", color="#ff8800", alpha=0.8)

        if "mean_confidence" in records_df:
            ax_features.plot(frames, records_df["mean_confidence"], label="Mean Confidence", color="#8833ff", alpha=0.8)

        ax_features.set_xlabel("Video Frame Index")
        ax_features.set_ylabel("Normalized Feature Value")
        ax_features.legend(loc="upper right")
        ax_features.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Metrics Data Table
        with st.expander("📄 View Frame-by-Frame Metrics Table"):
            st.dataframe(records_df, use_container_width=True)

else:
    st.info("👈 Upload a video in the sidebar and click **'Run Tracking'** to begin.")
