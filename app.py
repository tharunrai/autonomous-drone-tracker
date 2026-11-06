"""Streamlit UI for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking.

Layout:  sidebar = settings  ->  main page = intro, run status, then results in 3 tabs.
Each part of the page lives in its own small function so it is easy to find and edit.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.pipeline.pipeline import Mode, TrackingPipeline

UPLOAD_DIR = Path("data/uploads")

MODE_HELP = {
    "ADAPTIVE": "Switches between FAST and ROBUST on its own, based on how hard each moment looks.",
    "FAST": "Small model, low resolution, skips frames. Quickest, least careful.",
    "ROBUST": "Big model, high resolution, every frame. Slowest, most careful.",
}

st.set_page_config(page_title="UAV Tracker", page_icon=":material/flight:", layout="wide")

# A few small touches the theme file can't do (tighter sidebar title, stat card look).
st.html(
    """
    <style>
      [data-testid="stMetric"] {
          background: #efeadf; border: 1px solid #d9d2c3;
          border-radius: 6px; padding: 14px 18px;
      }
      [data-testid="stMetricLabel"] { opacity: .75; }
      .eyebrow { letter-spacing: .14em; text-transform: uppercase;
                 font-size: .75rem; color: #c2410c; font-weight: 700; margin: 0; }
    </style>
    """
)


# ----------------------------------------------------------------------------
# Sidebar: all the settings in one place
# ----------------------------------------------------------------------------
def sidebar_settings():
    """Draw the sidebar and return what the user picked."""
    with st.sidebar:
        st.markdown("### :material/flight: UAV Tracker")
        st.caption("Condition-aware perception for air-to-air tracking")
        st.divider()

        video = st.file_uploader("1. Flight video", type=["mp4", "avi", "mov", "mkv"])

        mode = st.segmented_control(
            "2. Mode", options=list(MODE_HELP), default="ADAPTIVE", selection_mode="single"
        ) or "ADAPTIVE"
        st.caption(MODE_HELP[mode])

        with st.expander("Fine tuning"):
            conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05,
                             help="Detections below this score are ignored.")
            limit = st.number_input("Only process first N frames", 0, 20000, 0, 50,
                                    help="0 means the whole video. Handy for quick tests.")

        run = st.button("Run tracking", type="primary", width="stretch",
                        icon=":material/play_arrow:", disabled=video is None)
        if video is None:
            st.caption("Upload a video to enable the button.")
    return video, mode, conf, (int(limit) or None), run


# ----------------------------------------------------------------------------
# Running the pipeline
# ----------------------------------------------------------------------------
def save_upload(video) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / video.name
    path.write_bytes(video.getbuffer())
    return path


def run_pipeline(video_path: Path, mode: str, conf: float, max_frames):
    """Run tracking with a live progress bar, then store results in session_state."""
    bar = st.progress(0.0, text="Starting...")

    def on_progress(current, total, fps):
        bar.progress(min(1.0, current / max(1, total)),
                     text=f"Frame {current} of {total}  ·  {fps:.0f} fps")

    try:
        pipeline = TrackingPipeline(mode=Mode(mode), custom_conf=conf)
        summary = pipeline.run_video(input_path=video_path, max_frames=max_frames,
                                     progress_callback=on_progress)
    except Exception as err:  # show a readable message instead of a stack trace
        bar.empty()
        st.error(f"Tracking failed: {err}")
        st.stop()

    bar.empty()
    st.session_state["result"] = {"summary": summary, "input": str(video_path), "mode": mode}


# ----------------------------------------------------------------------------
# Results: three tabs
# ----------------------------------------------------------------------------
def stat_row(summary, frames_df):
    peak = int(frames_df["num_tracks"].max()) if not frames_df.empty else 0
    cols = st.columns(4)
    cols[0].metric("Speed", f"{summary['avg_fps']} fps")
    cols[1].metric("Detector used on", f"{summary['detector_call_ratio'] * 100:.0f}% of frames",
                   help=f"{summary['detector_calls']} detector calls in total")
    cols[2].metric("Mode switches", summary["mode_switches"])
    cols[3].metric("Most targets at once", peak)


def tab_video(result):
    summary = result["summary"]
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        if Path(result["input"]).is_file():
            st.video(result["input"])
    with right:
        st.caption(f"Tracked · {result['mode'].title()} mode")
        if Path(summary["output_video"]).is_file():
            st.video(summary["output_video"])

    d1, d2, _ = st.columns([1, 1, 1])
    for col, key, label, mime in [
        (d1, "output_video", "Download video", "video/mp4"),
        (d2, "tracks_csv", "Download positions (CSV)", "text/csv"),
    ]:
        path = Path(summary.get(key, ""))
        if path.is_file():
            col.download_button(label, path.read_bytes(), file_name=path.name, mime=mime,
                                width="stretch", icon=":material/download:")


def tab_targets(tracks_df):
    if tracks_df.empty:
        st.info("No objects were tracked in this video. Try lowering the confidence in the sidebar.")
        return

    ids = sorted(tracks_df["track_id"].unique())
    st.write(f"**{len(ids)}** target(s) followed across **{len(tracks_df)}** observations.")

    choice = st.pills("Show targets", [f"#{i}" for i in ids], selection_mode="multi",
                      default=[f"#{i}" for i in ids][:6])
    shown = tracks_df[tracks_df["track_id"].isin([int(c[1:]) for c in choice])].copy()
    shown["target"] = "#" + shown["track_id"].astype(str)

    # Flight path in image coordinates (y is flipped so it matches the video).
    path = (
        alt.Chart(shown)
        .mark_line(point=alt.OverlayMarkDef(size=18))
        .encode(
            x=alt.X("center_x:Q", title="x (px)"),
            y=alt.Y("center_y:Q", title="y (px)", scale=alt.Scale(reverse=True)),
            color=alt.Color("target:N", title="Target"),
            order="frame:Q",
            tooltip=["target", "frame", "center_x", "center_y", "confidence"],
        )
        .properties(height=380)
        .interactive()
    )
    st.altair_chart(path, alt="Flight paths of tracked targets in image coordinates")

    st.markdown("**Last known position of each target**")
    last = tracks_df.sort_values("frame").groupby("track_id").tail(1)
    st.dataframe(
        last[["track_id", "frame", "center_x", "center_y", "bbox_width", "bbox_height", "confidence"]]
        .rename(columns={"track_id": "target"}),
        hide_index=True, alt="Last known position of each target",
    )
    with st.expander("Every observation"):
        st.dataframe(shown.drop(columns="target"), hide_index=True, alt="All tracked observations")


def tab_conditions(frames_df):
    if frames_df.empty:
        st.info("No per-frame data available.")
        return

    st.markdown("**Which mode was running**")
    modes = frames_df.assign(run=lambda d: (d["mode"] != d["mode"].shift()).cumsum())
    blocks = modes.groupby("run").agg(start=("frame", "min"), end=("frame", "max"), mode=("mode", "first"))
    timeline = (
        alt.Chart(blocks.reset_index())
        .mark_bar(height=26)
        .encode(
            x=alt.X("start:Q", title="frame"), x2="end:Q",
            color=alt.Color("mode:N", title=None,
                            scale=alt.Scale(domain=["FAST", "ROBUST"], range=["#2f6f6a", "#c2410c"])),
            tooltip=["mode", "start", "end"],
        )
        .properties(height=60)
    )
    st.altair_chart(timeline, alt="Timeline showing which mode was active over the video")

    st.markdown("**What the scene looked like** (each line scaled 0 to 1)")
    names = {"blur": "Sharpness", "camera_motion": "Camera motion", "mean_confidence": "Detection confidence"}
    cols = [c for c in names if c in frames_df]
    if cols:
        scaled = frames_df[cols] / frames_df[cols].max().clip(lower=1e-9)
        scaled.index = frames_df["frame"]
        st.line_chart(scaled.rename(columns=names), height=280)
    with st.expander("Per-frame table"):
        st.dataframe(frames_df, hide_index=True, alt="Per-frame telemetry")


def show_results(result):
    summary = result["summary"]
    frames_df = summary["records_df"]
    tracks_df = summary.get("tracks_df", pd.DataFrame())

    st.success(f"Done: {summary['total_frames']} frames in {summary['total_time_seconds']} s.",
               icon=":material/check_circle:")
    stat_row(summary, frames_df)

    video_tab, targets_tab, cond_tab = st.tabs(
        [":material/movie: Video", ":material/moving: Targets & paths", ":material/tune: Conditions"]
    )
    with video_tab:
        tab_video(result)
    with targets_tab:
        tab_targets(tracks_df)
    with cond_tab:
        tab_conditions(frames_df)


def show_empty_state():
    with st.container(border=True):
        st.markdown("#### How it works")
        a, b, c = st.columns(3)
        a.markdown("**1. Upload**  \nPick a drone flight video in the sidebar.")
        b.markdown("**2. Choose a mode**  \nLet *Adaptive* decide, or force *Fast* / *Robust*.")
        c.markdown("**3. Run**  \nWatch the tracked video, flight paths and mode timeline.")
    st.caption("FAST is quick but coarse. ROBUST is careful but slow. "
               "ADAPTIVE uses blur, camera motion, target size and confidence to pick between them.")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
video, mode, conf, max_frames, run = sidebar_settings()

st.html('<p class="eyebrow">Air-to-air tracking</p>')
st.title("Spend compute only where the scene is hard")

if video is not None and run:
    run_pipeline(save_upload(video), mode, conf, max_frames)

if "result" in st.session_state:
    show_results(st.session_state["result"])
else:
    show_empty_state()
