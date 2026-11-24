"""Streamlit UI for Condition-Aware Adaptive Perception for Air-to-Air UAV Tracking.

Layout:  main page = header, setup panel (upload, mode, run buttons), run status, then
results in 4 tabs.  Sidebar = fine tuning and saved runs.
Every run is kept for the session, so the same video can be compared across modes.
Each part of the page lives in its own small function so it is easy to find and edit.
"""

import time
from collections import deque
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.selector.selector import SELECTOR_LABELS, available_selectors, selector_model_path
from src.utils.config import load_config

# The tracking pipeline (torch + YOLO) is imported by the loading screen, not here,
# so the page can show something during the slow first import.

UPLOAD_DIR = Path("data/uploads")

MODE_HELP = {
    "ADAPTIVE": "A selector watches every moment and switches between FAST and ROBUST: "
                "a trained model, or the hand-set rules as a baseline.",
    "FAST": "Small model, low resolution, skips frames. Quickest, least careful.",
    "ROBUST": "Big model, high resolution, every frame. Slowest, most careful.",
}
# One color per mode, reused by every chart (matches the theme's categorical palette).
MODE_COLORS = {"ADAPTIVE": "#2563eb", "FAST": "#0d9488", "ROBUST": "#ea580c"}
BADGE_COLORS = {"ADAPTIVE": "blue", "FAST": "green", "ROBUST": "orange"}
SELECTOR_HELP = {
    "rules": "Hand-set thresholds on blur, motion, target size and confidence. The baseline.",
    "tree": "A decision tree trained to predict when FAST would miss targets ROBUST finds.",
    "forest": "A random forest (many trees voting) trained for the same job. Usually the most accurate.",
}
RESULT_TABS = [":material/movie: Video", ":material/moving: Targets & paths",
               ":material/tune: Conditions", ":material/compare_arrows: Compare runs"]
LIVE_MAX_UPDATES = 12  # live preview redraws per second
LIVE_MAX_WIDTH = 960   # live preview frames are shrunk to this width before sending

st.set_page_config(page_title="UAV Tracker", page_icon=":material/flight:", layout="wide")
st.session_state.setdefault("runs", {})  # label -> run dict, newest last


@st.cache_data(ttl=60, show_spinner=False)
def app_config() -> dict:
    return load_config("config.yaml")


@st.cache_resource(show_spinner=False, max_entries=4)
def selector_report(kind: str, mtime: float) -> dict:
    """Training report stored inside a selector bundle (mtime busts the cache on retrain)."""
    return joblib.load(selector_model_path(kind, app_config())).get("report", {})


def trained_selectors() -> list:
    """'rules' plus the learned selectors whose model files exist, best first."""
    found = available_selectors(app_config())
    return [k for k in ("forest", "tree", "rules") if k in found]


# ----------------------------------------------------------------------------
# Loading screen: shown once per server start while the libraries load and the
# models warm up (about 10 s cold). Later visits and runs skip it.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def engine_status() -> dict:
    """One shared flag per server process: has the tracker been warmed up yet?"""
    return {"ready": False}


def load_engine():
    status = engine_status()
    if status["ready"]:
        return

    screen = st.empty()
    with screen.container():
        st.space("large")
        _, middle, _ = st.columns([1, 2, 1])
        with middle.container(border=True, horizontal_alignment="center"):
            st.space("small")
            st.badge("Air-to-air tracking", icon=":material/flight:", color="primary")
            st.subheader("Getting the tracker ready", anchor=False, text_alignment="center")
            st.caption("Loading the detection libraries and warming up both models. "
                       "This only happens on first start and takes about 10 seconds.",
                       text_alignment="center")
            bar = st.progress(0.0, text="Step 1 of 3 · Loading detection libraries")

            try:
                with st.spinner("Importing torch and YOLO...", show_time=True):
                    from src.pipeline.pipeline import Mode, TrackingPipeline

                bar.progress(0.45, text="Step 2 of 3 · Warming up the fast model")
                with st.spinner("Running a first test frame...", show_time=True):
                    pipeline = TrackingPipeline(mode=Mode.ADAPTIVE, selector_type="rules")
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    pipeline.fast_detector.detect(blank)

                bar.progress(0.9, text="Step 3 of 3 · Warming up the robust model")
                with st.spinner("Running a first test frame...", show_time=True):
                    pipeline.robust_detector.detect(blank)
            except Exception as err:
                bar.empty()
                st.error(f"{err}\n\nCheck that the model files in `weights/` and "
                         "`config.yaml` are in place, then reload the page.",
                         title="The tracker couldn't start", icon=":material/error:")
                st.stop()

            bar.progress(1.0, text="Ready")
            time.sleep(0.3)  # let "Ready" show for a moment before the app appears

    status["ready"] = True
    screen.empty()


load_engine()


# ----------------------------------------------------------------------------
# Sidebar: fine tuning and saved runs
# ----------------------------------------------------------------------------
def sidebar_settings():
    """Draw the sidebar and return (conf, max_frames, live)."""
    with st.sidebar:
        st.markdown("### :material/flight: UAV Tracker")
        st.caption("Condition-aware perception for air-to-air tracking")

        st.markdown("**Settings**")
        conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05,
                         help="Detections below this score are ignored.")
        limit = st.number_input("Only process first N frames", 0, 20000, 0, 50,
                                help="0 means the whole video. Handy for quick tests.")
        live = st.toggle("Show tracking live", value=True,
                         help="Plays the tracked frames while the video is processed. "
                              "Turn off for a slightly faster run.")

        history_slot = st.container()  # filled by draw_history() once this run has finished

    return conf, int(limit) or None, live, history_slot


def draw_history(slot):
    """List the saved runs in the sidebar (called after any run so the list is current)."""
    runs = st.session_state["runs"]
    if not runs:
        return
    with slot:
        st.divider()
        st.markdown("**Saved runs**")
        for label, saved in runs.items():
            active = label == st.session_state.get("active_run")
            st.caption(f"{':material/check_circle:' if active else ':material/circle:'} {label}"
                       f" · {saved['summary']['avg_fps']} fps")
        if st.button("Clear saved runs", type="tertiary", icon=":material/delete:"):
            st.session_state["runs"] = {}
            st.rerun()


def setup_panel():
    """The main-page panel where a run is set up.

    Returns (video, mode, selector, threshold, run, compare).
    """
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            video = st.file_uploader("Flight video", type=["mp4", "avi", "mov", "mkv"],
                                     help="A drone flight clip. Large files are fine.")
        with right:
            mode = st.segmented_control(
                "Mode", options=list(MODE_HELP), default="ADAPTIVE", selection_mode="single",
                format_func=str.title, width="stretch",
            ) or "ADAPTIVE"
            selector, threshold = None, None
            if mode == "ADAPTIVE":
                options = trained_selectors()
                selector = st.segmented_control(
                    "Decided by", options=options, default=options[0], selection_mode="single",
                    format_func=SELECTOR_LABELS.get, width="stretch",
                ) or options[0]
                st.caption(SELECTOR_HELP[selector])
                if selector in ("tree", "forest"):
                    threshold = st.slider(
                        "Switch to Robust when the model is at least", 0.5, 0.95,
                        float((app_config().get("selector") or {}).get("threshold", 0.5)), 0.05,
                        format="%.2f",
                        help="The model scores each moment with the chance that FAST would "
                             "miss targets. Lower is safer (more ROBUST), higher saves "
                             "compute (more FAST).")
                if options == ["rules"]:
                    st.caption(":material/school: No trained selector yet. Train one with "
                               "`python -m src.training.train_selector`.")
            else:
                st.caption(MODE_HELP[mode])
            run = st.button("Run tracking", type="primary", width="stretch",
                            icon=":material/play_arrow:", disabled=video is None)
            compare = st.button("Compare all modes", width="stretch",
                                icon=":material/compare_arrows:", disabled=video is None,
                                help="Runs Fast, Robust, then Adaptive with every available "
                                     "selector (rules and each trained model) on this video, "
                                     "then opens the comparison.")
            if video is None:
                st.caption("Upload a video to enable the buttons.")
    return video, mode, selector, threshold, run, compare


def mode_title(mode: str, selector) -> str:
    return f"Adaptive · {SELECTOR_LABELS[selector]}" if mode == "ADAPTIVE" and selector else mode.title()


def run_label(settings) -> str:
    label = f"{settings['video_name']} · {mode_title(settings['mode'], settings.get('selector'))}"
    if settings.get("selector") in ("tree", "forest") and settings.get("threshold") not in (None, 0.5):
        label += f" @ {settings['threshold']:.2f}"
    if settings["conf"] != 0.25:
        label += f" · conf {settings['conf']:.2f}"
    if settings["max_frames"]:
        label += f" · first {settings['max_frames']}"
    return label


# ----------------------------------------------------------------------------
# Running the pipeline
# ----------------------------------------------------------------------------
def save_upload(video) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / video.name
    path.write_bytes(video.getbuffer())
    return path


def live_view():
    """Lay out the live player and return a callback that draws each new frame into it.

    Frames are shrunk and sent at most LIVE_MAX_UPDATES times a second, so the browser
    keeps up and the pipeline isn't slowed down by the preview.
    """
    frame_col, stats_col = st.columns([3, 1])
    with frame_col:
        image_slot = st.skeleton(height=360)
    with stats_col:
        mode_slot, fps_slot, tracks_slot = st.empty(), st.empty(), st.empty()

    recent_fps, recent_tracks = deque(maxlen=90), deque(maxlen=90)
    last_draw = [0.0]

    def on_frame(frame, info):
        recent_fps.append(info["fps"])
        recent_tracks.append(info["num_tracks"])
        now = time.perf_counter()
        is_last = 0 < info["total_frames"] <= info["frame"] + 1
        if now - last_draw[0] < 1 / LIVE_MAX_UPDATES and not is_last:
            return
        last_draw[0] = now

        h, w = frame.shape[:2]
        if w > LIVE_MAX_WIDTH:
            frame = cv2.resize(frame, (LIVE_MAX_WIDTH, int(h * LIVE_MAX_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
        image_slot.image(frame, channels="BGR", output_format="JPEG", width="stretch",
                         caption=f"Frame {info['frame'] + 1} of {info['total_frames']}",
                         alt="Live view of the video with tracked targets boxed")

        mode = info["mode"]
        mode_slot.metric("Mode now", mode.title(), border=True,
                         icon=":material/bolt:" if mode == "FAST" else ":material/shield:",
                         help="Which detector setup is running on this frame.")
        fps_slot.metric("Speed", f"{info['fps']:.0f} fps", border=True,
                        chart_data=list(recent_fps))
        tracks_slot.metric("Targets in view", info["num_tracks"], border=True,
                           chart_data=list(recent_tracks), chart_type="bar")

    return on_frame


def run_pipeline(video, settings, live: bool):
    """Run tracking inside a status box (optionally playing it live), then save the run."""
    from src.pipeline.pipeline import Mode, TrackingPipeline  # already loaded by load_engine

    label = run_label(settings)
    title = mode_title(settings["mode"], settings.get("selector"))
    with st.status(f"Tracking {settings['video_name']} · {title}...", expanded=True) as status:
        video_path = save_upload(video)
        bar = st.progress(0.0, text="Preparing models...")
        on_frame = live_view() if live else None
        st.caption("Changing a setting or uploading a file while this runs will stop it.")

        def on_progress(current, total, fps):
            bar.progress(min(1.0, current / max(1, total)),
                         text=f"Frame {current} of {total}  ·  {fps:.0f} fps")

        try:
            pipeline = TrackingPipeline(mode=Mode(settings["mode"]), custom_conf=settings["conf"],
                                        selector_type=settings.get("selector"),
                                        selector_threshold=settings.get("threshold"))
            summary = pipeline.run_video(input_path=video_path, max_frames=settings["max_frames"],
                                         progress_callback=on_progress, frame_callback=on_frame)
        except Exception as err:  # show a readable message instead of a stack trace
            bar.empty()
            status.update(label="Tracking failed", state="error", expanded=True)
            st.error(str(err), title="The pipeline stopped with an error")
            st.stop()

        bar.empty()
        status.update(label=f"Done: {summary['total_frames']} frames in "
                            f"{summary['total_time_seconds']} s",
                      state="complete", expanded=False)

    runs = st.session_state["runs"]
    runs.pop(label, None)  # re-running the same settings replaces the old run
    runs[label] = {**settings, "label": label, "input": str(video_path), "summary": summary}
    st.session_state["active_run"] = label


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def sparkline(values: pd.Series, points: int = 60) -> list:
    """Average a per-frame series down to a few points for a metric sparkline."""
    s = pd.Series(values).astype(float).reset_index(drop=True)
    if s.empty:
        return []
    step = max(1, len(s) // points)
    return s.groupby(s.index // step).mean().round(3).tolist()


def run_stats(run) -> dict:
    """Headline numbers for one run (also used by the comparison table)."""
    summary, frames = run["summary"], run["summary"]["records_df"]
    robust = (frames["mode"] == "ROBUST").mean() if not frames.empty else 0.0
    return {
        "run": run["label"],
        "mode": run["mode"],
        "decided_by": SELECTOR_LABELS.get(summary.get("selector"), "—"),
        "frames": summary["total_frames"],
        "seconds": summary["total_time_seconds"],
        "fps": summary["avg_fps"],
        "detector": summary["detector_call_ratio"],
        "robust": float(robust),
        "switches": summary["mode_switches"],
        "targets": summary.get("total_unique_targets", 0),
    }


def stat_row(run):
    summary, frames = run["summary"], run["summary"]["records_df"]
    stats = run_stats(run)
    peak = int(frames["num_tracks"].max()) if not frames.empty else 0
    has = not frames.empty

    cols = st.columns(4)
    cols[0].metric(
        "Speed", f"{stats['fps']} fps", border=True, height="stretch",
        chart_data=sparkline(frames["fps"]) if has else None,
        help="Average frames per second. The line shows speed over the video.",
    )
    cols[1].metric(
        "Detector used on", f"{stats['detector'] * 100:.0f}% of frames", border=True,
        height="stretch", chart_type="area",
        chart_data=sparkline(frames["detector_invoked"]) if has else None,
        help=f"{summary['detector_calls']} detector calls in total. "
             "Other frames reuse the tracker's prediction.",
    )
    cols[2].metric(
        "Time in robust mode", f"{stats['robust'] * 100:.0f}%", border=True,
        height="stretch", chart_type="area",
        chart_data=sparkline(frames["mode"] == "ROBUST") if has else None,
        help=f"{stats['switches']} mode switches. The area shows when ROBUST was running.",
    )
    cols[3].metric(
        "Most targets at once", peak, border=True, height="stretch", chart_type="bar",
        chart_data=sparkline(frames["num_tracks"]) if has else None,
        help=f"{stats['targets']} different targets seen in total.",
    )


def tab_video(run):
    summary = run["summary"]
    left, right = st.columns(2)
    with left:
        st.caption("Original")
        if Path(run["input"]).is_file():
            st.video(run["input"], alt="Original flight video")
        else:
            st.caption(":material/videocam_off: Original file is no longer on disk.")
    with right:
        st.caption(f"Tracked · {run['mode'].title()} mode")
        if Path(summary["output_video"]).is_file():
            st.video(summary["output_video"], alt="Flight video with tracked targets drawn on it")
        else:
            st.caption(":material/videocam_off: Tracked video was not written.")

    with st.container(horizontal=True):
        for key, label, mime in [
            ("output_video", "Tracked video", "video/mp4"),
            ("tracks_csv", "Target positions (CSV)", "text/csv"),
            ("metrics_csv", "Per-frame data (CSV)", "text/csv"),
        ]:
            path = Path(summary.get(key, ""))
            if path.is_file():
                # A callable is only read when clicked, so big videos don't slow every rerun.
                st.download_button(label, data=path.read_bytes if key == "output_video"
                                   else path.read_bytes(),
                                   file_name=path.name, mime=mime, icon=":material/download:",
                                   on_click="ignore", key=f"dl_{key}")


def tab_targets(run):
    tracks_df = run["summary"].get("tracks_df", pd.DataFrame())
    if tracks_df.empty:
        st.info("Try lowering the detection confidence in the sidebar.",
                title="No objects were tracked in this video", icon=":material/search_off:")
        return

    ids = sorted(tracks_df["track_id"].unique())
    labels = [f"#{i}" for i in ids]
    st.markdown(f"**{len(ids)}** target(s) followed across **{len(tracks_df)}** observations.")

    picker = st.pills if len(ids) <= 15 else st.multiselect
    choice = picker("Show targets", labels, default=labels[:6], key=f"targets_{run['label']}",
                    **({"selection_mode": "multi"} if picker is st.pills else {}))
    shown = tracks_df[tracks_df["track_id"].isin([int(c[1:]) for c in choice or []])].copy()
    shown["target"] = "#" + shown["track_id"].astype(str)

    if shown.empty:
        st.caption("Pick at least one target to draw its flight path.")
    else:
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
    px = st.column_config.NumberColumn(format="%.0f px")
    st.dataframe(
        last[["track_id", "frame", "center_x", "center_y", "bbox_width", "bbox_height", "confidence"]],
        hide_index=True, alt="Last known position of each target",
        column_config={
            "track_id": st.column_config.NumberColumn("Target", format="#%d"),
            "frame": "Last seen (frame)",
            "center_x": px, "center_y": px, "bbox_width": px, "bbox_height": px,
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0,
                                                          max_value=1.0, format="%.2f"),
        },
    )
    with st.expander("Every observation", icon=":material/table_rows:"):
        if shown.empty:
            st.caption("No targets selected.")
        else:
            st.dataframe(shown.drop(columns="target"), hide_index=True,
                         alt="All observations of the selected targets")


def tab_conditions(run):
    frames_df = run["summary"]["records_df"]
    if frames_df.empty:
        st.info("No per-frame data available.")
        return

    stats = run_stats(run)
    st.markdown("**Which mode was running**")
    st.caption(f"{stats['switches']} switches · ROBUST for {stats['robust'] * 100:.0f}% of frames")
    modes = frames_df.assign(run=lambda d: (d["mode"] != d["mode"].shift()).cumsum())
    blocks = modes.groupby("run").agg(start=("frame", "min"), end=("frame", "max"),
                                      mode=("mode", "first"))
    blocks["end"] += 1  # a block covers its last frame too, so 1-frame blocks stay visible
    timeline = (
        alt.Chart(blocks.reset_index())
        .mark_bar(height=26)
        .encode(
            x=alt.X("start:Q", title="frame"), x2="end:Q",
            color=alt.Color("mode:N", title=None,
                            scale=alt.Scale(domain=["FAST", "ROBUST"],
                                            range=[MODE_COLORS["FAST"], MODE_COLORS["ROBUST"]])),
            tooltip=["mode", "start", "end"],
        )
        .properties(height=60)
    )
    st.altair_chart(timeline, alt="Timeline showing which mode was active over the video")

    st.markdown("**What the scene looked like**")
    st.caption("Each line is scaled 0 to 1 so they fit on one chart.")
    names = {"blur": "Sharpness", "camera_motion": "Camera motion",
             "mean_confidence": "Detection confidence"}
    cols = [c for c in names if c in frames_df]
    if cols:
        scaled = frames_df[cols] / frames_df[cols].max().clip(lower=1e-9)
        scaled.index = frames_df["frame"]
        st.line_chart(scaled.rename(columns=names), height=280, x_label="frame",
                      alt="Scene sharpness, camera motion and detection confidence over the video")

    why_it_switched(run)
    with st.expander("Per-frame table", icon=":material/table_rows:"):
        st.dataframe(frames_df, hide_index=True, alt="Per-frame telemetry")


def why_it_switched(run):
    """For runs decided by a trained model: its per-frame confidence and what it relies on."""
    kind = run["summary"].get("selector")
    if kind not in ("tree", "forest"):
        return
    frames_df = run["summary"]["records_df"]
    st.markdown(f"**Why it switched · {SELECTOR_LABELS[kind]}**")
    left, right = st.columns([3, 2], gap="large")
    with left:
        if "robust_prob" in frames_df:
            thr = run.get("threshold") or 0.5
            st.caption("The model's estimate, frame by frame, that FAST would miss targets. "
                       f"At {thr:.2f} or above it asks for ROBUST; the smoother then waits a "
                       "few frames before actually switching.")
            prob = frames_df.set_index("frame")[["robust_prob"]].rename(
                columns={"robust_prob": "Chance ROBUST is needed"})
            st.area_chart(prob, height=220, x_label="frame", y_label="probability",
                          color=MODE_COLORS["ROBUST"],
                          alt="Model's probability that ROBUST is needed, per frame")
    with right:
        path = selector_model_path(kind, app_config())
        if not path.is_file():
            st.caption("The model file has been removed since this run.")
            return
        report = selector_report(kind, path.stat().st_mtime)
        importance = pd.DataFrame(list(report.get("feature_importance", {}).items())[:6],
                                  columns=["feature", "importance"])
        if not importance.empty:
            st.caption("What the model relies on most (feature importance from training).")
            st.altair_chart(
                alt.Chart(importance).mark_bar(color=MODE_COLORS["ADAPTIVE"]).encode(
                    x=alt.X("importance:Q", title=None),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".2f")],
                ).properties(height=alt.Step(26)),
                alt="Feature importance of the trained selector",
            )
        cv = report.get("cv") or {}
        if cv:
            st.caption(f"Cross-validated accuracy {cv['accuracy'] * 100:.0f}% · "
                       f"balanced {cv['balanced_accuracy'] * 100:.0f}% (tested on clips it "
                       "was not trained on).")


def tab_compare(runs):
    if len(runs) < 2:
        st.info("Run the same video again in another mode (for example Fast, then Robust, "
                "then Adaptive) and the runs will line up here.",
                title="Nothing to compare yet", icon=":material/compare_arrows:")
        return

    table = pd.DataFrame([run_stats(r) for r in runs.values()])
    st.markdown("**Speed of each run**")
    bars = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("fps:Q", title="average fps"),
            y=alt.Y("run:N", title=None, sort="-x"),
            color=alt.Color("mode:N", title="Mode",
                            scale=alt.Scale(domain=list(MODE_COLORS),
                                            range=list(MODE_COLORS.values()))),
            tooltip=["run", "fps", "seconds", "detector"],
        )
        .properties(height=alt.Step(34))
    )
    st.altair_chart(bars, alt="Average speed of every saved run, colored by mode")

    st.dataframe(
        table, hide_index=True, alt="Side-by-side numbers for every saved run",
        column_config={
            "run": st.column_config.TextColumn("Run", width="large"),
            "mode": "Mode", "decided_by": "Decided by", "frames": "Frames",
            "seconds": st.column_config.NumberColumn("Time", format="%.1f s"),
            "fps": st.column_config.NumberColumn("Speed", format="%.1f fps"),
            "detector": st.column_config.ProgressColumn("Detector used", min_value=0.0,
                                                        max_value=1.0, format="percent"),
            "robust": st.column_config.ProgressColumn("Time in robust", min_value=0.0,
                                                      max_value=1.0, format="percent"),
            "switches": "Mode switches", "targets": "Targets",
        },
    )
    st.caption("Fewer detector calls and less time in ROBUST means less compute. "
               "Compare the tracked videos to check nothing important was missed.")


def show_results(runs, settings):
    labels = list(runs)
    if st.session_state.get("active_run") not in runs:
        st.session_state["active_run"] = labels[-1]

    if len(labels) > 1:
        st.selectbox("Showing run", labels, key="active_run")
    run = runs[st.session_state["active_run"]]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(run["video_name"], anchor=False, width="content")
        st.badge(run["mode"].title(), color=BADGE_COLORS[run["mode"]])
        if run["summary"].get("selector"):
            st.badge(f"Decided by {SELECTOR_LABELS[run['summary']['selector']].lower()}",
                     icon=":material/psychology:", color="violet")
        st.badge(f"{run['summary']['total_frames']} frames", icon=":material/movie:", color="gray")
        st.badge(f"{run['summary']['total_time_seconds']} s", icon=":material/timer:", color="gray")

    current = run_label(settings) if settings["video_name"] else None
    if current and current not in runs:
        st.caption(":material/info: Your current settings don't match a saved run yet. "
                   "Press **Run tracking** to process them.")

    stat_row(run)

    tabs = st.tabs(RESULT_TABS, key="results_tab", on_change="rerun")
    # Only the open tab is drawn, so charts in hidden tabs cost nothing.
    for tab, draw, arg in zip(tabs, [tab_video, tab_targets, tab_conditions, tab_compare],
                              [run, run, run, runs]):
        if tab.open:
            with tab:
                draw(arg)


def show_empty_state():
    a, b, c = st.columns(3, border=True)
    a.markdown(":material/upload: **Upload**  \nDrop a drone flight video in the box above.")
    b.markdown(":material/alt_route: **Choose a mode**  \n"
               "Let *Adaptive* decide, or force *Fast* or *Robust*.")
    c.markdown(":material/compare_arrows: **Run and compare**  \n"
               "Run one mode, or press *Compare all modes* to see them side by side.")
    st.caption("FAST is quick but coarse. ROBUST is careful but slow. ADAPTIVE lets a trained "
               "model read blur, camera motion, target size and confidence to pick between them.")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
conf, max_frames, live, history_slot = sidebar_settings()

st.badge("Air-to-air tracking", icon=":material/flight:", color="primary")
st.title("Track targets in your flight video", anchor=False)
st.caption("A small fast model handles easy stretches and a bigger careful one takes over "
           "when the scene gets hard, so compute is spent only where it is needed.")

video, mode, selector, threshold, run_one, run_all = setup_panel()
settings = {"video_name": video.name if video else None, "mode": mode, "selector": selector,
            "threshold": threshold, "conf": conf, "max_frames": max_frames}

if video is not None and run_one:
    run_pipeline(video, settings, live)
elif video is not None and run_all:
    plan = [("FAST", None), ("ROBUST", None)]
    plan += [("ADAPTIVE", kind) for kind in reversed(trained_selectors())]  # rules first, then models
    for each_mode, each_selector in plan:
        run_pipeline(video, {**settings, "mode": each_mode, "selector": each_selector,
                             "threshold": threshold if each_selector in ("tree", "forest") else None},
                     live)
    best = trained_selectors()[0]
    st.session_state["active_run"] = run_label({
        **settings, "mode": "ADAPTIVE", "selector": best,
        "threshold": threshold if best in ("tree", "forest") else None})
    st.session_state["results_tab"] = RESULT_TABS[-1]  # open the comparison

if st.session_state["runs"]:
    show_results(st.session_state["runs"], settings)
else:
    show_empty_state()

draw_history(history_slot)  # last, so the sidebar list shows which run is active
