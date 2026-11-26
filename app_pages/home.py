"""Landing page: a product-style home for the project.

Light on purpose: it imports no torch or YOLO, so it appears instantly. Styling is in
assets/styles.css; the hero and call-to-action are native containers (styled by key) so
their buttons stay real Streamlit buttons.
"""

import joblib
import streamlit as st

from src.selector.selector import SELECTOR_LABELS, available_selectors, selector_model_path
from src.utils.config import load_config


@st.cache_data(ttl=300, show_spinner=False)
def selector_scores() -> dict:
    """Cross-validated accuracy of each trained selector (empty if none are trained)."""
    cfg = load_config("config.yaml")
    scores = {}
    for kind in available_selectors(cfg):
        if kind == "rules":
            continue
        cv = joblib.load(selector_model_path(kind, cfg)).get("report", {}).get("cv") or {}
        if cv:
            scores[kind] = cv
    return scores


def open_tracker(label: str, key: str, kind: str = "primary"):
    if st.button(label, type=kind, icon=":material/arrow_forward:", icon_position="right", key=key):
        st.switch_page("app_pages/tracker.py")


def section(eyebrow: str, title: str, text: str) -> str:
    return (f'<div class="sec-head"><div class="eyebrow">{eyebrow}</div>'
            f'<h2>{title}</h2><p>{text}</p></div>')


def card(icon: str, title: str, text: str, tone: str = "") -> str:
    return (f'<div class="card {tone}"><div class="ico"><span class="msr">{icon}</span></div>'
            f'<h3>{title}</h3><p>{text}</p></div>')


# --- Hero ----------------------------------------------------------------------
with st.container(key="hero"):
    left, right = st.columns([11, 9], gap="large", vertical_alignment="center")
    with left:
        st.badge("Air-to-air UAV tracking", icon=":material/flight:", color="primary")
        st.title("Condition-aware tracking for drones in flight", anchor=False)
        st.html('<p class="hero-lead">Run a fast detector when the scene is easy and a robust one '
                "when it isn't. A trained selector decides frame by frame, so compute goes "
                "only where it is needed.</p>")
        with st.container(horizontal=True):
            open_tracker("Launch tracker", "cta_top")
            st.link_button("See features", "#features", type="tertiary")
    with right:
        st.html(
            '<div class="mock"><div class="bar"><i></i><i></i><i></i>'
            '<span>ADAPTIVE</span></div>'
            '<div class="box one"><b>#1 · 0.94</b></div>'
            '<div class="box two"><b>#2 · 0.88</b></div>'
            '<div class="box three"><b>#3 · 0.71</b></div></div>')

# --- Stats ---------------------------------------------------------------------
scores = selector_scores()
stats = [("3", "Operating modes"), (str(len(scores) + 1), "Selectors available")]
if scores:
    best, cv = max(scores.items(), key=lambda kv: kv[1]["balanced_accuracy"])
    stats.append((f"{cv['balanced_accuracy'] * 100:.0f}%", f"{SELECTOR_LABELS[best]} balanced accuracy"))
st.html('<div class="stats">' + "".join(f"<div class='stat'><b>{n}</b><span>{t}</span></div>"
                                        for n, t in stats) + "</div>")

# --- Features ------------------------------------------------------------------
st.html(section("Features", "Everything you need to evaluate adaptive perception",
                "From a single upload to a side-by-side comparison of every mode."))
st.html('<div class="grid">' + "".join([
    card("auto_mode", "Adaptive switching",
         "Blur, camera motion, target size and confidence drive an automatic switch between "
         "Fast and Robust."),
    card("psychology", "Learned selector",
         "Choose hand-set rules, a decision tree or a random forest to make the call.", "teal"),
    card("moving", "Multi-target tracking",
         "Persistent IDs and flight paths for every target in view.", "orange"),
    card("live_tv", "Live preview",
         "Watch tracked frames, speed and target counts update while a video processes."),
    card("compare_arrows", "Run comparison",
         "Line up every mode on the same clip and see speed and compute saved.", "teal"),
    card("download", "Exportable results",
         "Download the tracked video, target positions and per-frame data.", "orange"),
]) + "</div>")

# --- Modes ---------------------------------------------------------------------
st.html(section("Modes", "Pick your trade-off",
                "Force a mode, or let the selector adapt as the footage changes."))
st.html('<div class="grid">' + "".join([
    card("auto_mode", "Adaptive", "Switches between Fast and Robust per moment. "
         "Best balance of speed and reliability."),
    card("bolt", "Fast", "Small model, low resolution, frames skipped. Quickest.", "teal"),
    card("shield", "Robust", "Big model, high resolution, every frame. Most careful.", "orange"),
]) + "</div>")

# --- Call to action ------------------------------------------------------------
with st.container(key="cta", horizontal_alignment="center"):
    st.header("Try it on your own footage", anchor=False, text_alignment="center")
    st.markdown("Upload a flight clip and compare every mode in one run.", text_alignment="center")
    open_tracker("Launch tracker", "cta_bottom")
