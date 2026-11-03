"""Benchmark results and trade-off visualization page."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")

st.title("📊 Benchmark Results & Compute Trade-Offs")
st.markdown(
    "Comparative evaluation of **Always FAST**, **Always ROBUST**, and **Condition-Aware Adaptive** pipelines "
    "across tracking quality metrics (HOTA, IDF1, MOTA, IDSW) and compute resource footprint."
)

# Dummy benchmark results table
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

