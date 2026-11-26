"""Entry point: sets up the page frame and routes between the landing page and the tracker.

Run with:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="UAV Tracker", page_icon=":material/flight:", layout="wide")

# Custom CSS lives in assets/styles.css; st.html renders it into the page for every view.
_css = Path(__file__).parent / "assets" / "styles.css"
if _css.is_file():
    st.html(f"<style>{_css.read_text(encoding='utf-8')}</style>")

home = st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True)
tracker = st.Page("app_pages/tracker.py", title="Tracker", icon=":material/flight:")

st.navigation([home, tracker], position="top").run()

# Footer, shown under every page.
st.html("""
<footer class="site-footer">
  <div class="cols">
    <div>
      <div class="brand"><span class="logo"><span class="msr" style="font-size:20px">flight</span></span>UAV Tracker</div>
      <p>Condition-aware adaptive perception for air-to-air UAV tracking.</p>
    </div>
    <div><h4>Product</h4><ul>
      <li><a href="/" target="_self">Home</a></li>
      <li><a href="/tracker" target="_self">Tracker</a></li></ul></div>
    <div><h4>Modes</h4><ul><li>Adaptive</li><li>Fast</li><li>Robust</li></ul></div>
    <div><h4>Built with</h4><ul><li>Streamlit</li><li>PyTorch</li>
      <li>Ultralytics YOLO</li><li>OpenCV</li></ul></div>
  </div>
  <div class="legal"><span>&copy; 2026 UAV Tracker. All rights reserved.</span>
    <span>Condition-Aware Adaptive Perception</span></div>
</footer>
""")
