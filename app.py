"""
app.py — Conv Efficiency Simulator  (modular edition)
------------------------------------------------------
University of Malta · Deep Learning Assignment
CPU-only: all metrics computed analytically; optional PyTorch CPU sandbox.

Entry-point responsibilities (and nothing else)
-----------------------------------------------
1. st.set_page_config  — must be the very first Streamlit call
2. inject_css          — global adaptive theme (Light + Dark compatible)
3. configure_matplotlib — transparent, theme-neutral figure defaults
4. init_session_state  — seed slider/input defaults once per browser session
5. render_sidebar      — Apple-style controls; returns canonical params dict
6. compute_all         — derive every metric from params (pure NumPy)
7. Hero header         — gradient title + six-metric summary row
8. Three-tab dashboard — delegate entirely to ui.dashboards view functions

Module map
----------
config.styles                  CSS injection + Matplotlib rcParams
core.math_engine               Pure analytical math (Std, DWS, MBv2)
core.torch_sandbox             Live PyTorch CPU validation (Tab 3)
ui.sidebar                     Two-way synced hyperparameter controls
ui.dashboards                  Tab 1 / Tab 2 / Tab 3 view renderers
visualization.static_plots     Matplotlib figure factories
visualization.animation_engine HTML5 Canvas kernel-sweep animation
"""

import streamlit as st

# ── Page config must come first ───────────────────────────────────────────────
st.set_page_config(
    page_title="Conv Efficiency Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports (after page config) ───────────────────────────────────────
from config.styles    import inject_css, configure_matplotlib
from core.math_engine import compute_all, fmt
from ui.sidebar       import init_session_state, render_sidebar
from ui.dashboards    import render_tab1, render_tab2, render_tab3

# ── Global setup ─────────────────────────────────────────────────────────────
inject_css()
configure_matplotlib()
init_session_state()

# ── Sidebar → canonical params ────────────────────────────────────────────────
params = render_sidebar()

# ── Compute all metrics ───────────────────────────────────────────────────────
results = compute_all(params)

M, N, Dk = params["M"], params["N"], params["Dk"]
mbv2     = params["mbv2_mode"]
t        = params["t"]

s_p     = results["s_p"]
s_f     = results["s_f"]
cmp_p   = results["cmp_p"]
cmp_f   = results["cmp_f"]
r_th    = results["r_th"]
ratio_p = results["ratio_p"]
ratio_f = results["ratio_f"]
saved_p = results["saved_p"]

# ── Hero header ───────────────────────────────────────────────────────────────
_page_title = (
    "MobileNetV2 Inverted Residual Block"
    if mbv2 else
    "Standard vs. Depthwise Separable Convolutions"
)
_page_subtitle = (
    "Inverted Residual Bottleneck Simulator"
    if mbv2 else
    "Interactive Efficiency Simulator"
)
_cmp_p_label   = "MBv2 Params"  if mbv2 else "DWS Params"
_cmp_f_label   = "MBv2 FLOPs"   if mbv2 else "DWS FLOPs"
_ratio_help    = "t(M+Dk²+N)/(Dk²·N)"  if mbv2 else "1/N + 1/Dk²"
_savings_label = "Params Overhead" if saved_p < 0 else "Params Saved"

st.markdown(
    f'<h1 class="grad-heading" style="font-size:2.5rem; text-align:center;">'
    f"{_page_title}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="text-align:center; opacity:0.55; font-size:1rem; '
    f'margin-bottom:1.4rem;">'
    f"{_page_subtitle} &nbsp;&#183;&nbsp; University of Malta"
    f" &nbsp;&#183;&nbsp; Deep Learning</p>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Std. Params",   fmt(s_p))
m2.metric(_cmp_p_label,    fmt(cmp_p["total"]),
          delta=f"{(ratio_p - 1) * 100:+.1f}% vs Std",
          delta_color="inverse")
m3.metric("Std. FLOPs",    fmt(s_f))
m4.metric(_cmp_f_label,    fmt(cmp_f["total"]),
          delta=f"{(ratio_f - 1) * 100:+.1f}% vs Std",
          delta_color="inverse")
m5.metric("Theory Ratio",  f"{r_th:.4f}", help=_ratio_help)
m6.metric(_savings_label,  f"{abs(saved_p):.1f}%")
st.markdown("---")

# ── Three-tab dashboard ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Visual Pipeline & Benchmarks",
    "📐  Mathematical Derivations",
    "🔬  PyTorch Sandbox",
])

with tab1:
    render_tab1(params, results)

with tab2:
    render_tab2(params, results)

with tab3:
    render_tab3(params, results)
