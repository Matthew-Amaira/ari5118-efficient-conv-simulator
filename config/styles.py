"""
config/styles.py
----------------
Centralised design-system for the Conv Efficiency Simulator.

Responsibilities
----------------
* inject_css()          — writes the full adaptive-theme CSS block into the
                          Streamlit page via st.markdown.
* configure_matplotlib() — applies transparent, theme-neutral rcParams to every
                           Matplotlib figure produced in this session.

Design Tokens
-------------
Primary accent  : #0ea5e9  (sky-500)
Secondary accent: #818cf8  (indigo-400)
Success         : #22c55e  (green-500)
Warning         : #f59e0b  (amber-500)
Danger/pink     : #db2777  (pink-600)
Chart text      : #555759  (mid-slate — WCAG AA on white and dark navy)
"""

import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Design tokens exposed to the rest of the package
# ---------------------------------------------------------------------------
CHART_TEXT = "#555759"   # mid-slate: readable on white AND dark backgrounds
CHART_GRID = "#94a3b8"   # soft blue-grey grid lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject the full adaptive-theme CSS block into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def configure_matplotlib() -> None:
    """Apply transparent, theme-neutral rcParams globally for this session."""
    plt.rcParams.update({
        "figure.facecolor":  (1, 1, 1, 0),   # fully transparent RGBA
        "axes.facecolor":    (1, 1, 1, 0),
        "axes.edgecolor":    CHART_GRID,
        "grid.color":        CHART_GRID,
        "grid.alpha":        0.30,
        "text.color":        CHART_TEXT,
        "axes.labelcolor":   CHART_TEXT,
        "xtick.color":       CHART_TEXT,
        "ytick.color":       CHART_TEXT,
        "axes.titlecolor":   CHART_TEXT,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
    })


# ---------------------------------------------------------------------------
# CSS source — uses Streamlit CSS custom properties so Light/Dark flip works
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ── honour Streamlit's own theme on the root canvas ── */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* ── sidebar inherits the secondary surface colour ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: var(--secondary-background-color);
    border-right: 1px solid rgba(128, 128, 128, 0.2);
}

/* ── metric cards: semi-translucent surface + soft border ── */
[data-testid="metric-container"] {
    background: rgba(128, 128, 128, 0.08) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    opacity: 0.65;
}

/* ── tab bar ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--secondary-background-color) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-color) !important;
    opacity: 0.55;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--background-color) !important;
    color: #38bdf8 !important;
    opacity: 1 !important;
    border-bottom: none !important;
}
[data-testid="stTabPanel"] {
    padding-top: 1.4rem !important;
}

/* ── tab icon glows — distinct neon accent per tab ── */
[data-testid="stTabs"] [data-baseweb="tab"]:nth-child(1) [data-testid="stIcon"] {
    font-size: 1.3rem;
    filter: drop-shadow(0 0 7px #0ea5e9);
}
[data-testid="stTabs"] [data-baseweb="tab"]:nth-child(2) [data-testid="stIcon"] {
    font-size: 1.3rem;
    filter: drop-shadow(0 0 7px #818cf8);
}
[data-testid="stTabs"] [data-baseweb="tab"]:nth-child(3) [data-testid="stIcon"] {
    font-size: 1.3rem;
    filter: drop-shadow(0 0 7px #22c55e);
}
/* Brighten the active tab's icon further */
[data-testid="stTabs"] [aria-selected="true"]:nth-child(1) [data-testid="stIcon"] {
    filter: drop-shadow(0 0 10px #0ea5e9) brightness(1.15);
}
[data-testid="stTabs"] [aria-selected="true"]:nth-child(2) [data-testid="stIcon"] {
    filter: drop-shadow(0 0 10px #818cf8) brightness(1.15);
}
[data-testid="stTabs"] [aria-selected="true"]:nth-child(3) [data-testid="stIcon"] {
    filter: drop-shadow(0 0 10px #22c55e) brightness(1.15);
}

/* ── tables ── */
.stTable table {
    background: rgba(128, 128, 128, 0.05) !important;
    color: var(--text-color) !important;
}
.stTable th {
    background: rgba(128, 128, 128, 0.12) !important;
    color: #38bdf8 !important;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25) !important;
}
.stTable td {
    border-bottom: 1px solid rgba(128, 128, 128, 0.12) !important;
}

/* ── dividers ── */
hr { border-color: rgba(128, 128, 128, 0.2) !important; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── progress bar fill ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0ea5e9, #818cf8) !important;
}

/* ── gradient heading ── */
.grad-heading {
    font-weight: 800;
    background: linear-gradient(90deg, #0ea5e9 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
}

/* ── section chip: sky-blue pill ── */
.section-chip {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0ea5e9;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 20px;
    padding: 2px 10px;
    margin-bottom: 0.5rem;
}

/* ── formula card ── */
.formula-card {
    background: rgba(128, 128, 128, 0.08);
    border-left: 3px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.92rem;
}

/* ── insight callout ── */
.insight-box {
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
    font-size: 0.93rem;
}

/* ── sidebar unified control tiles — Apple-style alignment ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
}
[data-testid="stSidebar"] .stSlider {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    margin-bottom: 0px !important;
}
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
    height: 2.1rem !important;
    font-family: monospace !important;
    font-weight: 600 !important;
    text-align: center !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] .stNumberInput [data-testid="stWidgetLabel"] {
    display: none !important;
}
</style>
"""
