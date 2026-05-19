"""
Interactive Education Simulator -- Adaptive Theme Edition
Depthwise Separable Convolutions vs. Standard Convolutions
University of Malta - Deep Learning Assignment
CPU-only: all metrics computed analytically; optional PyTorch CPU sandbox.
Fully compatible with Streamlit Light Mode and Dark Mode.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------------------------
# 1.  PAGE CONFIG  (must be first Streamlit call)
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Conv Efficiency Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2.  ADAPTIVE CSS  — uses Streamlit theme variables so Light & Dark both work
# ------------------------------------------------------------------------------
# Design principles applied here:
#   • var(--background-color) / var(--secondary-background-color) / var(--text-color)
#     are injected by Streamlit into :root and adapt automatically to the active theme.
#   • Custom surfaces (cards, chips, insight boxes) use rgba(128,128,128, α) overlays
#     so they read as a subtle tint on any base colour — white or navy alike.
#   • No !important overrides on generic text tags (p / span / div) — those were the
#     primary cause of Light Mode breakage.
#   • Accent colours (sky-blue #38bdf8, indigo #818cf8) are saturated enough to remain
#     legible on both white and dark-slate backgrounds without adjustment.

st.markdown("""
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
/* keep value prominent; inherit theme colour so it flips automatically */
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    opacity: 0.65;               /* readable tint on any background */
}

/* ── tab bar: secondary surface background so it floats above page ── */
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

/* ── tables: transparent surface, theme-driven text ── */
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

/* ── buttons: gradient accent unchanged, white label always readable on gradient ── */
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

/* ── gradient heading: works on any background colour ── */
.grad-heading {
    font-weight: 800;
    background: linear-gradient(90deg, #0ea5e9 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
}

/* ── section chip: translucent sky-blue pill ── */
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

/* ── formula card: thin left accent, translucent fill ── */
.formula-card {
    background: rgba(128, 128, 128, 0.08);
    border-left: 3px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.92rem;
}

/* ── insight callout: indigo left accent, translucent fill ── */
.insight-box {
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
    font-size: 0.93rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3.  MATPLOTLIB ADAPTIVE DEFAULTS
# ------------------------------------------------------------------------------
# Strategy: make every figure and axes background fully transparent so the plot
# floats naturally on whatever surface Streamlit provides (white in Light Mode,
# dark slate in Dark Mode).  All text is set to a single mid-tone slate (#555759)
# that achieves ≥ 4.5:1 WCAG contrast on a white canvas and ≥ 3.5:1 on a dark
# navy canvas — acceptable for chart annotation text at the rendered sizes.
# Accent colours on the bars and diagram boxes remain fully saturated, so they
# pop on both surfaces without any adjustment.

_CHART_TEXT  = "#555759"   # mid-slate: readable on white AND dark backgrounds
_CHART_GRID  = "#94a3b8"   # soft blue-grey grid lines

plt.rcParams.update({
    # Transparent backgrounds — figure floats on the Streamlit container colour.
    "figure.facecolor":  (1, 1, 1, 0),   # fully transparent (RGBA tuple)
    "axes.facecolor":    (1, 1, 1, 0),   # fully transparent

    # Spine and grid in neutral tones.
    "axes.edgecolor":    _CHART_GRID,
    "grid.color":        _CHART_GRID,
    "grid.alpha":        0.30,

    # All text rendered at the neutral mid-tone defined above.
    "text.color":        _CHART_TEXT,
    "axes.labelcolor":   _CHART_TEXT,
    "xtick.color":       _CHART_TEXT,
    "ytick.color":       _CHART_TEXT,
    "axes.titlecolor":   _CHART_TEXT,

    # Hide top/right spines for a clean, modern look.
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
})

# ------------------------------------------------------------------------------
# 4.  PURE-NUMPY COMPUTATION LAYER  (CPU-only, no torch required here)
# ------------------------------------------------------------------------------

def std_params(M: int, N: int, Dk: int) -> int:
    """Standard conv parameters: Dk^2 * M * N (no bias)."""
    return int(Dk * Dk * M * N)


def std_flops(M: int, N: int, Dk: int, H: int, W: int) -> int:
    """Standard conv FLOPs: 2 * Dk^2 * M * N * H * W (same padding)."""
    return int(2 * Dk * Dk * M * N * H * W)


def dws_params(M: int, N: int, Dk: int) -> dict:
    """DWS parameters: depthwise (Dk^2 * M) + pointwise (M * N)."""
    dw = int(Dk * Dk * M)
    pw = int(M * N)
    return {"dw": dw, "pw": pw, "total": dw + pw}


def dws_flops(M: int, N: int, Dk: int, H: int, W: int) -> dict:
    """DWS FLOPs split by stage."""
    dw = int(2 * Dk * Dk * M * H * W)
    pw = int(2 * M * N * H * W)
    return {"dw": dw, "pw": pw, "total": dw + pw}


def reduction_ratio(N: int, Dk: int) -> float:
    """Closed-form cost ratio: 1/N + 1/Dk^2  (Howard et al. 2017)."""
    return (1.0 / N) + (1.0 / (Dk ** 2))


# ── MobileNetV2 Inverted Residual Bottleneck ──────────────────────────────────
# The block consists of three sequential layers:
#   1. Expand   : 1×1 PW conv, M  → t·M   (widens the representation)
#   2. Depthwise: Dk×Dk DW conv, t·M      (spatial filtering in high-dim space)
#   3. Project  : 1×1 PW conv, t·M → N   (compresses back to target width)
# A residual skip is added when M == N (stride-1 blocks only).

def mbv2_params(M: int, N: int, Dk: int, t: int) -> dict:
    """
    Parameter counts for a MobileNetV2 Inverted Residual Bottleneck block.

    Stage 1 — Expand  : Params = M  × (t·M)  = t·M²
    Stage 2 — DW      : Params = Dk²× (t·M)
    Stage 3 — Project : Params = (t·M) × N   = t·M·N
    Total             : t·M·(M + Dk² + N)
    """
    tM      = t * M
    expand  = M  * tM          # 1×1: in_ch × out_ch
    dw      = Dk * Dk * tM     # DW : Dk² × ch
    project = tM * N           # 1×1: in_ch × out_ch
    return {"expand": expand, "dw": dw, "project": project, "total": expand + dw + project}


def mbv2_flops(M: int, N: int, Dk: int, H: int, W: int, t: int) -> dict:
    """
    FLOPs (2 × MACs) for a MobileNetV2 Inverted Residual Bottleneck block.

    Stage 1 — Expand  : 2·M·(t·M)·H·W
    Stage 2 — DW      : 2·Dk²·(t·M)·H·W
    Stage 3 — Project : 2·(t·M)·N·H·W
    Total             : 2·t·M·H·W·(M + Dk² + N)
    """
    tM      = t * M
    expand  = int(2 * M  * tM * H * W)
    dw      = int(2 * Dk * Dk * tM * H * W)
    project = int(2 * tM * N  * H * W)
    return {"expand": expand, "dw": dw, "project": project, "total": expand + dw + project}


def mbv2_reduction_ratio(M: int, N: int, Dk: int, t: int) -> float:
    """
    Cost ratio of MBv2 block vs an equivalent standard conv (same I/O shape).

    Derivation:
        Std  params = Dk² · M · N
        MBv2 params = t · M · (M + Dk² + N)
        Ratio       = t · (M + Dk² + N) / (Dk² · N)

    Values > 1 indicate MBv2 costs MORE parameters than a plain Dk×Dk conv;
    the advantage of MBv2 is architectural quality (residual flow, high-dim
    feature space for DW), not raw parameter compression.
    """
    return (t * (M + Dk ** 2 + N)) / (Dk ** 2 * N)


def fmt(n: int) -> str:
    """Human-readable magnitude string (K / M / B)."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.3f} M"
    if n >= 1_000:
        return f"{n / 1e3:.2f} K"
    return str(n)


# ------------------------------------------------------------------------------
# 5.  THEME-ADAPTIVE VISUALISATION HELPERS
# ------------------------------------------------------------------------------
# All figures use transparent backgrounds (set via rcParams above).
# Diagram box fills use light pastels — clearly visible on white and still
# readable against dark backgrounds because the saturated edge colour provides
# the perceptual boundary.  All annotation text is the neutral _CHART_TEXT grey.
# Badge overlays use RGBA tuples with low alpha so they work as tinted glass on
# any background colour.

# Shared neutral colour for all text annotations in plots.
_T = _CHART_TEXT          # mid-slate, sufficient contrast on light & dark
_ARROW = "#64748b"        # slightly darker than grid, visible on both themes


def _patch(ax, x, y, w, h, face, edge, lw=2.0, alpha=1.0, zorder=3):
    """Add a single rounded rectangle to an axis."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10",
        facecolor=face, edgecolor=edge,
        linewidth=lw, alpha=alpha, zorder=zorder,
    ))


def _stacked(ax, x, y, w, h, n, face, edge, label, sublabel):
    """Draw translucent stacked rectangles to suggest channel depth."""
    layers = min(n, 5)
    for i in range(layers):
        off = i * 0.08
        _patch(ax, x + off, y + off, w, h, face=face, edge=edge,
               lw=1.0, alpha=max(0.25, 0.85 - i * 0.13), zorder=2 + i)
    cx  = x + layers * 0.08 / 2 + w / 2
    top = y + layers * 0.08 + h
    # Label: use saturated edge colour so it pops against the box face.
    ax.text(cx, top + 0.28, label,    ha="center", va="bottom",
            fontsize=9,   fontweight="bold", color=edge, zorder=10)
    # Sublabel: neutral mid-tone — readable on both light and dark surfaces.
    ax.text(cx, top + 0.02, sublabel, ha="center", va="bottom",
            fontsize=7.5, color=_T, zorder=10)


def _kbox(ax, x, y, w, h, face, edge, top_txt, bot_txt):
    """Draw a single kernel block with two descriptor lines."""
    _patch(ax, x, y, w, h, face=face, edge=edge, lw=2.4)
    cy = y + h / 2
    ax.text(x + w / 2, cy + 0.25, top_txt,
            ha="center", fontsize=10, fontweight="bold", color=edge, zorder=6)
    ax.text(x + w / 2, cy - 0.30, bot_txt,
            ha="center", fontsize=7.5, color=_T, zorder=6)


def _arr(ax, x0, x1, y):
    """Draw a horizontal annotation arrow between two x positions."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=_ARROW, lw=1.8),
                zorder=8)


def draw_std_diagram(M: int, N: int, Dk: int, H: int, W: int) -> plt.Figure:
    """Block diagram for a standard 2-D convolution — theme-adaptive."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.axis("off")
    # Title inherits _CHART_TEXT from rcParams — no hard-coded colour here.
    ax.set_title("Standard Convolution  (1 operation)",
                 fontsize=13, fontweight="bold", pad=14)

    # Light pastel fills: clearly visible on white; edge colour carries contrast on dark.
    _stacked(ax, 0.3, 1.6, 1.7, 3.8, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}x{W}x{M}")
    _arr(ax, 2.15, 3.1, 3.5)
    _kbox(ax, 3.2, 2.4, 2.5, 2.2,
          "#fef3c7", "#d97706",          # amber pastel fill, dark amber edge
          f"{Dk}x{Dk}x{M}x{N}",
          f"Kernel - {fmt(Dk * Dk * M * N)} params")
    _arr(ax, 5.8, 6.7, 3.5)
    _stacked(ax, 6.8, 1.6, 1.7, 3.8, N,
             "#dcfce7", "#16a34a", "Output", f"{H}x{W}x{N}")

    # Badge: semi-transparent indigo tint — works as tinted glass on any bg.
    ax.text(5.0, 0.5, "SINGLE-PASS  -  ALL CHANNELS FUSED",
            ha="center", fontsize=9, fontweight="bold", color="#4f46e5",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.38, 0.34, 0.93, 0.12),   # indigo @ 12 % alpha
                      edgecolor="#6366f1", lw=1.4))
    plt.tight_layout()
    return fig


def draw_dws_diagram(M: int, N: int, Dk: int, H: int, W: int) -> plt.Figure:
    """Block diagram for depthwise separable convolution — theme-adaptive."""
    dp = dws_params(M, N, Dk)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.set_title("Depthwise Separable Convolution  (2 operations)",
                 fontsize=13, fontweight="bold", pad=14)

    _stacked(ax, 0.1, 1.6, 1.3, 3.8, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}x{W}x{M}")
    _arr(ax, 1.55, 2.35, 3.5)

    _kbox(ax, 2.45, 2.5, 1.9, 2.0,
          "#fce7f3", "#db2777",          # rose pastel fill, dark rose edge
          f"{Dk}x{Dk} per ch.",
          f"DW - {fmt(dp['dw'])} params")
    _arr(ax, 4.45, 5.05, 3.5)

    _stacked(ax, 5.1, 1.8, 1.1, 3.3, M,
             "#ede9fe", "#7c3aed", "Interm.", f"{H}x{W}x{M}")
    _arr(ax, 6.3, 6.85, 3.5)

    _kbox(ax, 6.9, 2.5, 1.7, 2.0,
          "#d1fae5", "#059669",          # emerald pastel fill, dark green edge
          "1x1  chan. mix",
          f"PW - {fmt(dp['pw'])} params")
    _arr(ax, 8.7, 9.3, 3.5)

    _stacked(ax, 9.35, 1.6, 0.45, 3.8, N,
             "#dcfce7", "#16a34a", "Output", f"{H}x{W}x{N}")

    # Badge: semi-transparent rose tint.
    ax.text(5.0, 0.5, "SPATIAL FILTERING  then  CHANNEL MIXING  (FACTORISED)",
            ha="center", fontsize=8.5, fontweight="bold", color="#be185d",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.86, 0.15, 0.47, 0.10),   # rose @ 10 % alpha
                      edgecolor="#db2777", lw=1.4))
    plt.tight_layout()
    return fig


def draw_bar_chart(sp: int, dp: int, sf: int, df: int,
                   cmp_label: str = "DW Separable\nConv") -> plt.Figure:
    """Side-by-side performance bar chart — theme-adaptive.

    cmp_label: x-axis label for the second (comparison) bar.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    cats   = ["Standard\nConv", cmp_label]
    c_pair = [("#6366f1", "#22c55e"), ("#f59e0b", "#ec4899")]

    for ax, vals, colors, title, unit in [
        (ax1, [sp, dp], c_pair[0], "Parameter Count",    "Params"),
        (ax2, [sf, df], c_pair[1], "FLOPs  (fwd. pass)", "FLOPs"),
    ]:
        bars = ax.bar(cats, vals, color=colors, width=0.42,
                      edgecolor="none",      # no border — avoids dark/light clash
                      linewidth=0, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", zorder=0)
        ax.tick_params(axis="both", labelsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.035,
                    fmt(v), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=_T)
    plt.tight_layout(pad=2.4)
    return fig


def draw_ratio_breakdown(M: int, N: int, Dk: int, H: int, W: int, s_f: int) -> plt.Figure:
    """Horizontal bar: per-stage DWS FLOPs as % of standard conv — theme-adaptive."""
    df     = dws_flops(M, N, Dk, H, W)
    stages = ["Depthwise\n(spatial)", "Pointwise\n(channel)"]
    fracs  = [df["dw"] / s_f * 100, df["pw"] / s_f * 100]
    colors = ["#db2777", "#059669"]

    fig, ax = plt.subplots(figsize=(7, 1.9))
    bars = ax.barh(stages, fracs, color=colors,
                   edgecolor="none", linewidth=0, height=0.5, zorder=3)
    ax.set_xlim(0, max(fracs) * 1.28)
    ax.set_xlabel("% of Standard Conv FLOPs", fontsize=9)
    ax.set_title("DWS Stage-wise FLOPs as % of Standard Conv",
                 fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="x", zorder=0)
    ax.tick_params(labelsize=8)
    for b, v in zip(bars, fracs):
        ax.text(v + max(fracs) * 0.02,
                b.get_y() + b.get_height() / 2,
                f"{v:.2f}%", va="center",
                fontsize=9, fontweight="bold", color=_T)
    plt.tight_layout()
    return fig


def draw_mbv2_diagram(M: int, N: int, Dk: int, H: int, W: int, t: int) -> plt.Figure:
    """
    Block diagram for a MobileNetV2 Inverted Residual Bottleneck — theme-adaptive.

    Shows the three sequential layers (Expand → Depthwise → Project) plus a
    dashed residual skip connection when M == N (stride-1 condition).
    """
    tM = t * M
    mp = mbv2_params(M, N, Dk, t)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.8)
    ax.axis("off")
    ax.set_title(f"MobileNetV2 Inverted Residual Block  (t = {t})",
                 fontsize=13, fontweight="bold", pad=14)

    # ── Input tensor stack ────────────────────────────────────────────────────
    _stacked(ax, 0.1, 2.0, 1.1, 3.5, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}x{W}x{M}")
    _arr(ax, 1.35, 1.95, 3.75)

    # ── Stage 1: Expand (1×1 PW, M → tM) ─────────────────────────────────────
    _kbox(ax, 2.0, 2.5, 1.7, 2.5,
          "#fef9c3", "#ca8a04",
          f"1×1  M→{tM}",
          f"Expand · {fmt(mp['expand'])} params")
    ax.text(2.85, 5.3, "① Expand", ha="center", fontsize=8,
            fontweight="bold", color="#ca8a04")
    _arr(ax, 3.75, 4.25, 3.75)

    # ── Stage 2: Depthwise (Dk×Dk, tM channels) ──────────────────────────────
    _kbox(ax, 4.3, 2.5, 1.7, 2.5,
          "#fce7f3", "#db2777",
          f"{Dk}×{Dk}  ch={tM}",
          f"DW · {fmt(mp['dw'])} params")
    ax.text(5.15, 5.3, "② Depthwise", ha="center", fontsize=8,
            fontweight="bold", color="#db2777")
    _arr(ax, 6.05, 6.55, 3.75)

    # ── Stage 3: Project (1×1 PW, tM → N) ────────────────────────────────────
    _kbox(ax, 6.6, 2.5, 1.7, 2.5,
          "#d1fae5", "#059669",
          f"1×1  {tM}→N",
          f"Project · {fmt(mp['project'])} params")
    ax.text(7.45, 5.3, "③ Project", ha="center", fontsize=8,
            fontweight="bold", color="#059669")
    _arr(ax, 8.35, 8.85, 3.75)

    # ── Output tensor stack ───────────────────────────────────────────────────
    _stacked(ax, 8.9, 2.0, 0.75, 3.5, N,
             "#dcfce7", "#16a34a", "Output", f"{H}x{W}x{N}")

    # ── Intermediate tensor dimension labels ──────────────────────────────────
    for xc in (4.0, 6.3):
        ax.text(xc, 1.7, f"H×W×{tM}", ha="center", fontsize=7,
                color=_T, style="italic")

    # ── Residual skip connection (shown only when M == N, stride-1 block) ─────
    if M == N:
        from matplotlib.patches import FancyArrowPatch
        skip = FancyArrowPatch(
            (0.65, 6.6), (9.2, 6.6),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.0",
            color="#6366f1", lw=1.8, linestyle="dashed",
        )
        ax.add_patch(skip)
        ax.text(4.95, 7.05,
                "Residual skip  (M = N, stride = 1  →  output += input)",
                ha="center", fontsize=7.5, color="#6366f1", style="italic")
    else:
        ax.text(4.95, 6.85, "No skip  (M ≠ N  →  projection-only block)",
                ha="center", fontsize=7.5, color=_T, style="italic")

    # ── Bottom badge ──────────────────────────────────────────────────────────
    ax.text(5.0, 0.75,
            f"3-STAGE INVERTED BOTTLENECK  ·  t = {t}  ·  expanded ch = {tM}",
            ha="center", fontsize=8.5, fontweight="bold", color="#7c3aed",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.49, 0.23, 0.93, 0.10),
                      edgecolor="#7c3aed", lw=1.4))
    plt.tight_layout()
    return fig


def draw_mbv2_stage_breakdown(M: int, N: int, Dk: int,
                               H: int, W: int, t: int, s_f: int) -> plt.Figure:
    """
    Horizontal bar chart: per-stage MBv2 FLOPs as % of a standard conv baseline.
    Provides an at-a-glance view of where the compute budget is spent.
    """
    mf     = mbv2_flops(M, N, Dk, H, W, t)
    stages = ["Expand\n(1×1 PW)", "Depthwise\n(Dk×Dk)", "Project\n(1×1 PW)"]
    fracs  = [mf["expand"] / s_f * 100,
              mf["dw"]     / s_f * 100,
              mf["project"] / s_f * 100]
    colors = ["#ca8a04", "#db2777", "#059669"]

    fig, ax = plt.subplots(figsize=(7, 2.2))
    bars = ax.barh(stages, fracs, color=colors,
                   edgecolor="none", linewidth=0, height=0.5, zorder=3)
    ax.set_xlim(0, max(fracs) * 1.30)
    ax.set_xlabel("% of Standard Conv FLOPs", fontsize=9)
    ax.set_title("MBv2 Stage-wise FLOPs as % of Standard Conv",
                 fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="x", zorder=0)
    ax.tick_params(labelsize=8)
    for b, v in zip(bars, fracs):
        ax.text(v + max(fracs) * 0.02,
                b.get_y() + b.get_height() / 2,
                f"{v:.2f}%", va="center",
                fontsize=9, fontweight="bold", color=_T)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# 6.  SIDEBAR -- hyperparameter controls
# ------------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<p class="grad-heading" style="font-size:1.3rem;">Simulator Controls</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Architecture Mode selector ─────────────────────────────────────────────
    # Drives which computation path and diagrams are rendered throughout the app.
    arch_mode = st.radio(
        "Architecture Mode",
        options=[
            "Basic Convolution (Standard vs. DWS)",
            "MobileNetV2 Inverted Residual Block",
        ],
        index=0,
        help=(
            "Basic: compares a standard Dk×Dk conv against a two-stage "
            "Depthwise Separable conv (Howard et al., MobileNets 2017).\n\n"
            "MobileNetV2: simulates the full Inverted Residual Bottleneck "
            "block (Sandler et al., 2018) — Expand → Depthwise → Project."
        ),
    )
    _mbv2_mode = (arch_mode == "MobileNetV2 Inverted Residual Block")

    st.markdown("---")
    st.markdown("**Hyper-Parameters**")

    # ── Core sliders (always visible) ─────────────────────────────────────────
    M  = st.slider("Input Channels  (M)",   1, 512,  32, 1,
                   help="Number of channels in the input feature map.")
    N  = st.slider("Output Channels  (N)",  1, 512,  64, 1,
                   help="Number of channels produced by the convolution.")
    Dk = st.slider("Kernel Size  (D_K)",    1,  11,   3, 2,
                   help="Square kernel dimension Dk × Dk. Odd values only.")
    H  = st.slider("Spatial Size  (H = W)", 4, 256,  56, 4,
                   help="Feature map height = width. Same padding assumed.")
    W  = H

    # ── Expansion factor (MBv2 mode only) ─────────────────────────────────────
    if _mbv2_mode:
        t = st.slider(
            "Expansion Factor  (t)",
            min_value=1, max_value=6, value=6, step=1,
            help=(
                "Channel multiplier for the internal expansion stage. "
                "MobileNetV2 uses t = 6 for most blocks (t = 1 for the "
                "first layer). Higher t gives richer intermediate features "
                "at the cost of more parameters and FLOPs."
            ),
        )
    else:
        t = 1   # unused in Basic mode; set to a valid neutral value

    # ── Active shape summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Active tensor shapes**")
    st.markdown(f"Input  : `{H} × {W} × {M}`")
    if _mbv2_mode:
        st.markdown(f"Expanded: `{H} × {W} × {t * M}`")
    st.markdown(f"Output : `{H} × {W} × {N}`")
    st.markdown(f"Kernel : `{Dk} × {Dk}`")
    st.markdown("---")
    st.caption("University of Malta · Deep Learning\nCPU-only simulator · No GPU required")


# ------------------------------------------------------------------------------
# 7.  LIVE CALCULATIONS
# ------------------------------------------------------------------------------

s_p = std_params(M, N, Dk)
s_f = std_flops(M, N, Dk, H, W)

# DWS always computed — needed by the PyTorch sandbox in Tab 3 regardless of mode.
d_p = dws_params(M, N, Dk)
d_f = dws_flops(M, N, Dk, H, W)

if not _mbv2_mode:
    cmp_p     = d_p
    cmp_f     = d_f
    cmp_label = "DW Separable"
    r_th      = reduction_ratio(N, Dk)
else:
    cmp_p     = mbv2_params(M, N, Dk, t)
    cmp_f     = mbv2_flops(M, N, Dk, H, W, t)
    cmp_label = "MobileNetV2"
    r_th      = mbv2_reduction_ratio(M, N, Dk, t)

ratio_p = cmp_p["total"] / s_p
ratio_f = cmp_f["total"] / s_f
saved_p = (1 - ratio_p) * 100
saved_f = (1 - ratio_f) * 100


# ------------------------------------------------------------------------------
# 8.  HERO HEADER
# ------------------------------------------------------------------------------

_page_title = (
    "MobileNetV2 Inverted Residual Block"
    if _mbv2_mode else
    "Standard vs. Depthwise Separable Convolutions"
)
_page_subtitle = (
    "Inverted Residual Bottleneck Simulator"
    if _mbv2_mode else
    "Interactive Efficiency Simulator"
)
_cmp_p_label  = "MBv2 Params"  if _mbv2_mode else "DWS Params"
_cmp_f_label  = "MBv2 FLOPs"   if _mbv2_mode else "DWS FLOPs"
_ratio_help   = "t(M+Dk²+N)/(Dk²·N)"  if _mbv2_mode else "1/N + 1/Dk²"
_savings_label = "Params Overhead" if saved_p < 0 else "Params Saved"

st.markdown(
    f'<h1 class="grad-heading" style="font-size:2.5rem; text-align:center;">'
    f"{_page_title}"
    f"</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="text-align:center; opacity:0.55; font-size:1rem; margin-bottom:1.4rem;">'
    f"{_page_subtitle} &nbsp;&#183;&nbsp; University of Malta"
    f" &nbsp;&#183;&nbsp; Deep Learning"
    f"</p>",
    unsafe_allow_html=True,
)

# Summary metric row — delta uses "inverse" so green = reduction, red = increase.
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


# ------------------------------------------------------------------------------
# 9.  THREE-TAB DASHBOARD
# ------------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊  Visual Pipeline & Benchmarks",
    "📐  Mathematical Derivations",
    "🔬  PyTorch Sandbox",
])


# ==============================================================================
# TAB 1 -- Visual Pipeline & Benchmarks
# ==============================================================================
with tab1:

    # ── Architecture diagrams ─────────────────────────────────────────────────
    st.markdown('<span class="section-chip">Architecture Diagrams</span>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        fig_s = draw_std_diagram(M, N, Dk, H, W)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)
    with col_r:
        if _mbv2_mode:
            fig_cmp = draw_mbv2_diagram(M, N, Dk, H, W, t)
        else:
            fig_cmp = draw_dws_diagram(M, N, Dk, H, W)
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

    st.markdown("---")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown('<span class="section-chip">Performance Comparison</span>',
                unsafe_allow_html=True)
    _bar_cmp_label = "MobileNetV2\nBlock" if _mbv2_mode else "DW Separable\nConv"
    fig_bars = draw_bar_chart(s_p, cmp_p["total"], s_f, cmp_f["total"],
                              cmp_label=_bar_cmp_label)
    st.pyplot(fig_bars, use_container_width=True)
    plt.close(fig_bars)

    st.markdown("---")

    # ── Stage-wise FLOPs breakdown ────────────────────────────────────────────
    if _mbv2_mode:
        st.markdown('<span class="section-chip">MBv2 Stage-wise FLOPs Breakdown</span>',
                    unsafe_allow_html=True)
        fig_rb = draw_mbv2_stage_breakdown(M, N, Dk, H, W, t, s_f)
    else:
        st.markdown('<span class="section-chip">DWS Stage-wise FLOPs Breakdown</span>',
                    unsafe_allow_html=True)
        fig_rb = draw_ratio_breakdown(M, N, Dk, H, W, s_f)
    st.pyplot(fig_rb, use_container_width=True)
    plt.close(fig_rb)

    st.markdown("---")

    # ── Numerical breakdown tables ────────────────────────────────────────────
    st.markdown('<span class="section-chip">Numerical Breakdown Tables</span>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Parameters**")
        if _mbv2_mode:
            st.table({
                "Component": [
                    "Standard Conv",
                    f"① Expand  (1×1  M→{t*M})",
                    f"② Depthwise  ({Dk}×{Dk}  ch={t*M})",
                    f"③ Project  (1×1  {t*M}→N)",
                    "MBv2 Total",
                ],
                "Count": [
                    fmt(s_p),
                    fmt(cmp_p["expand"]),
                    fmt(cmp_p["dw"]),
                    fmt(cmp_p["project"]),
                    fmt(cmp_p["total"]),
                ],
                "% of Std.": [
                    "100.00 %",
                    f"{cmp_p['expand'] / s_p * 100:.2f} %",
                    f"{cmp_p['dw']     / s_p * 100:.2f} %",
                    f"{cmp_p['project']/ s_p * 100:.2f} %",
                    f"{ratio_p * 100:.2f} %",
                ],
            })
        else:
            st.table({
                "Component": [
                    "Standard Conv",
                    f"Depthwise  ({Dk}×{Dk}×{M})",
                    f"Pointwise  (1×1×{M}×{N})",
                    "DWS Total",
                ],
                "Count": [
                    fmt(s_p),
                    fmt(cmp_p["dw"]),
                    fmt(cmp_p["pw"]),
                    fmt(cmp_p["total"]),
                ],
                "% of Std.": [
                    "100.00 %",
                    f"{cmp_p['dw'] / s_p * 100:.2f} %",
                    f"{cmp_p['pw'] / s_p * 100:.2f} %",
                    f"{ratio_p * 100:.2f} %",
                ],
            })
    with col_r:
        st.markdown("**FLOPs**")
        if _mbv2_mode:
            st.table({
                "Component": [
                    "Standard Conv",
                    "① Expand stage",
                    "② Depthwise stage",
                    "③ Project stage",
                    "MBv2 Total",
                ],
                "FLOPs": [
                    fmt(s_f),
                    fmt(cmp_f["expand"]),
                    fmt(cmp_f["dw"]),
                    fmt(cmp_f["project"]),
                    fmt(cmp_f["total"]),
                ],
                "% of Std.": [
                    "100.00 %",
                    f"{cmp_f['expand'] / s_f * 100:.2f} %",
                    f"{cmp_f['dw']     / s_f * 100:.2f} %",
                    f"{cmp_f['project']/ s_f * 100:.2f} %",
                    f"{ratio_f * 100:.2f} %",
                ],
            })
        else:
            st.table({
                "Component": [
                    "Standard Conv",
                    "Depthwise stage",
                    "Pointwise stage",
                    "DWS Total",
                ],
                "FLOPs": [
                    fmt(s_f),
                    fmt(cmp_f["dw"]),
                    fmt(cmp_f["pw"]),
                    fmt(cmp_f["total"]),
                ],
                "% of Std.": [
                    "100.00 %",
                    f"{cmp_f['dw'] / s_f * 100:.2f} %",
                    f"{cmp_f['pw'] / s_f * 100:.2f} %",
                    f"{ratio_f * 100:.2f} %",
                ],
            })

    st.markdown("---")

    # ── Cost fraction bar / callout ───────────────────────────────────────────
    st.markdown('<span class="section-chip">Compute Cost Fraction</span>',
                unsafe_allow_html=True)
    if _mbv2_mode:
        if r_th <= 1.0:
            st.markdown(
                f"MBv2 uses **{r_th * 100:.2f}%** of Standard Conv compute "
                f"(saving **{(1 - r_th) * 100:.2f}%**):"
            )
        else:
            st.markdown(
                f'<div class="insight-box">'
                f"With t = <strong>{t}</strong>, M = <strong>{M}</strong>, "
                f"N = <strong>{N}</strong>, the MBv2 block costs "
                f"<strong>{r_th * 100:.2f}%</strong> of a standard conv — "
                f"<strong>{(r_th - 1) * 100:.2f}%</strong> more expensive. "
                f"MBv2's efficiency advantage comes from architectural quality "
                f"(residual flow, high-dimensional DW space), not raw parameter "
                f"compression. Reduce t or increase N to bring the ratio below 1."
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"DWS uses **{r_th * 100:.2f}%** of Standard Conv compute "
            f"(saving **{(1 - r_th) * 100:.2f}%**):"
        )
    st.progress(float(np.clip(r_th, 0.0, 1.0)))


# ==============================================================================
# TAB 2 -- Mathematical Derivations
# ==============================================================================
with tab2:

    col_l, col_r = st.columns(2)

    # ── Standard Convolution (always shown) ───────────────────────────────────
    with col_l:
        st.markdown('<h3 class="grad-heading">Standard Convolution</h3>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="formula-card"><strong>Parameters</strong> — a single volumetric '
            'kernel of depth M is replicated N times (one per output channel).</div>',
            unsafe_allow_html=True)
        st.latex(r"\text{Params}_{\text{std}} = D_K^2 \times M \times N")
        st.latex(
            rf"\text{{Params}}_{{\text{{std}}}} "
            rf"= {Dk}^2 \times {M} \times {N} "
            rf"= \mathbf{{{fmt(s_p)}}}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="formula-card"><strong>FLOPs</strong> — at each H×W output '
            'location every output channel accumulates D_K² · M multiply-adds (×2 ops).</div>',
            unsafe_allow_html=True)
        st.latex(
            r"\text{FLOPs}_{\text{std}} = 2 \cdot D_K^2 \cdot M \cdot N \cdot H \cdot W")
        st.latex(
            rf"\text{{FLOPs}}_{{\text{{std}}}} "
            rf"= 2 \times {Dk}^2 \times {M} \times {N} \times {H} \times {W} "
            rf"= \mathbf{{{fmt(s_f)}}}")

    # ── Right column: DWS or MBv2 formulas ────────────────────────────────────
    with col_r:
        if not _mbv2_mode:
            # ── Depthwise Separable Convolution ──────────────────────────────
            st.markdown('<h3 class="grad-heading">Depthwise Separable Conv.</h3>',
                        unsafe_allow_html=True)

            st.markdown(
                '<div class="formula-card"><strong>Stage 1 — Depthwise</strong>: '
                'one Dk×Dk spatial filter per input channel (groups=M). '
                'No cross-channel mixing yet.</div>',
                unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{DW}} = D_K^2 \times M")
            st.latex(rf"\text{{Params}}_{{\text{{DW}}}} = {Dk}^2 \times {M} = {fmt(d_p['dw'])}")
            st.latex(r"\text{FLOPs}_{\text{DW}} = 2 \cdot D_K^2 \cdot M \cdot H \cdot W")
            st.latex(rf"\text{{FLOPs}}_{{\text{{DW}}}} = {fmt(d_f['dw'])}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="formula-card"><strong>Stage 2 — Pointwise</strong>: '
                '1×1 conv mixes the M depthwise outputs into N channels.</div>',
                unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{PW}} = M \times N")
            st.latex(rf"\text{{Params}}_{{\text{{PW}}}} = {M} \times {N} = {fmt(d_p['pw'])}")
            st.latex(r"\text{FLOPs}_{\text{PW}} = 2 \cdot M \cdot N \cdot H \cdot W")
            st.latex(rf"\text{{FLOPs}}_{{\text{{PW}}}} = {fmt(d_f['pw'])}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><strong>DWS Totals</strong></div>',
                        unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{DWS}} = D_K^2 M + MN")
            st.latex(
                rf"\text{{Params}}_{{\text{{DWS}}}} "
                rf"= {fmt(d_p['dw'])} + {fmt(d_p['pw'])} "
                rf"= \mathbf{{{fmt(d_p['total'])}}}")
            st.latex(r"\text{FLOPs}_{\text{DWS}} = 2MHW\!\left(D_K^2 + N\right)")
            st.latex(rf"\text{{FLOPs}}_{{\text{{DWS}}}} = \mathbf{{{fmt(d_f['total'])}}}")

        else:
            # ── MobileNetV2 Inverted Residual Block ───────────────────────────
            tM = t * M
            st.markdown('<h3 class="grad-heading">MobileNetV2 Inverted Residual</h3>',
                        unsafe_allow_html=True)

            st.markdown(
                f'<div class="formula-card"><strong>① Expand  (1×1 PW, M → tM = {tM})</strong>: '
                f'pointwise conv widens the channel dimension by expansion factor t.</div>',
                unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{exp}} = M \times tM = t M^2")
            st.latex(
                rf"\text{{Params}}_{{\text{{exp}}}} "
                rf"= {t} \times {M}^2 = {fmt(cmp_p['expand'])}")
            st.latex(r"\text{FLOPs}_{\text{exp}} = 2 \cdot M \cdot tM \cdot H \cdot W")
            st.latex(rf"\text{{FLOPs}}_{{\text{{exp}}}} = {fmt(cmp_f['expand'])}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f'<div class="formula-card"><strong>② Depthwise  (Dk×Dk, tM = {tM} channels)</strong>: '
                f'spatial filtering operates in the expanded high-dimensional space.</div>',
                unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{DW}} = D_K^2 \times tM")
            st.latex(
                rf"\text{{Params}}_{{\text{{DW}}}} "
                rf"= {Dk}^2 \times {tM} = {fmt(cmp_p['dw'])}")
            st.latex(r"\text{FLOPs}_{\text{DW}} = 2 \cdot D_K^2 \cdot tM \cdot H \cdot W")
            st.latex(rf"\text{{FLOPs}}_{{\text{{DW}}}} = {fmt(cmp_f['dw'])}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f'<div class="formula-card"><strong>③ Project  (1×1 PW, tM → N)</strong>: '
                f'pointwise conv compresses back to the target width N. '
                f'No activation (linear bottleneck).</div>',
                unsafe_allow_html=True)
            st.latex(r"\text{Params}_{\text{proj}} = tM \times N")
            st.latex(
                rf"\text{{Params}}_{{\text{{proj}}}} "
                rf"= {tM} \times {N} = {fmt(cmp_p['project'])}")
            st.latex(r"\text{FLOPs}_{\text{proj}} = 2 \cdot tM \cdot N \cdot H \cdot W")
            st.latex(rf"\text{{FLOPs}}_{{\text{{proj}}}} = {fmt(cmp_f['project'])}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><strong>MBv2 Block Totals</strong></div>',
                        unsafe_allow_html=True)
            st.latex(
                r"\text{Params}_{\text{MBv2}} = tM^2 + D_K^2 tM + tMN "
                r"= tM\!\left(M + D_K^2 + N\right)")
            st.latex(
                rf"\text{{Params}}_{{\text{{MBv2}}}} "
                rf"= {t} \times {M} \times ({M} + {Dk}^2 + {N}) "
                rf"= \mathbf{{{fmt(cmp_p['total'])}}}")
            st.latex(
                r"\text{FLOPs}_{\text{MBv2}} = 2 \cdot tM \cdot H \cdot W"
                r"\!\left(M + D_K^2 + N\right)")
            st.latex(
                rf"\text{{FLOPs}}_{{\text{{MBv2}}}} = \mathbf{{{fmt(cmp_f['total'])}}}")

    st.markdown("---")

    # ── Ratio derivation — adapts to active mode ───────────────────────────────
    if not _mbv2_mode:
        # ── DWS reduction ratio ────────────────────────────────────────────────
        st.markdown(
            '<h3 class="grad-heading" style="text-align:center;">'
            "Efficiency Reduction Ratio  (Howard et al., MobileNets 2017)"
            "</h3>",
            unsafe_allow_html=True)

        col_eq, col_num = st.columns([1.5, 0.5])
        with col_eq:
            st.latex(
                r"\frac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Std}}} "
                r"= \frac{D_K^2 \cdot M \cdot H \cdot W + M \cdot N \cdot H \cdot W}"
                r"{D_K^2 \cdot M \cdot N \cdot H \cdot W}")
            st.latex(
                r"= \frac{D_K^2 \cdot M \cdot H \cdot W}{D_K^2 \cdot M \cdot N \cdot H \cdot W}"
                r"+ \frac{M \cdot N \cdot H \cdot W}{D_K^2 \cdot M \cdot N \cdot H \cdot W}")
            st.latex(
                r"\boxed{\dfrac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Std}}} "
                r"= \dfrac{1}{N} + \dfrac{1}{D_K^2}}")
            st.latex(
                rf"\frac{{1}}{{{N}}} + \frac{{1}}{{{Dk}^2}} "
                rf"= {1/N:.5f} + {1/Dk**2:.5f} "
                rf"= \mathbf{{{r_th:.5f}}}")
            st.markdown(
                f'<div class="insight-box">'
                f"With <strong>N={N}</strong> output channels and a "
                f"<strong>{Dk}&times;{Dk}</strong> kernel, DWS costs only "
                f"<strong>{r_th * 100:.2f}%</strong> of standard conv — a "
                f"<strong>{(1 - r_th) * 100:.2f}%</strong> reduction in both parameters "
                f"and FLOPs.  As N grows the 1/N term vanishes and savings "
                f"asymptotically approach (1 − 1/D_K²)."
                f"</div>",
                unsafe_allow_html=True)

        with col_num:
            st.metric("Theory Ratio", f"{r_th:.5f}",  help="1/N + 1/Dk²")
            st.metric("Param Ratio",  f"{ratio_p:.5f}", help="DWS params / Std params")
            st.metric("FLOP Ratio",   f"{ratio_f:.5f}", help="DWS FLOPs / Std FLOPs")
            match = abs(r_th - ratio_p) < 1e-9
            if match:
                st.success("Ratios match analytically")
            else:
                st.error("Ratio mismatch")
            st.progress(float(np.clip(r_th, 0.0, 1.0)))
            st.caption(f"{r_th * 100:.2f}% of Std. cost")

    else:
        # ── MBv2 cost ratio derivation ─────────────────────────────────────────
        tM = t * M
        st.markdown(
            '<h3 class="grad-heading" style="text-align:center;">'
            "Cost Ratio vs. Standard Conv  (Sandler et al., MobileNetV2 2018)"
            "</h3>",
            unsafe_allow_html=True)

        col_eq, col_num = st.columns([1.5, 0.5])
        with col_eq:
            st.latex(
                r"\frac{\text{Params}_{\text{MBv2}}}{\text{Params}_{\text{Std}}} "
                r"= \frac{tM\!\left(M + D_K^2 + N\right)}{D_K^2 \cdot M \cdot N}")
            st.latex(
                r"\boxed{\text{Ratio} = \frac{t\!\left(M + D_K^2 + N\right)}{D_K^2 \cdot N}}")
            st.latex(
                rf"\text{{Ratio}} = "
                rf"\frac{{{t} \times ({M} + {Dk}^2 + {N})}}{{{Dk}^2 \times {N}}} "
                rf"= \mathbf{{{r_th:.5f}}}")

            if r_th > 1.0:
                st.markdown(
                    f'<div class="insight-box">'
                    f"The ratio <strong>{r_th:.4f} &gt; 1</strong> means this MBv2 "
                    f"block uses <strong>{(r_th - 1) * 100:.1f}% more</strong> parameters "
                    f"than a plain {Dk}&times;{Dk} standard conv with the same I/O shape. "
                    f"MobileNetV2's efficiency benefit is <em>architectural</em>: the "
                    f"inverted bottleneck maintains a rich high-dimensional space for "
                    f"spatial filtering and the residual connection (when M = N) preserves "
                    f"gradient flow — gains not captured by the raw parameter count alone. "
                    f"To bring the ratio below 1, reduce t or increase N."
                    f"</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="insight-box">'
                    f"The ratio <strong>{r_th:.4f} &lt; 1</strong> — this configuration "
                    f"is also cheaper in raw parameters than a standard conv. "
                    f"This occurs when N is large relative to t&times;(M + D_K² + N)."
                    f"</div>",
                    unsafe_allow_html=True)

        with col_num:
            st.metric("Ratio (MBv2/Std)", f"{r_th:.5f}", help="t(M+Dk²+N)/(Dk²·N)")
            st.metric("Param Ratio",      f"{ratio_p:.5f}", help="actual params ratio")
            st.metric("FLOP Ratio",       f"{ratio_f:.5f}", help="actual FLOPs ratio")
            match = abs(r_th - ratio_p) < 1e-9
            if match:
                st.success("Formula matches exactly")
            else:
                st.error("Ratio mismatch")
            skip_status = "Yes  (M = N)" if M == N else "No  (M ≠ N)"
            st.metric("Residual Skip", skip_status)
            st.progress(float(np.clip(r_th, 0.0, 1.0)))
            st.caption(f"{r_th * 100:.2f}% of Std. cost")


# ==============================================================================
# TAB 3 -- PyTorch CPU Sandbox
# ==============================================================================
with tab3:

    st.markdown('<h3 class="grad-heading">PyTorch CPU Verification Sandbox</h3>',
                unsafe_allow_html=True)
    st.markdown(
        f"Runs actual `torch.nn` layers on **CPU only** (`device='cpu'` hard-coded) "
        f"with a dummy input of shape `[1, {M}, {H}, {W}]` — driven by the active "
        f"sidebar sliders.  Verifies that tensor shapes and parameter counts match "
        f"the analytical formulas exactly."
    )
    st.markdown("---")

    # ------------------------------------------------------------------
    # Code walkthrough — mirrors the live execution below exactly.
    # All values are injected from the active sidebar sliders so the
    # displayed snippet always stays in sync with what will actually run.
    # ------------------------------------------------------------------
    col_code, col_run = st.columns([1.6, 0.4])
    with col_code:
        st.code(
            f'import torch\n'
            f'import torch.nn as nn\n\n'
            f'# ── Dummy input: shape driven by sidebar sliders ──────────\n'
            f'device = "cpu"                          # CPU-only hard constraint\n'
            f'x = torch.randn(1, {M}, {H}, {W}, device=device)\n\n'
            f'# ── Standard Convolution ──────────────────────────────────\n'
            f'std_layer = nn.Conv2d(\n'
            f'    in_channels={M},\n'
            f'    out_channels={N},\n'
            f'    kernel_size={Dk},\n'
            f'    padding={Dk // 2},\n'
            f'    bias=False,\n'
            f')\n\n'
            f'# ── Depthwise Separable: two-stage factorisation ──────────\n'
            f'dw_layer = nn.Conv2d(       # depthwise: groups=M isolates channels\n'
            f'    in_channels={M},\n'
            f'    out_channels={M},\n'
            f'    kernel_size={Dk},\n'
            f'    padding={Dk // 2},\n'
            f'    groups={M},\n'
            f'    bias=False,\n'
            f')\n'
            f'pw_layer = nn.Conv2d(       # pointwise: 1x1 mixes channels\n'
            f'    in_channels={M},\n'
            f'    out_channels={N},\n'
            f'    kernel_size=1,\n'
            f'    bias=False,\n'
            f')\n\n'
            f'# ── Forward passes ────────────────────────────────────────\n'
            f'with torch.no_grad():\n'
            f'    out_std = std_layer(x)\n'
            f'    inter   = dw_layer(x)   # intermediate: same spatial, M channels\n'
            f'    out_dws = pw_layer(inter)\n\n'
            f'# ── Parameter counts ──────────────────────────────────────\n'
            f'p_std = sum(p.numel() for p in std_layer.parameters())\n'
            f'p_dws = (\n'
            f'    sum(p.numel() for p in dw_layer.parameters()) +\n'
            f'    sum(p.numel() for p in pw_layer.parameters())\n'
            f')',
            language="python",
        )

    with col_run:
        st.markdown("**Expected results**")
        st.markdown(f"Std. params: `{fmt(s_p)}`")
        st.markdown(f"DWS params:  `{fmt(d_p['total'])}`")
        st.markdown(f"Output shape: `[1, {N}, {H}, {W}]`")
        st.markdown("---")
        run = st.button("▶  Run Validation", use_container_width=True)

    # ------------------------------------------------------------------
    # Live execution — all layer and tensor definitions are driven by
    # the current sidebar slider values (M, N, Dk, H, W).  The entire
    # block is wrapped in a try/except so an absent torch installation
    # degrades gracefully without breaking the rest of the application.
    # ------------------------------------------------------------------
    if run:
        st.markdown("---")
        try:
            import torch
            import torch.nn as nn

            # Device string — avoids creating a torch.device object before
            # confirming the import succeeded.
            device = "cpu"

            # Input tensor: shape = [batch=1, C=M, H, W], slider-driven.
            x = torch.randn(1, M, H, W, device=device)

            # Standard convolution layer — all kwargs match slider values.
            std_layer = nn.Conv2d(
                in_channels=M,
                out_channels=N,
                kernel_size=Dk,
                padding=Dk // 2,
                bias=False,
            )

            # Depthwise layer: groups=M forces one filter per input channel.
            dw_layer = nn.Conv2d(
                in_channels=M,
                out_channels=M,
                kernel_size=Dk,
                padding=Dk // 2,
                groups=M,
                bias=False,
            )

            # Pointwise layer: 1x1 kernel mixes M channels into N.
            pw_layer = nn.Conv2d(
                in_channels=M,
                out_channels=N,
                kernel_size=1,
                bias=False,
            )

            # Forward passes — no gradient computation needed.
            with torch.no_grad():
                out_std = std_layer(x)
                inter   = dw_layer(x)   # M-channel spatial-filtered intermediate
                out_dws = pw_layer(inter)

            # Count learnable parameters via numel().
            p_std = sum(p.numel() for p in std_layer.parameters())
            p_dw  = sum(p.numel() for p in dw_layer.parameters())
            p_pw  = sum(p.numel() for p in pw_layer.parameters())
            p_dws = p_dw + p_pw

            # Verification flags — compare against analytical formulas.
            shape_ok_std = list(out_std.shape) == [1, N, H, W]
            shape_ok_dws = list(out_dws.shape) == [1, N, H, W]
            param_ok_std = (p_std == s_p)
            param_ok_dws = (p_dws == d_p["total"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Standard Convolution")
                st.info(
                    f"Output shape : `{tuple(out_std.shape)}`  "
                    f"[{'OK' if shape_ok_std else 'FAIL'}]\n\n"
                    f"Torch params : `{fmt(p_std)}`  "
                    f"[{'OK' if param_ok_std else 'FAIL'}]\n\n"
                    f"Formula      : `{fmt(s_p)}`"
                )
            with col_b:
                st.markdown("#### Depthwise Separable Conv.")
                st.info(
                    f"Output shape  : `{tuple(out_dws.shape)}`  "
                    f"[{'OK' if shape_ok_dws else 'FAIL'}]\n\n"
                    f"Torch params  : `{fmt(p_dws)}`  "
                    f"[{'OK' if param_ok_dws else 'FAIL'}]\n\n"
                    f"Formula       : `{fmt(d_p['total'])}`\n\n"
                    f"  DW layer    : `{fmt(p_dw)}`  (formula `{fmt(d_p['dw'])}`)\n\n"
                    f"  PW layer    : `{fmt(p_pw)}`  (formula `{fmt(d_p['pw'])}`)"
                )

            st.markdown("#### Forward-pass Shape Trace")
            st.code(
                f"Input               : {tuple(x.shape)}\n"
                f"-> Std  Conv2d      : {tuple(out_std.shape)}\n\n"
                f"Input               : {tuple(x.shape)}\n"
                f"-> Depthwise Conv2d : {tuple(inter.shape)}\n"
                f"-> Pointwise Conv2d : {tuple(out_dws.shape)}",
                language="text",
            )

            all_ok = shape_ok_std and shape_ok_dws and param_ok_std and param_ok_dws
            if all_ok:
                st.success(
                    "All checks passed. "
                    "PyTorch CPU parameter counts match the analytical formulas exactly."
                )
            else:
                st.error("One or more checks failed -- review the result cards above.")

        except ImportError:
            # Soft warning: does not interrupt the rest of the application.
            # Guides the user to install PyTorch inside their active virtual
            # environment rather than globally, which is the most common
            # source of import failures in isolated Python environments.
            st.warning(
                "**PyTorch is not available in the current runtime.**\n\n"
                "The rest of the simulator (analytics, charts, LaTeX) continues "
                "to work without it — only this live sandbox requires PyTorch.\n\n"
                "To enable the sandbox, activate your project virtual environment "
                "and run the CPU-only install command:\n\n"
                "```bash\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "```\n\n"
                "Then restart the Streamlit server (`streamlit run app.py`) and "
                "click **Run Validation** again."
            )


# ------------------------------------------------------------------------------
# 10.  FOOTER
# ------------------------------------------------------------------------------

st.markdown(
    '<div style="text-align:center; opacity:0.45; font-size:0.80rem; '
    'padding:1.6rem 0 0.4rem;">'
    "CPU-only simulator -- metrics computed analytically via NumPy.  No CUDA required.<br>"
    "Reference: Howard et al. (2017) -- MobileNets: Efficient CNNs for Mobile Vision"
    " | University of Malta - Deep Learning Assignment"
    "</div>",
    unsafe_allow_html=True,
)
