"""
visualization/static_plots.py
------------------------------
All Matplotlib figure factories for the Conv Efficiency Simulator.

Every figure uses fully transparent figure/axes backgrounds (set via
rcParams in config.styles.configure_matplotlib) so it floats naturally on
Streamlit's Light and Dark surfaces.

All annotation text uses _T = '#555759' (mid-slate), which achieves the
WCAG AA contrast threshold on both white and dark-navy backgrounds at the
rendered font sizes used here.

Public API
----------
draw_std_diagram           — standard 2-D conv block diagram
draw_dws_diagram           — depthwise separable conv block diagram
draw_mbv2_diagram          — MobileNetV2 inverted residual block diagram
draw_bar_chart             — side-by-side Params / FLOPs bar chart
draw_ratio_breakdown       — DWS stage-wise FLOPs as % of std (horiz. bars)
draw_mbv2_stage_breakdown  — MBv2 stage-wise FLOPs as % of std (horiz. bars)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config.styles import CHART_TEXT
from core.math_engine import fmt, dws_params, dws_flops, mbv2_params, mbv2_flops

# ---------------------------------------------------------------------------
# Module-level colour aliases
# ---------------------------------------------------------------------------
_T     = CHART_TEXT    # mid-slate text: readable on light and dark
_ARROW = "#64748b"     # slightly darker than grid lines; visible on both themes


# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------

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
        _patch(ax, x + off, y + off, w, h,
               face=face, edge=edge, lw=1.0,
               alpha=max(0.25, 0.85 - i * 0.13), zorder=2 + i)
    cx  = x + layers * 0.08 / 2 + w / 2
    top = y + layers * 0.08 + h
    ax.text(cx, top + 0.28, label,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=edge, zorder=10)
    ax.text(cx, top + 0.02, sublabel,
            ha="center", va="bottom",
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


# ---------------------------------------------------------------------------
# Architecture diagrams
# ---------------------------------------------------------------------------

def draw_std_diagram(M: int, N: int, Dk: int, H: int, W: int) -> plt.Figure:
    """
    Block diagram for a standard 2-D convolution.

    Canvas: figsize=(8.5, 4.4), xlim=(0, 12).
    Layout distributes Input → Kernel → Output with uniform spacing.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("Standard Convolution  (1 operation)",
                 fontsize=13, fontweight="bold", pad=14)

    _stacked(ax, 0.35, 1.6, 2.0, 3.8, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}×{W}×{M}")
    _arr(ax, 2.55, 3.65, 3.5)
    _kbox(ax, 3.7, 2.4, 2.8, 2.2,
          "#fef3c7", "#d97706",
          f"{Dk}×{Dk}×{M}×{N}",
          f"Kernel — {fmt(Dk * Dk * M * N)} params")
    _arr(ax, 6.6, 7.65, 3.5)
    _stacked(ax, 7.7, 1.6, 2.0, 3.8, N,
             "#dcfce7", "#16a34a", "Output", f"{H}×{W}×{N}")

    ax.text(5.5, 0.5, "SINGLE-PASS  ·  ALL CHANNELS FUSED",
            ha="center", fontsize=9, fontweight="bold", color="#4f46e5",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.38, 0.34, 0.93, 0.12),
                      edgecolor="#6366f1", lw=1.4))
    plt.tight_layout(pad=1.2)
    return fig


def draw_dws_diagram(M: int, N: int, Dk: int, H: int, W: int) -> plt.Figure:
    """
    Block diagram for depthwise separable convolution.

    Canvas: figsize=(8.5, 4.4), xlim=(0, 14).
    Output stack positioned at x=12.25 so the stacked right edge (13.92)
    stays inside the xlim boundary on compact monitors.
    """
    dp = dws_params(M, N, Dk)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("Depthwise Separable Convolution  (2 operations)",
                 fontsize=13, fontweight="bold", pad=14)

    _stacked(ax, 0.15, 1.6, 1.6, 3.8, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}×{W}×{M}")
    _arr(ax, 1.95, 2.75, 3.5)
    _kbox(ax, 2.85, 2.5, 2.5, 2.0,
          "#fce7f3", "#db2777",
          f"{Dk}×{Dk} per ch.",
          f"DW — {fmt(dp['dw'])} params")
    _arr(ax, 5.45, 6.2, 3.5)
    _stacked(ax, 6.3, 1.8, 1.4, 3.3, M,
             "#ede9fe", "#7c3aed", "Interm.", f"{H}×{W}×{M}")
    _arr(ax, 7.9, 8.65, 3.5)
    _kbox(ax, 8.75, 2.5, 2.5, 2.0,
          "#d1fae5", "#059669",
          "1×1  chan. mix",
          f"PW — {fmt(dp['pw'])} params")
    _arr(ax, 11.35, 12.15, 3.5)
    _stacked(ax, 12.25, 1.6, 1.35, 3.8, N,
             "#dcfce7", "#16a34a", "Output", f"{H}×{W}×{N}")

    ax.text(7.0, 0.5,
            "SPATIAL FILTERING  ·  then  ·  CHANNEL MIXING  (FACTORISED)",
            ha="center", fontsize=8.5, fontweight="bold", color="#be185d",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.86, 0.15, 0.47, 0.10),
                      edgecolor="#db2777", lw=1.4))
    plt.tight_layout(pad=1.2)
    return fig


def draw_mbv2_diagram(M: int, N: int, Dk: int, H: int, W: int, t: int) -> plt.Figure:
    """
    Block diagram for a MobileNetV2 Inverted Residual Bottleneck.

    Canvas: figsize=(10.0, 5.6), xlim=(0, 14.5), ylim=(0, 9.5).
    Five elements (Input, Expand, DW, Project, Output) distributed with
    uniform 0.6-unit inter-element gaps.  Output right stacked edge = 12.97
    which stays clear of the 14.5 boundary on compact monitors.
    """
    tM = t * M
    mp = mbv2_params(M, N, Dk, t)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    ax.set_title(f"MobileNetV2 Inverted Residual Block  (t = {t})",
                 fontsize=13, fontweight="bold", pad=16)

    _stacked(ax, 0.25, 2.2, 1.3, 3.8, M,
             "#dbeafe", "#1d4ed8", "Input", f"{H}×{W}×{M}")
    _arr(ax, 1.95, 2.65, 4.1)

    _kbox(ax, 2.7, 2.7, 2.2, 2.8,
          "#fef9c3", "#ca8a04",
          f"1×1   M→{tM}",
          f"Expand · {fmt(mp['expand'])} params")
    ax.text(3.8, 5.75, "① Expand", ha="center", fontsize=8.5,
            fontweight="bold", color="#ca8a04")
    _arr(ax, 5.0, 5.5, 4.1)

    _kbox(ax, 5.55, 2.7, 2.2, 2.8,
          "#fce7f3", "#db2777",
          f"{Dk}×{Dk}   ch={tM}",
          f"DW · {fmt(mp['dw'])} params")
    ax.text(6.65, 5.75, "② Depthwise", ha="center", fontsize=8.5,
            fontweight="bold", color="#db2777")
    _arr(ax, 7.85, 8.35, 4.1)

    _kbox(ax, 8.4, 2.7, 2.2, 2.8,
          "#d1fae5", "#059669",
          f"1×1   {tM}→N",
          f"Project · {fmt(mp['project'])} params")
    ax.text(9.5, 5.75, "③ Project", ha="center", fontsize=8.5,
            fontweight="bold", color="#059669")
    _arr(ax, 10.7, 11.3, 4.1)

    _stacked(ax, 11.35, 2.2, 1.3, 3.8, N,
             "#dcfce7", "#16a34a", "Output", f"{H}×{W}×{N}")

    for xc in (5.25, 8.1):
        ax.text(xc, 1.9, f"H×W×{tM}",
                ha="center", fontsize=7.5, color=_T, style="italic")

    if M == N:
        skip = FancyArrowPatch(
            (0.85, 7.1), (12.0, 7.1),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.0",
            color="#6366f1", lw=1.8, linestyle="dashed",
        )
        ax.add_patch(skip)
        ax.text(6.4, 7.6,
                "Residual skip  (M = N, stride = 1  →  output += input)",
                ha="center", fontsize=7.5, color="#6366f1", style="italic")
    else:
        ax.text(6.4, 7.3,
                "No skip  (M ≠ N  →  projection-only block)",
                ha="center", fontsize=7.5, color=_T, style="italic")

    ax.text(6.4, 0.65,
            f"3-STAGE INVERTED BOTTLENECK  ·  t = {t}  ·  expanded ch = {tM}",
            ha="center", fontsize=8.5, fontweight="bold", color="#7c3aed",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(0.49, 0.23, 0.93, 0.10),
                      edgecolor="#7c3aed", lw=1.4))
    plt.tight_layout(pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# Bar charts and breakdown plots
# ---------------------------------------------------------------------------

def draw_bar_chart(
    sp: int, dp: int, sf: int, df: int,
    cmp_label: str = "DW Separable\nConv",
) -> plt.Figure:
    """Side-by-side Params / FLOPs comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    cats   = ["Standard\nConv", cmp_label]
    c_pair = [("#6366f1", "#22c55e"), ("#f59e0b", "#ec4899")]

    for ax, vals, colors, title, unit in [
        (ax1, [sp, dp], c_pair[0], "Parameter Count",    "Params"),
        (ax2, [sf, df], c_pair[1], "FLOPs  (fwd. pass)", "FLOPs"),
    ]:
        bars = ax.bar(cats, vals, color=colors, width=0.42,
                      edgecolor="none", linewidth=0, zorder=3)
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


def draw_ratio_breakdown(
    M: int, N: int, Dk: int, H: int, W: int, s_f: int,
) -> plt.Figure:
    """Horizontal bar: DWS stage-wise FLOPs as % of standard conv."""
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


def draw_mbv2_stage_breakdown(
    M: int, N: int, Dk: int, H: int, W: int, t: int, s_f: int,
) -> plt.Figure:
    """Horizontal bar: MBv2 stage-wise FLOPs as % of standard conv baseline."""
    mf     = mbv2_flops(M, N, Dk, H, W, t)
    stages = ["Expand\n(1×1 PW)", "Depthwise\n(Dk×Dk)", "Project\n(1×1 PW)"]
    fracs  = [
        mf["expand"]  / s_f * 100,
        mf["dw"]      / s_f * 100,
        mf["project"] / s_f * 100,
    ]
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
