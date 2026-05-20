"""
ui/sidebar.py
-------------
Apple-style sidebar with two-way synchronised hyperparameter controls.

Design contract
---------------
* init_session_state()  — must be called once per page load, BEFORE the
                          sidebar is rendered, to seed st.session_state.
* render_sidebar()      — renders the complete sidebar and returns a params
                          dict with the canonical current values.

Session-state key scheme  (three keys per parameter)
-----------------------------------------------------
  "M"      — canonical integer value consumed by all computation
  "_M_sl"  — slider widget key  (snapped to step grid)
  "_M_nb"  — number-input key   (exact typed value; may differ from slider)

Callback pattern
----------------
  _on_X_sl  : slider moved → write canonical + mirror to number-input
  _on_X_nb  : value typed  → clamp, write canonical, snap slider to tick

The _nearest_snap helper (imported from core.math_engine) rounds the typed
value to the nearest valid slider tick using integer arithmetic so that
Python's banker's rounding never produces an off-by-one surprise.
"""

from __future__ import annotations

import streamlit as st
from core.math_engine import _nearest_snap


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "M":  32, "_M_sl":  32, "_M_nb":  32,
    "N":  64, "_N_sl":  64, "_N_nb":  64,
    "H":  56, "_H_sl":  56, "_H_nb":  56,
    "Dk":  3, "_Dk_sl":  3, "_Dk_nb":  3,
    "t":   6, "_t_sl":   6, "_t_nb":   6,
}


def init_session_state() -> None:
    """Seed st.session_state with parameter defaults (idempotent)."""
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Two-way sync callbacks
# ---------------------------------------------------------------------------

def _on_M_sl():
    st.session_state["M"]     = st.session_state["_M_sl"]
    st.session_state["_M_nb"] = st.session_state["_M_sl"]

def _on_M_nb():
    v = max(1, min(512, int(st.session_state["_M_nb"])))
    st.session_state["M"]     = v
    st.session_state["_M_sl"] = _nearest_snap(v, 16, 16, 512)


def _on_N_sl():
    st.session_state["N"]     = st.session_state["_N_sl"]
    st.session_state["_N_nb"] = st.session_state["_N_sl"]

def _on_N_nb():
    v = max(1, min(512, int(st.session_state["_N_nb"])))
    st.session_state["N"]     = v
    st.session_state["_N_sl"] = _nearest_snap(v, 16, 16, 512)


def _on_Dk_sl():
    st.session_state["Dk"]     = st.session_state["_Dk_sl"]
    st.session_state["_Dk_nb"] = st.session_state["_Dk_sl"]

def _on_Dk_nb():
    v = max(1, min(11, int(st.session_state["_Dk_nb"])))
    st.session_state["Dk"]     = v
    st.session_state["_Dk_sl"] = _nearest_snap(v, 1, 2, 11)


def _on_H_sl():
    st.session_state["H"]     = st.session_state["_H_sl"]
    st.session_state["_H_nb"] = st.session_state["_H_sl"]

def _on_H_nb():
    v = max(1, min(512, int(st.session_state["_H_nb"])))
    st.session_state["H"]     = v
    st.session_state["_H_sl"] = _nearest_snap(v, 7, 7, 224)


def _on_t_sl():
    st.session_state["t"]     = st.session_state["_t_sl"]
    st.session_state["_t_nb"] = st.session_state["_t_sl"]

def _on_t_nb():
    v = max(1, min(6, int(st.session_state["_t_nb"])))
    st.session_state["t"]     = v
    st.session_state["_t_sl"] = v   # step=1: every integer is a valid tick


# ---------------------------------------------------------------------------
# Tile helper
# ---------------------------------------------------------------------------

def _tile_label(label: str, key: str, unit: str, color: str = "#0ea5e9") -> None:
    """Render a flex row: descriptive label (left) + live badge (right)."""
    val = st.session_state[key]
    bg  = color.replace("#", "")
    # Decode hex to RGB for the rgba background
    r   = int(bg[0:2], 16)
    g   = int(bg[2:4], 16)
    b   = int(bg[4:6], 16)
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;margin-bottom:4px;'>"
        f"<span style='font-weight:600;font-size:0.88rem;"
        f"color:var(--text-color);'>{label}</span>"
        f"<span style='font-family:monospace;font-size:0.85rem;font-weight:700;"
        f"color:{color};background:rgba({r},{g},{b},0.1);padding:2px 8px;"
        f"border-radius:6px;'>{unit} = {val}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _spacer() -> None:
    st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render the complete sidebar and return the canonical parameter dict.

    Returns
    -------
    dict with keys: M, N, Dk, H, W, t, mbv2_mode
    """
    with st.sidebar:
        st.markdown(
            '<p class="grad-heading" style="font-size:1.3rem;">Simulator Controls</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Architecture Mode ─────────────────────────────────────────────
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
        mbv2_mode = (arch_mode == "MobileNetV2 Inverted Residual Block")

        # ── Hyper-parameter controls ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<span class="section-chip">Hyper-Parameters</span>',
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.75rem;opacity:0.55;margin:0.2rem 0 0.6rem;'>"
            "Sliders snap to standard DL dimensions. "
            "Type any exact value in the right-hand box.</p>",
            unsafe_allow_html=True,
        )

        # Input Channels (M) — step=16 covers backbone widths 16→512
        _tile_label("Input Channels", "M", "M")
        _sl, _, _nb = st.columns([3.2, 0.15, 1.0])
        _sl.slider("M_slider", 16, 512, step=16,
                   key="_M_sl", on_change=_on_M_sl,
                   label_visibility="collapsed",
                   help="Snap step = 16 — standard widths: 16, 32, 64, 128, 256, 512.")
        _nb.number_input("M_input", 1, 512, step=1,
                         key="_M_nb", on_change=_on_M_nb,
                         label_visibility="collapsed",
                         help="Type any exact channel count. Slider snaps to nearest ×16.")
        M = st.session_state["M"]
        _spacer()

        # Output Channels (N)
        _tile_label("Output Channels", "N", "N")
        _sl, _, _nb = st.columns([3.2, 0.15, 1.0])
        _sl.slider("N_slider", 16, 512, step=16,
                   key="_N_sl", on_change=_on_N_sl,
                   label_visibility="collapsed",
                   help="Snap step = 16 — standard output widths.")
        _nb.number_input("N_input", 1, 512, step=1,
                         key="_N_nb", on_change=_on_N_nb,
                         label_visibility="collapsed",
                         help="Type any exact channel count. Slider snaps to nearest ×16.")
        N = st.session_state["N"]
        _spacer()

        # Kernel Size (Dk) — step=2 enforces odd values: 1, 3, 5, 7, 9, 11
        _tile_label("Kernel Size", "Dk", "Dₖ")
        _sl, _, _nb = st.columns([3.2, 0.15, 1.0])
        _sl.slider("Dk_slider", 1, 11, step=2,
                   key="_Dk_sl", on_change=_on_Dk_sl,
                   label_visibility="collapsed",
                   help="Odd values only: 1, 3, 5, 7, 9, 11.")
        _nb.number_input("Dk_input", 1, 11, step=2,
                         key="_Dk_nb", on_change=_on_Dk_nb,
                         label_visibility="collapsed",
                         help="Type any odd kernel size (1–11).")
        Dk = st.session_state["Dk"]
        _spacer()

        # Spatial Size (H = W) — step=7 hits ImageNet feature-map sizes
        _tile_label("Spatial Size", "H", "H")
        _sl, _, _nb = st.columns([3.2, 0.15, 1.0])
        _sl.slider("H_slider", 7, 224, step=7,
                   key="_H_sl", on_change=_on_H_sl,
                   label_visibility="collapsed",
                   help="Step = 7 — ImageNet sizes: 7, 14, 28, 56, 112, 224.")
        _nb.number_input("H_input", 1, 512, step=1,
                         key="_H_nb", on_change=_on_H_nb,
                         label_visibility="collapsed",
                         help="Type any exact spatial dimension.")
        H = st.session_state["H"]
        W = H
        _spacer()

        # Expansion Factor (t) — only in MBv2 mode
        if mbv2_mode:
            _tile_label("Expansion Factor", "t", "t", color="#818cf8")
            _sl, _, _nb = st.columns([3.2, 0.15, 1.0])
            _sl.slider("t_slider", 1, 6, step=1,
                       key="_t_sl", on_change=_on_t_sl,
                       label_visibility="collapsed",
                       help="MobileNetV2 uses t=6 for most blocks, t=1 for the first.")
            _nb.number_input("t_input", 1, 6, step=1,
                             key="_t_nb", on_change=_on_t_nb,
                             label_visibility="collapsed",
                             help="Expansion factor 1–6.")
            t = st.session_state["t"]
            _spacer()
        else:
            t = 1

        # ── Active tensor shape card ──────────────────────────────────────
        st.markdown("---")
        with st.container(border=True):
            st.markdown('<span class="section-chip">Active Shapes</span>',
                        unsafe_allow_html=True)
            _exp_row = (
                f"<tr><td style='opacity:0.55;padding:1px 8px 1px 0'>Expanded</td>"
                f"<td><code>{H}×{W}×{t * M}</code></td></tr>"
                if mbv2_mode else ""
            )
            st.markdown(
                f"<table style='width:100%;font-size:0.84rem;border-collapse:collapse;"
                f"margin-top:0.3rem'>"
                f"<tr><td style='opacity:0.55;padding:1px 8px 1px 0'>Input</td>"
                f"    <td><code>{H}×{W}×{M}</code></td></tr>"
                f"{_exp_row}"
                f"<tr><td style='opacity:0.55;padding:1px 8px 1px 0'>Output</td>"
                f"    <td><code>{H}×{W}×{N}</code></td></tr>"
                f"<tr><td style='opacity:0.55;padding:1px 8px 1px 0'>Kernel</td>"
                f"    <td><code>{Dk}×{Dk}</code></td></tr>"
                f"</table>",
                unsafe_allow_html=True,
            )

        st.caption("University of Malta · Deep Learning\nCPU-only simulator · No GPU required")

    return {
        "M": M, "N": N, "Dk": Dk,
        "H": H, "W": W, "t": t,
        "mbv2_mode": mbv2_mode,
    }
