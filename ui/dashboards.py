"""
ui/dashboards.py
----------------
Tab-level view renderers for the Conv Efficiency Simulator.

Each public function receives the current params dict and the pre-computed
results dict from core.math_engine.compute_all, then renders a complete
Streamlit tab.  No computation logic lives here — all numbers come from
the results dict so that the view layer remains a pure rendering concern.

Public API
----------
render_tab1(params, results)  — Visual Pipeline & Benchmarks
render_tab2(params, results)  — Mathematical Derivations
render_tab3(params, results)  — PyTorch CPU Sandbox
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from core.math_engine import fmt
from visualization.static_plots import (
    draw_std_diagram,
    draw_dws_diagram,
    draw_mbv2_diagram,
    draw_bar_chart,
    draw_ratio_breakdown,
    draw_mbv2_stage_breakdown,
)
from visualization.animation_engine import render_animation_panel
from core.torch_sandbox import render_sandbox

try:
    import matplotlib.pyplot as plt
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tab 1 — Visual Pipeline & Benchmarks
# ---------------------------------------------------------------------------

def render_tab1(params: dict, results: dict) -> None:
    """Render architecture diagrams, bar charts, tables, and animation panel."""
    M, N, Dk = params["M"], params["N"], params["Dk"]
    H, W, t  = params["H"], params["W"], params["t"]
    mbv2     = params["mbv2_mode"]

    s_p    = results["s_p"]
    s_f    = results["s_f"]
    cmp_p  = results["cmp_p"]
    cmp_f  = results["cmp_f"]
    r_th   = results["r_th"]
    ratio_p = results["ratio_p"]
    ratio_f = results["ratio_f"]

    # ── Architecture diagrams ─────────────────────────────────────────────
    st.markdown('<span class="section-chip">Architecture Diagrams</span>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        fig_s = draw_std_diagram(M, N, Dk, H, W)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)
    with col_r:
        fig_cmp = (draw_mbv2_diagram(M, N, Dk, H, W, t)
                   if mbv2 else draw_dws_diagram(M, N, Dk, H, W))
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

    st.markdown("---")

    # ── Interactive animation panel ───────────────────────────────────────
    render_animation_panel(params, mbv2)

    st.markdown("---")

    # ── Bar chart ─────────────────────────────────────────────────────────
    st.markdown('<span class="section-chip">Performance Comparison</span>',
                unsafe_allow_html=True)
    _bar_label = "MobileNetV2\nBlock" if mbv2 else "DW Separable\nConv"
    fig_bars = draw_bar_chart(s_p, cmp_p["total"], s_f, cmp_f["total"],
                              cmp_label=_bar_label)
    st.pyplot(fig_bars, use_container_width=True)
    plt.close(fig_bars)

    st.markdown("---")

    # ── Stage-wise FLOPs breakdown ────────────────────────────────────────
    if mbv2:
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

    # ── Numerical breakdown tables ────────────────────────────────────────
    st.markdown('<span class="section-chip">Numerical Breakdown Tables</span>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Parameters**")
        if mbv2:
            st.table({
                "Component": [
                    "Standard Conv",
                    f"① Expand  (1×1  M→{t*M})",
                    f"② Depthwise  ({Dk}×{Dk}  ch={t*M})",
                    f"③ Project  (1×1  {t*M}→N)",
                    "MBv2 Total",
                ],
                "Count": [
                    fmt(s_p), fmt(cmp_p["expand"]),
                    fmt(cmp_p["dw"]), fmt(cmp_p["project"]),
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
                    fmt(s_p), fmt(cmp_p["dw"]),
                    fmt(cmp_p["pw"]), fmt(cmp_p["total"]),
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
        if mbv2:
            st.table({
                "Component": [
                    "Standard Conv",
                    "① Expand stage",
                    "② Depthwise stage",
                    "③ Project stage",
                    "MBv2 Total",
                ],
                "FLOPs": [
                    fmt(s_f), fmt(cmp_f["expand"]),
                    fmt(cmp_f["dw"]), fmt(cmp_f["project"]),
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
                    fmt(s_f), fmt(cmp_f["dw"]),
                    fmt(cmp_f["pw"]), fmt(cmp_f["total"]),
                ],
                "% of Std.": [
                    "100.00 %",
                    f"{cmp_f['dw'] / s_f * 100:.2f} %",
                    f"{cmp_f['pw'] / s_f * 100:.2f} %",
                    f"{ratio_f * 100:.2f} %",
                ],
            })

    st.markdown("---")

    # ── Cost fraction ─────────────────────────────────────────────────────
    st.markdown('<span class="section-chip">Compute Cost Fraction</span>',
                unsafe_allow_html=True)
    if mbv2:
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


# ---------------------------------------------------------------------------
# Tab 2 — Mathematical Derivations
# ---------------------------------------------------------------------------

def render_tab2(params: dict, results: dict) -> None:
    """Render side-by-side LaTeX formula derivations and ratio analysis."""
    M, N, Dk = params["M"], params["N"], params["Dk"]
    H, W, t  = params["H"], params["W"], params["t"]
    mbv2     = params["mbv2_mode"]

    s_p    = results["s_p"]
    s_f    = results["s_f"]
    d_p    = results["d_p"]
    d_f    = results["d_f"]
    cmp_p  = results["cmp_p"]
    cmp_f  = results["cmp_f"]
    r_th   = results["r_th"]
    ratio_p = results["ratio_p"]
    ratio_f = results["ratio_f"]

    col_l, col_r = st.columns(2)

    # ── Standard Convolution (always left) ───────────────────────────────
    with col_l:
        st.markdown('<h3 class="grad-heading">Standard Convolution</h3>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="formula-card"><strong>Parameters</strong> — a single '
            'volumetric kernel of depth M is replicated N times (one per output '
            'channel).</div>', unsafe_allow_html=True)
        st.latex(r"\text{Params}_{\text{std}} = D_K^2 \times M \times N")
        st.latex(
            rf"\text{{Params}}_{{\text{{std}}}} "
            rf"= {Dk}^2 \times {M} \times {N} "
            rf"= \mathbf{{{fmt(s_p)}}}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="formula-card"><strong>FLOPs</strong> — at each H×W '
            'output location every output channel accumulates D_K²·M '
            'multiply-adds (×2 ops).</div>', unsafe_allow_html=True)
        st.latex(
            r"\text{FLOPs}_{\text{std}} = 2 \cdot D_K^2 \cdot M \cdot N \cdot H \cdot W")
        st.latex(
            rf"\text{{FLOPs}}_{{\text{{std}}}} "
            rf"= 2 \times {Dk}^2 \times {M} \times {N} \times {H} \times {W} "
            rf"= \mathbf{{{fmt(s_f)}}}")

    # ── Right column: DWS or MBv2 ────────────────────────────────────────
    with col_r:
        if not mbv2:
            _render_dws_formulas(M, N, Dk, H, W, d_p, d_f)
        else:
            _render_mbv2_formulas(M, N, Dk, H, W, t, cmp_p, cmp_f)

    st.markdown("---")

    # ── Ratio derivation ─────────────────────────────────────────────────
    if not mbv2:
        _render_dws_ratio(M, N, Dk, r_th, ratio_p, ratio_f)
    else:
        _render_mbv2_ratio(M, N, Dk, t, r_th, ratio_p, ratio_f)


def _render_dws_formulas(M, N, Dk, H, W, d_p, d_f):
    st.markdown('<h3 class="grad-heading">Depthwise Separable Conv.</h3>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="formula-card"><strong>Stage 1 — Depthwise</strong>: '
        'one Dk×Dk spatial filter per input channel (groups=M). '
        'No cross-channel mixing yet.</div>', unsafe_allow_html=True)
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


def _render_mbv2_formulas(M, N, Dk, H, W, t, cmp_p, cmp_f):
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
        f'spatial filtering in the expanded high-dimensional space.</div>',
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
        f'compresses back to target width N. No activation (linear bottleneck).</div>',
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
    st.latex(rf"\text{{FLOPs}}_{{\text{{MBv2}}}} = \mathbf{{{fmt(cmp_f['total'])}}}")


def _render_dws_ratio(M, N, Dk, r_th, ratio_p, ratio_f):
    st.markdown(
        '<h3 class="grad-heading" style="text-align:center;">'
        "Efficiency Reduction Ratio  (Howard et al., MobileNets 2017)"
        "</h3>", unsafe_allow_html=True)

    col_eq, col_num = st.columns([1.5, 0.5])
    with col_eq:
        st.latex(
            r"\frac{\text{Cost}_{\text{DWS}}}{\text{Cost}_{\text{Std}}} "
            r"= \frac{D_K^2 \cdot M \cdot H \cdot W + M \cdot N \cdot H \cdot W}"
            r"{D_K^2 \cdot M \cdot N \cdot H \cdot W}")
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
            f"<strong>{(1 - r_th) * 100:.2f}%</strong> reduction. "
            f"As N grows the 1/N term vanishes and savings asymptotically "
            f"approach (1 − 1/D_K²)."
            f"</div>", unsafe_allow_html=True)
    with col_num:
        st.metric("Theory Ratio", f"{r_th:.5f}",  help="1/N + 1/Dk²")
        st.metric("Param Ratio",  f"{ratio_p:.5f}", help="DWS params / Std params")
        st.metric("FLOP Ratio",   f"{ratio_f:.5f}", help="DWS FLOPs / Std FLOPs")
        match = abs(r_th - ratio_p) < 1e-9
        st.success("Ratios match analytically") if match else st.error("Ratio mismatch")
        st.progress(float(np.clip(r_th, 0.0, 1.0)))
        st.caption(f"{r_th * 100:.2f}% of Std. cost")


def _render_mbv2_ratio(M, N, Dk, t, r_th, ratio_p, ratio_f):
    tM = t * M
    st.markdown(
        '<h3 class="grad-heading" style="text-align:center;">'
        "Cost Ratio vs. Standard Conv  (Sandler et al., MobileNetV2 2018)"
        "</h3>", unsafe_allow_html=True)

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
                f"Ratio <strong>{r_th:.4f} &gt; 1</strong>: this MBv2 block uses "
                f"<strong>{(r_th - 1)*100:.1f}% more</strong> parameters than a "
                f"plain {Dk}&times;{Dk} standard conv. MobileNetV2's benefit is "
                f"<em>architectural</em>: the inverted bottleneck maintains a rich "
                f"high-dimensional feature space and the residual connection "
                f"(when M = N) preserves gradient flow — not captured by raw "
                f"parameter counts. Reduce t or increase N to bring ratio below 1."
                f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="insight-box">'
                f"Ratio <strong>{r_th:.4f} &lt; 1</strong> — this configuration "
                f"is also cheaper in raw parameters than a standard conv."
                f"</div>", unsafe_allow_html=True)
    with col_num:
        st.metric("Ratio (MBv2/Std)", f"{r_th:.5f}", help="t(M+Dk²+N)/(Dk²·N)")
        st.metric("Param Ratio",      f"{ratio_p:.5f}", help="actual params ratio")
        st.metric("FLOP Ratio",       f"{ratio_f:.5f}", help="actual FLOPs ratio")
        match = abs(r_th - ratio_p) < 1e-9
        st.success("Formula matches exactly") if match else st.error("Ratio mismatch")
        skip_status = "Yes  (M = N)" if M == N else "No  (M ≠ N)"
        st.metric("Residual Skip", skip_status)
        st.progress(float(np.clip(r_th, 0.0, 1.0)))
        st.caption(f"{r_th * 100:.2f}% of Std. cost")


# ---------------------------------------------------------------------------
# Tab 3 — PyTorch CPU Sandbox
# ---------------------------------------------------------------------------

def render_tab3(params: dict, results: dict) -> None:
    """Delegate to the isolated torch_sandbox renderer."""
    render_sandbox(params, results)
