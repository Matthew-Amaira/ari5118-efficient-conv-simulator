"""
core/math_engine.py
-------------------
Pure analytical computation layer — zero Streamlit dependencies.

All functions operate on plain Python integers/floats using NumPy where
needed.  No GPU, no autograd, no side effects.

Public API
----------
std_params / std_flops           — standard 2-D convolution
dws_params / dws_flops           — depthwise separable factorisation
reduction_ratio                  — Howard et al. 2017 closed-form ratio
mbv2_params / mbv2_flops         — MobileNetV2 inverted residual block
mbv2_reduction_ratio             — Sandler et al. 2018 cost ratio
fmt                              — human-readable magnitude string (K/M/B)
_nearest_snap                    — round-half-up snap to slider tick grid
compute_all                      — convenience wrapper; returns a single
                                   results dict consumed by all UI layers
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Standard 2-D Convolution
# ---------------------------------------------------------------------------

def std_params(M: int, N: int, Dk: int) -> int:
    """Learnable parameters: Dk² × M × N  (no bias term)."""
    return int(Dk * Dk * M * N)


def std_flops(M: int, N: int, Dk: int, H: int, W: int) -> int:
    """Forward-pass FLOPs: 2 × Dk² × M × N × H × W  (same-padding)."""
    return int(2 * Dk * Dk * M * N * H * W)


# ---------------------------------------------------------------------------
# Depthwise Separable Convolution  (Howard et al., MobileNets 2017)
# ---------------------------------------------------------------------------

def dws_params(M: int, N: int, Dk: int) -> dict:
    """
    Parameter split for the two-stage DWS factorisation.

    Stage 1 — Depthwise : Dk² × M   (one spatial filter per input channel)
    Stage 2 — Pointwise : M  × N    (1×1 cross-channel mixer)
    """
    dw = int(Dk * Dk * M)
    pw = int(M * N)
    return {"dw": dw, "pw": pw, "total": dw + pw}


def dws_flops(M: int, N: int, Dk: int, H: int, W: int) -> dict:
    """FLOPs split by DWS stage."""
    dw = int(2 * Dk * Dk * M * H * W)
    pw = int(2 * M * N * H * W)
    return {"dw": dw, "pw": pw, "total": dw + pw}


def reduction_ratio(N: int, Dk: int) -> float:
    """
    Closed-form efficiency ratio:  Cost_DWS / Cost_Std = 1/N + 1/Dk²

    Derived by cancelling the M·H·W factor common to both cost expressions.
    The ratio is independent of M and the spatial dimensions H, W.
    """
    return (1.0 / N) + (1.0 / (Dk ** 2))


# ---------------------------------------------------------------------------
# MobileNetV2 Inverted Residual Bottleneck  (Sandler et al., 2018)
# ---------------------------------------------------------------------------
# Three sequential operations:
#   ① Expand   : 1×1 PW conv   M  → t·M   (channel widening)
#   ② Depthwise: Dk×Dk DW conv t·M        (spatial filtering in high-dim space)
#   ③ Project  : 1×1 PW conv   t·M → N   (channel compression — linear bottleneck)
# Residual skip is added when M == N (stride-1 blocks only).

def mbv2_params(M: int, N: int, Dk: int, t: int) -> dict:
    """
    Parameter counts per MBv2 stage.

    Expand  : M  × (t·M)  = t·M²
    DW      : Dk²× (t·M)
    Project : (t·M) × N   = t·M·N
    Total   : t·M·(M + Dk² + N)
    """
    tM      = t * M
    expand  = M  * tM
    dw      = Dk * Dk * tM
    project = tM * N
    return {
        "expand":  expand,
        "dw":      dw,
        "project": project,
        "total":   expand + dw + project,
    }


def mbv2_flops(M: int, N: int, Dk: int, H: int, W: int, t: int) -> dict:
    """
    FLOPs (2 × MACs) per MBv2 stage.

    Expand  : 2·M·(t·M)·H·W
    DW      : 2·Dk²·(t·M)·H·W
    Project : 2·(t·M)·N·H·W
    Total   : 2·t·M·H·W·(M + Dk² + N)
    """
    tM      = t * M
    expand  = int(2 * M  * tM * H * W)
    dw      = int(2 * Dk * Dk * tM * H * W)
    project = int(2 * tM * N  * H * W)
    return {
        "expand":  expand,
        "dw":      dw,
        "project": project,
        "total":   expand + dw + project,
    }


def mbv2_reduction_ratio(M: int, N: int, Dk: int, t: int) -> float:
    """
    Cost ratio of one MBv2 block vs an equivalent standard conv (same I/O).

    Ratio = t·(M + Dk² + N) / (Dk²·N)

    Values > 1 indicate MBv2 costs MORE parameters than a plain Dk×Dk conv.
    The advantage of MBv2 is architectural quality (residual gradient flow,
    high-dimensional feature space for DW), not raw parameter compression.
    """
    return (t * (M + Dk ** 2 + N)) / (Dk ** 2 * N)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def fmt(n: int) -> str:
    """Human-readable magnitude string: 1 234 567 → '1.235 M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.3f} M"
    if n >= 1_000:
        return f"{n / 1e3:.2f} K"
    return str(n)


def _nearest_snap(v: int, lo: int, step: int, hi: int) -> int:
    """
    Round v to the nearest valid slider tick (lo + n·step), clamped to [lo, hi].

    Uses round-half-up (not Python's banker's rounding) so that ties always
    round away from lo.  This keeps the slider visually consistent with the
    value the user typed into the number-input box.
    """
    n = int((v - lo + step // 2) // step)
    return max(lo, min(hi, lo + n * step))


# ---------------------------------------------------------------------------
# Convenience batch computation
# ---------------------------------------------------------------------------

def compute_all(params: dict) -> dict:
    """
    Derive every metric needed by the UI from the raw hyperparameter dict.

    Parameters
    ----------
    params : dict with keys M, N, Dk, H, W, t, mbv2_mode

    Returns
    -------
    dict with keys:
        s_p, s_f            — standard conv params / FLOPs
        d_p, d_f            — DWS params / FLOPs  (always computed for Tab 3)
        cmp_p, cmp_f        — comparison-architecture params / FLOPs
        cmp_label           — short label string for chart axes
        r_th                — theoretical cost ratio
        ratio_p, ratio_f    — actual cmp / std ratios
        saved_p, saved_f    — % savings (negative = overhead)
    """
    M, N, Dk = params["M"], params["N"], params["Dk"]
    H, W, t  = params["H"], params["W"], params["t"]
    mbv2     = params["mbv2_mode"]

    s_p = std_params(M, N, Dk)
    s_f = std_flops(M, N, Dk, H, W)

    # DWS always computed — Tab 3 PyTorch sandbox references d_p / d_f
    # regardless of active architecture mode.
    d_p = dws_params(M, N, Dk)
    d_f = dws_flops(M, N, Dk, H, W)

    if mbv2:
        cmp_p     = mbv2_params(M, N, Dk, t)
        cmp_f     = mbv2_flops(M, N, Dk, H, W, t)
        cmp_label = "MobileNetV2"
        r_th      = mbv2_reduction_ratio(M, N, Dk, t)
    else:
        cmp_p     = d_p
        cmp_f     = d_f
        cmp_label = "DW Separable"
        r_th      = reduction_ratio(N, Dk)

    ratio_p = cmp_p["total"] / s_p
    ratio_f = cmp_f["total"] / s_f
    saved_p = (1 - ratio_p) * 100
    saved_f = (1 - ratio_f) * 100

    return {
        "s_p": s_p, "s_f": s_f,
        "d_p": d_p, "d_f": d_f,
        "cmp_p": cmp_p, "cmp_f": cmp_f,
        "cmp_label": cmp_label,
        "r_th": r_th,
        "ratio_p": ratio_p, "ratio_f": ratio_f,
        "saved_p": saved_p, "saved_f": saved_f,
    }
