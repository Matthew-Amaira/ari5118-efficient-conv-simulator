"""
core/torch_sandbox.py
---------------------
Isolated PyTorch CPU validation sandbox renderer.

Renders the entire Tab 3 content: a side-by-side code walkthrough and a
live execution panel that instantiates real torch.nn layers and verifies
that tensor shapes and parameter counts match the analytical formulas.

Hard constraint: device = "cpu" is hard-coded throughout.  No CUDA, no MPS,
no GPU dependency of any kind.

The entire execution block is wrapped in a try/except ImportError so that an
absent PyTorch installation degrades gracefully without breaking Tabs 1 or 2.
"""

from __future__ import annotations

import streamlit as st


def render_sandbox(params: dict, results: dict) -> None:
    """
    Render the PyTorch CPU verification sandbox inside the active Streamlit tab.

    Parameters
    ----------
    params  : dict — M, N, Dk, H, W, t, mbv2_mode
    results : dict — pre-computed metrics from core.math_engine.compute_all
    """
    M, N, Dk = params["M"], params["N"], params["Dk"]
    H, W     = params["H"], params["W"]
    s_p      = results["s_p"]
    d_p      = results["d_p"]

    from core.math_engine import fmt  # local import to avoid circular deps

    st.markdown('<h3 class="grad-heading">PyTorch CPU Verification Sandbox</h3>',
                unsafe_allow_html=True)
    st.markdown(
        f"Runs actual `torch.nn` layers on **CPU only** (`device='cpu'` hard-coded) "
        f"with a dummy input of shape `[1, {M}, {H}, {W}]` — driven by the active "
        f"sidebar sliders.  Verifies that tensor shapes and parameter counts match "
        f"the analytical formulas exactly."
    )
    st.markdown("---")

    col_code, col_run = st.columns([1.6, 0.4])

    with col_code:
        st.code(
            f'import torch\n'
            f'import torch.nn as nn\n\n'
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
            f'dw_layer = nn.Conv2d(       # groups=M isolates channels\n'
            f'    in_channels={M},\n'
            f'    out_channels={M},\n'
            f'    kernel_size={Dk},\n'
            f'    padding={Dk // 2},\n'
            f'    groups={M},\n'
            f'    bias=False,\n'
            f')\n'
            f'pw_layer = nn.Conv2d(       # 1×1 mixes channels\n'
            f'    in_channels={M},\n'
            f'    out_channels={N},\n'
            f'    kernel_size=1,\n'
            f'    bias=False,\n'
            f')\n\n'
            f'with torch.no_grad():\n'
            f'    out_std = std_layer(x)\n'
            f'    inter   = dw_layer(x)\n'
            f'    out_dws = pw_layer(inter)\n\n'
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

    if run:
        st.markdown("---")
        _execute_validation(M, N, Dk, H, W, s_p, d_p, fmt)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _execute_validation(
    M: int, N: int, Dk: int, H: int, W: int,
    s_p: int, d_p: dict, fmt_fn
) -> None:
    """Attempt live PyTorch execution; degrade gracefully if torch absent."""
    try:
        import torch
        import torch.nn as nn

        device = "cpu"
        x = torch.randn(1, M, H, W, device=device)

        std_layer = nn.Conv2d(M, N, Dk, padding=Dk // 2, bias=False)
        dw_layer  = nn.Conv2d(M, M, Dk, padding=Dk // 2, groups=M, bias=False)
        pw_layer  = nn.Conv2d(M, N, 1, bias=False)

        with torch.no_grad():
            out_std = std_layer(x)
            inter   = dw_layer(x)
            out_dws = pw_layer(inter)

        p_std = sum(p.numel() for p in std_layer.parameters())
        p_dw  = sum(p.numel() for p in dw_layer.parameters())
        p_pw  = sum(p.numel() for p in pw_layer.parameters())
        p_dws = p_dw + p_pw

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
                f"Torch params : `{fmt_fn(p_std)}`  "
                f"[{'OK' if param_ok_std else 'FAIL'}]\n\n"
                f"Formula      : `{fmt_fn(s_p)}`"
            )
        with col_b:
            st.markdown("#### Depthwise Separable Conv.")
            st.info(
                f"Output shape  : `{tuple(out_dws.shape)}`  "
                f"[{'OK' if shape_ok_dws else 'FAIL'}]\n\n"
                f"Torch params  : `{fmt_fn(p_dws)}`  "
                f"[{'OK' if param_ok_dws else 'FAIL'}]\n\n"
                f"Formula       : `{fmt_fn(d_p['total'])}`\n\n"
                f"  DW layer    : `{fmt_fn(p_dw)}`  (formula `{fmt_fn(d_p['dw'])}`)\n\n"
                f"  PW layer    : `{fmt_fn(p_pw)}`  (formula `{fmt_fn(d_p['pw'])}`)"
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
                "All checks passed.  "
                "PyTorch CPU parameter counts match the analytical formulas exactly."
            )
        else:
            st.error("One or more checks failed — review the result cards above.")

    except ImportError:
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
