"""
visualization/animation_engine.py
----------------------------------
Responsive HTML5 Canvas animation panel for the Conv Efficiency Simulator.

Rendered via st.components.v1.html — zero external JS dependencies.

Design contract
---------------
1. Zero hardcoded pixel anchors.  Every coordinate is derived from CW/CH and
   the three equal column widths (COL_W = floor(CW/3)).
2. Unified clock.  A single delta-time accumulator (subTs, 0..1 per step)
   drives ALL visual motion: kernel sweep, pointwise wave, arrow pulse, panel
   glow.  The raw frame counter is never used for animation logic.
3. Pointwise mixing visualisation.  The centre panel shows a column of
   stacked channel bars that animate as a sequential wave (indexed by subTs).
   Bezier convergence lines shoot from each bar's right edge to the active
   output-cell centre — lit/dimmed in sync with their bar's wave peak.
4. Neon accent glows.  ctx.shadowBlur + per-phase accent colours (#6366f1
   standard, #db2777 depthwise, #059669 pointwise, #ca8a04 expand) give
   bright glow rings around active sweep kernels and output cells.

Canvas layout  (680 × 390 px)
──────────────────────────────
  [0 .. 72 px]  Phase label bar + indicator dots + progress strip
  [90 .. 370 px] Three equal columns, grids/panel top-aligned at GRID_TOP=90
    Col 0  [0   .. 226]  Input Grid
    Col 1  [226 .. 452]  Kernel Panel (sweep: Dk×Dk; pointwise: channel bars)
    Col 2  [452 .. 680]  Output Grid

Animation phases
────────────────
DWS mode  : Standard Conv sweep → DWS Depthwise sweep → DWS Pointwise mix
MBv2 mode : Expand PW mix       → Depthwise sweep      → Project PW mix
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_animation_panel(params: dict, mbv2_mode: bool) -> None:
    """Inject the interactive canvas animation inside a Streamlit expander."""
    with st.expander(
        "🎬  Interactive Convolution Animation  (click to expand)",
        expanded=False,
    ):
        mode = "mbv2" if mbv2_mode else "dws"
        st.markdown(
            "<p style='font-size:0.82rem;opacity:0.58;margin-bottom:0.5rem;'>"
            "Watch the kernel sweep across the feature map in real time. "
            "Phases auto-advance and loop continuously. "
            "Use the controls to pause or adjust speed."
            "</p>",
            unsafe_allow_html=True,
        )
        components.html(_build_html(params, mode), height=460, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# HTML / JS builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(params: dict, mode: str) -> str:
    """Return a self-contained HTML document with the canvas animation."""
    M  = int(params["M"])
    N  = int(params["N"])
    Dk = int(params["Dk"])
    H  = int(params["H"])
    t  = int(params["t"])
    tM = t * M

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px;
    overflow: hidden;
  }}
  canvas {{ display: block; border-radius: 10px; }}
  #controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 7px;
    padding: 7px 14px;
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #334155;
    width: 680px;
  }}
  button {{
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    min-width: 74px;
    white-space: nowrap;
  }}
  button:hover {{ opacity: 0.85; }}
  #resetBtn {{ background: #334155; min-width: 58px; }}
  #phaseInfo {{ font-size: 10.5px; color: #64748b; white-space: nowrap; }}
  label {{
    font-size: 10.5px;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
    white-space: nowrap;
  }}
  input[type=range] {{ accent-color: #0ea5e9; width: 88px; }}
</style>
</head>
<body>
<canvas id="anim" width="680" height="390"></canvas>
<div id="controls">
  <button id="playPauseBtn">⏸ Pause</button>
  <button id="resetBtn">↺ Reset</button>
  <span id="phaseInfo">Phase 1 / 3</span>
  <label>Speed <input type="range" id="speedSlider" min="1" max="12" value="5"></label>
</div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// §1 · INJECTED PYTHON HYPERPARAMETERS
// ═══════════════════════════════════════════════════════════════════════════
const M    = {M};
const N    = {N};
const Dk   = {Dk};
const H    = {H};
const tM   = {tM};
const MODE = "{mode}";   // "dws" | "mbv2"

// ═══════════════════════════════════════════════════════════════════════════
// §2 · ADAPTIVE LAYOUT GEOMETRY  (zero hardcoded pixel anchors)
// ═══════════════════════════════════════════════════════════════════════════
const CW = 680, CH = 390;

// ── Vertical zones ──────────────────────────────────────────────────────────
const PB_H      = 72;          // phase-bar height (label + dots + progress)
const GRID_TOP  = 90;          // grids/panel start Y  (leaves 18 px for labels)
const GRID_BOT  = CH - 20;     // grids/panel end Y    (leaves 20 px for counters)
const GRID_AVH  = GRID_BOT - GRID_TOP;   // 280 px available for content

// ── Horizontal thirds ────────────────────────────────────────────────────────
const COL_W   = Math.floor(CW / 3);   // ≈226 px per column
const COL_PAD = 9;                     // inner horizontal padding
const COL_USW = COL_W - COL_PAD * 2;  // usable column width ≈208 px

// ── Grid display caps ────────────────────────────────────────────────────────
const DISP     = Math.min(H, 8);               // display grid dim (max 8)
const VALID_SW = Math.max(1, DISP - Dk + 1);  // valid sweep positions / axis
const TOTAL_SW = VALID_SW * VALID_SW;
const TOTAL_PW = DISP * DISP;

// ── Cell sizes  (fit within column + height; cap at MAX_CELL) ────────────────
const MAX_CELL = 36;
const CELL_IN  = Math.min(MAX_CELL,
                   Math.max(4, Math.floor(Math.min(COL_USW, GRID_AVH) / DISP)));
const CELL_SW  = Math.min(MAX_CELL,
                   Math.max(4, Math.floor(Math.min(COL_USW, GRID_AVH) / Math.max(VALID_SW, 1))));

// ── Grid pixel spans ─────────────────────────────────────────────────────────
const LG_SPAN = CELL_IN * DISP;       // input grid w/h in px
const SW_SPAN = CELL_SW * VALID_SW;   // sweep output w/h in px

// ── Column 0: Input grid — centred in col 0, top-aligned ────────────────────
const LG_X = Math.floor((COL_W - LG_SPAN) / 2);
const LG_Y = GRID_TOP;

// ── Column 2: Sweep output — centred in col 2 ───────────────────────────────
const SW_X = COL_W * 2 + Math.floor((COL_W - SW_SPAN) / 2);
const SW_Y = GRID_TOP;

// ── Column 2: Pointwise output — same cell size as input ────────────────────
const PW_X = COL_W * 2 + Math.floor((COL_W - LG_SPAN) / 2);
const PW_Y = GRID_TOP;

// ── Column 1: Kernel panel ───────────────────────────────────────────────────
const KP_X = COL_W + COL_PAD;
const KP_Y = GRID_TOP;
const KP_W = COL_W - COL_PAD * 2;
const KP_H = GRID_AVH;

// ═══════════════════════════════════════════════════════════════════════════
// §3 · COLOUR PALETTE
// ═══════════════════════════════════════════════════════════════════════════
const C = {{
  bg:       "#0f172a",
  surface:  "#1e293b",
  border:   "#334155",
  cellBase: "#283548",
  cellDone: "rgba(34,197,94,0.36)",
  text:     "#cbd5e1",
  sub:      "#64748b",
}};

// Neon accent per operation type
const ACCENT = {{
  std: "#6366f1",   // standard conv  — indigo
  dw:  "#db2777",   // depthwise      — pink
  pw:  "#059669",   // pointwise      — emerald
  exp: "#ca8a04",   // expand         — amber
}};

// ═══════════════════════════════════════════════════════════════════════════
// §4 · PHASE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════
function buildPhases() {{
  if (MODE === "mbv2") {{
    return [
      {{ type:"pointwise", color:ACCENT.exp,
         label:`① Expand — 1×1 PW · M=${{M}} → tM=${{tM}} (t=${{Math.round(tM/Math.max(M,1))}})`,
         inNote:`${{H}}×${{H}}×${{M}}`, outNote:`${{H}}×${{H}}×${{tM}}`,
         kNote:`1×1 · ${{M}}→${{tM}}`, inCh:M, outCh:tM }},
      {{ type:"sweep",     color:ACCENT.dw,
         label:`② Depthwise — ${{Dk}}×${{Dk}} per channel · ${{tM}} independent spatial filters`,
         inNote:`${{H}}×${{H}}×${{tM}}`, outNote:`${{H}}×${{H}}×${{tM}}`,
         kNote:`${{Dk}}×${{Dk}} · ${{tM}} ch`, inCh:tM, outCh:tM }},
      {{ type:"pointwise", color:ACCENT.pw,
         label:`③ Project — 1×1 PW · tM=${{tM}} → N=${{N}} (linear bottleneck)`,
         inNote:`${{H}}×${{H}}×${{tM}}`, outNote:`${{H}}×${{H}}×${{N}}`,
         kNote:`1×1 · ${{tM}}→${{N}}`, inCh:tM, outCh:N }},
    ];
  }} else {{
    return [
      {{ type:"sweep",     color:ACCENT.std,
         label:`Standard Conv — ${{Dk}}×${{Dk}} kernel · ALL ${{M}} input channels fused simultaneously`,
         inNote:`${{H}}×${{H}}×${{M}}`, outNote:`${{H}}×${{H}}×${{N}}`,
         kNote:`${{Dk}}×${{Dk}}×${{M}}×${{N}}`, inCh:M, outCh:N }},
      {{ type:"sweep",     color:ACCENT.dw,
         label:`① DWS Depthwise — ${{Dk}}×${{Dk}} per channel · ${{M}} independent spatial filters`,
         inNote:`${{H}}×${{H}}×${{M}}`, outNote:`${{H}}×${{H}}×${{M}}`,
         kNote:`${{Dk}}×${{Dk}} · ${{M}} ch`, inCh:M, outCh:M }},
      {{ type:"pointwise", color:ACCENT.pw,
         label:`② DWS Pointwise — 1×1 mix · M=${{M}} → N=${{N}} channels`,
         inNote:`${{H}}×${{H}}×${{M}}`, outNote:`${{H}}×${{H}}×${{N}}`,
         kNote:`1×1 · ${{M}}→${{N}}`, inCh:M, outCh:N }},
    ];
  }}
}}

const phases = buildPhases();

// ═══════════════════════════════════════════════════════════════════════════
// §5 · ANIMATION STATE
// ═══════════════════════════════════════════════════════════════════════════
let phaseIdx = 0;    // active phase index (0 .. phases.length-1)
let stepPos  = 0;    // discrete step within the current phase
let subTs    = 0;    // fractional sub-step progress [0..1), delta-time driven
let lastTs   = 0;    // previous rAF timestamp (ms)
let running  = true;
let speed    = 5;    // speed slider value (1..12)

const canvas = document.getElementById("anim");
const ctx    = canvas.getContext("2d");

// ═══════════════════════════════════════════════════════════════════════════
// §6 · DRAWING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/** Trace a rounded-rectangle path. Automatically clamps r to half cell. */
function roundRect(x, y, w, h, r) {{
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y,     x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x,     y + h, r);
  ctx.arcTo(x,     y + h, x,     y,     r);
  ctx.arcTo(x,     y,     x + w, y,     r);
  ctx.closePath();
}}

/** Convert a "#rrggbb" hex string + float alpha [0..1] → "rgba(r,g,b,a)". */
function hexA(hex, a) {{
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${{r}},${{g}},${{b}},${{Math.max(0, Math.min(1, a)).toFixed(3)}})`;
}}

/**
 * Draw an animated dashed arrow from (x1,y1) to (x2,y2).
 * alpha: overall opacity driven by the caller (should be a subTs function).
 */
function drawArrow(x1, y1, x2, y2, color, alpha) {{
  if (alpha < 0.02) return;
  ctx.save();
  ctx.strokeStyle = hexA(color, alpha);
  ctx.lineWidth   = 1.6;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = hexA(color, alpha * 0.9);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 10 * Math.cos(a - 0.38), y2 - 10 * Math.sin(a - 0.38));
  ctx.lineTo(x2 - 10 * Math.cos(a + 0.38), y2 - 10 * Math.sin(a + 0.38));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}}

// ═══════════════════════════════════════════════════════════════════════════
// §7 · PHASE BAR  (top strip: label + indicator dots + progress strip)
// ═══════════════════════════════════════════════════════════════════════════
function drawPhaseBar(phase, progress) {{
  // Background pill
  ctx.fillStyle = C.surface;
  roundRect(8, 6, CW - 16, 44, 8);
  ctx.fill();
  ctx.strokeStyle = hexA(phase.color, 0.65);
  ctx.lineWidth   = 1.5;
  roundRect(8, 6, CW - 16, 44, 8);
  ctx.stroke();

  // Phase label (maxWidth prevents overflow)
  ctx.fillStyle    = phase.color;
  ctx.font         = "bold 11px -apple-system, sans-serif";
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(phase.label, CW / 2, 28, CW - 48);

  // Indicator dots — one per phase
  const dotDx = 20;
  const dotsW = phases.length * dotDx - 6;
  const dotX0 = (CW - dotsW) / 2;
  const dotY  = 57;
  for (let i = 0; i < phases.length; i++) {{
    const active = (i === phaseIdx);
    ctx.fillStyle = active ? phases[i].color : C.border;
    ctx.beginPath();
    ctx.arc(dotX0 + i * dotDx + 5, dotY, active ? 5.5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  }}

  // Progress strip (smooth due to subTs in the progress fraction)
  const barY = 66, barH = 4;
  ctx.fillStyle = C.border;
  ctx.fillRect(18, barY, CW - 36, barH);
  ctx.fillStyle = phase.color;
  ctx.fillRect(18, barY, (CW - 36) * Math.min(1, Math.max(0, progress)), barH);
}}

// ═══════════════════════════════════════════════════════════════════════════
// §8 · GRID RENDERER  (handles both input and output grids)
// ═══════════════════════════════════════════════════════════════════════════
/**
 * Draw a dim×dim grid of cells.
 *   activeCells : Set of flat indices shown with accent-colour fill
 *   doneCells   : Set of flat indices shown in completed-green
 *   glowKernel  : if truthy, draw a neon glow bounding box around activeCells
 */
function drawGrid(gx, gy, dim, cell, activeCells, doneCells, color, glowKernel) {{
  const span = dim * cell;

  // Dark background for the entire grid area
  ctx.fillStyle = "#1a2538";
  roundRect(gx - 1, gy - 1, span + 2, span + 2, 4);
  ctx.fill();

  // Individual cell fills
  for (let r = 0; r < dim; r++) {{
    for (let c = 0; c < dim; c++) {{
      const idx = r * dim + c;
      const px  = gx + c * cell;
      const py  = gy + r * cell;

      if (doneCells.has(idx)) {{
        ctx.fillStyle = C.cellDone;
      }} else if (activeCells.has(idx)) {{
        ctx.fillStyle = hexA(color, 0.70);
      }} else {{
        ctx.fillStyle = C.cellBase;
      }}
      roundRect(px + 1, py + 1, cell - 2, cell - 2, 2);
      ctx.fill();
    }}
  }}

  // Thin grid-line separators (only when cells are large enough to show them)
  if (cell >= 8) {{
    ctx.strokeStyle = hexA("#64748b", 0.25);
    ctx.lineWidth   = 0.5;
    for (let r = 1; r < dim; r++) {{
      ctx.beginPath();
      ctx.moveTo(gx, gy + r * cell);
      ctx.lineTo(gx + span, gy + r * cell);
      ctx.stroke();
    }}
    for (let c = 1; c < dim; c++) {{
      ctx.beginPath();
      ctx.moveTo(gx + c * cell, gy);
      ctx.lineTo(gx + c * cell, gy + span);
      ctx.stroke();
    }}
  }}

  // Outer border
  ctx.strokeStyle = hexA(color, 0.32);
  ctx.lineWidth   = 1.2;
  roundRect(gx - 1, gy - 1, span + 2, span + 2, 4);
  ctx.stroke();

  // Neon glow bounding box around all active cells
  if (glowKernel && activeCells.size > 0) {{
    let minR = dim, maxR = 0, minC = dim, maxC = 0;
    for (const idx of activeCells) {{
      const r = Math.floor(idx / dim), c = idx % dim;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
    }}
    const gx2 = gx + minC * cell;
    const gy2 = gy + minR * cell;
    const gw2 = (maxC - minC + 1) * cell;
    const gh2 = (maxR - minR + 1) * cell;

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur  = 16;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    roundRect(gx2, gy2, gw2, gh2, 3);
    ctx.stroke();
    ctx.shadowBlur  = 0;
    ctx.fillStyle   = hexA(color, 0.11);
    roundRect(gx2, gy2, gw2, gh2, 3);
    ctx.fill();
    ctx.restore();
  }}
}}

// ═══════════════════════════════════════════════════════════════════════════
// §9 · GRID LABELS  (two lines centred above each grid)
// ═══════════════════════════════════════════════════════════════════════════
function gridLabel(gx, gSpan, topLine, botLine, color) {{
  const cx = gx + gSpan / 2;
  ctx.textAlign    = "center";
  ctx.textBaseline = "bottom";

  ctx.fillStyle = color;
  ctx.font      = "bold 10px sans-serif";
  ctx.fillText(topLine, cx, GRID_TOP - 11, gSpan + 12);

  ctx.fillStyle = C.sub;
  ctx.font      = "9px monospace";
  ctx.fillText(botLine, cx, GRID_TOP - 1, gSpan + 12);
}}

// ═══════════════════════════════════════════════════════════════════════════
// §10 · SWEEP KERNEL PANEL
//   Shows a Dk×Dk kernel grid that pulses gently via subTs.
// ═══════════════════════════════════════════════════════════════════════════
function drawSweepKernelPanel(phase) {{
  // Panel background
  ctx.fillStyle = C.surface;
  roundRect(KP_X, KP_Y, KP_W, KP_H, 9);
  ctx.fill();
  ctx.strokeStyle = hexA(phase.color, 0.45);
  ctx.lineWidth   = 1.5;
  roundRect(KP_X, KP_Y, KP_W, KP_H, 9);
  ctx.stroke();

  // Panel title
  ctx.fillStyle    = C.text;
  ctx.font         = "bold 10px monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Kernel Filter", KP_X + KP_W / 2, KP_Y + 7, KP_W - 8);

  // Dk×Dk kernel cell grid (capped at 7 to stay readable)
  const kDisp  = Math.min(Dk, 7);
  const maxKSz = Math.min(KP_W - 24, KP_H - 50);
  const kCell  = Math.max(6, Math.floor(maxKSz / kDisp));
  const kgW    = kCell * kDisp;
  const kgX    = KP_X + (KP_W - kgW) / 2;
  const kgY    = KP_Y + 24;
  // subTs-driven pulse: brightness oscillates between 0.5 and 1.0
  const pulse  = 0.5 + 0.5 * Math.abs(Math.sin(subTs * Math.PI));

  for (let r = 0; r < kDisp; r++) {{
    for (let c = 0; c < kDisp; c++) {{
      ctx.fillStyle   = hexA(phase.color, 0.18 + 0.42 * pulse);
      roundRect(kgX + c * kCell + 1, kgY + r * kCell + 1, kCell - 2, kCell - 2, 2);
      ctx.fill();
      ctx.strokeStyle = hexA(phase.color, 0.50 * pulse);
      ctx.lineWidth   = 0.9;
      roundRect(kgX + c * kCell + 1, kgY + r * kCell + 1, kCell - 2, kCell - 2, 2);
      ctx.stroke();
    }}
  }}

  // Bottom label
  ctx.fillStyle    = "#e2e8f0";
  ctx.font         = "bold 9.5px monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(phase.kNote,     KP_X + KP_W / 2, KP_Y + KP_H - 15, KP_W - 8);
  ctx.fillStyle = C.sub;
  ctx.font      = "8.5px monospace";
  ctx.fillText("spatial sweep", KP_X + KP_W / 2, KP_Y + KP_H -  3, KP_W - 8);
}}

// ═══════════════════════════════════════════════════════════════════════════
// §11 · POINTWISE KERNEL PANEL  (channel-stack + bezier convergence lines)
//
//   Shows NUM_BARS stacked bars (one per input channel, capped at 8).
//   A brightness wave travels top-to-bottom as subTs advances 0→1.
//   Each bar fires a bezier line toward (outCellCx, outCellCy) — the canvas
//   centre of the currently active output cell — at an opacity proportional
//   to that bar's wave intensity.  ALL timing is derived from subTs.
// ═══════════════════════════════════════════════════════════════════════════
function drawPointwiseKernelPanel(phase, outCellCx, outCellCy) {{
  // Panel background
  ctx.fillStyle = C.surface;
  roundRect(KP_X, KP_Y, KP_W, KP_H, 9);
  ctx.fill();
  ctx.strokeStyle = hexA(phase.color, 0.45);
  ctx.lineWidth   = 1.5;
  roundRect(KP_X, KP_Y, KP_W, KP_H, 9);
  ctx.stroke();

  // Panel title
  ctx.fillStyle    = C.text;
  ctx.font         = "bold 10px monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Channel Mix", KP_X + KP_W / 2, KP_Y + 7, KP_W - 8);

  // Channel bars
  const NUM_BARS = Math.max(1, Math.min(8, phase.inCh));
  const barAreaY = KP_Y + 22;
  const barAreaH = KP_H - 46;
  const slotH    = Math.max(8, Math.floor(barAreaH / NUM_BARS));
  const barH     = Math.max(5, slotH - 3);
  const barX     = KP_X + 12;
  const barW     = KP_W - 24;
  // Right edge of bar area — convergence lines originate here
  const barRightX = barX + barW;

  for (let ch = 0; ch < NUM_BARS; ch++) {{
    const barY = barAreaY + ch * slotH;

    // Wave crest passes each bar at subTs = (ch + 0.5) / NUM_BARS
    // Spread controls how wide the wave envelope is relative to the bar slot
    const crest  = (ch + 0.5) / NUM_BARS;
    const spread = 0.75 / NUM_BARS;
    const dist   = Math.abs(subTs - crest);
    // alpha: 0.08 baseline + 0.92 wave peak, clamped to [0, 1]
    const waveA  = Math.max(0, Math.min(1, 1 - dist / spread));
    const barA   = 0.08 + 0.92 * waveA;

    // Bar fill — width proportional to activity
    ctx.fillStyle = hexA(phase.color, barA * 0.75);
    ctx.fillRect(barX, barY, barW * Math.max(0.05, waveA), barH);

    // Bar border outline (always visible at low opacity)
    ctx.strokeStyle = hexA(phase.color, barA * 0.35 + 0.12);
    ctx.lineWidth   = 0.8;
    ctx.strokeRect(barX, barY, barW, barH);

    // Channel index label (only when bars are tall enough)
    if (barH >= 9) {{
      ctx.fillStyle    = hexA(C.text, barA * 0.65 + 0.15);
      ctx.font         = `${{Math.min(8, barH - 2)}}px monospace`;
      ctx.textAlign    = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`ch${{ch}}`, barX + 3, barY + barH / 2);
    }}

    // Bezier convergence line: bar right-edge → active output cell centre
    // Opacity tied directly to the wave peak for this bar.
    const lineA = waveA * 0.75;
    if (lineA > 0.04) {{
      const sx = barRightX;
      const sy = barY + barH / 2;
      const tx = outCellCx;
      const ty = outCellCy;
      // Control points: exit horizontally from bar, then bend toward target
      const cpx1 = sx + (tx - sx) * 0.40;
      const cpy1 = sy;
      const cpx2 = sx + (tx - sx) * 0.60;
      const cpy2 = ty;

      ctx.save();
      ctx.strokeStyle = hexA(phase.color, lineA);
      ctx.lineWidth   = 1.3;
      ctx.shadowColor = phase.color;
      ctx.shadowBlur  = 5;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, tx, ty);
      ctx.stroke();
      ctx.restore();
    }}
  }}

  // Bottom label
  ctx.fillStyle    = "#e2e8f0";
  ctx.font         = "bold 9.5px monospace";
  ctx.textAlign    = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(phase.kNote,    KP_X + KP_W / 2, KP_Y + KP_H - 15, KP_W - 8);
  ctx.fillStyle = C.sub;
  ctx.font      = "8.5px monospace";
  ctx.fillText("channel mix",  KP_X + KP_W / 2, KP_Y + KP_H -  3, KP_W - 8);
}}

// ═══════════════════════════════════════════════════════════════════════════
// §12 · SWEEP PHASE RENDERER
// ═══════════════════════════════════════════════════════════════════════════
function renderSweep(phase) {{
  const valid = VALID_SW;
  const pos   = stepPos % TOTAL_SW;
  const kr    = Math.floor(pos / valid);
  const kc    = pos % valid;

  // Input grid: cells covered by the sliding kernel window
  const inputActive = new Set();
  for (let dr = 0; dr < Dk; dr++)
    for (let dc = 0; dc < Dk; dc++)
      if (kr + dr < DISP && kc + dc < DISP)
        inputActive.add((kr + dr) * DISP + (kc + dc));

  // Output grid: previous positions done (green), current active (accent)
  const outDone   = new Set();
  const outActive = new Set([pos]);
  for (let p = 0; p < pos; p++) outDone.add(p);

  // Draw grids
  drawGrid(LG_X, LG_Y, DISP,  CELL_IN, inputActive, new Set(), phase.color, true);
  drawGrid(SW_X, SW_Y, valid,  CELL_SW, outActive,   outDone,   phase.color, true);

  // Labels above each grid
  gridLabel(LG_X, LG_SPAN, "Input",  phase.inNote,  phase.color);
  gridLabel(SW_X, SW_SPAN, "Output", phase.outNote,  phase.color);

  // Sweep kernel panel (centre)
  drawSweepKernelPanel(phase);

  // Arrows (pulse driven by subTs — in sync with the rest of the animation)
  const arrowA  = 0.30 + 0.55 * Math.abs(Math.sin(subTs * Math.PI));
  const lgMidY  = LG_Y + LG_SPAN / 2;
  const swMidY  = SW_Y + SW_SPAN / 2;
  drawArrow(LG_X + LG_SPAN + 3, lgMidY, KP_X - 3,       lgMidY, phase.color, arrowA);
  drawArrow(KP_X + KP_W    + 3, swMidY, SW_X     - 3,   swMidY, phase.color, arrowA);

  // Step counter
  ctx.fillStyle    = C.sub;
  ctx.font         = "8.5px monospace";
  ctx.textAlign    = "left";
  ctx.textBaseline = "top";
  ctx.fillText(
    `step ${{pos + 1}}/${{TOTAL_SW}}  (r${{kr + 1}} c${{kc + 1}}) → out[${{kr + 1}}][${{kc + 1}}]`,
    LG_X, LG_Y + LG_SPAN + 5, LG_SPAN + 4
  );
}}

// ═══════════════════════════════════════════════════════════════════════════
// §13 · POINTWISE PHASE RENDERER
// ═══════════════════════════════════════════════════════════════════════════
function renderPointwise(phase) {{
  const pos = stepPos % TOTAL_PW;
  const pr  = Math.floor(pos / DISP);   // current pixel row
  const pc  = pos % DISP;               // current pixel col

  // Input and output grids: current pos active, previous done
  const active = new Set([pos]);
  const done   = new Set();
  for (let p = 0; p < pos; p++) done.add(p);

  drawGrid(LG_X, LG_Y, DISP, CELL_IN, active, done, phase.color, true);
  drawGrid(PW_X, PW_Y, DISP, CELL_IN, active, done, phase.color, true);

  gridLabel(LG_X, LG_SPAN, "Input",  phase.inNote,  phase.color);
  gridLabel(PW_X, LG_SPAN, "Output", phase.outNote,  phase.color);

  // Canvas centre of the active output cell (target for convergence lines)
  const outCellCx = PW_X + pc * CELL_IN + CELL_IN / 2;
  const outCellCy = PW_Y + pr * CELL_IN + CELL_IN / 2;

  // Pointwise kernel panel with channel bars and convergence lines
  drawPointwiseKernelPanel(phase, outCellCx, outCellCy);

  // Arrows track the active pixel row for visual coherence
  const arrowA  = 0.30 + 0.55 * Math.abs(Math.sin(subTs * Math.PI));
  const rowMidY = LG_Y + pr * CELL_IN + CELL_IN / 2;
  drawArrow(LG_X + LG_SPAN + 3, rowMidY, KP_X - 3,     rowMidY, phase.color, arrowA);
  drawArrow(KP_X + KP_W    + 3, rowMidY, PW_X  - 3,    rowMidY, phase.color, arrowA);

  // Step counter
  ctx.fillStyle    = C.sub;
  ctx.font         = "8.5px monospace";
  ctx.textAlign    = "left";
  ctx.textBaseline = "top";
  ctx.fillText(
    `pixel ${{pos + 1}}/${{TOTAL_PW}}  (r${{pr + 1}} c${{pc + 1}})`,
    LG_X, LG_Y + LG_SPAN + 5, LG_SPAN + 4
  );
}}

// ═══════════════════════════════════════════════════════════════════════════
// §14 · MAIN RENDER LOOP  (unified delta-time clock)
//
// subTs accumulates elapsed / msPerStep each frame while running.
// Every discrete step increment is detected via while(subTs >= 1.0).
// NO animation logic anywhere in this file reads a raw frame counter.
// ═══════════════════════════════════════════════════════════════════════════
function render(ts) {{
  // ── Clock update ─────────────────────────────────────────────────────────
  if (running) {{
    if (lastTs === 0) lastTs = ts;
    const elapsed   = ts - lastTs;
    lastTs          = ts;
    const msPerStep = Math.max(16, 800 / speed);
    subTs          += elapsed / msPerStep;

    // Advance by as many full steps as elapsed time covers
    while (subTs >= 1.0) {{
      subTs  -= 1.0;
      stepPos++;
      const ph    = phases[phaseIdx];
      const total = ph.type === "sweep" ? TOTAL_SW : TOTAL_PW;
      if (stepPos >= total) {{
        stepPos  = 0;
        phaseIdx = (phaseIdx + 1) % phases.length;
        document.getElementById("phaseInfo").textContent =
          `Phase ${{phaseIdx + 1}} / ${{phases.length}}`;
      }}
    }}
  }} else {{
    // Keep lastTs current so resuming doesn't produce a time-jump spike
    lastTs = ts;
  }}

  // ── Clear ────────────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, CW, CH);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, CW, CH);

  // ── Phase bar ────────────────────────────────────────────────────────────
  const phase    = phases[phaseIdx];
  const total    = phase.type === "sweep" ? TOTAL_SW : TOTAL_PW;
  // Smooth fractional progress (stepPos + subTs) / total
  const progress = (stepPos + subTs) / total;
  drawPhaseBar(phase, progress);

  // ── Dispatch ─────────────────────────────────────────────────────────────
  if (phase.type === "sweep") {{
    renderSweep(phase);
  }} else {{
    renderPointwise(phase);
  }}

  requestAnimationFrame(render);
}}

// ═══════════════════════════════════════════════════════════════════════════
// §15 · CONTROLS
// ═══════════════════════════════════════════════════════════════════════════
document.getElementById("playPauseBtn").onclick = function () {{
  running = !running;
  this.textContent = running ? "⏸ Pause" : "▶ Play";
}};

document.getElementById("resetBtn").onclick = function () {{
  phaseIdx = 0; stepPos = 0; subTs = 0; lastTs = 0; running = true;
  document.getElementById("playPauseBtn").textContent = "⏸ Pause";
  document.getElementById("phaseInfo").textContent    = `Phase 1 / ${{phases.length}}`;
}};

document.getElementById("speedSlider").oninput = function () {{
  speed = parseInt(this.value, 10);
}};

// ── Boot ─────────────────────────────────────────────────────────────────
document.getElementById("phaseInfo").textContent = `Phase 1 / ${{phases.length}}`;
requestAnimationFrame(render);
</script>
</body>
</html>"""
