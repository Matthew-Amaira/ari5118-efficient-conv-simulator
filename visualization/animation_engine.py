"""
visualization/animation_engine.py
----------------------------------
Fluid HTML5 Canvas animation panel for the Conv Efficiency Simulator.

Renders an interactive kernel-sweep animation inside a Streamlit expander
using st.components.v1.html (no external dependencies beyond Streamlit).

Architecture
------------
* render_animation_panel(params, mbv2_mode)
      Builds the parameterised HTML/JS payload and injects it into a
      Streamlit expander.

Animation phases
----------------
Basic mode  (Standard vs. DWS)
    Phase 0: Standard Conv   — Dk×Dk kernel sweeps ALL channels at once
    Phase 1: DWS Stage 1     — Dk×Dk kernel per channel (independent filters)
    Phase 2: DWS Stage 2     — 1×1 pointwise channel mixing

MobileNetV2 mode
    Phase 0: Expand          — 1×1 PW:  M  → tM  (channel widening)
    Phase 1: Depthwise       — Dk×Dk:   tM spatial filters
    Phase 2: Project         — 1×1 PW:  tM → N   (linear bottleneck)

Each phase auto-advances on completion and loops continuously.
Controls: Play/Pause · Reset · Speed slider (all inside the iframe).

Canvas layout  (680 × 400 px)
    ┌──────────────── Phase label bar + progress strip ─────────────────┐
    │  Input grid  │  Kernel / op info  │  Output grid                  │
    │  (DISP×DISP) │   (centre panel)   │  (VALID×VALID or DISP×DISP)  │
    └───────────── Play/Pause · Reset · Speed ──────────────────────────┘

Grid cap: DISP = min(H, 8) — rendering a 224×224 grid would be unusable.
Output grid size = VALID × VALID = (DISP − Dk + 1)² for sweep phases,
                   DISP × DISP   for 1×1 pointwise phases (Dk=1, VALID=DISP).
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_animation_panel(params: dict, mbv2_mode: bool) -> None:
    """
    Render an interactive kernel-sweep animation inside a Streamlit expander.

    Parameters
    ----------
    params    : dict with keys M, N, Dk, H, W, t
    mbv2_mode : bool — True → MBv2 three-phase animation; False → Std + DWS
    """
    with st.expander("🎬  Interactive Convolution Animation  (click to expand)",
                     expanded=False):

        mode = "mbv2" if mbv2_mode else "dws"

        st.markdown(
            "<p style='font-size:0.82rem;opacity:0.6;margin-bottom:0.6rem;'>"
            "Watch the kernel window sweep across the feature map in real time. "
            "Phases auto-advance and loop. Use the controls to pause or change speed."
            "</p>",
            unsafe_allow_html=True,
        )

        html = _build_html(params, mode)
        components.html(html, height=440, scrolling=False)


# ---------------------------------------------------------------------------
# HTML / JavaScript builder
# ---------------------------------------------------------------------------

def _build_html(params: dict, mode: str) -> str:
    """
    Return a self-contained HTML string with the Canvas animation.

    All Python hyperparameter values are injected as JS constants via
    f-string substitution before the string is sent to the browser.
    """
    M  = params["M"]
    N  = params["N"]
    Dk = params["Dk"]
    H  = params["H"]
    t  = params["t"]
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
  }}
  canvas {{
    display: block;
    border-radius: 10px;
    background: #0f172a;
  }}
  #controls {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 8px;
    padding: 8px 16px;
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #334155;
    width: 680px;
  }}
  button {{
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    min-width: 80px;
  }}
  button:hover {{ opacity: 0.85; }}
  #resetBtn {{
    background: #334155;
  }}
  label {{
    font-size: 11px;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }}
  input[type=range] {{
    accent-color: #0ea5e9;
    width: 90px;
  }}
  #phaseInfo {{
    font-size: 11px;
    color: #64748b;
    margin-left: 8px;
  }}
</style>
</head>
<body>
<canvas id="anim" width="680" height="390"></canvas>
<div id="controls">
  <button id="playPauseBtn">⏸ Pause</button>
  <button id="resetBtn">↺ Reset</button>
  <span id="phaseInfo">Phase 1/3</span>
  <label>
    Speed
    <input type="range" id="speedSlider" min="1" max="12" value="5">
  </label>
</div>

<script>
// ─── Injected hyperparameters ────────────────────────────────────────────────
const M    = {M};
const N    = {N};
const Dk   = {Dk};
const H    = {H};
const tM   = {tM};
const MODE = "{mode}";   // "dws" or "mbv2"

// ─── Display constants ───────────────────────────────────────────────────────
const CW   = 680;
const CH   = 390;

// Max display grid dimension — capped to avoid unreadable tiny cells
const DISP = Math.min(H, 8);

// Valid kernel positions per axis for a spatial sweep (Dk×Dk kernel)
const VALID_SW = Math.max(1, DISP - Dk + 1);
const TOTAL_SW = VALID_SW * VALID_SW;

// Pointwise (1×1) covers the full grid
const VALID_PW = DISP;
const TOTAL_PW = VALID_PW * VALID_PW;

// Cell sizes
const CELL_IN  = Math.floor(188 / DISP);
const CELL_OUT_SW = Math.floor(188 / Math.max(VALID_SW, 1));
const CELL_OUT_PW = CELL_IN;

// Grid anchor positions on canvas
const LG_X = 18;
const LG_Y = 80;
const RG_X = 448;
const RG_Y = 80;

// ─── Phase definitions ───────────────────────────────────────────────────────
function buildPhases() {{
  if (MODE === "mbv2") {{
    return [
      {{
        label:   `① Expand — 1×1 PW  ·  M=${{M}} → tM=${{tM}} channels  (expansion t=${{Math.round(tM/M)}})`,
        color:   "#ca8a04",
        type:    "pointwise",
        inNote:  `Input ${{H}}×${{H}}×${{M}}`,
        outNote: `Output ${{H}}×${{H}}×${{tM}}`,
        kNote:   `1×1 · ${{M}}→${{tM}}`,
      }},
      {{
        label:   `② Depthwise — ${{Dk}}×${{Dk}} per channel  ·  ${{tM}} independent spatial filters`,
        color:   "#db2777",
        type:    "sweep",
        inNote:  `Expanded ${{H}}×${{H}}×${{tM}}`,
        outNote: `Filtered ${{H}}×${{H}}×${{tM}}`,
        kNote:   `${{Dk}}×${{Dk}} · ${{tM}} ch`,
      }},
      {{
        label:   `③ Project — 1×1 PW  ·  tM=${{tM}} → N=${{N}} channels  (linear bottleneck)`,
        color:   "#059669",
        type:    "pointwise",
        inNote:  `Filtered ${{H}}×${{H}}×${{tM}}`,
        outNote: `Output ${{H}}×${{H}}×${{N}}`,
        kNote:   `1×1 · ${{tM}}→${{N}}`,
      }},
    ];
  }} else {{
    return [
      {{
        label:   `Standard Conv — ${{Dk}}×${{Dk}} kernel  ·  ALL ${{M}} input channels fused simultaneously`,
        color:   "#6366f1",
        type:    "sweep",
        inNote:  `Input ${{H}}×${{H}}×${{M}}`,
        outNote: `Output ${{H}}×${{H}}×${{N}}`,
        kNote:   `${{Dk}}×${{Dk}}×${{M}}×${{N}}`,
      }},
      {{
        label:   `① DWS Depthwise — ${{Dk}}×${{Dk}} per channel  ·  ${{M}} independent spatial filters`,
        color:   "#db2777",
        type:    "sweep",
        inNote:  `Input ${{H}}×${{H}}×${{M}}`,
        outNote: `Interm. ${{H}}×${{H}}×${{M}}`,
        kNote:   `${{Dk}}×${{Dk}} · ${{M}} ch`,
      }},
      {{
        label:   `② DWS Pointwise — 1×1 linear mix  ·  ${{M}} → ${{N}} channels`,
        color:   "#059669",
        type:    "pointwise",
        inNote:  `Interm. ${{H}}×${{H}}×${{M}}`,
        outNote: `Output ${{H}}×${{H}}×${{N}}`,
        kNote:   `1×1 · ${{M}}→${{N}}`,
      }},
    ];
  }}
}}

const phases     = buildPhases();
let   phaseIdx   = 0;
let   stepPos    = 0;
let   running    = true;
let   speed      = 5;
let   lastTs     = 0;
let   frameCount = 0;

const canvas = document.getElementById("anim");
const ctx    = canvas.getContext("2d");

// ─── Colour palette ──────────────────────────────────────────────────────────
const C = {{
  bg:        "#0f172a",
  surface:   "#1e293b",
  border:    "#334155",
  cellBase:  "#1e293b",
  cellCover: "rgba(249,115,22,0.75)",
  cellDone:  "rgba(34,197,94,0.55)",
  text:      "#cbd5e1",
  sub:       "#64748b",
}};

// ─── Drawing utilities ───────────────────────────────────────────────────────
function rr(ctx, x, y, w, h, r) {{
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}}

function drawPhaseBar(phase, progress) {{
  // Background pill
  ctx.fillStyle = "#1e293b";
  rr(ctx, 8, 8, CW - 16, 40, 8);
  ctx.fill();
  ctx.strokeStyle = phase.color;
  ctx.lineWidth = 1.5;
  rr(ctx, 8, 8, CW - 16, 40, 8);
  ctx.stroke();

  // Label text
  ctx.fillStyle = phase.color;
  ctx.font = "bold 11.5px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(phase.label, CW / 2, 28);

  // Phase indicator dots
  const dotY = 56;
  const totalW = phases.length * 22 - 6;
  const dotX0 = (CW - totalW) / 2;
  for (let i = 0; i < phases.length; i++) {{
    ctx.fillStyle = i === phaseIdx ? phases[i].color : "#334155";
    ctx.beginPath();
    ctx.arc(dotX0 + i * 22 + 5, dotY, i === phaseIdx ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
  }}

  // Progress strip
  const barY = 66, barH = 4;
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(18, barY, CW - 36, barH);
  ctx.fillStyle = phase.color;
  ctx.fillRect(18, barY, (CW - 36) * Math.min(progress, 1.0), barH);
}}

function drawGrid(gx, gy, dim, cellSz, covered, done, accentColor) {{
  for (let r = 0; r < dim; r++) {{
    for (let c = 0; c < dim; c++) {{
      const key = r * dim + c;
      const px = gx + c * cellSz;
      const py = gy + r * cellSz;
      const isActive = covered.has(key);
      const isDone   = done.has(key);

      ctx.fillStyle = isDone   ? C.cellDone
                    : isActive ? C.cellCover
                    : C.cellBase;
      rr(ctx, px + 1, py + 1, cellSz - 2, cellSz - 2, 3);
      ctx.fill();

      if (!isDone && !isActive && cellSz >= 16) {{
        ctx.fillStyle = "#2d3f55";
        ctx.fillRect(px + 1, py + 1, cellSz - 2, cellSz - 2);
      }}
    }}
  }}

  // Grid border
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  rr(ctx, gx - 2, gy - 2, dim * cellSz + 4, dim * cellSz + 4, 5);
  ctx.stroke();
}}

function drawKernelPanel(phase) {{
  const kx = 238, ky = 95, kw = 176, kh = 155;
  ctx.fillStyle = "#1e293b";
  rr(ctx, kx, ky, kw, kh, 10);
  ctx.fill();
  ctx.strokeStyle = phase.color;
  ctx.lineWidth = 1.8;
  rr(ctx, kx, ky, kw, kh, 10);
  ctx.stroke();

  // Kernel grid visualization (capped at 5×5 display)
  const kDisp = Math.min(Dk, 5);
  const kCell = Math.floor(72 / kDisp);
  const kgx   = kx + (kw - kDisp * kCell) / 2;
  const kgy   = ky + 12;

  for (let r = 0; r < kDisp; r++) {{
    for (let c = 0; c < kDisp; c++) {{
      ctx.fillStyle = phase.color + "66";
      rr(ctx, kgx + c * kCell + 1, kgy + r * kCell + 1,
         kCell - 2, kCell - 2, 2);
      ctx.fill();
      ctx.strokeStyle = phase.color + "aa";
      ctx.lineWidth = 0.8;
      rr(ctx, kgx + c * kCell + 1, kgy + r * kCell + 1,
         kCell - 2, kCell - 2, 2);
      ctx.stroke();
    }}
  }}

  // Kernel note text
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(phase.kNote, kx + kw / 2, ky + kh - 22);

  ctx.fillStyle = C.sub;
  ctx.font = "9px monospace";
  ctx.fillText(phase.type === "sweep" ? "spatial sweep" : "channel mix", kx + kw / 2, ky + kh - 8);
}}

function drawArrow(x1, y1, x2, y2, color) {{
  ctx.save();
  ctx.strokeStyle = color + "aa";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = color + "cc";
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 9 * Math.cos(angle - 0.35), y2 - 9 * Math.sin(angle - 0.35));
  ctx.lineTo(x2 - 9 * Math.cos(angle + 0.35), y2 - 9 * Math.sin(angle + 0.35));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}}

function gridLabel(x, y, w, h, topLine, botLine, color) {{
  ctx.fillStyle = color;
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(topLine, x + w / 2, y - 22);
  ctx.fillStyle = C.sub;
  ctx.font = "8.5px monospace";
  ctx.fillText(botLine, x + w / 2, y - 10);
}}

// ─── Sweep phase renderer ────────────────────────────────────────────────────
function renderSweep(phase) {{
  const total   = TOTAL_SW;
  const valid   = VALID_SW;
  const cellOut = CELL_OUT_SW;
  const pos     = stepPos % total;
  const kr = Math.floor(pos / valid);
  const kc = pos % valid;

  // Input grid — covered cells
  const covered = new Set();
  for (let dr = 0; dr < Dk && kr + dr < DISP; dr++)
    for (let dc = 0; dc < Dk && kc + dc < DISP; dc++)
      covered.add((kr + dr) * DISP + (kc + dc));

  // Output grid — done positions
  const done = new Set();
  for (let p = 0; p < pos; p++) done.add(p);
  const outActive = new Set([pos]);

  drawGrid(LG_X, LG_Y, DISP, CELL_IN, covered, new Set(), phase.color);
  drawGrid(RG_X, RG_Y, valid, cellOut, outActive, done, phase.color);

  gridLabel(LG_X, LG_Y, DISP * CELL_IN, DISP * CELL_IN,
            "Input", phase.inNote, phase.color);
  gridLabel(RG_X, RG_Y, valid * cellOut, valid * cellOut,
            "Output", phase.outNote, phase.color);

  // Kernel window highlight on input grid
  const kwPx = Dk * CELL_IN, khPx = Dk * CELL_IN;
  ctx.strokeStyle = phase.color;
  ctx.lineWidth = 2.5;
  rr(ctx, LG_X + kc * CELL_IN, LG_Y + kr * CELL_IN, kwPx, khPx, 3);
  ctx.stroke();

  // Arrows
  const lgMidY = LG_Y + DISP * CELL_IN / 2;
  const rgMidY = RG_Y + valid * cellOut / 2;
  const midY   = (lgMidY + rgMidY) / 2;
  drawArrow(LG_X + DISP * CELL_IN + 4, lgMidY, 238, midY, phase.color);
  drawArrow(238 + 176 + 4, midY, RG_X - 4, rgMidY, phase.color);

  // Position counter
  ctx.fillStyle = C.sub;
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`pos ${{pos+1}}/${{total}}  (r${{kr+1}},c${{kc+1}})`, LG_X, LG_Y + DISP * CELL_IN + 6);
}}

// ─── Pointwise phase renderer ────────────────────────────────────────────────
function renderPointwise(phase) {{
  const total   = TOTAL_PW;
  const pos     = stepPos % total;
  const pr = Math.floor(pos / VALID_PW);
  const pc = pos % VALID_PW;

  // Full input grid always lit; output fills in
  const covered = new Set([pos]);
  const done    = new Set();
  for (let p = 0; p < pos; p++) done.add(p);

  drawGrid(LG_X, LG_Y, DISP, CELL_IN, covered, done, phase.color);
  drawGrid(RG_X, RG_Y, DISP, CELL_IN, covered, done, phase.color);

  // Animated "mixing" pulse — vertical bar sweeping down the centre panel
  const pulse = (frameCount % 20) / 20;
  const barX  = 238 + 176 / 2 - 10;
  const barY  = LG_Y + DISP * CELL_IN * pulse;
  ctx.fillStyle = phase.color + "55";
  ctx.fillRect(barX, barY, 20, Math.min(30, DISP * CELL_IN * 0.15));

  gridLabel(LG_X, LG_Y, DISP * CELL_IN, DISP * CELL_IN,
            "Input", phase.inNote, phase.color);
  gridLabel(RG_X, RG_Y, DISP * CELL_IN, DISP * CELL_IN,
            "Output", phase.outNote, phase.color);

  const lgMidY = LG_Y + DISP * CELL_IN / 2;
  drawArrow(LG_X + DISP * CELL_IN + 4, lgMidY, 238, lgMidY, phase.color);
  drawArrow(238 + 176 + 4, lgMidY, RG_X - 4, lgMidY, phase.color);

  ctx.fillStyle = C.sub;
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`pixel ${{pos+1}}/${{total}}  (${{H}}×${{H}} spatial)`, LG_X, LG_Y + DISP * CELL_IN + 6);
}}

// ─── Main render loop ────────────────────────────────────────────────────────
function render(ts) {{
  frameCount++;

  if (running) {{
    const msPerStep = Math.max(16, 700 / speed);
    if (ts - lastTs >= msPerStep) {{
      lastTs = ts;
      const phase = phases[phaseIdx];
      const total = phase.type === "sweep" ? TOTAL_SW : TOTAL_PW;
      stepPos++;
      if (stepPos >= total) {{
        stepPos = 0;
        phaseIdx = (phaseIdx + 1) % phases.length;
        document.getElementById("phaseInfo").textContent =
          `Phase ${{phaseIdx + 1}}/${{phases.length}}`;
      }}
    }}
  }}

  ctx.clearRect(0, 0, CW, CH);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, CW, CH);

  const phase   = phases[phaseIdx];
  const total   = phase.type === "sweep" ? TOTAL_SW : TOTAL_PW;
  const progress = (stepPos + 1) / total;

  drawPhaseBar(phase, progress);
  drawKernelPanel(phase);

  if (phase.type === "sweep") {{
    renderSweep(phase);
  }} else {{
    renderPointwise(phase);
  }}

  requestAnimationFrame(render);
}}

// ─── Controls ────────────────────────────────────────────────────────────────
document.getElementById("playPauseBtn").onclick = function() {{
  running = !running;
  this.textContent = running ? "⏸ Pause" : "▶ Play";
}};

document.getElementById("resetBtn").onclick = function() {{
  phaseIdx = 0;
  stepPos  = 0;
  running  = true;
  document.getElementById("playPauseBtn").textContent = "⏸ Pause";
  document.getElementById("phaseInfo").textContent = `Phase 1/${{phases.length}}`;
}};

document.getElementById("speedSlider").oninput = function() {{
  speed = parseInt(this.value);
}};

// Kick off
document.getElementById("phaseInfo").textContent = `Phase 1/${{phases.length}}`;
requestAnimationFrame(render);
</script>
</body>
</html>"""
