# Convolution Efficiency Simulator: Deep Learning Interactive Learning Pack

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=flat-square&logo=numpy&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square)
![Course](https://img.shields.io/badge/ARI5118-Deep%20Learning%20for%20CV-CC0000?style=flat-square)

---

This is an interactive, browser-based simulator that lets you see, in real time, exactly how different convolution strategies compare in terms of parameter counts and computational cost. It breaks down three architectures — Standard Convolution, Depthwise Separable Convolution (Howard et al., 2017), and the MobileNetV2 Inverted Residual Bottleneck (Sandler et al., 2018) — so you can drag a slider and instantly watch the numbers change instead of just reading about them. The core insight this tool teaches is that clever factorisation of a single heavy convolution into cheaper, sequential operations is what makes deploying deep learning models on phones and embedded devices actually possible.

---

## 📦 Repository Contents

Everything you need for this assignment is organised into four companion pieces. Start with the simulator to build intuition, then deepen your understanding through the notebook and readings.

| Deliverable | Location | What it does |
|---|---|---|
| 🖥️ **Streamlit Simulator** | `simulator/` ← *you are here* | Live interactive dashboard with HTML5 Canvas animations, Matplotlib architecture diagrams, analytical benchmarks, and a CPU PyTorch sandbox that validates every formula in real hardware |
| 📓 **Annotated Tutorial** | `../walkthrough.ipynb` | Step-by-step Jupyter notebook walking through PyTorch implementations of each convolution type, with a built-in FLOPs profiler that measures real wall-clock cost on CPU |
| 📝 **Adversarial Quiz** | `../quiz_with_rationale.pdf` | Ten deliberately tricky exam-style questions covering edge cases in efficiency analysis, each with a full written rationale explaining the reasoning behind the correct answer |
| 📁 **Core Literature** | `../further_reading/` | Curated collection of the mandatory primary research papers — MobileNetV1, MobileNetV2, Xception, and ShuffleNet — that form the theoretical foundation of all three simulator modes |

---

## 🧠 The Core Concepts (Explained Simply)

### Why does convolution efficiency matter?

A standard convolution is like using one enormous multi-layered stamp: every time you press it down on the image, it reads every spatial neighbourhood *and* every input channel simultaneously, and it does this for every output channel you want to produce. It is powerful, but the cost — in both parameters and multiply-add operations — scales with the product of all those dimensions at once.

Depthwise Separable Convolution (the engine of MobileNet) splits that single stamp into two much cheaper steps. First, a **depthwise** pass processes each input channel completely independently with its own small spatial filter — think of slicing a sandwich and seasoning each layer on its own. Then a **pointwise** `1×1` pass mixes those individual results across channels, combining the flavours at the very end. The spatial and channel work are decoupled, and the savings are dramatic.

MobileNetV2's **Inverted Residual Bottleneck** adds one more elegant trick: before the depthwise spatial filter runs, the channel dimension is *widened* by an expansion factor `t` (typically `t = 6`). This gives the spatial filter more representational room to work in, then a final projection `1×1` compresses everything back down to the target channel width. Unlike a classical bottleneck (wide → narrow → wide), this one goes narrow → wide → narrow, which is why it is called *inverted*. When input and output dimensions match, a residual skip connection is added, enabling efficient gradient flow just like ResNet — but at a fraction of the cost.

---

### The Efficiency Formula

For a standard `Dk × Dk` convolution mapping `M` input channels to `N` output channels, the cost ratio of Depthwise Separable Convolution is:

$$\frac{\text{DWS Cost}}{\text{Standard Cost}} = \frac{1}{N} + \frac{1}{D_k^2}$$

**In plain English:** with a `3×3` kernel and 64 output channels, DWS is roughly **8–9× cheaper** than the standard equivalent, with negligible accuracy loss. That single equation is the practical justification for an entire family of efficient mobile architectures.

For MobileNetV2's three-stage block the ratio generalises to:

$$\frac{\text{MBv2 Cost}}{\text{Standard Cost}} = \frac{t \left(M + D_k^2 + N\right)}{D_k^2 \cdot N}$$

where `t` is the expansion factor. The simulator computes both ratios live as you move the sliders.

---

## 🚀 Quick Start Guide

Follow these steps exactly, in order. Every command is ready to copy and paste into a **Windows PowerShell** terminal opened inside the `simulator/` directory.

### Step 1 — Create an isolated Python environment

This keeps the project's dependencies completely separate from anything else installed on your machine.

```powershell
python -m venv .venv
```

### Step 2 — Activate the environment

PowerShell requires explicit activation before it will use the local packages.

```powershell
.venv\Scripts\Activate.ps1
```

> **Permission error?** Run PowerShell as Administrator and execute `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then repeat Step 2.

### Step 3 — Install dependencies

This installs Streamlit, NumPy, Matplotlib, and the CPU build of PyTorch. No GPU is required at any point.

```powershell
pip install -r requirements.txt
```

### Step 4 — Launch the simulator

Streamlit will open the dashboard automatically in your default browser at `http://localhost:8501`.

```powershell
streamlit run app.py
```

That is all. Adjust the sliders in the left panel and every chart, formula, and animation updates instantly.

---

## 🗂️ Simulator Module Map

The simulator is structured as a clean Python package. Each file has exactly one responsibility.

```
simulator/
├── app.py                          ← Entry point: page config, hero header, tab shell
│
├── config/
│   └── styles.py                   ← Global CSS (Light/Dark adaptive) + Matplotlib rcParams
│
├── core/
│   ├── math_engine.py              ← Pure NumPy analytics: all parameter & FLOPs formulas
│   └── torch_sandbox.py            ← Live CPU PyTorch validation (Tab 3)
│
├── ui/
│   ├── sidebar.py                  ← Two-way synced hyperparameter controls (Apple-style tiles)
│   └── dashboards.py               ← Tab 1 / Tab 2 / Tab 3 view renderers
│
└── visualization/
    ├── static_plots.py             ← Matplotlib architecture diagrams & benchmark charts
    └── animation_engine.py         ← HTML5 Canvas kernel-sweep animation (no external JS)
```

---

## 🎛️ Simulator Features at a Glance

| Feature | Where to find it |
|---|---|
| Live parameter & FLOPs counters | Hero metric row at the top of every page |
| Architecture diagrams (Std, DWS, MBv2) | Tab 1 → Architecture Diagrams |
| HTML5 Canvas kernel-sweep animation | Tab 1 → Interactive Convolution Animation (expander) |
| Bar chart: Std vs DWS/MBv2 breakdown | Tab 1 → Benchmark Charts |
| Step-by-step formula derivations | Tab 2 → Mathematical Derivations |
| Ratio derivation with LaTeX | Tab 2 → Reduction Ratio |
| Live CPU PyTorch layer validation | Tab 3 → PyTorch Sandbox |
| Two-way synced sliders + number inputs | Left sidebar — drag *or* type exact values |
| MobileNetV2 Inverted Residual mode | Sidebar → Architecture Mode radio toggle |

---

## 🔬 CPU-Only Execution

This project runs entirely on CPU. There is no CUDA requirement and no GPU dependency anywhere in the codebase. The PyTorch sandbox in Tab 3 explicitly sets `device = "cpu"` and validates shapes against the analytical formulas derived in `core/math_engine.py`. If PyTorch is not installed in the active environment, Tab 3 degrades gracefully with an informative message rather than crashing the app.

---

## 📐 Academic Alignment

This project is a direct submission artefact for **ARI5118 — Deep Learning for Computer Vision** at the **University of Malta**. The simulator, notebook, and quiz collectively address the Topic 5 syllabus criteria on efficient convolutional architectures, covering the theoretical derivation of parameter reduction ratios, the architectural motivation for depthwise factorisation, and the design principles of the MobileNetV2 Inverted Residual Bottleneck as described in the primary literature.

All formulas implemented in `core/math_engine.py` are analytically derived from first principles and cross-validated against live PyTorch layer measurements in `core/torch_sandbox.py`.

---

*University of Malta · Faculty of ICT · ARI5118 Deep Learning for Computer Vision*
