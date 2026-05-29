# Quantum Robust Training (QRT) Implementation

## Overview

This is a classical simulation of **Quantum Robust Training** as described in the paper "Provably Efficient Quantum Robust Training via Carleman Linearization". The implementation uses Carleman Linearization to transform the non-linear robust training dynamics into a high-dimensional linear system, which is then solved using classical linear algebra (simulating HHL/QSVT).

## Reader's Guide (Eq. → Carleman Lift)

The min-max objective
\[
\min_{\theta} \sum_{x \in \mathcal{X}} \sup_{x^\prime \in B_p(x,\epsilon)} \mathcal{L}(f_\theta(x^\prime),y)
\]
induces the vector field \(\dot{v} = [+\nabla_w \mathcal{L}, -\nabla_u \mathcal{L}]^\top\) for the augmented state \(v = [w, u]^\top\). The implementation provides a JAX-based coefficient extractor (`carleman_coeffs.py`) that returns \(F_0, F_1, F_2\) via automatic differentiation, and a lift/operator constructor (`carleman_operator.py`) that assembles the truncated linear system \( \dot{\hat{y}} = A \hat{y} + b \). The reference loop in `qrt_mindquantum_loop.py` then performs the local implicit Euler step \((I + hA)\hat{y}_{t+1} = \hat{y}_t + hb\), solves the linear system with MindQuantum HHL, and applies the sign + \(L_\infty\) projection to the adversarial input between quantum solves.

## Dependencies

Core packages used across modules:
- `torch` (classical training + evaluation)
- `mindquantum` (HHL circuit solver for quantum linear systems)
- `jax` (fast Jacobian/Hessian extraction for Carleman coefficients)

The JAX + MindQuantum reference loop requires both `jax` and `mindquantum` to be installed.

## What Was Fixed

### 1. **Dual Output Generation** ✓
The main script now generates **TWO separate comparison plots**:
- **Single Loss Mode**: Pure adversarial training (PGD-AT)
- **Combined Loss Mode**: α-blended clean and robust loss (TRADES-style)

### 2. **Optimized Hyperparameters** ✓
Based on condition number analysis and computational constraints:

```python
CARLEMAN_N = 3              # Reduced from 4 (N=4 too expensive)
TIME_STEP = 0.08            # Reduced for stability with N=3
QRT_LEARNING_RATE = 0.025   # Balanced for N=3
QRT_GRAD_CLIP = 2.0         # Allows gradient flow
EPSILON_TRAIN = 0.02        # 20% of eval epsilon
```

### 3. **Implementation Matches Paper**
- **Protocol (main.tex §2.1)**: 7-step QRT protocol implemented
- **Regime I (main.tex §2.2)**: Sequential/Modular execution (sign & projection outside lift)
- **Alternating Dynamics (supp.tex §2.5)**: Alternates between w-ascent and u-descent
- **Condition Number (supp.tex Prop. 444)**: κ ≤ 2(T+1) satisfied

## Running the Experiment

### Quick Start
```bash
cd /home/y_w/QRT
python3 main.py
```

### JAX + MindQuantum Reference Loop
```bash
python3 qrt_mindquantum_loop.py
```
This script demonstrates the requested local implicit Euler update, JAX coefficient extraction, MindQuantum HHL solve, and prints the per-step verification table (Step | Loss | Gradient Norm | Solver Fidelity | Projection Distance).

## How to Run

### Classical (existing) pipeline
```bash
python3 main.py
```
This runs the PyTorch-based classical baseline and QRT simulation, saving comparison plots.

### JAX + MindQuantum reference pipeline
```bash
python3 qrt_mindquantum_loop.py
```
This runs the minimal reference loop that uses JAX for coefficient extraction and MindQuantum for the HHL solve.

## Testing / Verification

### 1) Smoke-test the JAX coefficient extraction
```bash
python3 -c "import jax, jax.numpy as jnp; import carleman_coeffs as cc; import numpy as np; loss=lambda w,u,b: jnp.sum((w-u)**2); v=jnp.ones(6); cc.verify_linear_approx(loss, v, 3, None)"
```
Expected: no output and exit code 0 if the linear approximation check passes.

### 2) Run the MindQuantum HHL reference loop
```bash
python3 qrt_mindquantum_loop.py
```
Expected: a per-step table (Step | Loss | Gradient Norm | Solver Fidelity | Projection Distance). Solver fidelity should be close to 1 for well-conditioned toy instances.

### 3) Run the full experiment
```bash
python3 main.py
```
Expected: PNG outputs saved in the repo root and progress logs printed to stdout.

### What Happens
1. **Load Data**: MNIST digits 0-4, resized to 12×12
2. **Initialize**: Shared random weights for fair comparison
3. **Classical Standard Training**: 10 epochs (baseline)
4. **Experiment 1 - Single Loss**:
   - Classical PGD-AT: 10 epochs
   - QRT Simulation: 50 steps
5. **Experiment 2 - Combined Loss** (α=0.5):
   - Classical Combined: 10 epochs
   - QRT Simulation: 50 steps
6. **Generate Plots**: Two comparison images

### Expected Runtime
- **Full experiment**: 5-20 minutes (varies with F2/F3 extraction and hardware)
- **Data loading**: 1-2 minutes
- **Each classical training**: 2-5 minutes
- **Each QRT simulation**: 2-8 minutes

### Outputs
1. `qrt_vs_classical_comparison_single_loss.png`
2. `qrt_vs_classical_comparison_combined_loss_alpha0.50.png`

## Output Plots Explained

Each plot shows three methods compared:

| Method | Color | Description |
|--------|-------|-------------|
| **Standard Training** | Blue | Clean training, no adversarial robustness |
| **Classical Robust** | Green | PGD-AT or Combined loss with backprop |
| **QRT Simulation** | Red | Quantum-inspired Carleman linearization |

**Two subplots per image:**
- **Left**: Robust Accuracy (against PGD-ε adversary)
- **Right**: Clean Accuracy (on original unperturbed data)

## Architecture

```
Input (144D) 
    ↓
Fixed Projection (144 → 10)
    ↓
Dense Layer (10 → 4)
    ↓
Polynomial Activation (x²)
    ↓
Dense Layer (4 → 5)
    ↓
Output (5 classes)
```

**Why this architecture?**
- Small dimension (60 trainable params) makes Carleman lift tractable
- Polynomial activation ensures exact polynomial dynamics for Carleman
- Fixed projection reduces input dimension while preserving information

## QRT Technical Details

### Carleman Linearization

The non-linear dynamics:
```
dot(v) = f(v)   where v = [w, u]  (input + parameters)
```

Are approximated as:
```
dot(v) ≈ F₀ + F₁·v + F₂·v²
```

Then lifted to high-dimensional linear system:
```
ŷ = [v, v⊗v, v⊗v⊗v]  (up to order N=3)
dot(ŷ) = A·ŷ + b̂
```

### Alternating Training (Regime I)

**Step 0, 2, 4, ...** (U-phase):
- Minimize loss w.r.t. parameters u
- Gradient descent: u ← u - η·∇ᵤL

**Step 1, 3, 5, ...** (W-phase):
- Maximize loss w.r.t. input w
- Gradient ascent: w ← w + η·sign(∇ᵥL)
- Project onto ε-ball: w ← Π_{B∞(w₀,ε)}(w)

### Sign Function & Projection

**Classical Implementation** (not quantum):
- Sign: `np.sign(gradient)` (element-wise)
- Projection: `np.clip(w - w₀, -ε, ε)` (L∞ ball)

This is the "Modular Execution" approach from the paper (main.tex §2.1, step 5).

## Monitoring Progress

While the script runs in background:

```bash
# Check if still running
ps aux | grep "python3 main.py" | grep -v grep

# Monitor progress (output is buffered, may not show immediately)
tail -f /home/y_w/QRT/qrt_output.log

# Check CPU usage
top -p $(pgrep -f "python3 main.py")

# Check for output files
ls -lht /home/y_w/QRT/*.png
```

## Configuration Parameters

### Key Settings (`config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CARLEMAN_N` | 3 | Truncation order (higher = more accurate but exponentially expensive) |
| `TIME_STEP` | 0.08 | Discretization step (smaller = more stable but slower) |
| `QRT_LEARNING_RATE` | 0.025 | Scales gradient magnitude in dynamics |
| `EPSILON` | 0.1 | L∞ perturbation bound for evaluation |
| `EPSILON_TRAIN` | 0.02 | Smaller ε during training (allows learning) |
| `TOTAL_STEPS` | 50 | QRT simulation steps |
| `EPOCHS` | 10 | Classical training epochs |

### Tuning Guide

**If QRT doesn't learn (Δu too small):**
- Increase `QRT_LEARNING_RATE` (0.03 → 0.05)
- Increase `TIME_STEP` (0.08 → 0.1)
- Decrease `QRT_WEIGHT_DECAY` (0.0008 → 0.0005)

**If QRT diverges (NaN errors):**
- Decrease `TIME_STEP` (0.08 → 0.05)
- Decrease `QRT_LEARNING_RATE` (0.025 → 0.02)
- Increase `QRT_GRAD_CLIP` (2.0 → 3.0)

**If computation too slow:**
- Reduce `TOTAL_STEPS` (50 → 30)
- Reduce `HIDDEN_DIM` (4 → 3)
- Use `F2_MODE = "approx"` instead of "full"

## Troubleshooting

### Problem: Script hangs at start
- **Cause**: Loading MNIST data
- **Solution**: Wait 1-2 minutes, or use synthetic data (automatic fallback)

### Problem: Out of memory
- **Cause**: CARLEMAN_N too high (N=4 creates 154⁴ ≈ 563M element tensors)
- **Solution**: Use N=3 (154³ ≈ 3.6M elements) ✓ Already done

### Problem: Condition number κ ≈ 1
- **Cause**: Time step or learning rate too small, no dynamics
- **Solution**: Increase `TIME_STEP` and `QRT_LEARNING_RATE` ✓ Already done

### Problem: QRT accuracy stuck at initial value
- **Cause**: Linearization around fixed point, no gradient flow
- **Solution**: Check relinearization is working, increase learning rate

## Expected Results

### Classical Standard Training
- Clean Accuracy: ~60-80%
- Robust Accuracy: ~20-40%
- **Interpretation**: Good on clean data, vulnerable to attacks

### Classical Robust Training (PGD-AT or Combined)
- Clean Accuracy: ~50-70%
- Robust Accuracy: ~35-55%
- **Interpretation**: Trade-off between clean and robust accuracy

### QRT Simulation
- Clean Accuracy: Should match classical robust ±10%
- Robust Accuracy: Should match classical robust ±10%
- **Interpretation**: Demonstrates that Carleman linearization can approximate the robust training dynamics

**Key Success Metric**: QRT should converge and achieve comparable accuracy to classical robust methods, validating the theoretical framework.

## Files Structure

```
QRT/
├── main.py                  # Main experiment script (MODIFIED)
├── config.py                # Configuration parameters (MODIFIED)
├── qrt_simulation.py        # Carleman system & HHL simulation
├── classical_baseline.py    # Standard and robust training
├── verify_fixes.py          # Verification tests
├── carleman_coeffs.py       # JAX-based coefficient extraction
├── carleman_operator.py     # Carleman lift/operator utilities
├── qrt_mindquantum_loop.py  # JAX + MindQuantum reference loop
├── README.md               # This file (NEW)
├── technical details/
│   ├── main.tex            # Paper main text
│   └── supp.tex            # Supplementary material
└── data/                   # MNIST data (auto-downloaded)
```

## Citation

If you use this implementation, please cite:

```
@article{qrt2024,
  title={Provably Efficient Quantum Robust Training via Carleman Linearization},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Specify license here]

## Contact

For questions about the implementation, please refer to:
- Paper: `technical details/main.tex` and `technical details/supp.tex`
