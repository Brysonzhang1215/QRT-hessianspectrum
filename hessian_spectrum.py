#!/usr/bin/env python3
"""
Hessian Eigenvalue Spectral Density via Stochastic Lanczos Quadrature (SLQ).

Parameters are evolved through Carleman linearization (with truncation error)
so the Hessian is computed on the actual QRT parameter trajectory. The truncation
error in Carleman linearization changes the parameter values, which in turn
affects the Hessian spectrum — this captures the true loss landscape that the
quantum system navigates, rather than the idealized classical gradient descent path.

Methodology:
  - Carleman linearization evolves parameters (truncation order N from config)
  - Implicit Euler time-stepping solved via HHL/GMRES
  - PyTorch autograd HVPs for Hessian-vector products (two-pass)
  - Lanczos tridiagonalization per random probe
  - Gaussian kernel smoothing with fixed sigma
  - Stable monotonically-narrowing grid bounds
  - Log₁₀ density ridgeline plots with dual y-axes
  - Magenta→Cyan color gradient

Produces stacked ridge plots for three loss modes:
  1. Clean loss
  2. Adversarial (single/robust) loss
  3. Combined loss (TRADES-style, α=0.5)

Requires: PyTorch, NumPy, Matplotlib, SciPy, JAX (for Carleman coefficients).

Usage:
    conda run -n qrt310 python hessian_spectrum.py
"""

from __future__ import annotations

import time
import gzip

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import zoom as scipy_zoom

import torch
import torch.nn as nn

import config
from classical_baseline import PolyMLP, RobustTrainer, create_dataset_iterator
from qrt_simulation import train_qrt

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

NUM_LANCZOS     = 60        # Lanczos iterations per probe (= param dim for exact spectrum)
NUM_PROBES      = 30        # Random probe vectors for SLQ averaging
TRAIN_EPOCHS    = 20        # Number of snapshot intervals (= "epochs" for plotting)
SEED            = 42

# Classical adversarial pre-training before Carleman (warm-up)
PRETRAIN_EPOCHS = 20        # PGD-AT epochs before QRT takes over
PRETRAIN_SNAP_EVERY = 5     # Capture Hessian snapshot every N pretrain epochs

# QRT / Carleman steps used in this script (overrides config.TOTAL_STEPS locally)
QRT_TOTAL_STEPS  = 100      # Total Carleman evolution steps per mode

# SLQ configuration (matching robustness.ipynb exactly)
SLQ_SPECTRUM_BINS     = 256     # Discretization bins
SLQ_LOG_DENSITY       = True    # Display log₁₀(density)
SLQ_STABLE_GRID       = True    # Keep grid bounds monotonically narrowing
SLQ_FIX_SIGMA         = True    # Fix kernel bandwidth over time
SLQ_SIGMA_FRACTION    = 0.01   # sigma as fraction of initial span
SLQ_NORMALIZE_DENSITY = True    # Normalize density to integrate to 1

# Plot style
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════

def load_data(n_train: int = 12000, n_test: int = 3000):
    """Load MNIST subset (digits 0-4), downscale to 12x12, normalize."""
    try:
        with gzip.open("data/train-images-idx3-ubyte.gz", "rb") as f:
            imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open("data/train-labels-idx1-ubyte.gz", "rb") as f:
            lbls = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open("data/t10k-images-idx3-ubyte.gz", "rb") as f:
            test_imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open("data/t10k-labels-idx1-ubyte.gz", "rb") as f:
            test_lbls = np.frombuffer(f.read(), np.uint8, offset=8)

        def process(images, labels, n):
            mask = np.isin(labels, config.TARGET_DIGITS)
            images, labels = images[mask][:n], labels[mask][:n]
            scale = (config.IMG_HEIGHT / 28, config.IMG_WIDTH / 28)
            images = np.stack([scipy_zoom(im, scale, order=1) for im in images])
            X = (images.reshape(len(images), -1).astype(np.float32) / 255.0 - 0.5) * 2.0
            lmap = {d: i for i, d in enumerate(config.TARGET_DIGITS)}
            y = np.eye(config.NUM_CLASSES, dtype=np.float32)[
                np.vectorize(lmap.get)(labels)]
            return X, y

        X_train, y_train = process(imgs, lbls, n_train)
        X_test, y_test = process(test_imgs, test_lbls, n_test)
        print(f"[data] Loaded MNIST: {len(X_train)} train, {len(X_test)} test")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"[data] MNIST load failed ({e}), generating synthetic data")
        rng = np.random.default_rng(SEED)
        X_train = rng.normal(size=(n_train, config.INPUT_DIM)).astype(np.float32) * 0.3
        y_train = np.eye(config.NUM_CLASSES, dtype=np.float32)[
            rng.integers(0, config.NUM_CLASSES, n_train)]
        X_test = rng.normal(size=(500, config.INPUT_DIM)).astype(np.float32) * 0.3
        y_test = np.eye(config.NUM_CLASSES, dtype=np.float32)[
            rng.integers(0, config.NUM_CLASSES, 500)]
        return X_train, y_train, X_test, y_test


# ════════════════════════════════════════════════════════════════════
# 2. PyTorch HVP AND SLQ (matching robustness.ipynb exactly)
# ════════════════════════════════════════════════════════════════════

def _flat_params_count(params):
    """Total number of scalar parameters."""
    return sum(p.numel() for p in params)


def _make_pytorch_loss_fn(model, X_np, y_np, mode="clean", epsilon=None):
    """
    Build a closure that computes loss on the given data using current model params.
    Uses MSE loss to match the QRT codebase (classical_baseline.py).
    """
    x_t = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    criterion = nn.MSELoss()

    if mode == "clean":
        def loss_fn():
            return criterion(model(x_t), y_t)
        return loss_fn

    if epsilon is None:
        epsilon = float(config.EPSILON_TRAIN)

    if mode == "adversarial":
        def loss_fn():
            # FGSM adversarial examples (one-step, matching the QRT approach)
            x_adv = x_t.detach().requires_grad_(True)
            clean_loss = criterion(model(x_adv), y_t)
            grad_x = torch.autograd.grad(clean_loss, x_adv, retain_graph=False)[0]
            x_adv_final = (x_t + epsilon * torch.sign(grad_x.detach())).detach()
            return criterion(model(x_adv_final), y_t)
        return loss_fn

    if mode == "combined":
        alpha = float(getattr(config, "LOSS_ALPHA", 0.5))
        def loss_fn():
            l_clean = criterion(model(x_t), y_t)
            x_adv = x_t.detach().requires_grad_(True)
            clean_for_grad = criterion(model(x_adv), y_t)
            grad_x = torch.autograd.grad(clean_for_grad, x_adv, retain_graph=False)[0]
            x_adv_final = (x_t + epsilon * torch.sign(grad_x.detach())).detach()
            l_adv = criterion(model(x_adv_final), y_t)
            return alpha * l_clean + (1.0 - alpha) * l_adv
        return loss_fn

    raise ValueError(f"Unknown mode: {mode}")


def slq_spectral_density(model, X_np, y_np, mode="clean",
                         num_probes=NUM_PROBES, num_lanczos=NUM_LANCZOS,
                         grid_bins=SLQ_SPECTRUM_BINS, epsilon=None,
                         state=None):
    """
    Estimate Hessian spectral density via SLQ using PyTorch autograd HVPs.

    Matches robustness.ipynb methodology:
    - Two-pass autograd: grad → dot with probe → second grad
    - Multiple random probes averaged
    - Gaussian kernel smoothing with fixed bandwidth
    - Stable grid bounds (monotonically narrowing across calls)
    - Normalized density

    Args:
        model: PolyMLP instance with current weights loaded
        X_np, y_np: numpy arrays for Hessian computation
        mode: 'clean', 'adversarial', or 'combined'
        state: dict carrying stable grid bounds and fixed sigma across calls

    Returns:
        eig_grid: array of eigenvalue grid points (may be stable-narrowed across calls)
        density: array of spectral density values
        raw_bounds: dict with "lam_lo_raw", "lam_hi_raw" = actual spectral bounds at this snapshot
    """
    if state is None:
        state = {}

    was_training = model.training
    model.eval()

    # Build loss closure for this mode
    loss_fn = _make_pytorch_loss_fn(model, X_np, y_np, mode=mode, epsilon=epsilon)

    # Get all trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    dim = _flat_params_count(params)
    num_lanczos = min(num_lanczos, dim)

    def hvp(v_flat):
        """
        Hessian-vector product via two-pass autograd.
        Matches robustness.ipynb exactly:
          1. Compute loss → grad w.r.t. params (with create_graph=True)
          2. Flatten grads → dot product with probe vector
          3. Grad of dot product w.r.t. params → HVP
        """
        # Disable autocast for numerical stability (matching notebook)
        with torch.autocast(device_type="cpu", enabled=False):
            loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([g.contiguous().view(-1) for g in grads])
        dot = (g_flat * v_flat).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])
        return hv_flat

    # === Lanczos tridiagonalization for each probe ===
    tri_alphas = []
    tri_betas = []

    for _ in range(num_probes):
        v = torch.randn(dim)
        v = v / (v.norm() + 1e-12)
        alphas, betas = [], []
        v_prev = torch.zeros_like(v)
        beta_prev = 0.0

        for k in range(num_lanczos):
            w = hvp(v)
            if k > 0:
                w = w - beta_prev * v_prev
            alpha = torch.dot(v, w)
            w = w - alpha * v
            beta = w.norm()

            alphas.append(alpha.item())
            betas.append(beta.item())

            if beta.item() < 1e-10:
                break
            v_prev, v = v, w / (beta + 1e-12)
            beta_prev = beta.item()

        tri_alphas.append(alphas)
        tri_betas.append(betas)

    # === Determine spectral bounds from all probes' tridiagonals ===
    all_eigs = []
    for alphas, betas in zip(tri_alphas, tri_betas):
        m = len(alphas)
        T = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            T[i, i] = alphas[i]
            if i + 1 < m:
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]
        evals = np.linalg.eigvalsh(T)
        all_eigs.append(evals)

    lam_lo_raw = float(min(e[0] for e in all_eigs))
    lam_hi_raw = float(max(e[-1] for e in all_eigs))
    lam_lo, lam_hi = lam_lo_raw, lam_hi_raw

    # === Stable grid bounds (monotonically narrowing, matching notebook) ===
    if SLQ_STABLE_GRID:
        lo_key, hi_key = f"lo_{mode}", f"hi_{mode}"
        if lo_key not in state:
            state[lo_key] = lam_lo
            state[hi_key] = lam_hi
        else:
            state[lo_key] = max(state[lo_key], lam_lo)
            state[hi_key] = min(state[hi_key], lam_hi)
        lam_lo, lam_hi = state[lo_key], state[hi_key]

    eig_grid = np.linspace(lam_lo, lam_hi, grid_bins)
    density = np.zeros_like(eig_grid)

    # === Fixed sigma (matching notebook) ===
    sigma_key = f"sigma_{mode}"
    if SLQ_FIX_SIGMA:
        if sigma_key not in state:
            initial_span = max(1e-12, lam_hi - lam_lo)
            state[sigma_key] = max(1e-3, SLQ_SIGMA_FRACTION * initial_span)
        sigma = state[sigma_key]
    else:
        sigma = max(1e-3, 0.01 * (lam_hi - lam_lo + 1e-12))

    # === Accumulate SLQ density (Gauss quadrature from tridiag, matching notebook) ===
    for alphas, betas in zip(tri_alphas, tri_betas):
        m = len(alphas)
        T = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            T[i, i] = alphas[i]
            if i + 1 < m:
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]
        evals, evecs = np.linalg.eigh(T)
        # First basis vector e1 determines quadrature weights (|e1^T q_j|^2)
        w0 = evecs[0, :] ** 2
        for lam, weight in zip(evals, w0):
            density += weight * np.exp(-0.5 * ((eig_grid - lam) / sigma) ** 2) / \
                       (np.sqrt(2 * np.pi) * sigma)

    density /= max(1, len(tri_alphas))

    if SLQ_NORMALIZE_DENSITY:
        area = np.trapezoid(density, eig_grid)
        if area > 0:
            density /= area

    if was_training:
        model.train()
    raw_bounds = {"lam_lo_raw": lam_lo_raw, "lam_hi_raw": lam_hi_raw}
    return eig_grid, density, raw_bounds


# ════════════════════════════════════════════════════════════════════
# 3. TRAINING VIA CARLEMAN LINEARIZATION WITH SNAPSHOT CAPTURE
# ════════════════════════════════════════════════════════════════════

def extract_weights(model):
    """Flatten all trainable weights to a single numpy vector."""
    return np.concatenate([p.detach().cpu().numpy().flatten()
                           for p in model.trainable_params()])


def _compute_snapshot_loss(u_vec, X_data, y_data, max_samples=500):
    """Compute MSE loss at given weights for logging."""
    m = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM,
                initial_weights=u_vec)
    m.eval()
    n = min(max_samples, len(X_data))
    x_t = torch.from_numpy(X_data[:n].astype(np.float32))
    y_t = torch.from_numpy(y_data[:n].astype(np.float32))
    with torch.no_grad():
        return float(nn.MSELoss()(m(x_t), y_t))


def run_qrt_for_mode(X_train, y_train, pretrained_u, mode="clean",
                     num_steps=None, num_snapshots=None):
    """
    Run the real train_qrt() from qrt_simulation.py (same code path as main.py)
    and convert its snapshots into (epoch, u_vec, loss) tuples for spectrum analysis.

    Args:
        X_train, y_train: Training data.
        pretrained_u: Flat weight vector from adversarial pre-training.
        mode: 'clean', 'adversarial', or 'combined'.
        num_steps: Override config.TOTAL_STEPS for this run.
        num_snapshots: Max number of evenly-spaced snapshots to keep.

    Returns:
        snapshots: list of (epoch_label, u_vector, loss) tuples.
    """
    if num_steps is None:
        num_steps = QRT_TOTAL_STEPS
    if num_snapshots is None:
        num_snapshots = TRAIN_EPOCHS

    orig_combined = getattr(config, "USE_COMBINED_LOSS", False)
    orig_alpha = float(getattr(config, "LOSS_ALPHA", 0.5))
    orig_total_steps = int(getattr(config, "TOTAL_STEPS", 24000))

    if mode == "clean":
        config.USE_COMBINED_LOSS = False
    elif mode == "adversarial":
        config.USE_COMBINED_LOSS = True
        config.LOSS_ALPHA = 0.0
    elif mode == "combined":
        config.USE_COMBINED_LOSS = True
        config.LOSS_ALPHA = 0.5
    else:
        raise ValueError(f"Unknown mode: {mode}")

    config.TOTAL_STEPS = num_steps

    try:
        history, u_final, qrt_snapshots = train_qrt(
            X_train, y_train,
            initial_weights=pretrained_u,
            eval_interval=config.QRT_EVAL_INTERVAL,
        )

        # Subsample to num_snapshots evenly-spaced entries
        if len(qrt_snapshots) > num_snapshots + 1:
            indices = np.round(
                np.linspace(0, len(qrt_snapshots) - 1, num_snapshots + 1)
            ).astype(int)
            qrt_snapshots = [qrt_snapshots[i] for i in indices]

        snapshots = []
        for idx, (step, u_vec) in enumerate(qrt_snapshots):
            loss = _compute_snapshot_loss(u_vec, X_train, y_train)
            snapshots.append((idx, u_vec.copy(), loss))
            print(f"    [{mode:>12s}] Snapshot {idx} (step {step}) | "
                  f"Loss = {loss:.4f} | |u| = {np.linalg.norm(u_vec):.4f}")

        return snapshots

    finally:
        config.USE_COMBINED_LOSS = orig_combined
        config.LOSS_ALPHA = orig_alpha
        config.TOTAL_STEPS = orig_total_steps


# ════════════════════════════════════════════════════════════════════
# 4. COMPUTE SPECTRAL DENSITY AT ALL SNAPSHOTS
# ════════════════════════════════════════════════════════════════════

def compute_spectra_at_snapshots(snapshots, X_hess, y_hess, mode, epsilon):
    """
    For each weight snapshot, compute the SLQ spectral density.
    Returns list of dicts {'epoch', 'eigs_grid', 'density', 'stats'}.
    """
    results = []
    slq_state = {}  # per-mode state for stable grid & fixed sigma

    for idx, (epoch, u_np, train_loss) in enumerate(snapshots):
        t0 = time.time()
        model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM,
                        initial_weights=u_np)
        model.eval()

        eig_grid, density, raw_bounds = slq_spectral_density(
            model, X_hess, y_hess, mode=mode,
            num_probes=NUM_PROBES, num_lanczos=NUM_LANCZOS,
            epsilon=epsilon, state=slq_state)

        # Summary statistics: use actual spectral bounds at this snapshot (not grid)
        stats = {}
        if len(eig_grid) > 0:
            stats["min_eig"] = float(raw_bounds["lam_lo_raw"])
            stats["max_eig"] = float(raw_bounds["lam_hi_raw"])
            stats["avg_density"] = float(density.mean())

        gap = stats["max_eig"] - stats["min_eig"]
        elapsed = time.time() - t0
        print(f"    [{mode:>12s}] Epoch {epoch:>2d} | SLQ {elapsed:.1f}s | "
              f"λ∈[{stats['min_eig']:.3e},{stats['max_eig']:.3e}] gap={gap:.3e} | "
              f"avg density={density.mean():.3e}")

        results.append({
            "epoch": epoch,
            "eigs_grid": eig_grid,
            "density": density,
            "stats": stats,
        })

    return results


# ════════════════════════════════════════════════════════════════════
# 5. RIDGELINE PLOT (matching robustness.ipynb exactly)
# ════════════════════════════════════════════════════════════════════

def plot_ridge(slq_records, mode_label, filename):
    """
    Ridgeline plot matching robustness.ipynb:
    - Log₁₀ density
    - Magenta → Cyan color gradient
    - Dual y-axes (epoch on left, density scale on right)
    - Faint dashed grid lines at each tracked epoch
    """
    if not slq_records:
        print(f"  [warn] No records to plot for {mode_label}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    steps = [r["epoch"] for r in slq_records]
    n = len(steps)

    ymin = min(steps) - 1
    ymax = max(steps) + 2

    # Faint dashed grid lines at each tracked step
    for s in steps:
        ax.axhline(s, color=(0, 0, 0, 0.15), linestyle="--", linewidth=0.8, zorder=0)

    scale = (steps[1] - steps[0]) * 0.8 if n > 1 else 1.0
    max_abs_density = 0

    for i, rec in enumerate(slq_records):
        x = rec["eigs_grid"]
        y = rec["density"].copy()
        if SLQ_LOG_DENSITY:
            y = np.log10(np.maximum(y, 1e-12))
        cur_max = np.max(np.abs(y)) + 1e-12
        max_abs_density = max(max_abs_density, cur_max)
        y_scaled = y / cur_max * scale

        # Magenta → Cyan gradient (matching notebook exactly)
        c = (1 - i / max(n - 1, 1), i / max(n - 1, 1), 1.0)
        ax.plot(x, y_scaled + rec["epoch"], lw=1.5, color=c)
        ax.fill_between(x, rec["epoch"], y_scaled + rec["epoch"],
                        alpha=0.18, color=c)

    ax.set_xlabel("Eigenvalue", fontsize=13)
    ax.set_ylabel("Epoch", fontsize=13)
    ax.set_title(f"Hessian spectral density (SLQ) — {mode_label}", fontsize=14)
    ax.set_ylim(ymin, ymax)

    # Secondary (right) y-axis: density scale (matching notebook)
    ax2 = ax.twinx()
    density_per_unit = max_abs_density / scale if scale > 0 else 1
    ax2.set_ylim(0, (ymax - ymin) * density_per_unit)
    ax2.set_ylabel("Log density" if SLQ_LOG_DENSITY else "Density", fontsize=13)

    fig.tight_layout()
    plt.savefig(filename)
    print(f"  ✓ Ridge plot saved: {filename}")
    plt.close()


def plot_comparison(all_records, filename):
    """Overlay the FINAL spectral density for all three modes on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"clean": "tab:blue", "adversarial": "tab:red", "combined": "tab:green"}
    labels = {"clean": "Clean Loss", "adversarial": "Adversarial Loss",
              "combined": "Combined Loss"}

    for mode, records in all_records.items():
        if not records:
            continue
        final = records[-1]
        x = final["eigs_grid"]
        y = final["density"].copy()
        if SLQ_LOG_DENSITY:
            y = np.log10(np.maximum(y, 1e-12))

        ax.fill_between(x, 0, y, alpha=0.25, color=colors[mode])
        ax.plot(x, y, color=colors[mode], linewidth=2,
                label=f"{labels[mode]} (Epoch {final['epoch']})")

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Eigenvalue (λ)")
    ax.set_ylabel("Log Spectral Density" if SLQ_LOG_DENSITY else "Spectral Density")
    ax.set_title("Hessian Spectrum Comparison — Final Epoch",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"  ✓ Comparison plot saved: {filename}")
    plt.close()


def plot_eigenvalue_stats(all_records, filename):
    """Plot evolution of key spectral statistics over training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"clean": "tab:blue", "adversarial": "tab:red", "combined": "tab:green"}
    labels_map = {"clean": "Clean", "adversarial": "Adversarial",
                  "combined": "Combined"}

    for mode, records in all_records.items():
        if not records:
            continue
        epochs = [r["epoch"] for r in records]
        min_eigs = [r["stats"]["min_eig"] for r in records]
        max_eigs = [r["stats"]["max_eig"] for r in records]

        # λ_max and λ_min (actual per-snapshot bounds, not grid)
        axes[0].plot(epochs, max_eigs, "o-", color=colors[mode], linewidth=1.5,
                     markersize=4, label=f"{labels_map[mode]} λ_max")
        axes[0].plot(epochs, min_eigs, "s--", color=colors[mode], linewidth=1.2,
                     markersize=3, alpha=0.7, label=f"{labels_map[mode]} λ_min")

        # Spectral gap
        gaps = [mx - mn for mx, mn in zip(max_eigs, min_eigs)]
        axes[1].plot(epochs, gaps, "o-", color=colors[mode], linewidth=1.5,
                     markersize=4, label=labels_map[mode])

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Extreme Eigenvalues")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.2)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Spectral Gap (λ_max − λ_min)")
    axes[1].set_title("Spectral Gap Evolution")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.suptitle("Hessian Spectral Statistics Over Training",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  ✓ Stats plot saved: {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════

def print_summary_table(all_records):
    """Print a formatted summary table of final-epoch spectral statistics."""
    print("\n" + "=" * 80)
    print("HESSIAN SPECTRUM SUMMARY (Final Epoch)")
    print("=" * 80)
    header = f"{'Mode':<15s} {'λ_min':>12s} {'λ_max':>12s} {'Gap':>12s} {'Avg Density':>14s}"
    print(header)
    print("-" * 80)

    for mode in ["clean", "adversarial", "combined"]:
        records = all_records.get(mode, [])
        if not records:
            continue
        final = records[-1]
        lam_lo = final["stats"]["min_eig"]
        lam_hi = final["stats"]["max_eig"]
        gap = lam_hi - lam_lo
        avg_d = float(final["density"].mean())
        print(f"{mode:<15s} "
              f"{lam_lo:>12.4e} "
              f"{lam_hi:>12.4e} "
              f"{gap:>12.4e} "
              f"{avg_d:>14.4e}")

    print("=" * 80)
    print("Note: If the spectrum does not contract over snapshots, Carleman-evolved")
    print("trajectories can differ from gradient descent and may not reach flatter minima.")
    print("=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════
# 7. MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  HESSIAN SPECTRAL DENSITY via Stochastic Lanczos Quadrature")
    print("  Pipeline: Adversarial pre-train → train_qrt (Carleman + GMRES)")
    print("            → Hessian spectrum at weight snapshots")
    print("  Carleman truncation N=%d" % config.CARLEMAN_N)
    print("=" * 70)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Load data ──
    X_train, y_train, X_test, y_test = load_data(
        n_train=config.TRAIN_SIZE, n_test=config.TEST_SIZE)

    # ── Shared random initialisation ──
    model_init = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    initial_u = extract_weights(model_init)
    print(f"[model] Param dim = {initial_u.shape[0]}, "
          f"|u₀| = {np.linalg.norm(initial_u):.4f}")

    epsilon = float(config.EPSILON_TRAIN)
    alpha = float(getattr(config, "LOSS_ALPHA", 0.5))
    print(f"[config] ε_train = {epsilon}, α = {alpha}, "
          f"Lanczos = {NUM_LANCZOS}, Probes = {NUM_PROBES}")
    print(f"[config] Carleman N = {config.CARLEMAN_N}, "
          f"QRT steps = {QRT_TOTAL_STEPS}, "
          f"relinearize every = {config.RELINEARIZE_INTERVAL}")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1 — Classical adversarial pre-training (warm-up)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"  PHASE 1: ADVERSARIAL PRE-TRAINING ({PRETRAIN_EPOCHS} epochs, PGD-AT)")
    print(f"{'─' * 60}")

    trainer = RobustTrainer(model_init)
    train_ds = list(create_dataset_iterator(
        X_train, y_train, batch_size=config.CLASSICAL_BATCH_SIZE))
    test_ds = list(create_dataset_iterator(
        X_test, y_test, batch_size=config.EVAL_BATCH_SIZE))

    # Capture snapshots during pre-training for Hessian analysis
    pretrain_snapshots = []
    u_ep0 = extract_weights(model_init)
    loss_ep0 = _compute_snapshot_loss(u_ep0, X_train, y_train)
    pretrain_snapshots.append((0, u_ep0.copy(), loss_ep0))
    print(f"  [pretrain] Epoch 0 (init) | loss={loss_ep0:.4f}")

    for ep in range(PRETRAIN_EPOCHS):
        loss = trainer.train_epoch_robust(train_ds, epoch=ep)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [pretrain] Epoch {ep+1}/{PRETRAIN_EPOCHS} | loss={loss:.4f}")
            trainer.evaluate(test_ds)
        if (ep + 1) % PRETRAIN_SNAP_EVERY == 0 or ep == PRETRAIN_EPOCHS - 1:
            u_ep = extract_weights(model_init)
            snap_loss = _compute_snapshot_loss(u_ep, X_train, y_train)
            pretrain_snapshots.append((ep + 1, u_ep.copy(), snap_loss))

    pretrained_u = extract_weights(model_init)
    print(f"\n[pretrain] Done. |u_pretrained| = {np.linalg.norm(pretrained_u):.4f}")
    print(f"[pretrain] Captured {len(pretrain_snapshots)} snapshots "
          f"(epochs: {[s[0] for s in pretrain_snapshots]})")

    # ── Data batch for Hessian evaluation ──
    rng_hess = np.random.default_rng(config.RANDOM_SEED)
    batch_sz = int(getattr(config, "QRT_BATCH_SIZE", 5))
    hess_idx = rng_hess.choice(len(X_train),
                               size=min(batch_sz, len(X_train)), replace=False)
    X_hess = X_train[hess_idx]
    y_hess = y_train[hess_idx]
    print(f"[hessian] Using {len(X_hess)} samples for Hessian evaluation\n")

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2 — QRT (Carleman + GMRES) per loss mode
    # ══════════════════════════════════════════════════════════════
    all_records = {}

    for mode in ["clean", "adversarial", "combined"]:
        print(f"\n{'─' * 60}")
        print(f"  PHASE 2 [{mode.upper()}]: train_qrt "
              f"({QRT_TOTAL_STEPS} steps, N={config.CARLEMAN_N})")
        print(f"{'─' * 60}")

        # 1) Run the real train_qrt (same code path as main.py)
        qrt_snapshots = run_qrt_for_mode(
            X_train, y_train, pretrained_u, mode=mode,
            num_steps=QRT_TOTAL_STEPS, num_snapshots=TRAIN_EPOCHS)

        # Merge: pre-training snapshots (labelled by epoch) + QRT snapshots
        # (labelled continuing from last pretrain epoch)
        last_pretrain_epoch = pretrain_snapshots[-1][0]
        snapshots = list(pretrain_snapshots)
        for idx, (_, u_vec, loss_val) in enumerate(qrt_snapshots):
            snapshots.append((last_pretrain_epoch + 1 + idx, u_vec, loss_val))

        # 2) Compute Hessian spectral density at each snapshot
        print(f"\n  [slq] Computing SLQ at {len(snapshots)} snapshots "
              f"({len(pretrain_snapshots)} pretrain + {len(qrt_snapshots)} QRT)...")
        records = compute_spectra_at_snapshots(
            snapshots, X_hess, y_hess, mode=mode, epsilon=epsilon)
        all_records[mode] = records

        # 3) Ridge plot for this mode
        plot_ridge(records, f"{mode.capitalize()} Loss", f"hessian_ridge_{mode}.png")

    # ── Cross-mode comparison ──
    print(f"\n{'─' * 60}")
    print(f"  GENERATING COMPARISON PLOTS")
    print(f"{'─' * 60}")

    plot_comparison(all_records, "hessian_comparison_final.png")
    plot_eigenvalue_stats(all_records, "hessian_stats_evolution.png")

    # ── Summary table ──
    print_summary_table(all_records)

    elapsed = time.time() - t_start
    print(f"\n✓ Total time: {elapsed:.1f}s")
    print("✓ Output files:")
    print("    hessian_ridge_clean.png         — Clean loss ridge plot")
    print("    hessian_ridge_adversarial.png   — Adversarial loss ridge plot")
    print("    hessian_ridge_combined.png      — Combined loss ridge plot")
    print("    hessian_comparison_final.png    — All three final spectra overlaid")
    print("    hessian_stats_evolution.png     — Spectral statistics over training")


if __name__ == "__main__":
    main()
