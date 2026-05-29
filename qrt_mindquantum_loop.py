"""Reference QRT loop using JAX coefficients and MindQuantum HHL."""

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np

_JAX_SPEC = importlib.util.find_spec("jax")
if _JAX_SPEC is None:
    raise RuntimeError("jax is required to run qrt_mindquantum_loop.py")

_MQ_SPEC = importlib.util.find_spec("mindquantum")
if _MQ_SPEC is None:
    raise RuntimeError("mindquantum is required to run qrt_mindquantum_loop.py")

import jax
import jax.numpy as jnp

from carleman_coeffs import compute_coeffs, verify_linear_approx
from carleman_operator import build_lifted_system, lift_state
from quantum_hhl import MindQuantumHHL


def toy_loss(w: jnp.ndarray, u: jnp.ndarray, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Simple squared loss: (w·u - target)^2 / 2."""
    target = batch["target"]
    pred = jnp.dot(w, u)
    return 0.5 * jnp.square(pred - target)


def project_lp(
    x: np.ndarray,
    center: np.ndarray,
    epsilon: float,
    p: float = np.inf,
) -> np.ndarray:
    """Project x to the Lp ball around center."""
    delta = x - center
    if p == np.inf:
        delta = np.clip(delta, -epsilon, epsilon)
        return center + delta
    if p == 2:
        norm = np.linalg.norm(delta)
        if norm > epsilon:
            delta = delta * (epsilon / (norm + 1e-12))
        return center + delta
    raise ValueError("Only p=inf or p=2 projections are supported.")


def verify_hhl_solver() -> None:
    """Verify the HHL solver against a simple 2x2 linear system."""
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    try:
        solver = MindQuantumHHL(precision=1e-6)
        hhl_solution, fidelity = solver.solve(matrix, b)
        classical_solution = np.linalg.solve(matrix, b)
        error = np.linalg.norm(hhl_solution - classical_solution)
    except RuntimeError as exc:
        print("HHL verifier skipped:", exc)
        return

    print("HHL verifier:")
    print("  HHL solution:", np.real_if_close(hhl_solution))
    print("  Classical solution:", np.real_if_close(classical_solution))
    backend = getattr(solver, "last_backend", None)
    err_detail = getattr(solver, "last_error", None)
    if backend:
        print(f"  Backend: {backend}")
    if err_detail:
        print(f"  Backend error: {err_detail}")
    print(f"  Fidelity: {fidelity:.6f} | Error norm: {error:.6e}")


def run_qrt(
    steps: int = 5,
    step_size: float = 0.1,
    epsilon: float = 0.2,
    truncation: int = 2,
    seed: int = 0,
) -> None:
    verify_hhl_solver()
    rng = np.random.default_rng(seed)
    dim = 6
    w = rng.normal(size=dim)
    u = rng.normal(size=dim)
    w_center = w.copy()

    v = np.concatenate([w, u])
    v_op = v.copy()
    w_dim = dim
    batch = {"target": jnp.asarray(0.5)}

    # Lift the deviation delta_v = v - v_op (Carleman state is in deviations).
    y = lift_state(np.zeros_like(v), truncation=truncation)
    solver = MindQuantumHHL(precision=1e-6)

    header = f"{'Step':>4} | {'Loss':>10} | {'Grad Norm':>10} | {'Fidelity':>9} | {'Proj Dist':>10}"
    print(header)
    print("-" * len(header))

    for step in range(steps):
        v_jax = jnp.asarray(v)
        coeffs = compute_coeffs(toy_loss, v_jax, w_dim, batch, max_order=truncation)
        verify_linear_approx(toy_loss, v_jax, w_dim, batch)

        F0 = coeffs["F0"]
        F1 = coeffs["F1"]
        F2 = coeffs.get("F2")

        A, b = build_lifted_system(F0, F1, F2, truncation=truncation)
        system_matrix = np.eye(A.shape[0]) + step_size * A
        rhs = y + step_size * b

        y_next, fidelity = solver.solve(system_matrix, rhs)
        if np.isnan(fidelity):
            err_detail = getattr(solver, "last_error", None)
            if err_detail:
                print("  [warn] solver fell back to classical:", err_detail.splitlines()[-1])
        delta_v = y_next[: v.shape[0]]

        grad_w = F0[:w_dim]
        w_raw = w + step_size * np.sign(grad_w)
        w_projected = project_lp(w_raw, w_center, epsilon, p=np.inf)
        proj_dist = np.linalg.norm(w_projected - w_raw)

        # y_next is the *next lifted state*, so its first block is delta_v(t+1),
        # not an increment to add to the current v.
        v = v_op + delta_v
        # Robust components are applied classically between quantum linear-solver steps.
        v[:w_dim] = w_projected
        w = v[:w_dim]
        u = v[w_dim:]
        y = lift_state(v - v_op, truncation=truncation)

        loss_val = float(toy_loss(jnp.asarray(w), jnp.asarray(u), batch))
        grad_norm = float(np.linalg.norm(F0))
        print(f"{step:4d} | {loss_val:10.4f} | {grad_norm:10.4f} | {fidelity:9.4f} | {proj_dist:10.4f}")


if __name__ == "__main__":
    run_qrt()
