"""Quantum-aware linear solver helpers.

This module centralizes the logic for turning linear updates that arise in the
Carleman linearization into circuits that can be executed by the MindSpore
Quantum package (``mindquantum``). When ``mindquantum`` is unavailable, we
fall back to a well-conditioned classical solver so that the codebase remains
executable in minimal environments while still exposing a single entry point
for HHL-style solves.
"""

from __future__ import annotations

import math
from typing import Callable

import importlib
import importlib.util
import traceback
import inspect

import numpy as np
import scipy.sparse.linalg as spla

_MQ_SPEC = None
_MQ_SIM_SPEC = None
MQ_AVAILABLE = False
_HHL_IMPORT_ERROR: str | None = None
Simulator = None
hhl = None

try:
    _MQ_SPEC = importlib.util.find_spec("mindquantum")
    _MQ_SIM_SPEC = importlib.util.find_spec("mindquantum.simulator")
    if _MQ_SPEC is not None and _MQ_SIM_SPEC is not None:
        _mq_simulator = importlib.import_module("mindquantum.simulator")
        Simulator = getattr(_mq_simulator, "Simulator", None)
        try:
            from hhl_provider import hhl as hhl  # type: ignore[assignment]
        except Exception as exc:
            _HHL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
            hhl = None
        MQ_AVAILABLE = hhl is not None and Simulator is not None
except (ModuleNotFoundError, ImportError):
    # mindquantum not installed: use GMRES fallback only
    pass


def _dense_from_matvec(matvec: Callable[[np.ndarray], np.ndarray], dim: int) -> np.ndarray:
    """Materialize a dense matrix from a linear matvec oracle.

    The HHL circuit needs access to the full matrix. When only a callable is
    available we probe it on the standard basis to reconstruct the columns.
    This is acceptable because the lifted system size is intentionally small.
    """

    basis = np.eye(dim, dtype=np.float64)
    cols = [matvec(col) for col in basis]
    return np.stack(cols, axis=1)


def _gmres_solve(
    lin_op: spla.LinearOperator,
    b: np.ndarray,
    precision: float,
    *,
    restart: int = 50,
    maxiter: int = 200,
) -> tuple[np.ndarray, int]:
    """Call SciPy GMRES with signature-compatible tolerance arguments."""
    gmres_params = inspect.signature(spla.gmres).parameters
    if "tol" in gmres_params:
        return spla.gmres(lin_op, b, tol=precision, restart=restart, maxiter=maxiter)
    # Newer SciPy uses rtol/atol
    return spla.gmres(lin_op, b, rtol=precision, atol=0.0, restart=restart, maxiter=maxiter)


def _run_quantum_hhl(matrix: np.ndarray, b: np.ndarray, precision: float) -> np.ndarray:
    """Run HHL on ``matrix`` and ``b`` using mindquantum when available."""

    if not MQ_AVAILABLE:
        detail = f" (hhl_provider import error: {_HHL_IMPORT_ERROR})" if _HHL_IMPORT_ERROR else ""
        raise ImportError(
            "mindquantum with the local hhl_provider implementation is required for the quantum HHL branch"
            f"{detail}"
        )

    def _is_hermitian(mat: np.ndarray, atol: float = 1e-10) -> bool:
        return np.allclose(mat, mat.conj().T, atol=atol, rtol=0.0)

    def _spd_lift(mat: np.ndarray, rhs: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
        """Map A x=b to (A^H A + ridge I) x = A^H b (Hermitian PD for ridge>0)."""
        ah = mat.conj().T
        spd = ah @ mat
        spd = 0.5 * (spd + spd.conj().T)  # symmetrize against numerical noise
        spd = spd + ridge * np.eye(spd.shape[0], dtype=spd.dtype)
        rhs2 = ah @ rhs
        return spd, rhs2

    matrix = np.asarray(matrix, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    original_dim = int(matrix.shape[0])

    # If the matrix is not Hermitian PD, fall back to normal equations so HHL's
    # Hermitian+PD preconditions are satisfied. This changes conditioning (κ²)
    # and solves a regularized least-squares variant when ridge>0.
    if not _is_hermitian(matrix):
        ridge = float(max(precision, 1e-12))
        matrix, b = _spd_lift(matrix, b, ridge=ridge)

    # mindquantum's reference implementation returns a tuple containing the
    # circuit and the classical post-processing helper. We follow the public
    # API documented in the library examples.
    hhl_circ, state_prep, result_decoder = hhl(matrix, b)

    num_qubits = hhl_circ.n_qubits
    sim = Simulator("mqvector", num_qubits)

    # mindquantum: get_qs(True) returns a ket string; we need the numeric statevector.
    state = sim.get_qs()
    state = state_prep(state)
    sim.reset()
    sim.set_qs(state)

    sim.apply_circuit(hhl_circ)
    final_state = sim.get_qs()
    solution = result_decoder(final_state, precision)
    solution = np.asarray(solution).reshape(-1)
    # hhl_provider pads the matrix/vector to a power-of-two dimension internally.
    # For downstream callers (and fidelity checks), we return the solution in the
    # original problem dimension.
    if solution.size >= original_dim:
        solution = solution[:original_dim]
    return np.real_if_close(solution)


def solve_linear_system_quantum(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    max_dense_dim: int = 4096,
) -> np.ndarray:
    """Solve ``A x = b`` using an HHL-style routine.

    Args:
        matvec: Callable implementing ``A @ x`` for the implicit system matrix.
        b: Right-hand-side vector.
        precision: Target solver tolerance.
        noise_scale: Optional noise scale applied to mimic quantum readout
            noise; useful when running in a purely classical fallback mode.
    """

    dim = len(b)

    # Important: materializing an implicit operator into a dense matrix is O(dim^2)
    # in both time and memory. The Carleman-lifted state can be large even when the
    # base state dimension is modest, so we hard-cap dense reconstruction.
    if dim > max_dense_dim:
        lin_op = spla.LinearOperator((dim, dim), matvec=matvec)
        x, _ = _gmres_solve(lin_op, b, precision, restart=50, maxiter=200)
        return x

    dense_matrix = _dense_from_matvec(matvec, dim)

    try:
        x = _run_quantum_hhl(dense_matrix, b, precision)
    except Exception:
        # Graceful fallback: use GMRES while preserving the interface.
        lin_op = spla.LinearOperator((dim, dim), matvec=matvec)
        x, _ = _gmres_solve(lin_op, b, precision, restart=50, maxiter=200)

    return x


def solve_linear_system_quantum_with_info(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    precision: float,
    *,
    max_dense_dim: int = 4096,
) -> tuple[np.ndarray, dict[str, object]]:
    """Like :func:`solve_linear_system_quantum`, but returns backend + residual info."""
    dim = len(b)
    info: dict[str, object] = {"dim": dim, "precision": float(precision)}

    def _residual(x: np.ndarray) -> float:
        r = matvec(x) - b
        denom = float(np.linalg.norm(b) + 1e-12)
        return float(np.linalg.norm(r) / denom)

    if dim > max_dense_dim:
        lin_op = spla.LinearOperator((dim, dim), matvec=matvec)
        x, gmres_info = _gmres_solve(lin_op, b, precision, restart=50, maxiter=200)
        info["backend"] = "gmres"
        info["gmres_info"] = int(gmres_info)
        info["rel_residual"] = _residual(x)
        return x, info

    dense_matrix = _dense_from_matvec(matvec, dim)
    try:
        x = _run_quantum_hhl(dense_matrix, b, precision)
        info["backend"] = "mindquantum_hhl"
        info["rel_residual"] = _residual(x)
        return x, info
    except Exception as exc:
        lin_op = spla.LinearOperator((dim, dim), matvec=matvec)
        x, gmres_info = _gmres_solve(lin_op, b, precision, restart=50, maxiter=200)
        info["backend"] = "gmres_fallback"
        info["gmres_info"] = int(gmres_info)
        info["hhl_error"] = f"{type(exc).__name__}: {exc}"
        info["rel_residual"] = _residual(x)
        return x, info


class MindQuantumHHL:
    """Run MindQuantum's HHL solver and report fidelity to a classical solve."""

    def __init__(self, precision: float = 1e-6) -> None:
        if not MQ_AVAILABLE:
            detail = f" (hhl_provider import error: {_HHL_IMPORT_ERROR})" if _HHL_IMPORT_ERROR else ""
            raise RuntimeError(
                "mindquantum with the local hhl_provider implementation is not available; "
                "please ensure hhl_provider.py is importable."
                f"{detail}"
            )
        self.precision = precision
        self.last_backend: str | None = None
        self.last_error: str | None = None

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve ``matrix @ x = rhs`` and return (solution, fidelity)."""
        classical = np.linalg.solve(matrix, rhs)
        try:
            solution = _run_quantum_hhl(np.asarray(matrix), np.asarray(rhs), self.precision)
            fidelity = _solution_fidelity(np.asarray(solution), np.asarray(classical))
            self.last_backend = "quantum"
            self.last_error = None
            return solution, fidelity
        except Exception as exc:
            # Robust fallback: return the classical answer and mark fidelity NaN.
            self.last_backend = "classical"
            self.last_error = traceback.format_exc()
            return classical, float("nan")


def _solution_fidelity(quantum_solution: np.ndarray, classical_solution: np.ndarray) -> float:
    """Compute the cosine-similarity-style fidelity between two vectors."""
    q = quantum_solution.astype(np.complex128)
    c = classical_solution.astype(np.complex128)
    q_norm = np.linalg.norm(q)
    c_norm = np.linalg.norm(c)
    if q_norm == 0 or c_norm == 0:
        return 0.0
    overlap = np.vdot(q / q_norm, c / c_norm)
    return float(np.abs(overlap))
