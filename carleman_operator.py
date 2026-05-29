"""Carleman lift and operator construction utilities."""

from __future__ import annotations

import numpy as np


def build_lifted_operator(
    F1: np.ndarray,
    F2: np.ndarray | None = None,
    truncation: int = 2,
) -> np.ndarray:
    """Build the lifted operator A for truncation orders 1 or 2."""
    if truncation not in (1, 2):
        raise ValueError("Only truncation orders 1 and 2 are supported.")

    dim = F1.shape[0]
    if truncation == 1:
        return F1.copy()

    if F2 is None:
        raise ValueError("F2 is required for truncation order 2.")

    block11 = F1
    block12 = F2.reshape(dim, dim * dim)
    block22 = np.kron(F1, np.eye(dim)) + np.kron(np.eye(dim), F1)
    zeros21 = np.zeros((dim * dim, dim))

    return np.block([[block11, block12], [zeros21, block22]])


def build_lifted_system(
    F0: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray | None = None,
    truncation: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, b) for the lifted linear system ydot = A y + b."""
    A = build_lifted_operator(F1, F2, truncation=truncation)
    if truncation == 1:
        b = F0.copy()
    else:
        b = np.concatenate([F0, np.zeros(F1.shape[0] ** 2)])
    return A, b


def lift_state(delta_v: np.ndarray, truncation: int = 2) -> np.ndarray:
    """Lift delta_v into [delta_v, delta_v ⊗ delta_v, ...] up to order 2."""
    if truncation not in (1, 2):
        raise ValueError("Only truncation orders 1 and 2 are supported.")

    if truncation == 1:
        return delta_v.copy()
    return np.concatenate([delta_v, np.kron(delta_v, delta_v)])
