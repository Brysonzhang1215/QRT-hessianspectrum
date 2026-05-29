"""Lightweight interop helpers between Torch / JAX / NumPy.

Goal: keep datatype conversions explicit and *minimal*.

Notes
- DLPack conversions are typically zero-copy when devices match.
- JAX arrays are immutable; treat shared-memory conversions as read-only.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


def torch_to_numpy(x: Any) -> np.ndarray:
    """Best-effort torch.Tensor -> np.ndarray (CPU copy if needed)."""
    # Avoid importing torch unless used.
    import torch  # type: ignore

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().cpu().numpy()


def numpy_to_torch(x: np.ndarray, *, device: str = "cpu") -> Any:
    """np.ndarray -> torch.Tensor."""
    import torch  # type: ignore

    return torch.from_numpy(np.asarray(x)).to(device)


def torch_to_jax(x: Any) -> Any:
    """torch.Tensor -> jax.Array via DLPack when possible."""
    import torch  # type: ignore

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if importlib.util.find_spec("jax") is None:
        raise RuntimeError("jax is not installed; cannot convert torch -> jax.")

    import jax  # type: ignore
    import jax.dlpack  # type: ignore

    # Ensure contiguous storage for predictable sharing.
    if not x.is_contiguous():
        x = x.contiguous()
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


def jax_to_torch(x: Any) -> Any:
    """jax.Array -> torch.Tensor via DLPack when possible."""
    import torch  # type: ignore

    if importlib.util.find_spec("jax") is None:
        raise RuntimeError("jax is not installed; cannot convert jax -> torch.")

    import jax  # type: ignore
    import jax.dlpack  # type: ignore

    # `jax.dlpack.to_dlpack` exists for modern JAX; keep this minimal.
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


