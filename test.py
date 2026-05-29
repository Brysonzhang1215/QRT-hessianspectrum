"""Quick stress test for per-sample perturbation dimension in QRT."""

from __future__ import annotations

import numpy as np

import config
from qrt_simulation import train_qrt


def main() -> None:
    # Configure per-sample perturbation: w has dimension B * INPUT_DIM.
    config.QRT_OBJECTIVE_MODE = "batch_perturbation"
    config.QRT_W_DIM_MODE = "per_sample"
    config.QRT_BATCH_SIZE = 5
    config.TOTAL_STEPS = 1
    config.RELINEARIZE_INTERVAL = 1

    rng = np.random.default_rng(config.RANDOM_SEED)
    X_train = rng.normal(size=(config.QRT_BATCH_SIZE, config.INPUT_DIM)).astype(np.float32)
    y_idx = rng.integers(0, config.NUM_CLASSES, size=(config.QRT_BATCH_SIZE,))
    y_train = np.eye(config.NUM_CLASSES, dtype=np.float32)[y_idx]

    print(
        "[test] batch_size="
        f"{config.QRT_BATCH_SIZE} | input_dim={config.INPUT_DIM} | "
        f"w_dim={config.QRT_BATCH_SIZE * config.INPUT_DIM}"
    )

    history, u_final, snapshots = train_qrt(X_train, y_train, eval_interval=1)
    print(f"[test] completed steps={len(history)} | u_norm={np.linalg.norm(u_final):.4f} | snapshots={len(snapshots)}")


if __name__ == "__main__":
    main()

