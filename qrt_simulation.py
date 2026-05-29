import importlib.util
import time
from typing import Optional

import numpy as np
import torch

import config
from classical_baseline import PolyMLP, RobustTrainer
from quantum_hhl import solve_linear_system_quantum, solve_linear_system_quantum_with_info

_JAX_SPEC = importlib.util.find_spec("jax")
if _JAX_SPEC is None:
    raise RuntimeError("jax is required for qrt_simulation.py (JAX-based Carleman coefficients).")

import jax
import jax.numpy as jnp

from carleman_coeffs import build_vector_field, compute_coeffs


def _resolve_w_dim(objective_mode: str, batch_size: int) -> int:
    if objective_mode == "batch_perturbation":
        w_dim_mode = str(getattr(config, "QRT_W_DIM_MODE", "shared")).lower()
        if w_dim_mode == "per_sample":
            return int(batch_size) * int(config.INPUT_DIM)
    return int(config.INPUT_DIM)

class CarlemanSystem:
    """
    Implements the implicit Carleman Linearization logic.
    Dynamics: dot(v) = F_0 + F_1 v + F_2 v^2 (Quadratic approximation)
    """
    def __init__(self, model, data, labels, v_operating_point=None):
        self.model = model
        self.data = data # Fixed batch for dynamics
        self.labels = labels
        self.objective_mode = str(getattr(config, "QRT_OBJECTIVE_MODE", "single_sample")).lower()
        if self.objective_mode == "batch_perturbation" and isinstance(self.data, np.ndarray):
            self.batch_size = int(self.data.shape[0])
        else:
            self.batch_size = 1
        self.w_dim = _resolve_w_dim(self.objective_mode, self.batch_size)
        # Use only trainable parameters (exclude fixed projection) for Carleman state.
        self.u_params = list(model.trainable_params())
        self.u_dim = int(sum(int(p.numel()) for p in self.u_params))
        self.dim = self.w_dim + self.u_dim
        self.truncation = config.CARLEMAN_N
        
        # Flattened parameter shapes for reconstruction
        self.param_shapes = [tuple(int(d) for d in p.shape) for p in self.u_params]
        self.param_sizes = [int(p.numel()) for p in self.u_params]
        self.proj = np.asarray(getattr(self.model, "proj").detach().cpu().numpy(), dtype=np.float32)
        
        # Operating point for linearization (v_0)
        # We linearize around v_0 such that delta_v = v - v_0.
        # If not provided, default to the *current* state: w = first sample,
        # u = current model parameters. This avoids the degenerate v_op=0 case
        # where gradients can be exactly zero (and truncation diagnostics become
        # meaningless).
        if v_operating_point is None:
            if self.objective_mode == "batch_perturbation":
                # w is interpreted as a shared additive perturbation, so the natural operating point is 0.
                w0 = np.zeros(self.w_dim, dtype=np.float32)
            else:
                if isinstance(self.data, np.ndarray) and self.data.size > 0:
                    w0 = np.asarray(self.data[0].flatten(), dtype=np.float32)
                else:
                    w0 = np.zeros(self.w_dim, dtype=np.float32)
            u0 = np.concatenate([np.asarray(p.detach().cpu().numpy(), dtype=np.float32).flatten() for p in self.u_params])
            self.v_op = np.concatenate([w0, u0])
        else:
            self.v_op = np.asarray(v_operating_point, dtype=np.float32)
        
        print(f"Initializing Carleman System with Dimension {self.dim} (w:{self.w_dim}, u:{self.u_dim}) and Truncation N={self.truncation}")
        
        # Precompute dynamics coefficients F0, F1, F2 around v_op
        self._extract_coefficients(phase="both")
        
        # Diagnostic output for debugging
        print(f"  F0 norm: {np.linalg.norm(self.F0):.6f}")
        print(f"  F1 norm: {np.linalg.norm(self.F1):.6f}")
        print(f"  F1 spectral radius: {np.max(np.abs(np.linalg.eigvals(self.F1))):.6f}")
        
    def _extract_coefficients(self, phase: str = "u_only"):
        """
        Extract F0, F1, F2 around v_op using JAX automatic differentiation.
        Let delta_v be the state.
        dot(delta_v) = V(v_op + delta_v)
        Approximate RHS as polynomial in delta_v:
        = V(v_op) + V'(v_op) delta_v + 1/2 V''(v_op) delta_v^2
        
        So:
        F0 = V(v_op)
        F1 = V'(v_op)
        F2 = 1/2 V''(v_op)
        
        Args:
            phase: 'u_only' for parameter descent only (recommended for QRT)
                   'both' for coupled w ascent + u descent
        """
        # Loss target(s)
        if self.objective_mode == "batch_perturbation":
            label = np.asarray(self.labels, dtype=np.float32)  # (B, C)
            x_clean = np.asarray(self.data, dtype=np.float32)  # (B, INPUT_DIM)
        else:
            label = np.asarray(self.labels[0], dtype=np.float32)  # (C,)
            x_clean = None
        use_combined = bool(getattr(config, "USE_COMBINED_LOSS", False))
        alpha = float(getattr(config, "LOSS_ALPHA", 0.5))
        eps_local = float(getattr(config, "EPSILON_TRAIN", 0.03))
        weight_decay = float(getattr(config, "QRT_WEIGHT_DECAY", 0.0))

        # IMPORTANT (JAX): keep parameter shapes/sizes out of JIT inputs.
        # We capture them as Python tuples so slicing/reshaping is static and
        # never involves traced values.
        param_shapes = tuple(tuple(int(d) for d in s) for s in self.param_shapes)
        param_sizes = tuple(int(sz) for sz in self.param_sizes)
        proj = jnp.asarray(self.proj)
        y = jnp.asarray(label)
        x_batch = jnp.asarray(x_clean) if x_clean is not None else None
        input_dim = int(config.INPUT_DIM)
        batch_size = int(self.batch_size)
        w_dim_mode = str(getattr(config, "QRT_W_DIM_MODE", "shared")).lower()

        def _unpack(u_in: jnp.ndarray) -> list[jnp.ndarray]:
            mats: list[jnp.ndarray] = []
            pointer = 0
            for shape, size in zip(param_shapes, param_sizes):
                mats.append(jnp.reshape(u_in[pointer:pointer + size], shape))
                pointer += size
            return mats

        # JAX loss function matching the polynomial network used in classical_baseline.PolyMLP.
        # We keep it purely in JAX so higher-order derivatives are well-defined and efficient.
        def _apply_activation(hidden: jnp.ndarray) -> jnp.ndarray:
            act = str(getattr(config, "ACTIVATION", "softmax")).lower()
            if act == "relu":
                return jax.nn.relu(hidden)
            if act == "tanh":
                return jnp.tanh(hidden)
            if act == "softmax":
                return jax.nn.softmax(hidden, axis=-1)
            if act == "square":
                return jnp.square(hidden)
            raise ValueError(f"Unsupported activation: {act}")

        def loss_fn(w: jnp.ndarray, u: jnp.ndarray, _batch: object) -> jnp.ndarray:
            def forward(w_in: jnp.ndarray, u_in: jnp.ndarray) -> jnp.ndarray:
                # Unpack weights (assumes two bias-free dense layers: fc1 and fc2).
                mats = _unpack(u_in)
                fc1_w = mats[0]
                fc2_w = mats[1] if len(mats) > 1 else None

                if self.objective_mode == "batch_perturbation":
                    # Interpret w as a shared perturbation applied to every x in the batch.
                    # x_batch: (B, INPUT_DIM), w_in: (INPUT_DIM,)
                    if w_dim_mode == "per_sample":
                        w_perturb = jnp.reshape(w_in, (batch_size, input_dim))
                        x_adv = x_batch + w_perturb
                    else:
                        x_adv = x_batch + w_in[None, :]
                    x_proj = jnp.matmul(x_adv, proj)  # (B, PROJ_DIM)
                    hidden = jnp.matmul(x_proj, fc1_w.T)  # (B, HIDDEN_DIM)
                    hidden = _apply_activation(hidden)
                    logits = jnp.matmul(hidden, fc2_w.T) if fc2_w is not None else hidden  # (B, C)
                    diff = logits - y
                    return jnp.mean(jnp.square(diff))

                # Legacy: single-sample objective where w itself is the input vector.
                w_proj = jnp.matmul(w_in, proj)  # (PROJ_DIM,)
                hidden = jnp.dot(fc1_w, w_proj)  # (HIDDEN_DIM,)
                hidden = _apply_activation(hidden)
                logits = jnp.dot(fc2_w, hidden) if fc2_w is not None else hidden
                diff = logits - y
                return jnp.mean(jnp.square(diff))

            loss_clean = forward(w, u)
            if not use_combined:
                return loss_clean + 0.5 * weight_decay * jnp.sum(jnp.square(u))

            # Adversarial component: w_adv = w + eps * sign(grad_w loss_clean)
            # Stop gradients through the sign to keep robust components classical.
            grad_w_clean = jax.grad(forward, argnums=0)(w, u)
            perturb = jax.lax.stop_gradient(jnp.sign(grad_w_clean))
            w_adv = w + eps_local * perturb
            loss_adv = forward(w_adv, u)
            return alpha * loss_clean + (1.0 - alpha) * loss_adv + 0.5 * weight_decay * jnp.sum(jnp.square(u))

        # Compute coefficients for the full augmented vector field.
        v_op_jax = jnp.asarray(self.v_op, dtype=jnp.float32)
        coeffs = compute_coeffs(loss_fn, v_op_jax, self.w_dim, None, max_order=2)
        F0 = coeffs["F0"].astype(np.float32)
        F1 = coeffs["F1"].astype(np.float32)
        F2 = coeffs["F2"].astype(np.float32)

        lr = float(getattr(config, "QRT_LEARNING_RATE", 1.0))
        F0 = lr * F0
        F1 = lr * F1
        F2 = lr * F2

        grad_clip = float(getattr(config, "QRT_GRAD_CLIP", 0.0))
        if grad_clip > 0.0:
            f0_norm = float(np.linalg.norm(F0))
            if f0_norm > grad_clip:
                F0 = F0 * (grad_clip / (f0_norm + 1e-8))

        # Store full coefficients (no phase gating) so we can apply phase by zeroing rows
        # between relinearizations without calling JAX again (fix for Problem 1).
        self._F0_full = F0.copy()
        self._F1_full = F1.copy()
        self._F2_full = F2.copy()

        # Apply phase gating (Regime I: modular execution).
        # 'w_only': update w, freeze u.  'u_only': update u, freeze w.  'both': no zeroing.
        if phase == "w_only":
            F0[self.w_dim:] = 0.0
            F1[self.w_dim:, :] = 0.0
            F2[self.w_dim:, :, :] = 0.0
        elif phase == "u_only":
            F0[: self.w_dim] = 0.0
            F1[: self.w_dim, :] = 0.0
            F2[: self.w_dim, :, :] = 0.0

        self.F0 = F0
        self.F1 = F1
        self.F2_dense = F2
        self.f2_mode = "full"
        self.F3_dense = None

        # Export a numpy-friendly vector field for diagnostics / truncation verification.
        vf_core = build_vector_field(loss_fn, self.w_dim)
        vf_compiled = jax.jit(lambda v_in: vf_core(v_in, None))
        self._phase_mode = phase

        def _vf_np(v_np: np.ndarray) -> np.ndarray:
            out = np.asarray(vf_compiled(jnp.asarray(v_np, dtype=jnp.float32)), dtype=np.float32)
            out = lr * out
            if phase == "w_only":
                out[self.w_dim:] = 0.0
            elif phase == "u_only":
                out[: self.w_dim] = 0.0
            return out

        self.vector_field_fn = _vf_np

    # NOTE: The previous MindSpore-based coefficient extraction and finite-difference
    # fallbacks have been intentionally removed. Coefficients are now extracted
    # solely through JAX autodiff to match the project requirements.

    def apply_phase_gating(self, phase: str) -> None:
        """
        Set F0, F1, F2 from stored full coefficients by zeroing rows for the given phase.
        No JAX call; use between relinearizations to switch w_only / u_only (Problem 1 fix).
        """
        if not hasattr(self, "_F0_full") or self._F0_full is None:
            return
        self.F0 = self._F0_full.copy()
        self.F1 = self._F1_full.copy()
        self.F2_dense = self._F2_full.copy()
        if phase == "w_only":
            self.F0[self.w_dim:] = 0.0
            self.F1[self.w_dim:, :] = 0.0
            self.F2_dense[self.w_dim:, :, :] = 0.0
        elif phase == "u_only":
            self.F0[: self.w_dim] = 0.0
            self.F1[: self.w_dim, :] = 0.0
            self.F2_dense[: self.w_dim, :, :] = 0.0
        self._phase_mode = phase

    def get_F2_action(self, delta_v):
        """
        Computes F2 * delta_v^2 implicitly.
        F2 delta_v^2 = 0.5 * (V(v_op + delta_v) + V(v_op - delta_v) - 2 V(v_op))
        """
        val_pos = self.vector_field_fn(self.v_op + delta_v)
        val_neg = self.vector_field_fn(self.v_op - delta_v)
        return 0.5 * (val_pos + val_neg - 2 * self.F0)

    def get_F1_action(self, delta_v):
        """
        Computes F1 * delta_v implicitly.
        """
        # Or use precomputed F1 matrix if fast
        return self.F1 @ delta_v

    def matvec(self, y_hat):
        """
        Implicit A * y_hat supporting truncation N in {2,3,4}.
        We retain the quadratic vector field structure (F0 + F1 y + F2 y^2).
        For N>=3, higher-order F2 contributions that would feed beyond y^(N+1) are dropped
        (standard Carleman truncation).
        """
        # Ensure float math to avoid int-casting errors from external solvers.
        y_hat = np.asarray(y_hat, dtype=np.float32)
        D = self.dim
        N = self.truncation

        # --- Parse levels ---
        levels = []
        offset = 0
        size = D
        for _ in range(N):
            levels.append(y_hat[offset:offset + size])
            offset += size
            size *= D

        z_levels = [np.zeros_like(levels[0], dtype=np.float32)]
        if N >= 2:
            z_levels.append(np.zeros_like(levels[1], dtype=np.float32))
        if N >= 3:
            z_levels.append(np.zeros_like(levels[2], dtype=np.float32))
        if N >= 4:
            z_levels.append(np.zeros_like(levels[3], dtype=np.float32))

        y1 = levels[0]

        # --- Row 1: dot(y) = F0 + F1·y + F2·y² + F3·y³ + ... ---
        z1 = z_levels[0]
        z1 += self.get_F1_action(y1)

        if N >= 2:
            y2 = levels[1].reshape(D, D)
            if getattr(self, "f2_mode", "approx") == "full" and hasattr(self, "F2_dense"):
                z1 += np.einsum('ijk,jk->i', self.F2_dense, y2)
            else:
                diag_y2 = np.diag(y2)
                z1 += np.einsum('ij,j->i', self.F2_diag, diag_y2)
                if hasattr(self, "F2_cross"):
                    for (i, j, vec) in self.F2_cross:
                        z1 += vec * y2[i, j]
        
        # Add F3 contribution for N>=3
        if N >= 3 and hasattr(self, "F3_dense") and self.F3_dense is not None:
            y3 = levels[2].reshape(D, D, D)
            for j in range(D):
                z1 += self.F3_dense[:, j, j, j] * y3[j, j, j]

        # --- Row 2: dot(y^2) ---
        if N >= 2:
            z2 = z_levels[1]
            term_21 = np.kron(self.F0, y1) + np.kron(y1, self.F0)
            z2 += term_21

            y2_mat = y2
            term_22 = self.F1 @ y2_mat + y2_mat @ self.F1.T
            z2 += term_22.flatten()

        # --- Row 3: dot(y^3) (truncated, ignore F2->y4 terms) ---
        if N >= 3:
            z3 = z_levels[2]
            y3_tensor = levels[2].reshape(D, D, D)

            term3 = np.einsum('li,ijk->ljk', self.F1, y3_tensor)
            term3 += np.einsum('lj,ijk->ilk', self.F1, y3_tensor)
            term3 += np.einsum('lk,ijk->ijl', self.F1, y3_tensor)
            z3 += term3.reshape(-1)

        # --- Row 4: dot(y^4) (truncated, ignore F2->y5 terms) ---
        if N >= 4:
            z4 = z_levels[3]
            y4_tensor = levels[3].reshape(D, D, D, D)

            # Apply F1 at each of the 4 tensor positions
            # Use 'm' for output index from F1 to avoid conflicts
            term4 = np.einsum('mi,ijkl->mjkl', self.F1, y4_tensor)
            term4 += np.einsum('mj,ijkl->imkl', self.F1, y4_tensor)
            term4 += np.einsum('mk,ijkl->ijml', self.F1, y4_tensor)
            term4 += np.einsum('ml,ijkl->ijkm', self.F1, y4_tensor)
            z4 += term4.reshape(-1)

        # Flatten all levels back
        out = []
        for lvl in z_levels:
            out.append(lvl.flatten())
        return np.concatenate(out)

    def precompute_dense_F2(self):
        """
        Helper to compute dense F2. 
        Only if we really need it.
        """
        pass

def verify_carleman_truncation(cs, test_point, num_tests=5, perturb_scale: Optional[float] = None):
    """
    Verify if Carleman truncation order is sufficient by comparing
    vector field reconstruction at different orders.
    
    Returns: average relative error over num_tests random perturbations
    """
    errors = []
    
    if perturb_scale is None:
        perturb_scale = float(getattr(config, "CARLEMAN_VERIFY_SCALE", 1e-2))

    for test_idx in range(num_tests):
        # Test at small random perturbation around operating point.
        # Note: in high dimension, scale=0.1 is extremely large and will
        # inevitably make a quadratic truncation look poor.
        v_test = cs.v_op + np.random.randn(cs.dim).astype(np.float32) * perturb_scale
        
        # Ground truth: full vector field evaluation
        v_true = cs.vector_field_fn(v_test)
        
        # Order-N approximation: F0 + F1*v + F2*v^2 + F3*v^3 + ...
        delta_v = v_test - cs.v_op
        v_recon = cs.F0 + cs.F1 @ delta_v
        
        # Add F2 contribution if available
        if hasattr(cs, 'f2_mode'):
            if cs.f2_mode == "full" and hasattr(cs, 'F2_dense'):
                delta_v2 = np.outer(delta_v, delta_v)
                v_recon += np.einsum('ijk,jk->i', cs.F2_dense, delta_v2)
            elif hasattr(cs, 'F2_diag'):
                v_recon += np.einsum('ij,j->i', cs.F2_diag, delta_v * delta_v)
        
        # Add F3 contribution if available (diagonal terms only)
        # Note: F3 is usually disabled due to numerical instability
        if hasattr(cs, 'F3_dense') and cs.F3_dense is not None:
            for j in range(len(delta_v)):
                v_recon += cs.F3_dense[:, j, j, j] * (delta_v[j]**3)
        
        rel_error = np.linalg.norm(v_true - v_recon) / (np.linalg.norm(v_true) + 1e-8)
        errors.append(rel_error)
    
    avg_error = np.mean(errors)
    return avg_error, errors

def _estimate_cond_level1(F1, dt):
    """
    Rough condition number estimate on the first-order block only:
    cond(I - F1*dt). This is cheap (size = dim x dim) and serves as
    a monitor; it does not reflect the lifted block.
    """
    A_lvl1 = np.eye(F1.shape[0]) - F1 * dt
    try:
        return float(np.linalg.cond(A_lvl1))
    except Exception:
        return float("nan")


def hhl_solve_noisy(A_matvec, b, dim, precision):
    """
    Solve Ax = b through a quantum-friendly interface.

    The implementation delegates to :func:`solve_linear_system_quantum`, which
    uses MindQuantum HHL when available and gracefully falls back to a classical
    GMRES solver otherwise.
    """
    if bool(getattr(config, "SOLVER_DIAGNOSTICS", False)):
        x, info = solve_linear_system_quantum_with_info(A_matvec, b, precision)
        backend = info.get("backend", "unknown")
        rel_res = info.get("rel_residual", None)
        gmres_info = info.get("gmres_info", None)
        if gmres_info is not None:
            print(f"    [solver] backend={backend} gmres_info={gmres_info} rel_res={rel_res}")
        else:
            print(f"    [solver] backend={backend} rel_res={rel_res}")
        return x
    return solve_linear_system_quantum(A_matvec, b, precision)

def _project_linf(x: np.ndarray, center: np.ndarray, epsilon: float) -> np.ndarray:
    delta = x - center
    delta = np.clip(delta, -epsilon, epsilon)
    return center + delta

def train_qrt(X_train=None, y_train=None, initial_weights=None, eval_interval=5):
    """
    Executes the Quantum Robust Training loop (Regime I: Modular).
    Returns:
        history: List of metric values (convergence).
        final_model_params: List of numpy arrays representing trained weights u.
        snapshots: List of tuples (step_idx, u_vec) captured every eval_interval steps (and final).
    """
    # 1. Setup System
    rng = np.random.default_rng(config.RANDOM_SEED)
    objective_mode = str(getattr(config, "QRT_OBJECTIVE_MODE", "single_sample")).lower()
    if X_train is None or y_train is None:
        # Fallback to mock data if not provided
        print("[QRT] Warning: No training data provided. Using mock data.")
        data_use = np.random.randn(1, config.INPUT_DIM).astype(np.float32)
        label_use = np.array([[1.0]]).astype(np.float32)
    else:
        # Use a random mini-batch for a more representative linearization
        BATCH_SIZE = int(getattr(config, "QRT_BATCH_SIZE", 64))
        idx = rng.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
        data_use = X_train[idx]
        label_use = y_train[idx]
    
    model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    
    # Initialize weights
    w_init_vec = None
    u_init_vec = None
    
    if initial_weights is not None:
        # Load provided initial weights (same random init as classical for fair comparison)
        model._load_flat_weights(initial_weights)
        u_init_vec = initial_weights
        print(f"[QRT] Initialized with provided weights (norm: {np.linalg.norm(u_init_vec):.4f})")
    else:
        # Fresh random weights (decouple from classical seed)
        rng_local = np.random.default_rng(int(time.time()))
        scale = 0.05
        with torch.no_grad():
            for param in model.trainable_params():
                shape = tuple(int(d) for d in param.shape)
                p_data = rng_local.normal(0.0, scale, size=shape).astype(np.float32)
                param.copy_(torch.from_numpy(p_data))
        u_init_vec = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.trainable_params()])
        print(f"[QRT] Initialized with fresh random weights (norm: {np.linalg.norm(u_init_vec):.4f})")

    # Initial inputs
    if objective_mode == "batch_perturbation":
        # w is an additive perturbation applied across the batch.
        batch_size = int(data_use.shape[0]) if isinstance(data_use, np.ndarray) else 1
        w_dim = _resolve_w_dim(objective_mode, batch_size)
        w_init_vec = np.zeros(w_dim, dtype=np.float32)
        w_center = np.zeros_like(w_init_vec)
    else:
        w_init_vec = data_use[0].flatten()
        # Anchor/center for the threat model ball (L_inf around clean sample).
        # We keep this fixed within a linearization window to avoid "center drift".
        w_center = w_init_vec.copy()
    
    # Operating point (v_0)
    v_op = np.concatenate([w_init_vec, u_init_vec])

    # Initialize Carleman System around v_op
    cs = CarlemanSystem(model, data_use, label_use, v_operating_point=v_op)
    
    # 2. Initialize State Deviation (delta_v)
    # At t=0, delta_v = 0
    delta_v_curr = np.zeros(cs.dim)
    
    # Lift state: y = [delta_v, delta_v^2, delta_v^3 (if enabled)]
    y_total = 0
    block_size = cs.dim
    for _ in range(cs.truncation):
        y_total += block_size
        block_size *= cs.dim
    y_curr = np.zeros(y_total)
    
    history = []
    snapshots = []
    print(f"[QRT] Starting Time-Stepping for {config.TOTAL_STEPS} steps...")
    
    # Verify Carleman truncation is sufficient
    print("\n=== Carleman Truncation Verification ===")
    avg_error, errors = verify_carleman_truncation(cs, v_op, num_tests=5)
    for idx, err in enumerate(errors):
        print(f"  Test {idx+1}: Relative Error = {err:.4%}")
    print(f"  Average Error: {avg_error:.4%}")
    if avg_error > 0.10:
        print(f"  WARNING: Error > 10%! Consider increasing CARLEMAN_N to {cs.truncation + 1}")
    else:
        print(f"  ✓ Truncation order N={cs.truncation} is sufficient")
    print()
    
    # Reconstruct v_curr for tracking
    v_curr = v_op + delta_v_curr
    # Capture initial snapshot before any updates
    snapshots.append((-1, u_init_vec.copy()))
    
    # Relinearization interval (allow dynamics to converge between resets)
    relinearize_interval = config.RELINEARIZE_INTERVAL

    for step in range(config.TOTAL_STEPS):
        # Determine phase: alternate between w-ascent (maximize) and u-descent (minimize)
        is_w_phase = (step % 2 == 1)
        phase_mode = 'w_only' if is_w_phase else 'u_only'

        # Re-linearize periodically with new batch and correct phase
        if step > 0 and (step % relinearize_interval == 0):
            if X_train is not None and y_train is not None:
                idx = rng.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
                data_use = X_train[idx]
                label_use = y_train[idx]
            # Use current state as operating point (continuous evolution)
            v_op_new = v_curr.copy()
            if objective_mode == "batch_perturbation":
                w_center = np.zeros(cs.w_dim, dtype=np.float32)
            else:
                w_center = data_use[0].flatten()  # Track clean sample center
            
            cs = CarlemanSystem(model, data_use, label_use, v_operating_point=v_op_new)
            cs._extract_coefficients(phase="both")  # compute once per relinearization
            cs.apply_phase_gating(phase_mode)      # gate for current step (no JAX)
            
            print(f"[QRT][step {step}] Re-linearized around current state (phase={phase_mode})")
            print(f"  F0 norm: {np.linalg.norm(cs.F0):.6f}")
            print(f"  F1 spectral: {np.max(np.abs(np.linalg.eigvals(cs.F1))):.6f}")

            # Reset deviation relative to new operating point
            delta_v_curr = np.zeros(cs.dim)
            v_curr = cs.v_op.copy()
            y_curr = np.zeros(y_total)
        
        # Apply phase gating from stored full coefficients (no JAX between relinearizations).
        # Step 0: init used phase="both", so gate for current phase. Relinearization steps already gated above.
        if step == 0 or (step % relinearize_interval != 0):
            cs.apply_phase_gating(phase_mode)

        # --- Step 1: Linear Evolution (HHL) ---
        b_vec = np.zeros_like(y_curr)
        b_vec[:cs.dim] = cs.F0

        rhs = y_curr + b_vec * config.TIME_STEP
        
        def system_operator(x):
            # Requested local solve form: (I - hA) y_{t+1} = y_t + h b (Implicit Euler)
            # Corrected from (I + hA) which was unstable for descent dynamics
            return x - cs.matvec(x) * config.TIME_STEP
            
        y_next = hhl_solve_noisy(
            system_operator, 
            rhs, 
            len(y_curr), 
            config.HHL_PRECISION
        )

        # Monitor (rough) condition number on first-order block
        cond_lvl1 = _estimate_cond_level1(cs.F1, config.TIME_STEP)
        
        # --- Step 2: Extract Updated State and Project ---
        delta_v_next = y_next[:cs.dim]
        
        # Reconstruct actual state (relative to current operating point)
        v_next = cs.v_op + delta_v_next
        
        # Extract w and u
        w_next = v_next[:cs.w_dim].copy()
        u_next = v_next[cs.w_dim:].copy()

        # Robust components (classical): sign + L_inf projection between linear-solver steps.
        eps_train = float(getattr(config, "EPSILON_TRAIN", 0.03))
        if is_w_phase:
            # Use the current local gradient (F0) as the ascent direction at the operating point.
            grad_w = cs.F0[:cs.w_dim]
            w_raw = w_next + config.TIME_STEP * np.sign(grad_w)
            w_projected = _project_linf(w_raw, w_center, eps_train)
            v_next[:cs.w_dim] = w_projected
        else:
            # Keep w on the threat-model ball even during u updates.
            v_next[:cs.w_dim] = _project_linf(w_next, w_center, eps_train)
        
        v_next[cs.w_dim:] = u_next
        
        # Update delta_v with projected state
        delta_v_next = v_next - cs.v_op
        
        # Track parameter change for diagnostics
        if step % 10 == 0 or step < 5:
            u_curr_norm = np.linalg.norm(v_next[cs.w_dim:])
            delta_u_norm = np.linalg.norm(v_next[cs.w_dim:] - v_curr[cs.w_dim:])
            delta_w_norm = np.linalg.norm(v_next[:cs.w_dim] - v_curr[:cs.w_dim])
            phase_str = "W-ASCENT" if is_w_phase else "U-DESCENT"
            print(f"[QRT][step {step}][{phase_str}] cond≈{cond_lvl1:.2e} | Δu={delta_u_norm:.6f} | Δw={delta_w_norm:.6f} | u_norm={u_curr_norm:.4f}")
        
        # --- Step 3: Re-encoding (populate lifted state with NORMALIZATION) ---
        # CRITICAL FIX: Normalize tensor products to prevent exponential blow-up
        offset = 0
        block = cs.dim
        
        # Level 1: delta_v (no normalization needed)
        y_next[offset:offset + block] = delta_v_next
        offset += block
        
        # Level 2: delta_v^2 with normalization
        if cs.truncation >= 2:
            block_size = cs.dim * cs.dim
            delta_v2 = np.kron(delta_v_next, delta_v_next)
            # Normalize: ||v⊗v|| scales as ||v||², but we want ||v|| scaling
            norm_factor = np.linalg.norm(delta_v_next) + 1e-8
            delta_v2_normalized = delta_v2 / norm_factor
            y_next[offset:offset + block_size] = delta_v2_normalized
            offset += block_size
        
        # Level 3: delta_v^3 with normalization
        if cs.truncation >= 3:
            block_size = cs.dim * cs.dim * cs.dim
            delta_v3 = np.kron(np.kron(delta_v_next, delta_v_next), delta_v_next)
            # Normalize: ||v⊗v⊗v|| scales as ||v||³, but we want ||v|| scaling
            norm_factor_cubed = (np.linalg.norm(delta_v_next) ** 2) + 1e-8
            delta_v3_normalized = delta_v3 / norm_factor_cubed
            y_next[offset:offset + block_size] = delta_v3_normalized
            offset += block_size
        
        # Level 4: delta_v^4 with normalization
        if cs.truncation >= 4:
            block_size = cs.dim ** 4
            delta_v4 = np.kron(np.kron(np.kron(delta_v_next, delta_v_next), delta_v_next), delta_v_next)
            # Normalize: ||v⊗v⊗v⊗v|| scales as ||v||⁴, but we want ||v|| scaling
            norm_factor_4th = (np.linalg.norm(delta_v_next) ** 3) + 1e-8
            delta_v4_normalized = delta_v4 / norm_factor_4th
            y_next[offset:offset + block_size] = delta_v4_normalized
        
        y_curr = y_next
        v_curr = v_next
        
        # Track convergence proxy (magnitude of parameters)
        history.append(np.linalg.norm(v_curr[cs.w_dim:]))

        # Capture snapshot for later external evaluation (every eval_interval or final)
        if (step % eval_interval == 0) or (step == config.TOTAL_STEPS - 1):
            snapshots.append((step, v_curr[cs.w_dim:].copy()))
    
    # Extract final weights
    u_final = v_curr[cs.w_dim:]
    
    return history, u_final, snapshots
