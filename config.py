# config.py
# -*- coding: utf-8 -*-
import numpy as np

# Data
IMG_HEIGHT = 12
IMG_WIDTH = 12
INPUT_DIM = IMG_HEIGHT * IMG_WIDTH
NUM_CLASSES = 5  # Classes: 0,1,2,3,4
TARGET_DIGITS = (0, 1, 2, 3, 4)
TRAIN_SIZE = 12000   # modest to match smaller model and full F2
TEST_SIZE = 3000

# Network
# Reduce projection to shrink Carleman state dimension.
PROJ_DIM = 10   # Fixed, non-trainable projection to reduce effective input dim
HIDDEN_DIM = 4  # With PROJ_DIM=10, params = 10*4 + 4*5 = 60 (no biases)
OUTPUT_DIM = NUM_CLASSES
# Params ~ INPUT_DIM*HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM*OUTPUT + OUTPUT

# Carleman Linearization
CARLEMAN_N = 2  # Truncation order (N=2 => quadratic, ~4e4 vars)
POLY_DEGREE = 3 # Degree of the polynomial vector field (non-polynomial activations are approximated)

# Training Dynamics
TIME_STEP = 0.12  # Step size for implicit Euler time-stepping
EPOCHS = 20      # Classical training epochs
TOTAL_STEPS = 24000   # QRT simulation steps (batch size 5 => 24k, batch size 10 => 12k)
RELINEARIZE_INTERVAL = 5  # Re-linearize more frequently to stay near v_op

# Learning rate for QRT dynamics (scales the gradient magnitude)
QRT_LEARNING_RATE = 0.05  # Requested learning rate

# Gradient clipping for QRT dynamics (prevents exponential blowup)
QRT_GRAD_CLIP = 1.0  # Requested gradient clip

# Weight decay for QRT (L2 regularization, stabilizes dynamics)
QRT_WEIGHT_DECAY = 0.0008  # Moderate damping for stability (reduced for better learning)

# Robustness
EPSILON = 0.025  # Evaluation epsilon (L_inf perturbation bound)
EPSILON_TRAIN = 0.025  # Training epsilon
ATTACK_STEP_SIZE = 0.01  # Step size for PGD attack
ATTACK_STEPS = 10  # Number of PGD iterations

# Model / dynamics options
# - "relu": standard ReLU activation
# - "tanh": standard tanh activation
# - "softmax": softmax activation (applied to hidden layer)
ACTIVATION = "softmax"

# Debug / instrumentation
SOLVER_DIAGNOSTICS = False  # Disable verbose solver output for long run

# QRT objective / batching
# - "single_sample": w is the actual adversarial example (legacy behavior; very local)
# - "batch_perturbation": w is a shared additive perturbation applied to a batch; loss is batch-mean
QRT_OBJECTIVE_MODE = "batch_perturbation"
QRT_BATCH_SIZE = 5
QRT_BATCH_SIZES = (5, 10)
# - "shared": w is a single perturbation shared across the batch
# - "per_sample": w is a concatenated per-sample perturbation (B * INPUT_DIM)
QRT_W_DIM_MODE = "per_sample"

# Classical training batching
CLASSICAL_BATCH_SIZE = 5
EVAL_BATCH_SIZE = 100

# Combined Loss Training (TRADES-style)
USE_COMBINED_LOSS = True  # If True, use alpha*Loss_clean + (1-alpha)*Loss_robust
LOSS_ALPHA = 0.5  # Weight for clean loss (0.5 = balanced, higher = favor clean accuracy)

# Simulation
HHL_PRECISION = 1e-6 # Much tighter tolerance
NOISE_SCALE = 0.0 # Disable solver-noise during debugging
NOISE_KAPPA_CAP = 1e3  # cap heuristic kappa used to scale noise
RANDOM_SEED = 42

# QRT evaluation granularity (evaluate robust accuracy every K steps)
QRT_EVAL_INTERVAL = 5

# Carleman / F2 handling
# Options: "approx" (skip heavy full tensor); "full" for richer curvature
F2_MODE = "full"
F2_FULL_DIM_LIMIT = 400  # allow full F2 at current dim ~200
F2_EPS = 1e-3  # finite-difference step for 2nd derivatives in "full" mode
# Approximation params (kept for compatibility)
F2_APPROX_LEVEL = 0.0
F2_TOP_CROSS = 0

# F3 (cubic terms) - Enabled with optimized step size
# Third-order finite differences require LARGER step size than F2!
# Optimal h ≈ (machine_epsilon)^(1/4) ≈ 6e-3 for float32
# We use 3e-3 as a compromise (more conservative)
EXTRACT_F3 = True
F3_EPS = 3e-3  # Larger step size for F3 (reduces numerical error)
F3_USE_FLOAT64 = True  # Use float64 for F3 computation (critical!)

