# main.py
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import config
from classical_baseline import PolyMLP, RobustTrainer, create_dataset_iterator
from qrt_simulation import train_qrt

import gzip
import os
from scipy.ndimage import zoom

def load_or_generate_data():
    """
    Loads MNIST digits in config.TARGET_DIGITS, resizes to config.IMG_HEIGHT x config.IMG_WIDTH, returns data.
    """
    print(f"Loading MNIST data (digits {config.TARGET_DIGITS})...")
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    try:
        train_images = load_images('data/train-images-idx3-ubyte.gz')
        train_labels = load_labels('data/train-labels-idx1-ubyte.gz')
        test_images = load_images('data/t10k-images-idx3-ubyte.gz')
        test_labels = load_labels('data/t10k-labels-idx1-ubyte.gz')
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Falling back to synthetic data...")
        print(f"Generating synthetic {config.IMG_HEIGHT}x{config.IMG_WIDTH} binary data...")
        X = (np.random.rand(config.TRAIN_SIZE + config.TEST_SIZE, config.INPUT_DIM).astype(np.float32) - 0.5) * 2.0
        radius_sq = np.sum(X**2, axis=1)
        threshold = np.median(radius_sq)
        y = (radius_sq > threshold).astype(np.int32)
        y_one_hot = np.eye(config.NUM_CLASSES, dtype=np.float32)[y]
        X_train, X_test = X[:config.TRAIN_SIZE], X[config.TRAIN_SIZE:]
        y_train, y_test = y_one_hot[:config.TRAIN_SIZE], y_one_hot[config.TRAIN_SIZE:]
        return (X_train, y_train), (X_test, y_test)

    # Use first NUM_CLASSES; convert labels to one-hot
    def process_all(images, labels):
        # Filter to target digits
        mask = np.isin(labels, np.array(config.TARGET_DIGITS))
        images = images[mask]
        labels = labels[mask]

        # Downscale to configured resolution for tractability
        if images.shape[1] != config.IMG_HEIGHT or images.shape[2] != config.IMG_WIDTH:
            scale = (config.IMG_HEIGHT / images.shape[1], config.IMG_WIDTH / images.shape[2])
            images = np.stack([zoom(img, scale, order=1) for img in images], axis=0)

        X = images.reshape(len(images), -1).astype(np.float32)
        X = (X / 255.0 - 0.5) * 2.0  # normalize to [-1,1]

        # Map labels to {0,1}
        label_map = {d: i for i, d in enumerate(config.TARGET_DIGITS)}
        mapped = np.vectorize(label_map.get)(labels)
        y = np.eye(config.NUM_CLASSES, dtype=np.float32)[mapped]
        return X, y

    X_train, y_train = process_all(train_images, train_labels)
    X_test, y_test = process_all(test_images, test_labels)
    
    # Shuffle before slicing to ensure balance
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    
    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]
    
    print(f"Loaded {len(X_train)} train samples, {len(X_test)} test samples (digits {config.TARGET_DIGITS}).")
    
    # Slice to config sizes
    # Ensure we have enough data
    if len(X_train) < config.TRAIN_SIZE:
        print(f"Warning: Not enough train samples ({len(X_train)} < {config.TRAIN_SIZE}). Using all.")
    else:
        X_train = X_train[:config.TRAIN_SIZE]
        y_train = y_train[:config.TRAIN_SIZE]
        
    if len(X_test) < config.TEST_SIZE:
        print(f"Warning: Not enough test samples ({len(X_test)} < {config.TEST_SIZE}). Using all.")
    else:
        X_test = X_test[:config.TEST_SIZE]
        y_test = y_test[:config.TEST_SIZE]
    
    return (X_train, y_train), (X_test, y_test)

def run_classical_experiment(X_train, y_train, X_test, y_test, initial_u=None):
    """Standard (clean) training baseline."""
    print("\n--- Starting Classical Standard Training (Adam) ---")
    start_time = time.time()
    
    model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM, initial_weights=initial_u)
    trainer = RobustTrainer(model)
    
    rob_history = []
    clean_history = []
    clean_loss_history = []
    robust_loss_history = []
    
    dataset = list(create_dataset_iterator(X_train, y_train, batch_size=config.CLASSICAL_BATCH_SIZE))
    
    print("Evaluating Initial Model (Epoch 0)...")
    init_clean, init_rob, init_clean_loss, init_robust_loss = trainer.evaluate(
        list(create_dataset_iterator(X_test, y_test, batch_size=config.EVAL_BATCH_SIZE))
    )
    print(f"Initial Clean Acc = {init_clean:.2%} | Robust Acc = {init_rob:.2%}")
    rob_history.append(init_rob)
    clean_history.append(init_clean)
    clean_loss_history.append(init_clean_loss)
    robust_loss_history.append(init_robust_loss)
    
    for epoch in range(config.EPOCHS):
        loss = trainer.train_epoch(dataset)
        acc_clean, acc_rob, clean_loss, robust_loss = trainer.evaluate(
            list(create_dataset_iterator(X_test, y_test, batch_size=config.EVAL_BATCH_SIZE))
        )
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Clean Acc = {acc_clean:.2%}, Robust Acc = {acc_rob:.2%}")
        rob_history.append(acc_rob) # Plot Robust Accuracy
        clean_history.append(acc_clean)
        clean_loss_history.append(clean_loss)
        robust_loss_history.append(robust_loss)
        
    duration = time.time() - start_time
    print(f"Classical Standard Training Time: {duration:.2f}s")
    return rob_history, clean_history, clean_loss_history, robust_loss_history, duration, trainer.model


def run_classical_robust_experiment(X_train, y_train, X_test, y_test, initial_u=None):
    """
    Classical Robust Training (PGD-AT): Adversarial Training baseline.
    This is the standard approach from Madry et al. (2018) where we train on 
    adversarial examples generated via PGD.
    
    If USE_COMBINED_LOSS is enabled, uses: α*Loss(clean) + (1-α)*Loss(adv)
    """
    if getattr(config, 'USE_COMBINED_LOSS', False):
        alpha = getattr(config, 'LOSS_ALPHA', 0.5)
        print(f"\n--- Starting Classical Robust Training (Combined Loss: α={alpha}) ---")
    else:
        print("\n--- Starting Classical Robust Training (PGD-AT) ---")
    start_time = time.time()
    
    model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM, initial_weights=initial_u)
    trainer = RobustTrainer(model)
    
    rob_history = []
    clean_history = []
    clean_loss_history = []
    robust_loss_history = []
    
    print("Evaluating Initial Model (Epoch 0)...")
    init_clean, init_rob, init_clean_loss, init_robust_loss = trainer.evaluate(
        list(create_dataset_iterator(X_test, y_test, batch_size=config.EVAL_BATCH_SIZE))
    )
    print(f"Initial Clean Acc = {init_clean:.2%} | Robust Acc = {init_rob:.2%}")
    rob_history.append(init_rob)
    clean_history.append(init_clean)
    clean_loss_history.append(init_clean_loss)
    robust_loss_history.append(init_robust_loss)
    
    for epoch in range(config.EPOCHS):
        # Use robust training (train on adversarial examples)
        dataset = list(create_dataset_iterator(X_train, y_train, batch_size=config.CLASSICAL_BATCH_SIZE))
        loss = trainer.train_epoch_robust(dataset, epoch=epoch)
        acc_clean, acc_rob, clean_loss, robust_loss = trainer.evaluate(
            list(create_dataset_iterator(X_test, y_test, batch_size=config.EVAL_BATCH_SIZE))
        )
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Clean Acc = {acc_clean:.2%}, Robust Acc = {acc_rob:.2%}")
        rob_history.append(acc_rob)
        clean_history.append(acc_clean)
        clean_loss_history.append(clean_loss)
        robust_loss_history.append(robust_loss)
        
    duration = time.time() - start_time
    print(f"Classical Robust Training (PGD-AT) Time: {duration:.2f}s")
    return rob_history, clean_history, clean_loss_history, robust_loss_history, duration, trainer.model

def evaluate_model_weights(u_flat, X_test, y_test, model_struct):
    """
    Evaluates a flattened weight vector u using the provided model structure.
    Returns: accuracy, predictions, indices of failures
    """
    # Load weights
    model_struct._load_flat_weights(u_flat)
    
    # Eval
    model_struct.set_train(False)
    correct = 0
    total = len(X_test)
    failures = []
    
    logits_list = []
    
    for i in range(total):
        x = torch.from_numpy(np.asarray(X_test[i:i+1], dtype=np.float32))
        y = y_test[i]
        
        # PGD Attack for evaluation (Robust Accuracy)
        trainer = RobustTrainer(model_struct)
        y_tensor = torch.from_numpy(np.asarray(y_test[i:i+1], dtype=np.float32))
        x_adv = trainer.pgd_attack(x, y_tensor, config.EPSILON, config.ATTACK_STEP_SIZE, config.ATTACK_STEPS)
        
        logits = model_struct(x_adv)
        logits_np = logits.detach().cpu().numpy()
        pred = int(np.argmax(logits_np, axis=1).item())
        label_idx = int(np.argmax(y))
        
        logits_list.append(logits_np[0])
        
        if pred == label_idx:
            correct += 1
        else:
            failures.append(i)
            
    acc = correct / total
    return acc, failures, logits_list

def evaluate_model_weights_batch(u_flat, X_test, y_test, model_struct, batch_size=100, return_loss=False):
    """
    Evaluation for plotting: loads flattened weights and returns (clean_acc, robust_acc),
    with optional clean/robust loss from RobustTrainer.evaluate().
    """
    # Load weights
    model_struct._load_flat_weights(u_flat)

    trainer = RobustTrainer(model_struct)
    clean_acc, robust_acc, clean_loss, robust_loss = trainer.evaluate(
        list(create_dataset_iterator(X_test, y_test, batch_size=batch_size))
    )
    if return_loss:
        return clean_acc, robust_acc, clean_loss, robust_loss
    return clean_acc, robust_acc

def run_qrt_experiment(X_train, y_train, X_test, y_test, initial_u=None, start_accuracy=None):
    if getattr(config, 'USE_COMBINED_LOSS', False):
        alpha = getattr(config, 'LOSS_ALPHA', 0.5)
        print(f"\n--- Starting Quantum Robust Training (Combined Loss: α={alpha}) ---")
    else:
        print("\n--- Starting Quantum Robust Training (Simulation) ---")
    start_time = time.time()
    
    # Run the HHL simulation loop
    history, u_final, snapshots = train_qrt(
        X_train, y_train, initial_weights=initial_u, eval_interval=config.QRT_EVAL_INTERVAL
    )
    
    # Evaluate the final "Quantum Trained" model (both clean + robust)
    model_qrt = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    qrt_clean_acc, qrt_robust_acc, qrt_clean_loss, qrt_robust_loss = evaluate_model_weights_batch(
        u_final, X_test, y_test, model_qrt, batch_size=config.EVAL_BATCH_SIZE, return_loss=True
    )
    
    print(f"QRT Final Clean Accuracy:  {qrt_clean_acc:.2%}")
    print(f"QRT Final Robust Accuracy: {qrt_robust_acc:.2%}")

    # Build real accuracy trajectory by evaluating captured snapshots
    qrt_acc_traj = []  # (step_idx, clean_acc, robust_acc)
    qrt_clean_loss_traj = []  # (step_idx, clean_loss)
    qrt_robust_loss_traj = []  # (step_idx, robust_loss)
    for step_idx, u_vec in snapshots:
        m = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
        c_acc, r_acc, c_loss, r_loss = evaluate_model_weights_batch(
            u_vec, X_test, y_test, m, batch_size=config.EVAL_BATCH_SIZE, return_loss=True
        )
        qrt_acc_traj.append((step_idx, c_acc, r_acc))
        qrt_clean_loss_traj.append((step_idx, c_loss))
        qrt_robust_loss_traj.append((step_idx, r_loss))
    
    duration = time.time() - start_time
    print(f"QRT Simulation Time: {duration:.2f}s")
    
    return (
        history,
        duration,
        u_final,
        qrt_clean_acc,
        qrt_robust_acc,
        qrt_clean_loss,
        qrt_robust_loss,
        qrt_acc_traj,
        qrt_clean_loss_traj,
        qrt_robust_loss_traj,
    )

def save_plot_data(filename, **arrays):
    os.makedirs("plot_data", exist_ok=True)
    path = os.path.join("plot_data", filename)
    sanitized = {}
    for key, value in arrays.items():
        if isinstance(value, list) or isinstance(value, tuple):
            sanitized[key] = np.asarray(value)
        else:
            sanitized[key] = value
    np.savez_compressed(path, **sanitized)
    print(f"✓ Plot data saved: {path}")

def main(run_mode='both'):
    """
    Main experiment function.
    
    Args:
        run_mode: 'both' (default) - run both single and combined loss experiments
                  'single' - only run single robust loss experiment
                  'combined' - only run combined loss experiment
    """
    # 1. Data
    (X_train, y_train), (X_test, y_test) = load_or_generate_data()

    # 2. Generate Random Initialization (shared across all experiments for fair comparison)
    dummy_model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    initial_u = np.concatenate([p.detach().cpu().numpy().flatten() for p in dummy_model.trainable_params()])
    print(f"\n{'='*60}")
    print(f"Generated Random Initial Weights (Norm: {np.linalg.norm(initial_u):.4f})")
    print(f"{'='*60}\n")

    # 3. Classical Standard Training Baseline (from initial_u) - SHARED BASELINE
    classical_rob, classical_clean, classical_clean_loss, classical_robust_loss, classical_time, model_classical = run_classical_experiment(
        X_train, y_train, X_test, y_test, initial_u=initial_u
    )

    # Store original config
    original_use_combined = getattr(config, 'USE_COMBINED_LOSS', False)
    original_loss_alpha = float(getattr(config, "LOSS_ALPHA", 0.5))

    # Determine which experiments to run
    if run_mode == 'both':
        run_single = True
        run_combined = True
    elif run_mode == 'single':
        run_single = True
        run_combined = False
    elif run_mode == 'combined':
        run_single = False
        run_combined = True
    else:
        # If called from run_single_loss.py or run_combined_loss.py, use config setting
        if original_use_combined:
            run_single = False
            run_combined = True
        else:
            run_single = True
            run_combined = False

    run_clean = True

    # Classical robust baselines (do once)
    robust_single_rob = robust_single_clean = robust_single_clean_loss = robust_single_robust_loss = None
    robust_combined_rob = robust_combined_clean = robust_combined_clean_loss = robust_combined_robust_loss = None

    if run_single:
        print(f"\n{'='*60}")
        print("EXPERIMENT: SINGLE ROBUST LOSS MODE (Pure Adversarial Training)")
        print(f"{'='*60}\n")
        config.USE_COMBINED_LOSS = False
        robust_single_rob, robust_single_clean, robust_single_clean_loss, robust_single_robust_loss, robust_single_time, model_robust_single = run_classical_robust_experiment(
            X_train, y_train, X_test, y_test, initial_u=initial_u
        )

    if run_combined:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: COMBINED LOSS MODE (α={config.LOSS_ALPHA})")
        print(f"{'='*60}\n")
        config.USE_COMBINED_LOSS = True
        robust_combined_rob, robust_combined_clean, robust_combined_clean_loss, robust_combined_robust_loss, robust_combined_time, model_robust_combined = run_classical_robust_experiment(
            X_train, y_train, X_test, y_test, initial_u=initial_u
        )

    # Restore original config
    config.USE_COMBINED_LOSS = original_use_combined

    # QRT runs for each batch size
    default_batch_size = int(getattr(config, "QRT_BATCH_SIZE", 5))
    batch_sizes = list(getattr(config, "QRT_BATCH_SIZES", (default_batch_size,)))
    seen = set()
    batch_sizes = [bs for bs in batch_sizes if not (bs in seen or seen.add(bs))]
    original_total_steps = int(getattr(config, "TOTAL_STEPS", 0))

    for batch_size in batch_sizes:
        config.QRT_BATCH_SIZE = int(batch_size)
        if batch_size == 5:
            config.TOTAL_STEPS = 24000
        elif batch_size == 10:
            config.TOTAL_STEPS = 12000
        else:
            config.TOTAL_STEPS = original_total_steps
        batch_label = f"bs{batch_size}"
        total_steps = int(getattr(config, "TOTAL_STEPS", 0))

        qrt_clean_final_clean = qrt_clean_final_rob = None
        qrt_clean_final_clean_loss = qrt_clean_final_robust_loss = None
        qrt_clean_acc_traj = []
        qrt_clean_clean_loss_traj = []
        qrt_clean_robust_loss_traj = []

        qrt_robust_final_clean = qrt_robust_final_rob = None
        qrt_robust_final_clean_loss = qrt_robust_final_robust_loss = None
        qrt_robust_acc_traj = []
        qrt_robust_clean_loss_traj = []
        qrt_robust_robust_loss_traj = []

        qrt_combined_final_clean = qrt_combined_final_rob = None
        qrt_combined_final_clean_loss = qrt_combined_final_robust_loss = None
        qrt_combined_acc_traj = []
        qrt_combined_clean_loss_traj = []
        qrt_combined_robust_loss_traj = []

        # ============================================================
        # QRT: CLEAN LOSS ONLY
        # ============================================================
        if run_clean:
            config.USE_COMBINED_LOSS = False
            print(f"\n{'='*60}")
            print(f"QRT SIMULATION: CLEAN LOSS ONLY (batch size={batch_size})")
            print(f"{'='*60}\n")
            qrt_clean_history, qrt_clean_time, u_final_qrt_clean, qrt_clean_final_clean, qrt_clean_final_rob, qrt_clean_final_clean_loss, qrt_clean_final_robust_loss, qrt_clean_acc_traj, qrt_clean_clean_loss_traj, qrt_clean_robust_loss_traj = run_qrt_experiment(
                X_train, y_train, X_test, y_test, initial_u=initial_u, start_accuracy=classical_rob[-1]
            )

        # ============================================================
        # QRT: ROBUST LOSS ONLY
        # ============================================================
        if run_single:
            config.USE_COMBINED_LOSS = True
            config.LOSS_ALPHA = 0.0
            print(f"\n{'='*60}")
            print(f"QRT SIMULATION: ROBUST LOSS ONLY (batch size={batch_size})")
            print(f"{'='*60}\n")
            qrt_robust_history, qrt_robust_time, u_final_qrt_robust, qrt_robust_final_clean, qrt_robust_final_rob, qrt_robust_final_clean_loss, qrt_robust_final_robust_loss, qrt_robust_acc_traj, qrt_robust_clean_loss_traj, qrt_robust_robust_loss_traj = run_qrt_experiment(
                X_train, y_train, X_test, y_test, initial_u=initial_u, start_accuracy=classical_rob[-1]
            )

        # ============================================================
        # QRT: COMBINED LOSS
        # ============================================================
        if run_combined:
            config.USE_COMBINED_LOSS = True
            config.LOSS_ALPHA = original_loss_alpha
            print(f"\n{'='*60}")
            print(f"QRT SIMULATION: COMBINED LOSS α={config.LOSS_ALPHA} (batch size={batch_size})")
            print(f"{'='*60}\n")
            qrt_combined_history, qrt_combined_time, u_final_qrt_combined, qrt_combined_final_clean, qrt_combined_final_rob, qrt_combined_final_clean_loss, qrt_combined_final_robust_loss, qrt_combined_acc_traj, qrt_combined_clean_loss_traj, qrt_combined_robust_loss_traj = run_qrt_experiment(
                X_train, y_train, X_test, y_test, initial_u=initial_u, start_accuracy=classical_rob[-1]
            )

        # Restore original config
        config.USE_COMBINED_LOSS = original_use_combined
        config.LOSS_ALPHA = original_loss_alpha

        # ============================================================
        # COMPARATIVE ANALYSIS & SUMMARY
        # ============================================================
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS SUMMARY (QRT batch size={batch_size})")
        print(f"{'='*60}\n")

        if run_clean:
            print("=" * 60)
            print("CLEAN LOSS ONLY (Standard Training)")
            print("=" * 60)
            print(f"Classical Standard: Clean={classical_clean[-1]:.2%}, Robust={classical_rob[-1]:.2%}")
            print(f"QRT Clean Loss:     Clean={qrt_clean_final_clean:.2%}, Robust={qrt_clean_final_rob:.2%}")

        if run_single:
            print("\n" + "=" * 60)
            print("ROBUST LOSS ONLY (Pure Adversarial Training)")
            print("=" * 60)
            print(f"Classical Robust (PGD-AT): Clean={robust_single_clean[-1]:.2%}, Robust={robust_single_rob[-1]:.2%}")
            print(f"QRT Robust Loss:           Clean={qrt_robust_final_clean:.2%}, Robust={qrt_robust_final_rob:.2%}")

        if run_combined:
            print("\n" + "=" * 60)
            print(f"COMBINED LOSS (α={original_loss_alpha} Clean + {1-original_loss_alpha:.1f} Robust)")
            print("=" * 60)
            print(f"Classical Combined (α={original_loss_alpha}): Clean={robust_combined_clean[-1]:.2%}, Robust={robust_combined_rob[-1]:.2%}")
            print(f"QRT Combined (α={original_loss_alpha}):       Clean={qrt_combined_final_clean:.2%}, Robust={qrt_combined_final_rob:.2%}")

        print("=" * 60 + "\n")

        def _calc_qrt_epochs(q_steps):
            return (q_steps / config.RELINEARIZE_INTERVAL) * batch_size / config.TRAIN_SIZE

        epochs_range = np.arange(len(classical_rob))

        # ============================================================
        # PLOT GENERATION: CLEAN LOSS ONLY
        # ============================================================
        if run_clean:
            print("\n[Generating Plot: Clean Loss Only Comparison...]")
            plt.figure(figsize=(18, 5))

            q_steps = np.array([])
            qrt_epochs = np.array([])
            q_clean_accs = np.array([])
            q_rob_accs = np.array([])
            if len(qrt_clean_acc_traj) > 0:
                q_steps, q_clean_accs, q_rob_accs = zip(*qrt_clean_acc_traj)
                q_steps = np.array(q_steps)
                qrt_epochs = _calc_qrt_epochs(q_steps)

            # Subplot 1: Robust Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, classical_rob, label=f'Classical Standard (Final: {classical_rob[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if run_single:
                plt.plot(
                    epochs_range,
                    robust_single_rob,
                    label=f'Classical Robust (Final: {robust_single_rob[-1]:.2%})',
                    marker='s',
                    color='tab:green',
                    linewidth=2,
                )
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_rob_accs, label=f'QRT Clean Loss (Final: {qrt_clean_final_rob:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Robust Accuracy (Clean Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Robust Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 2: Clean Accuracy
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, classical_clean, label=f'Classical Standard (Final: {classical_clean[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if run_single:
                plt.plot(
                    epochs_range,
                    robust_single_clean,
                    label=f'Classical Robust (Final: {robust_single_clean[-1]:.2%})',
                    marker='s',
                    color='tab:green',
                    linewidth=2,
                )
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_clean_accs, label=f'QRT Clean Loss (Final: {qrt_clean_final_clean:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Clean Accuracy (Clean Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Clean Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 3: Clean Loss
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, classical_clean_loss, label="Classical Standard", marker='o', color='tab:blue', linewidth=2)
            if run_single:
                plt.plot(epochs_range, robust_single_clean_loss, label="Classical Robust", marker='s', color='tab:green', linewidth=2)
            q_loss_steps = np.array([])
            q_clean_losses = np.array([])
            if len(qrt_clean_clean_loss_traj) > 0:
                q_loss_steps, q_clean_losses = zip(*qrt_clean_clean_loss_traj)
                q_loss_steps = np.array(q_loss_steps)
                if qrt_epochs.size == 0:
                    qrt_epochs = _calc_qrt_epochs(q_loss_steps)
                plt.plot(qrt_epochs, q_clean_losses, label="QRT Clean Loss", marker='x', color='tab:red', linewidth=2)
            plt.title(f"Clean Loss (Clean Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Clean Loss", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_clean_filename = (
                f"qrt_vs_classical_comparison_clean_loss_only_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.png"
            )
            plt.savefig(plot_clean_filename, dpi=150)
            print(f"✓ Plot saved: {plot_clean_filename}")
            plt.close()

            data_clean_filename = (
                f"plot_data_clean_loss_only_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.npz"
            )
            save_plot_data(
                data_clean_filename,
                batch_size=batch_size,
                total_steps=total_steps,
                relinearize_interval=config.RELINEARIZE_INTERVAL,
                train_size=config.TRAIN_SIZE,
                epochs_range=epochs_range,
                classical_clean=np.asarray(classical_clean),
                classical_rob=np.asarray(classical_rob),
                classical_clean_loss=np.asarray(classical_clean_loss),
                classical_robust_loss=np.asarray(classical_robust_loss),
                qrt_steps=np.asarray(q_steps),
                qrt_epochs=np.asarray(qrt_epochs),
                qrt_clean_accs=np.asarray(q_clean_accs),
                qrt_rob_accs=np.asarray(q_rob_accs),
                qrt_clean_losses=np.asarray(q_clean_losses),
                qrt_robust_losses=np.asarray([loss for _, loss in qrt_clean_robust_loss_traj]) if len(qrt_clean_robust_loss_traj) > 0 else np.asarray([]),
            )

        # ============================================================
        # PLOT GENERATION: ROBUST LOSS ONLY
        # ============================================================
        if run_single:
            print("\n[Generating Plot: Robust Loss Only Comparison...]")
            plt.figure(figsize=(18, 5))

            q_steps = np.array([])
            qrt_epochs = np.array([])
            q_clean_accs = np.array([])
            q_rob_accs = np.array([])
            if len(qrt_robust_acc_traj) > 0:
                q_steps, q_clean_accs, q_rob_accs = zip(*qrt_robust_acc_traj)
                q_steps = np.array(q_steps)
                qrt_epochs = _calc_qrt_epochs(q_steps)

            # Subplot 1: Robust Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, robust_single_rob, label=f'Classical Robust (Final: {robust_single_rob[-1]:.2%})', marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_rob, label=f'Classical Standard (Final: {classical_rob[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_rob_accs, label=f'QRT Robust Loss (Final: {qrt_robust_final_rob:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Robust Accuracy (Robust Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Robust Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 2: Clean Accuracy
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, robust_single_clean, label=f'Classical Robust (Final: {robust_single_clean[-1]:.2%})', marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_clean, label=f'Classical Standard (Final: {classical_clean[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_clean_accs, label=f'QRT Robust Loss (Final: {qrt_robust_final_clean:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Clean Accuracy (Robust Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Clean Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 3: Clean Loss
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, robust_single_robust_loss, label="Classical Robust", marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_robust_loss, label="Classical Standard", marker='o', color='tab:blue', linewidth=2)
            q_loss_steps = np.array([])
            q_rob_losses = np.array([])
            if len(qrt_robust_robust_loss_traj) > 0:
                q_loss_steps, q_rob_losses = zip(*qrt_robust_robust_loss_traj)
                q_loss_steps = np.array(q_loss_steps)
                if qrt_epochs.size == 0:
                    qrt_epochs = _calc_qrt_epochs(q_loss_steps)
                plt.plot(qrt_epochs, q_rob_losses, label="QRT Robust Loss", marker='x', color='tab:red', linewidth=2)
            plt.title(f"Robust Loss (Robust Loss Only, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Robust Loss", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_robust_filename = (
                f"qrt_vs_classical_comparison_robust_loss_only_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.png"
            )
            plt.savefig(plot_robust_filename, dpi=150)
            print(f"✓ Plot saved: {plot_robust_filename}")
            plt.close()

            data_robust_filename = (
                f"plot_data_robust_loss_only_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.npz"
            )
            save_plot_data(
                data_robust_filename,
                batch_size=batch_size,
                total_steps=total_steps,
                relinearize_interval=config.RELINEARIZE_INTERVAL,
                train_size=config.TRAIN_SIZE,
                epochs_range=epochs_range,
                classical_clean=np.asarray(robust_single_clean),
                classical_rob=np.asarray(robust_single_rob),
                classical_clean_loss=np.asarray(robust_single_clean_loss),
                classical_robust_loss=np.asarray(robust_single_robust_loss),
                qrt_steps=np.asarray(q_steps),
                qrt_epochs=np.asarray(qrt_epochs),
                qrt_clean_accs=np.asarray(q_clean_accs),
                qrt_rob_accs=np.asarray(q_rob_accs),
                qrt_clean_losses=np.asarray([loss for _, loss in qrt_robust_clean_loss_traj]) if len(qrt_robust_clean_loss_traj) > 0 else np.asarray([]),
                qrt_robust_losses=np.asarray(q_rob_losses),
            )

        # ============================================================
        # PLOT GENERATION: COMBINED LOSS
        # ============================================================
        if run_combined:
            print("\n[Generating Plot: Combined Loss Comparison...]")
            plt.figure(figsize=(18, 5))
            alpha = original_loss_alpha

            q_steps = np.array([])
            qrt_epochs = np.array([])
            q_clean_accs = np.array([])
            q_rob_accs = np.array([])
            if len(qrt_combined_acc_traj) > 0:
                q_steps, q_clean_accs, q_rob_accs = zip(*qrt_combined_acc_traj)
                q_steps = np.array(q_steps)
                qrt_epochs = _calc_qrt_epochs(q_steps)

            # Subplot 1: Robust Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, robust_combined_rob, label=f'Classical Combined (Final: {robust_combined_rob[-1]:.2%})', marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_rob, label=f'Classical Standard (Final: {classical_rob[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_rob_accs, label=f'QRT Combined (Final: {qrt_combined_final_rob:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Robust Accuracy (Combined Loss α={alpha}, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Robust Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 2: Clean Accuracy
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, robust_combined_clean, label=f'Classical Combined (Final: {robust_combined_clean[-1]:.2%})', marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_clean, label=f'Classical Standard (Final: {classical_clean[-1]:.2%})', marker='o', color='tab:blue', linewidth=2)
            if qrt_epochs.size > 0:
                plt.plot(qrt_epochs, q_clean_accs, label=f'QRT Combined (Final: {qrt_combined_final_clean:.2%})', marker='x', color='tab:red', linewidth=2)
            plt.title(f"Clean Accuracy (Combined Loss α={alpha}, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Clean Accuracy", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Subplot 3: Clean Loss
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, robust_combined_robust_loss, label=f"Classical Combined α={alpha}", marker='s', color='tab:green', linewidth=2)
            plt.plot(epochs_range, classical_robust_loss, label="Classical Standard", marker='o', color='tab:blue', linewidth=2)
            q_loss_steps = np.array([])
            q_rob_losses = np.array([])
            if len(qrt_combined_robust_loss_traj) > 0:
                q_loss_steps, q_rob_losses = zip(*qrt_combined_robust_loss_traj)
                q_loss_steps = np.array(q_loss_steps)
                if qrt_epochs.size == 0:
                    qrt_epochs = _calc_qrt_epochs(q_loss_steps)
                plt.plot(qrt_epochs, q_rob_losses, label="QRT Combined", marker='x', color='tab:red', linewidth=2)
            plt.title(f"Robust Loss (Combined Loss α={alpha}, batch={batch_size})", fontsize=12, fontweight='bold')
            plt.xlabel("Training Epochs (Effective)", fontsize=11)
            plt.ylabel("Robust Loss", fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_combined_filename = (
                f"qrt_vs_classical_comparison_combined_loss_alpha{alpha:.2f}_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.png"
            )
            plt.savefig(plot_combined_filename, dpi=150)
            print(f"✓ Plot saved: {plot_combined_filename}")
            plt.close()

            data_combined_filename = (
                f"plot_data_combined_loss_alpha{alpha:.2f}_{batch_label}_steps{total_steps}"
                f"_lr{config.QRT_LEARNING_RATE:.3f}_eps{config.EPSILON:.3f}"
                f"_epsT{config.EPSILON_TRAIN:.3f}_atk{config.ATTACK_STEP_SIZE:.3f}_atksteps{config.ATTACK_STEPS}.npz"
            )
            save_plot_data(
                data_combined_filename,
                batch_size=batch_size,
                total_steps=total_steps,
                relinearize_interval=config.RELINEARIZE_INTERVAL,
                train_size=config.TRAIN_SIZE,
                loss_alpha=alpha,
                epochs_range=epochs_range,
                classical_clean=np.asarray(robust_combined_clean),
                classical_rob=np.asarray(robust_combined_rob),
                classical_clean_loss=np.asarray(robust_combined_clean_loss),
                classical_robust_loss=np.asarray(robust_combined_robust_loss),
                qrt_steps=np.asarray(q_steps),
                qrt_epochs=np.asarray(qrt_epochs),
                qrt_clean_accs=np.asarray(q_clean_accs),
                qrt_rob_accs=np.asarray(q_rob_accs),
                qrt_clean_losses=np.asarray([loss for _, loss in qrt_combined_clean_loss_traj]) if len(qrt_combined_clean_loss_traj) > 0 else np.asarray([]),
                qrt_robust_losses=np.asarray(q_rob_losses),
            )

        print(f"\n{'='*60}")
        print(f"✓ Batch size {batch_size} completed successfully!")
        print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print("✓ Experiment(s) completed successfully!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # When run directly, execute both experiments
    main(run_mode='both')

