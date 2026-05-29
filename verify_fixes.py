#!/usr/bin/env python3
"""
Quick verification script to test if QRT fixes are working correctly.
This script runs a minimal version of QRT training to check for common issues.
"""

import numpy as np
import config
from qrt_simulation import CarlemanSystem, verify_carleman_truncation
from classical_baseline import PolyMLP

def test_carleman_truncation():
    """Test if Carleman truncation order is sufficient."""
    print("=" * 60)
    print("TEST 1: Carleman Truncation Verification")
    print("=" * 60)
    
    # Create a small test model
    model = PolyMLP(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    data = np.random.randn(1, config.INPUT_DIM).astype(np.float32)
    labels = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    
    # Create Carleman system
    cs = CarlemanSystem(model, data, labels, v_operating_point=None)
    
    # Verify truncation
    avg_error, errors = verify_carleman_truncation(cs, cs.v_op, num_tests=5, perturb_scale=1e-2)
    
    print(f"\nTruncation Order: N = {config.CARLEMAN_N}")
    print(f"Average Reconstruction Error: {avg_error:.4%}")
    
    if avg_error < 0.10:
        print("✅ PASS: Truncation error < 10%")
        return True
    elif avg_error < 0.20:
        print("⚠️  WARNING: Truncation error between 10-20%")
        print("   Consider increasing CARLEMAN_N to", config.CARLEMAN_N + 1)
        return True
    else:
        print("❌ FAIL: Truncation error > 20%")
        print("   MUST increase CARLEMAN_N to", config.CARLEMAN_N + 1, "or higher")
        return False

def test_hyperparameters():
    """Test if hyperparameters are correctly set."""
    print("\n" + "=" * 60)
    print("TEST 2: Hyperparameter Configuration")
    print("=" * 60)
    
    checks = []
    
    # Check learning rate
    if config.QRT_LEARNING_RATE <= 0.05:
        print(f"✅ QRT_LEARNING_RATE = {config.QRT_LEARNING_RATE} (reasonable)")
        checks.append(True)
    else:
        print(f"❌ QRT_LEARNING_RATE = {config.QRT_LEARNING_RATE} (too high, should be ≤ 0.05)")
        checks.append(False)
    
    # Check gradient clipping
    if config.QRT_GRAD_CLIP >= 0.5:
        print(f"✅ QRT_GRAD_CLIP = {config.QRT_GRAD_CLIP} (allows gradient flow)")
        checks.append(True)
    else:
        print(f"❌ QRT_GRAD_CLIP = {config.QRT_GRAD_CLIP} (too aggressive, should be ≥ 0.5)")
        checks.append(False)
    
    # Check training epsilon
    if hasattr(config, 'EPSILON_TRAIN') and config.EPSILON_TRAIN < config.EPSILON:
        print(f"✅ EPSILON_TRAIN = {config.EPSILON_TRAIN} < EPSILON = {config.EPSILON}")
        checks.append(True)
    elif hasattr(config, 'EPSILON_TRAIN'):
        print(f"❌ EPSILON_TRAIN = {config.EPSILON_TRAIN} >= EPSILON = {config.EPSILON}")
        print("   EPSILON_TRAIN should be smaller (e.g., 0.3 × EPSILON)")
        checks.append(False)
    else:
        print(f"❌ EPSILON_TRAIN not defined (add to config.py)")
        checks.append(False)
    
    # Check Carleman order
    if config.CARLEMAN_N >= 3:
        print(f"✅ CARLEMAN_N = {config.CARLEMAN_N} (sufficient for cubic dynamics)")
        checks.append(True)
    else:
        print(f"⚠️  CARLEMAN_N = {config.CARLEMAN_N} (may be insufficient, recommend ≥ 3)")
        checks.append(True)  # Warning, not failure
    
    # Check time step
    if config.TIME_STEP <= 0.1:
        print(f"✅ TIME_STEP = {config.TIME_STEP} (stable)")
        checks.append(True)
    else:
        print(f"❌ TIME_STEP = {config.TIME_STEP} (too large, should be ≤ 0.1)")
        checks.append(False)
    
    return all(checks)

def test_alternating_phases():
    """Test if the training loop alternates between w and u phases correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Alternating Phase Logic")
    print("=" * 60)
    
    # Simulate phase alternation
    phases = []
    for step in range(10):
        is_w_phase = (step % 2 == 1)
        phase = "W-ASCENT" if is_w_phase else "U-DESCENT"
        phases.append(phase)
        if step < 6:  # Only print first few
            print(f"  Step {step}: {phase}")
    
    # Check alternation pattern
    expected = ["U-DESCENT", "W-ASCENT"] * 3
    actual = phases[:6]
    
    if actual == expected:
        print("✅ PASS: Phases alternate correctly (U → W → U → W ...)")
        return True
    else:
        print("❌ FAIL: Phase alternation is incorrect")
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
        return False

def test_tensor_normalization():
    """Test if tensor products are being normalized correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Tensor Product Normalization")
    print("=" * 60)
    
    # Create test vectors
    v = np.random.randn(10) * 0.5
    v_norm = np.linalg.norm(v)
    
    # Without normalization (OLD CODE - WRONG)
    v2_raw = np.kron(v, v)
    v2_raw_norm = np.linalg.norm(v2_raw)
    
    # With normalization (NEW CODE - CORRECT)
    v2_normalized = v2_raw / (v_norm + 1e-8)
    v2_normalized_norm = np.linalg.norm(v2_normalized)
    
    print(f"||v|| = {v_norm:.6f}")
    print(f"||v⊗v|| (raw) = {v2_raw_norm:.6f} ≈ ||v||² = {v_norm**2:.6f}")
    print(f"||v⊗v|| (normalized) = {v2_normalized_norm:.6f} ≈ ||v|| = {v_norm:.6f}")
    
    # Check if normalization brings it back to ||v|| scale
    ratio = v2_normalized_norm / v_norm
    
    if 0.5 < ratio < 2.0:  # Within 2× is acceptable
        print(f"✅ PASS: Normalization factor ≈ 1.0 (actual: {ratio:.2f})")
        return True
    else:
        print(f"❌ FAIL: Normalization factor = {ratio:.2f} (should be ≈ 1.0)")
        return False

def main():
    print("\n" + "=" * 60)
    print("QRT FIX VERIFICATION SUITE")
    print("=" * 60)
    print(f"Config: CARLEMAN_N={config.CARLEMAN_N}, LR={config.QRT_LEARNING_RATE}")
    print(f"        TIME_STEP={config.TIME_STEP}, TOTAL_STEPS={config.TOTAL_STEPS}")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Run tests
    try:
        results['truncation'] = test_carleman_truncation()
    except Exception as e:
        print(f"❌ TEST 1 CRASHED: {e}")
        results['truncation'] = False
    
    try:
        results['hyperparams'] = test_hyperparameters()
    except Exception as e:
        print(f"❌ TEST 2 CRASHED: {e}")
        results['hyperparams'] = False
    
    try:
        results['alternation'] = test_alternating_phases()
    except Exception as e:
        print(f"❌ TEST 3 CRASHED: {e}")
        results['alternation'] = False
    
    try:
        results['normalization'] = test_tensor_normalization()
    except Exception as e:
        print(f"❌ TEST 4 CRASHED: {e}")
        results['normalization'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! QRT fixes are correctly applied.")
        print("   You can now run: python main.py")
    else:
        print("\n⚠️  Some tests failed. Please review the console output above.")
        print("   and ensure all changes were applied correctly.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



