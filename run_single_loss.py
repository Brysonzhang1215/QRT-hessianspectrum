#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script with SINGLE ROBUST LOSS (Pure PGD-AT)
Only trains on adversarial examples using robust loss.
"""

import sys
import os

# Ensure config is imported properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Override config for single loss training
config.USE_COMBINED_LOSS = False
config.LOSS_ALPHA = 0.0  # Not used in single loss mode

print("=" * 60)
print("TRAINING MODE: SINGLE ROBUST LOSS (Pure PGD-AT)")
print("=" * 60)
print("Loss Function: Adversarial Loss Only")
print("Attack Step Size: {}".format(config.ATTACK_STEP_SIZE))
print("Training Epsilon: {}".format(config.EPSILON_TRAIN))
print("Evaluation Epsilon: {}".format(config.EPSILON))
print("=" * 60)
print()

# Import and run main with single mode
from main import main
main(run_mode='single')
