#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script with COMBINED LOSS (TRADES-style)
Trains on: alpha * L(clean) + (1-alpha) * L(robust)
"""

import sys
import os

# Ensure config is imported properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Override config for combined loss training
config.USE_COMBINED_LOSS = True
config.LOSS_ALPHA = 0.5  # Balanced: equal weight to clean and robust

print("=" * 60)
print("TRAINING MODE: COMBINED LOSS (TRADES-style)")
print("=" * 60)
print("Loss Alpha: {}".format(config.LOSS_ALPHA))
print("  -> {} * L(clean) + {} * L(robust)".format(config.LOSS_ALPHA, 1.0 - config.LOSS_ALPHA))
print("Attack Step Size: {}".format(config.ATTACK_STEP_SIZE))
print("Training Epsilon: {}".format(config.EPSILON_TRAIN))
print("Evaluation Epsilon: {}".format(config.EPSILON))
print("=" * 60)
print()

# Import and run main with combined mode
from main import main
main(run_mode='combined')
