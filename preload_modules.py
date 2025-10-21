#!/usr/bin/env python3
"""
Preload heavy modules to avoid KeyboardInterrupt during import.
This fixes the issue where scipy.optimize import gets interrupted.
"""

print("Preloading heavy modules...")

import warnings
warnings.filterwarnings('ignore')

# Preload all heavy modules that might cause import interrupts
print("1. Loading numpy...")
import numpy as np

print("2. Loading scipy...")
import scipy
import scipy.stats
import scipy.optimize

print("3. Loading sklearn...")
import sklearn
from sklearn.metrics import roc_curve

print("4. Loading transformers...")
from transformers import AutoModel, AutoTokenizer, AutoConfig

print("5. Loading torch...")
import torch

print("6. Loading sentence_transformers...")
import sentence_transformers

print("✓ All heavy modules preloaded successfully!")
print("Imports should now be instant.")
