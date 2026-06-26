from __future__ import annotations

import os

import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = ["setting_1", "setting_2", "setting_3"]

WINDOW = 15
K = 2
EPOCHS = 4000
LR = 5e-3
LAMBDA_MONO = 5.0
LAMBDA_SMOOTH = 2.0
TEST_SIZE = 0.2
TREND_DYNAMIC = 0.20
TREND_INFORMATIVE = 0.50

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
FIG_BASE = os.path.join(RESULTS_DIR, "figures")
