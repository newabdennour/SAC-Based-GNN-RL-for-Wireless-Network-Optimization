from pathlib import Path
import torch

# Paths
DATA_FILE = Path("dataset/5g_network_data.csv")
ARTIFACT_DIR = Path("sac_graph5g_outputs")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment Configurations
TARGET_COLUMN = None
SEEDS = [42]
MAX_FEATURES = 256

# Search & Training Budgets
SEARCH_BUDGET = 24
SEARCH_EPOCHS = 45
SEARCH_PATIENCE = 12
FINAL_EPOCHS = 180
FINAL_PATIENCE = 30

# Feature Flags for Search Controllers
RUN_RANDOM_SEARCH = True
RUN_OPTUNA_SEARCH = True
RUN_SAC_SEARCH = True

# GNN Parameters Default
DEFAULT_GNN_PARAMS = {
    "k": 10,
    "hidden_dim": 128,
    "depth": 3,
    "dropout": 0.25,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
