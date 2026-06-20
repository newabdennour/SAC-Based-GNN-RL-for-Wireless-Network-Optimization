import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sac_graph5g.config import DEVICE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def metric_bundle(y_true, y_pred, probs=None, num_classes=3):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if probs is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, probs[:, 1]) if num_classes == 2 else roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        except Exception:
            out["roc_auc"] = np.nan
    return out
