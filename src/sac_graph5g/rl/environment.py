import numpy as np

from sac_graph5g.config import MAX_FEATURES

def class_entropy(y, num_classes):
    p = np.bincount(y, minlength=num_classes).astype(np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(max(num_classes, 2)))

def sac_state(ctx, ep, budget, last_reward, best_reward, last_acc, last_f1, slope, last_action, num_classes):
    counts = np.bincount(ctx["y"][ctx["train_idx"]], minlength=num_classes)
    imbalance = float(counts.max() / max(counts.min(), 1))
    dispersion = float(np.mean(np.std(ctx["X"], axis=0)))
    
    base = np.array([
        ep / max(budget, 1), 
        last_reward, 
        best_reward, 
        last_acc, 
        last_f1, 
        slope,
        min(len(ctx["y"]) / 10000, 1.0), 
        min(ctx["X"].shape[1] / MAX_FEATURES, 1.0),
        class_entropy(ctx["y"][ctx["train_idx"]], num_classes), 
        min(imbalance / 50, 1.0),
        min(dispersion, 5.0) / 5.0, 
        len(ctx["train_idx"]) / len(ctx["y"]), 
        len(ctx["val_idx"]) / len(ctx["y"]),
    ], dtype=np.float32)
    
    return np.concatenate([base, np.asarray(last_action, dtype=np.float32)])

def reward_from_metrics(metrics, params):
    complexity = np.mean([params["hidden_dim"] / 256, params["depth"] / 5, params["k"] / 30])
    return float(0.65 * metrics["f1_macro"] + 0.30 * metrics["accuracy"] - 0.03 * complexity)

ACTION_DIM = 6
HIDDEN_OPTIONS = np.array([32, 64, 96, 128, 192, 256])

def decode_action(action):
    u = (np.clip(np.asarray(action, dtype=np.float32), -1, 1) + 1) / 2
    return {
        "k": int(round(3 + u[0] * 27)),
        "hidden_dim": int(HIDDEN_OPTIONS[min(len(HIDDEN_OPTIONS) - 1, int(u[1] * len(HIDDEN_OPTIONS)))]),
        "depth": int(round(2 + u[2] * 3)),
        "dropout": float(0.05 + u[3] * 0.45),
        "lr": float(10 ** (-4.3 + u[4] * 1.8)),
        "weight_decay": float(10 ** (-6.0 + u[5] * 4.5)),
    }
