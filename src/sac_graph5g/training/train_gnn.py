import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sac_graph5g.config import DEVICE
from sac_graph5g.models.gnn import ResidualGraphSAGE
from sac_graph5g.data.dataset import build_knn_graph
from sac_graph5g.utils import metric_bundle, set_seed
from sac_graph5g.rl.environment import reward_from_metrics

@torch.no_grad()
def evaluate_gnn(model, adj, ctx, num_classes, split="val"):
    model.eval()
    idx_t = ctx[f"{split}_idx_t"]
    logits = model(ctx["x_tensor"], adj)
    probs = F.softmax(logits[idx_t], dim=1).detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    true = ctx["y_tensor"][idx_t].detach().cpu().numpy()
    
    out = metric_bundle(true, pred, probs, num_classes=num_classes)
    out.update({"y_true": true, "y_pred": pred, "probs": probs})
    return out

def train_gnn(ctx, params, class_names, max_epochs=120, patience=20, verbose=False):
    set_seed(ctx["seed"])
    num_classes = len(class_names)
    
    adj = build_knn_graph(ctx["X"], k=params["k"])
    model = ResidualGraphSAGE(
        ctx["X"].shape[1], 
        params["hidden_dim"], 
        num_classes, 
        params["depth"], 
        params["dropout"]
    ).to(DEVICE)
    
    counts = np.bincount(ctx["y"][ctx["train_idx"]], minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best_state, best_reward, best_epoch, history = None, -np.inf, 0, []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        
        logits = model(ctx["x_tensor"], adj)
        loss = F.cross_entropy(logits[ctx["train_idx_t"]], ctx["y_tensor"][ctx["train_idx_t"]], weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        
        val = evaluate_gnn(model, adj, ctx, num_classes, "val")
        tr = evaluate_gnn(model, adj, ctx, num_classes, "train")
        
        history.append({
            "epoch": epoch, 
            "loss": float(loss.detach().cpu()), 
            "train_accuracy": tr["accuracy"], 
            "val_accuracy": val["accuracy"], 
            "val_f1_macro": val["f1_macro"]
        })
        
        score = reward_from_metrics(val, params)
        if score > best_reward:
            best_reward, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"epoch={epoch:03d} loss={history[-1]['loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1_macro']:.4f}")
            
        if epoch - best_epoch >= patience:
            break
            
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        
    return model, adj, pd.DataFrame(history), best_reward

def final_train_from_params(ctx, method_name, params, class_names, final_epochs, final_patience):
    import time
    start = time.time()
    import json
    model, adj, hist, _ = train_gnn(ctx, params, class_names, final_epochs, final_patience)
    test = evaluate_gnn(model, adj, ctx, len(class_names), "test")
    
    row = {
        "method": method_name, 
        "seed": ctx["seed"], 
        "seconds": time.time() - start, 
        "selected_params": json.dumps(params), 
        **{k: test[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]}
    }
    return row, hist, test
