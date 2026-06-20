import numpy as np
import pandas as pd
import optuna
import torch

from sac_graph5g.config import DEVICE, SEARCH_BUDGET, SEARCH_EPOCHS, SEARCH_PATIENCE
from sac_graph5g.utils import set_seed
from sac_graph5g.training.train_gnn import train_gnn, evaluate_gnn
from sac_graph5g.rl.environment import decode_action, ACTION_DIM, reward_from_metrics, sac_state
from sac_graph5g.rl.sac import SACActor, SACCritic, ReplayBuffer, sac_update

def evaluate_params_short(ctx, params, class_names):
    model, adj, hist, _ = train_gnn(ctx, params, class_names, SEARCH_EPOCHS, SEARCH_PATIENCE)
    val = evaluate_gnn(model, adj, ctx, len(class_names), "val")
    return reward_from_metrics(val, params), val, hist

def random_search(ctx, class_names, budget=SEARCH_BUDGET):
    rng = np.random.default_rng(ctx["seed"] + 1000)
    best, rows = {"reward": -np.inf, "params": None}, []
    for trial in range(1, budget + 1):
        params = decode_action(rng.uniform(-1, 1, ACTION_DIM))
        reward, val, _ = evaluate_params_short(ctx, params, class_names)
        if reward > best["reward"]:
            best = {"reward": reward, "params": params}
        rows.append({
            "controller": "RandomSearch", "seed": ctx["seed"], "trial": trial, 
            "reward": reward, "val_accuracy": val["accuracy"], 
            "val_f1_macro": val["f1_macro"], **params
        })
    return best, pd.DataFrame(rows)

def optuna_search(ctx, class_names, budget=SEARCH_BUDGET):
    def objective(trial):
        params = {
            "k": trial.suggest_int("k", 3, 30),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192, 256]),
            "depth": trial.suggest_int("depth", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.05, 0.50),
            "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True),
        }
        reward, val, _ = evaluate_params_short(ctx, params, class_names)
        trial.set_user_attr("val_accuracy", val["accuracy"])
        trial.set_user_attr("val_f1_macro", val["f1_macro"])
        return reward

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=ctx["seed"]))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    rows = [{
        "controller": "OptunaTPE", "seed": ctx["seed"], "trial": t.number + 1, 
        "reward": t.value, "val_accuracy": t.user_attrs.get("val_accuracy"), 
        "val_f1_macro": t.user_attrs.get("val_f1_macro"), **t.params
    } for t in study.trials]
    return {"reward": study.best_value, "params": study.best_params}, pd.DataFrame(rows)

def sac_search(ctx, class_names, budget=SEARCH_BUDGET):
    set_seed(ctx["seed"])
    num_classes = len(class_names)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    state = sac_state(ctx, 0, budget, 0, 0, 0, 0, 0, last_action, num_classes)
    
    actor = SACActor(len(state), ACTION_DIM).to(DEVICE)
    q1 = SACCritic(len(state), ACTION_DIM).to(DEVICE)
    q2 = SACCritic(len(state), ACTION_DIM).to(DEVICE)
    tq1 = SACCritic(len(state), ACTION_DIM).to(DEVICE)
    tq2 = SACCritic(len(state), ACTION_DIM).to(DEVICE)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=3e-4)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=3e-4)
    
    replay, rewards, rows = ReplayBuffer(), [], []
    best = {"reward": -np.inf, "params": None}
    warmup = max(6, budget // 4)
    
    for ep in range(1, budget + 1):
        if ep <= warmup:
            action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor.sample(torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))[0].squeeze(0).cpu().numpy()
                
        params = decode_action(action)
        reward, val, _ = evaluate_params_short(ctx, params, class_names)
        rewards.append(reward)
        
        if reward > best["reward"]:
            best = {"reward": reward, "params": params}
            
        slope = 0.0 if len(rewards) < 6 else float(np.mean(rewards[-3:]) - np.mean(rewards[-6:-3]))
        next_state = sac_state(ctx, ep, budget, reward, best["reward"], val["accuracy"], val["f1_macro"], slope, action, num_classes)
        
        replay.add(state, action, reward, next_state, 1 if ep == budget else 0)
        
        for _ in range(10):
            sac_update(actor, q1, q2, tq1, tq2, actor_opt, q1_opt, q2_opt, replay)
            
        state = next_state
        rows.append({
            "controller": "SAC-GRAPH5G", "seed": ctx["seed"], "trial": ep, 
            "reward": reward, "val_accuracy": val["accuracy"], 
            "val_f1_macro": val["f1_macro"], **params
        })
        
    return best, pd.DataFrame(rows)
