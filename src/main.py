import argparse
import json
import pandas as pd
from pathlib import Path

from sac_graph5g.config import (
    DATA_FILE, ARTIFACT_DIR, SEEDS, MAX_FEATURES, SEARCH_BUDGET, 
    SEARCH_EPOCHS, FINAL_EPOCHS, FINAL_PATIENCE,
    RUN_RANDOM_SEARCH, RUN_OPTUNA_SEARCH, RUN_SAC_SEARCH, DEFAULT_GNN_PARAMS
)
from sac_graph5g.data.dataset import load_and_preprocess_data, prepare_context
from sac_graph5g.models.baselines import run_classical_baselines
from sac_graph5g.training.search import random_search, optuna_search, sac_search
from sac_graph5g.training.train_gnn import final_train_from_params

def main():
    parser = argparse.ArgumentParser(description="Run SAC-GRAPH5G Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast smoke test with reduced budgets")
    args = parser.parse_args()

    # Apply smoke test overrides
    if args.smoke_test:
        global SEEDS, SEARCH_BUDGET, SEARCH_EPOCHS, FINAL_EPOCHS, FINAL_PATIENCE
        SEEDS = [42]
        SEARCH_BUDGET = 2
        SEARCH_EPOCHS = 2
        FINAL_EPOCHS = 2
        FINAL_PATIENCE = 2
        print("Running in smoke-test mode...")

    # 1. Load Data
    X_raw_all, y_all, class_names = load_and_preprocess_data(DATA_FILE)
    print(f"Loaded {len(X_raw_all)} rows. Classes: {class_names}")

    all_results, search_tables, final_histories, final_predictions = [], [], {}, {}

    # 2. Iterate through seeds
    for seed in SEEDS:
        print(f"\n{'=' * 90}\nSeed: {seed}")
        ctx = prepare_context(seed, X_raw_all, y_all)
        
        # Baselines
        rows = run_classical_baselines(ctx, len(class_names))
        all_results.extend(rows)
        for r in rows:
            print(f"{r['method']}: acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f}")

        # Default GraphSAGE
        row, hist, _ = final_train_from_params(ctx, "Default-GraphSAGE", DEFAULT_GNN_PARAMS, class_names, FINAL_EPOCHS, FINAL_PATIENCE)
        all_results.append(row)
        final_histories[(seed, "Default-GraphSAGE")] = hist
        print(f"Default-GraphSAGE: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}")

        if RUN_RANDOM_SEARCH:
            best, hist_s = random_search(ctx, class_names, SEARCH_BUDGET)
            search_tables.append(hist_s)
            row, hist, _ = final_train_from_params(ctx, "RandomSearch-GraphSAGE", best["params"], class_names, FINAL_EPOCHS, FINAL_PATIENCE)
            all_results.append(row)
            final_histories[(seed, "RandomSearch-GraphSAGE")] = hist
            print(f"RandomSearch-GraphSAGE: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}")

        if RUN_OPTUNA_SEARCH:
            best, hist_s = optuna_search(ctx, class_names, SEARCH_BUDGET)
            search_tables.append(hist_s)
            row, hist, _ = final_train_from_params(ctx, "OptunaTPE-GraphSAGE", best["params"], class_names, FINAL_EPOCHS, FINAL_PATIENCE)
            all_results.append(row)
            final_histories[(seed, "OptunaTPE-GraphSAGE")] = hist
            print(f"OptunaTPE-GraphSAGE: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}")

        if RUN_SAC_SEARCH:
            best, hist_s = sac_search(ctx, class_names, SEARCH_BUDGET)
            search_tables.append(hist_s)
            row, hist, test = final_train_from_params(ctx, "SAC-GRAPH5G", best["params"], class_names, FINAL_EPOCHS, FINAL_PATIENCE)
            all_results.append(row)
            final_histories[(seed, "SAC-GRAPH5G")] = hist
            final_predictions[seed] = {
                "test_idx": ctx["test_idx"], 
                "y_true": test["y_true"], 
                "y_pred": test["y_pred"], 
                "probs": test["probs"], 
                "params": best["params"]
            }
            print(f"SAC-GRAPH5G: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}")

    # 3. Export Artifacts
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(ARTIFACT_DIR / "all_seed_results.csv", index=False)
    
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]
    summary_rows = []
    for method, g in results_df.groupby("method"):
        row = {"method": method, "n_seeds": g["seed"].nunique()}
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro_mean", ascending=False)
    summary_df.to_csv(ARTIFACT_DIR / "mean_std_summary.csv", index=False)
    
    if search_tables:
        search_history_df = pd.concat(search_tables, ignore_index=True)
        search_history_df.to_csv(ARTIFACT_DIR / "controller_search_history.csv", index=False)
        
    for seed, pack in final_predictions.items():
        pred_df = pd.DataFrame({
            "row_index": pack["test_idx"],
            "y_true": [class_names[i] for i in pack["y_true"]],
            "y_pred": [class_names[i] for i in pack["y_pred"]],
        })
        for i, name in enumerate(class_names):
            pred_df[f"prob_{name}"] = pack["probs"][:, i]
        pred_df.to_csv(ARTIFACT_DIR / f"sac_graph5g_test_predictions_seed_{seed}.csv", index=False)
        
    metadata = {
        "data_file": str(DATA_FILE.resolve()),
        "rows": int(len(y_all)),
        "classes": class_names,
        "seeds": SEEDS,
        "search_budget": SEARCH_BUDGET,
        "search_epochs": SEARCH_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
        "method": "SAC-GRAPH5G: residual GraphSAGE with Soft Actor-Critic graph/training-policy search",
    }
    with open(ARTIFACT_DIR / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("Execution complete. Artifacts saved to:", ARTIFACT_DIR.resolve())

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
