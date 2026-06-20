import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sac_graph5g.utils import metric_bundle

def run_classical_baselines(ctx, num_classes):
    X, y = ctx["X"], ctx["y"]
    X_train, y_train = X[ctx["train_idx"]], y[ctx["train_idx"]]
    X_test, y_test = X[ctx["test_idx"]], y[ctx["test_idx"]]
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", random_state=ctx["seed"], n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=ctx["seed"], early_stopping=True),
    }
    
    rows = []
    for name, clf in models.items():
        start = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        
        metrics = metric_bundle(y_test, pred, probs, num_classes=num_classes)
        rows.append({
            "method": name, 
            "seed": ctx["seed"], 
            "seconds": time.time() - start, 
            **metrics
        })
    return rows
