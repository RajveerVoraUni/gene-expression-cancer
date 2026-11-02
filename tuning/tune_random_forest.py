import sys
import time
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, RANDOM_STATE
from src.data_loader import load_data
from src.create_splits import load_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions


def tune_random_forest():
    # load data and training split
    X, y, gene_names, _, _ = load_data()
    train_idx, test_idx = load_split(0)

    # fit preprocessing on training data
    prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE)
    X_train, _, sel_genes = prep.fit_transform(X[train_idx], gene_names)
    y_train = y[train_idx]
    X_test = prep.transform(X[test_idx])
    y_test = y[test_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = [
        {"n_estimators": 100, "max_depth": 16, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 32, "max_features": "log2", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 32, "max_features": "log2", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 32, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": None, "max_features": "log2", "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 32, "max_features": "log2", "min_samples_leaf": 1},
    ]

    best_params = param_grid[0]
    best_score = -1.0
    best_loss = float("inf")
    cv_results = []

    # evaluate each candidate configuration across 5 folds
    for p in param_grid:
        f1_scores = []
        losses = []
        for cv_tr, cv_val in skf.split(X_train, y_train):
            clf = RandomForestClassifier(**p, random_state=RANDOM_STATE, n_jobs=-1)
            clf.fit(X_train[cv_tr], y_train[cv_tr])
            preds = clf.predict(X_train[cv_val])
            probs = clf.predict_proba(X_train[cv_val])
            m = evaluate_predictions(y_train[cv_val], preds, probs)
            f1_scores.append(m["macro_f1"])
            losses.append(m["log_loss"])

        mean_f1 = np.mean(f1_scores)
        mean_loss = np.mean(losses)
        cv_results.append({"params": p, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss)})

        if mean_f1 > best_score or (mean_f1 == best_score and mean_loss < best_loss):
            best_score = float(mean_f1)
            best_loss = float(mean_loss)
            best_params = p

    # evaluate winning configuration on held-out test split
    clf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_metrics = evaluate_predictions(y_test, clf.predict(X_test), clf.predict_proba(X_test))

    best_rf = {
        "score": best_score,
        "best_loss": best_loss,
        "params": best_params,
        "cv_results": cv_results,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_log_loss": test_metrics["log_loss"],
    }

    # save results
    out_file = root_dir / "tuning" / "rf_tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(best_rf, f, indent=2)


if __name__ == "__main__":
    tune_random_forest()
