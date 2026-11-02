import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent if (Path(__file__).resolve().parent / 'src').exists() else Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import numpy as np
import pandas as pd
from src.config import (
    P_GRID,
    RESULTS_DIR,
    RANDOM_STATE,
)
from src.data_loader import load_data
from src.create_splits import load_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions
from src.models import (
    get_linear_model,
    get_tree_model,
    SparseGroupLassoMLPNet,
    train_pytorch_model,
    predict_pytorch_model,
)

def run_p_grid_experiment(n_splits_to_eval: int = 5):
    print(f"=== Starting p-Grid Scaling Experiment ({n_splits_to_eval} splits, p in {P_GRID[:-1]}) ===")
    X, y, gene_names, _, le = load_data()

    # We evaluate p in [50, 100, 500, 1000, 5000]
    eval_p_grid = [p for p in P_GRID if p < 20000]

    records = []
    models_to_test = ["Lasso", "ElasticNet", "RandomForest", "XGBoost", "Sparse_NN"]

    for p in eval_p_grid:
        print(f"\n--- Testing Feature Dimension p = {p} ---")
        for split_idx in range(n_splits_to_eval):
            train_idx, test_idx = load_split(split_idx)
            X_train_raw, y_train = X[train_idx], y[train_idx]
            X_test_raw, y_test = X[test_idx], y[test_idx]

            prep = LeakageFreePreprocessor(p=p, apply_pca=False, random_state=RANDOM_STATE + split_idx)
            X_train, _, _ = prep.fit_transform(X_train_raw, gene_names)
            X_test = prep.transform(X_test_raw)

            for model_name in models_to_test:
                t0 = time.time()
                if model_name in ["Lasso", "ElasticNet"]:
                    clf = get_linear_model(model_name)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)
                elif model_name in ["RandomForest", "XGBoost"]:
                    clf = get_tree_model(model_name)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)
                elif model_name == "Sparse_NN":
                    nn_model = SparseGroupLassoMLPNet(in_features=p, hidden=48)
                    train_pytorch_model(nn_model, X_train, y_train, epochs=40, lr=0.005, lambda_group=0.03)
                    y_pred, y_prob = predict_pytorch_model(nn_model, X_test)

                fit_time = time.time() - t0
                metrics = evaluate_predictions(y_test, y_pred, y_prob)

                records.append({
                    "split_idx": split_idx,
                    "p": p,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "fit_time_s": fit_time,
                })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "p_grid_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved p-grid results to: {csv_path}")

if __name__ == "__main__":
    run_p_grid_experiment(n_splits_to_eval=5)
