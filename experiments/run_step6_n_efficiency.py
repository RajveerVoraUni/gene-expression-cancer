import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, N_TRAIN_GRID, RESULTS_DIR, RANDOM_STATE
from src.data_loader import load_data
from src.create_splits import load_n_efficiency_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions
from src.models import get_linear_model, get_tree_model


def run_n_efficiency_non_neural(n_splits: int = 5):
    print(f"Step 6: Sample efficiency stress test ({n_splits} splits, n in {N_TRAIN_GRID})")

    # load dataset
    X, y, gene_names, _, _ = load_data()
    records = []

    models = ["Softmax", "Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]

    for n_train in N_TRAIN_GRID:
        print(f"\nTesting sample size n = {n_train}")
        for split_idx in range(n_splits):
            train_idx, test_idx = load_n_efficiency_split(split_idx, n_train)
            X_train_raw, y_train = X[train_idx], y[train_idx]
            X_test_raw, y_test = X[test_idx], y[test_idx]

            # fit preprocessing on subsampled training data
            prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE + split_idx)
            X_train, _, _ = prep.fit_transform(X_train_raw, gene_names)
            X_test = prep.transform(X_test_raw)

            # evaluate each model under scarce sample conditions
            for m_name in models:
                t0 = time.time()
                if m_name in ["Softmax", "Ridge", "Lasso", "ElasticNet"]:
                    clf = get_linear_model(m_name)
                else:
                    clf = get_tree_model(m_name)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)
                metrics = evaluate_predictions(y_test, y_pred, y_prob)

                records.append({
                    "split_idx": split_idx,
                    "n_train": n_train,
                    "model": m_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "fit_time_s": time.time() - t0,
                })

    # save sample efficiency results to csv
    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "n_efficiency_comparison.csv"
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    run_n_efficiency_non_neural(n_splits=5)
