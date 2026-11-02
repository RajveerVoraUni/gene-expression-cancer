import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, N_SPLITS, RESULTS_DIR, RANDOM_STATE
from src.data_loader import load_data
from src.create_splits import load_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions
from src.models import get_linear_model, extract_active_linear_features


def main():
    print(f"Step 1: Running linear models across {N_SPLITS} stratified splits (p={DEFAULT_P})")

    # load dataset
    X, y, gene_names, _, _ = load_data()

    records = []
    active_features = {"Softmax": [], "Ridge": [], "Lasso": [], "ElasticNet": []}

    for split_idx in range(N_SPLITS):
        train_idx, test_idx = load_split(split_idx)
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        # fit variance filter and scaler strictly on training split
        prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE + split_idx)
        X_train, sel_idx, sel_genes = prep.fit_transform(X_train_raw, gene_names)
        X_test = prep.transform(X_test_raw)

        # evaluate each linear classifier
        for model_name in ["Softmax", "Ridge", "Lasso", "ElasticNet"]:
            t0 = time.time()
            clf = get_linear_model(model_name)
            clf.fit(X_train, y_train)
            fit_time = time.time() - t0

            # evaluate on held-out test split
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            # record active features for sparse models
            active = extract_active_linear_features(clf, sel_genes)
            n_active = len(active) if model_name in ["Lasso", "ElasticNet"] else DEFAULT_P
            active_features[model_name].append(list(active))

            rec = {
                "split_idx": split_idx,
                "model": model_name,
                "category": "Linear",
                "p": DEFAULT_P,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "fit_time_s": fit_time,
                "active_features": n_active,
            }
            records.append(rec)
            print(f"Split {split_idx+1:2d}/10 | {model_name:10s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f} | Loss: {metrics['log_loss']:.4f} | Active: {n_active}")

    # save split results to csv
    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "linear_results.csv"
    df.to_csv(out_csv, index=False)

    # save active feature signatures for stability analysis
    feat_path = RESULTS_DIR / "linear_active_features.json"
    with open(feat_path, "w") as f:
        json.dump(active_features, f, indent=2)


if __name__ == "__main__":
    main()
