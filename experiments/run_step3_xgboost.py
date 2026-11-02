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
from src.models import get_tree_model, extract_top_k_tree_features


def main():
    print(f"Step 3: Running XGBoost across {N_SPLITS} stratified splits (p={DEFAULT_P})")

    # load dataset
    X, y, gene_names, _, _ = load_data()

    records = []
    active_features = {"XGBoost": []}

    for split_idx in range(N_SPLITS):
        train_idx, test_idx = load_split(split_idx)
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        # fit variance filter and scaler on training split
        prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE + split_idx)
        X_train, sel_idx, sel_genes = prep.fit_transform(X_train_raw, gene_names)
        X_test = prep.transform(X_test_raw)

        # train gradient boosted trees with histogram algorithm
        t0 = time.time()
        xgb = get_tree_model("XGBoost")
        xgb.fit(X_train, y_train)
        fit_time = time.time() - t0

        # evaluate test set predictions and calibrated probabilities
        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)
        metrics = evaluate_predictions(y_test, y_pred, y_prob)

        # store top-50 features ranked by gain importance
        top_genes = extract_top_k_tree_features(xgb, sel_genes, k=50)
        active_features["XGBoost"].append(list(top_genes))

        rec = {
            "split_idx": split_idx,
            "model": "XGBoost",
            "category": "Tree Ensemble",
            "p": DEFAULT_P,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "log_loss": metrics["log_loss"],
            "fit_time_s": fit_time,
            "active_features": DEFAULT_P,
        }
        records.append(rec)
        print(f"Split {split_idx+1:2d}/10 | XGBoost      | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f} | Loss: {metrics['log_loss']:.4f}")

    # save results table to disk
    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "xgb_results.csv"
    df.to_csv(out_csv, index=False)

    # save feature importance sets for stability comparison
    feat_path = RESULTS_DIR / "xgb_active_features.json"
    with open(feat_path, "w") as f:
        json.dump(active_features, f, indent=2)


if __name__ == "__main__":
    main()
