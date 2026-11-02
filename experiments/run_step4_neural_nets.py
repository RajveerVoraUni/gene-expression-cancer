import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, N_SPLITS, RESULTS_DIR, RANDOM_STATE
from src.data_loader import load_data
from src.create_splits import load_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions
from src.neural_models import (
    PCAMLPNet,
    SparseGroupLassoMLPNet,
    train_pytorch_model,
    predict_pytorch_model,
    extract_active_nn_features,
)


def main():
    # use apple silicon gpu if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Step 4: Running neural networks across {N_SPLITS} stratified splits on {device.upper()}")

    # load dataset
    X, y, gene_names, _, _ = load_data()

    records = []
    active_features = {"PCA_MLP": [], "Sparse_NN": []}

    for split_idx in range(N_SPLITS):
        train_idx, test_idx = load_split(split_idx)
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        # preprocessing for pca baseline
        prep_pca = LeakageFreePreprocessor(p=None, apply_pca=True, pca_variance=0.95, random_state=RANDOM_STATE + split_idx)
        X_train_pca, _, _ = prep_pca.fit_transform(X_train_raw)
        X_test_pca = prep_pca.transform(X_test_raw)

        # preprocessing for sparse group lasso network
        prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE + split_idx)
        X_train, sel_idx, sel_genes = prep.fit_transform(X_train_raw, gene_names)
        X_test = prep.transform(X_test_raw)

        # 1. train dense pca mlp benchmark
        t0 = time.time()
        pca_mlp = PCAMLPNet(in_features=X_train_pca.shape[1])
        train_pytorch_model(pca_mlp, X_train_pca, y_train, epochs=40, lr=0.003, device=device)
        fit_time_pca = time.time() - t0

        y_pred, y_prob = predict_pytorch_model(pca_mlp, X_test_pca, device=device)
        metrics_pca = evaluate_predictions(y_test, y_pred, y_prob)

        rec_pca = {
            "split_idx": split_idx,
            "model": "PCA_MLP",
            "category": "Neural Network (Dense)",
            "p": DEFAULT_P,
            "accuracy": metrics_pca["accuracy"],
            "macro_f1": metrics_pca["macro_f1"],
            "log_loss": metrics_pca["log_loss"],
            "fit_time_s": fit_time_pca,
            "active_features": X_train_pca.shape[1],
        }
        records.append(rec_pca)
        print(f"Split {split_idx+1:2d}/10 | PCA_MLP     | Acc: {metrics_pca['accuracy']:.4f} | F1: {metrics_pca['macro_f1']:.4f} | Loss: {metrics_pca['log_loss']:.4f}")

        # 2. train sparse group lasso network
        t0 = time.time()
        sparse_nn = SparseGroupLassoMLPNet(in_features=DEFAULT_P, hidden=64)
        train_pytorch_model(sparse_nn, X_train, y_train, epochs=40, lr=0.005, lambda_group=0.02, device=device)
        fit_time_sparse = time.time() - t0

        y_pred, y_prob = predict_pytorch_model(sparse_nn, X_test, device=device)
        metrics_sparse = evaluate_predictions(y_test, y_pred, y_prob)

        # extract retained genes
        active_nn_genes = extract_active_nn_features(sparse_nn, sel_genes, threshold=1e-4)
        n_active_nn = len(active_nn_genes)
        active_features["Sparse_NN"].append(list(active_nn_genes))

        rec_sparse = {
            "split_idx": split_idx,
            "model": "Sparse_NN",
            "category": "Neural Network (Sparse Group-Lasso)",
            "p": DEFAULT_P,
            "accuracy": metrics_sparse["accuracy"],
            "macro_f1": metrics_sparse["macro_f1"],
            "log_loss": metrics_sparse["log_loss"],
            "fit_time_s": fit_time_sparse,
            "active_features": n_active_nn,
        }
        records.append(rec_sparse)
        print(f"Split {split_idx+1:2d}/10 | Sparse_NN   | Acc: {metrics_sparse['accuracy']:.4f} | F1: {metrics_sparse['macro_f1']:.4f} | Loss: {metrics_sparse['log_loss']:.4f}")

    # save neural network results
    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "nn_results.csv"
    df.to_csv(out_csv, index=False)

    feat_path = RESULTS_DIR / "nn_active_features.json"
    with open(feat_path, "w") as f:
        json.dump(active_features, f, indent=2)


if __name__ == "__main__":
    main()
