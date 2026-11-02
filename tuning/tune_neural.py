import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, RANDOM_STATE
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


def tune_neural_models():
    # enable mac gpu acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # load data and training split
    X, y, gene_names, _, _ = load_data()
    train_idx, test_idx = load_split(0)

    # preprocessing strictly on training data
    prep_pca = LeakageFreePreprocessor(p=None, apply_pca=True, pca_variance=0.95, random_state=RANDOM_STATE)
    X_train_pca, _, _ = prep_pca.fit_transform(X[train_idx])
    X_test_pca = prep_pca.transform(X[test_idx])

    prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE)
    X_train, _, sel_genes = prep.fit_transform(X[train_idx], gene_names)
    X_test = prep.transform(X[test_idx])

    y_train = y[train_idx]
    y_test = y[test_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuning_summary = {}

    # 1. tune dense pca mlp
    pca_grid = [
        {"hidden1": 128, "hidden2": 64, "lr": 0.003, "epochs": 40},
        {"hidden1": 128, "hidden2": 64, "lr": 0.001, "epochs": 45},
        {"hidden1": 64, "hidden2": 32, "lr": 0.005, "epochs": 40},
    ]

    best_pca_params = pca_grid[0]
    best_pca_score = -1.0
    best_pca_loss = float("inf")
    cv_pca_results = []

    for p in pca_grid:
        f1_scores = []
        losses = []
        for cv_tr, cv_val in skf.split(X_train_pca, y_train):
            net = PCAMLPNet(in_features=X_train_pca.shape[1], hidden1=p["hidden1"], hidden2=p["hidden2"])
            train_pytorch_model(net, X_train_pca[cv_tr], y_train[cv_tr], epochs=p["epochs"], lr=p["lr"], device=device)
            preds, probs = predict_pytorch_model(net, X_train_pca[cv_val], device=device)
            m = evaluate_predictions(y_train[cv_val], preds, probs)
            f1_scores.append(m["macro_f1"])
            losses.append(m["log_loss"])

        mean_f1 = np.mean(f1_scores)
        mean_loss = np.mean(losses)
        cv_pca_results.append({"params": p, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss)})

        if mean_f1 > best_pca_score or (mean_f1 == best_pca_score and mean_loss < best_pca_loss):
            best_pca_score = float(mean_f1)
            best_pca_loss = float(mean_loss)
            best_pca_params = p

    # evaluate best pca mlp on test set
    net = PCAMLPNet(in_features=X_train_pca.shape[1], hidden1=best_pca_params["hidden1"], hidden2=best_pca_params["hidden2"])
    train_pytorch_model(net, X_train_pca, y_train, epochs=best_pca_params["epochs"], lr=best_pca_params["lr"], device=device)
    preds, probs = predict_pytorch_model(net, X_test_pca, device=device)
    m_test = evaluate_predictions(y_test, preds, probs)
    tuning_summary["PCA_MLP"] = {
        "score": best_pca_score,
        "best_loss": best_pca_loss,
        "params": best_pca_params,
        "cv_results": cv_pca_results,
        "test_accuracy": m_test["accuracy"],
        "test_macro_f1": m_test["macro_f1"],
        "test_log_loss": m_test["log_loss"],
    }

    # 2. tune sparse group lasso network
    sparse_grid = [
        {"hidden": 64, "lambda_group": 0.02, "lr": 0.005, "epochs": 40},
        {"hidden": 64, "lambda_group": 0.04, "lr": 0.005, "epochs": 45},
        {"hidden": 64, "lambda_group": 0.08, "lr": 0.003, "epochs": 45},
        {"hidden": 32, "lambda_group": 0.04, "lr": 0.005, "epochs": 40},
    ]

    best_sparse_params = sparse_grid[0]
    best_sparse_score = -1.0
    best_sparse_loss = float("inf")
    best_sparse_active = 0.0
    cv_sparse_results = []

    for p in sparse_grid:
        f1_scores = []
        losses = []
        actives = []
        for cv_tr, cv_val in skf.split(X_train, y_train):
            net = SparseGroupLassoMLPNet(in_features=DEFAULT_P, hidden=p["hidden"])
            train_pytorch_model(net, X_train[cv_tr], y_train[cv_tr], epochs=p["epochs"], lr=p["lr"], lambda_group=p["lambda_group"], device=device)
            preds, probs = predict_pytorch_model(net, X_train[cv_val], device=device)
            m = evaluate_predictions(y_train[cv_val], preds, probs)
            act = extract_active_nn_features(net, sel_genes, threshold=1e-4)
            f1_scores.append(m["macro_f1"])
            losses.append(m["log_loss"])
            actives.append(len(act))

        mean_f1 = np.mean(f1_scores)
        mean_loss = np.mean(losses)
        mean_act = np.mean(actives)
        cv_sparse_results.append({"params": p, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss), "active_genes": float(mean_act)})

        if mean_f1 > best_sparse_score or (mean_f1 == best_sparse_score and mean_loss < best_sparse_loss):
            best_sparse_score = float(mean_f1)
            best_sparse_loss = float(mean_loss)
            best_sparse_params = p
            best_sparse_active = float(mean_act)

    # evaluate best sparse network on test set
    net = SparseGroupLassoMLPNet(in_features=DEFAULT_P, hidden=best_sparse_params["hidden"])
    train_pytorch_model(net, X_train, y_train, epochs=best_sparse_params["epochs"], lr=best_sparse_params["lr"], lambda_group=best_sparse_params["lambda_group"], device=device)
    preds, probs = predict_pytorch_model(net, X_test, device=device)
    m_test = evaluate_predictions(y_test, preds, probs)
    act_test = extract_active_nn_features(net, sel_genes, threshold=1e-4)
    tuning_summary["Sparse_NN"] = {
        "score": best_sparse_score,
        "best_loss": best_sparse_loss,
        "params": best_sparse_params,
        "active_genes": best_sparse_active,
        "cv_results": cv_sparse_results,
        "test_accuracy": m_test["accuracy"],
        "test_macro_f1": m_test["macro_f1"],
        "test_log_loss": m_test["log_loss"],
        "test_active_genes": len(act_test),
    }

    # save results
    out_file = root_dir / "tuning" / "neural_tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(tuning_summary, f, indent=2)


if __name__ == "__main__":
    tune_neural_models()
