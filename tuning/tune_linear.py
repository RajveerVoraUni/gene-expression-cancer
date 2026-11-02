import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEFAULT_P, RANDOM_STATE, RESULTS_DIR
from src.data_loader import load_data
from src.create_splits import load_split
from src.preprocessor import LeakageFreePreprocessor
from src.evaluate import evaluate_predictions
from src.models import extract_active_linear_features


def tune_linear_models():
    # load data and training split
    X, y, gene_names, _, _ = load_data()
    train_idx, test_idx = load_split(0)

    # fit feature selection and scaler strictly on training samples
    prep = LeakageFreePreprocessor(p=DEFAULT_P, apply_pca=False, random_state=RANDOM_STATE)
    X_train, _, sel_genes = prep.fit_transform(X[train_idx], gene_names)
    y_train = y[train_idx]
    X_test = prep.transform(X[test_idx])
    y_test = y[test_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuning_summary = {}

    # 1. tune ridge regularizer C
    c_grid_ridge = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_ridge = {"score": -1, "C": None, "cv_results": []}

    for c in c_grid_ridge:
        f1_scores = []
        losses = []
        for cv_tr, cv_val in skf.split(X_train, y_train):
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=500, random_state=RANDOM_STATE)
            clf.fit(X_train[cv_tr], y_train[cv_tr])
            preds = clf.predict(X_train[cv_val])
            probs = clf.predict_proba(X_train[cv_val])
            m = evaluate_predictions(y_train[cv_val], preds, probs)
            f1_scores.append(m["macro_f1"])
            losses.append(m["log_loss"])

        mean_f1 = np.mean(f1_scores)
        mean_loss = np.mean(losses)
        best_ridge["cv_results"].append({"C": c, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss)})

        if mean_f1 > best_ridge["score"] or (mean_f1 == best_ridge["score"] and mean_loss < best_ridge.get("best_loss", 999)):
            best_ridge["score"] = float(mean_f1)
            best_ridge["best_loss"] = float(mean_loss)
            best_ridge["C"] = c

    # evaluate best ridge on held-out test split
    clf_ridge = LogisticRegression(C=best_ridge["C"], solver="lbfgs", max_iter=500, random_state=RANDOM_STATE)
    clf_ridge.fit(X_train, y_train)
    test_metrics = evaluate_predictions(y_test, clf_ridge.predict(X_test), clf_ridge.predict_proba(X_test))
    best_ridge["test_accuracy"] = test_metrics["accuracy"]
    best_ridge["test_macro_f1"] = test_metrics["macro_f1"]
    best_ridge["test_log_loss"] = test_metrics["log_loss"]
    tuning_summary["Ridge"] = best_ridge

    # 2. tune lasso sparsity penalty C
    c_grid_lasso = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_lasso = {"score": -1, "C": None, "cv_results": []}

    for c in c_grid_lasso:
        f1_scores, losses, n_actives = [], [], []
        for cv_tr, cv_val in skf.split(X_train, y_train):
            clf = LogisticRegression(penalty="l1", C=c, solver="saga", max_iter=500, tol=1e-3, random_state=RANDOM_STATE)
            clf.fit(X_train[cv_tr], y_train[cv_tr])
            preds = clf.predict(X_train[cv_val])
            probs = clf.predict_proba(X_train[cv_val])
            m = evaluate_predictions(y_train[cv_val], preds, probs)
            act = extract_active_linear_features(clf, sel_genes)
            f1_scores.append(m["macro_f1"])
            losses.append(m["log_loss"])
            n_actives.append(len(act))

        mean_f1 = np.mean(f1_scores)
        mean_loss = np.mean(losses)
        mean_act = np.mean(n_actives)
        best_lasso["cv_results"].append({"C": c, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss), "active_genes": float(mean_act)})

        if mean_f1 > best_lasso["score"] or (mean_f1 == best_lasso["score"] and mean_loss < best_lasso.get("best_loss", 999)):
            best_lasso["score"] = float(mean_f1)
            best_lasso["best_loss"] = float(mean_loss)
            best_lasso["C"] = c
            best_lasso["active_genes"] = float(mean_act)

    # evaluate best lasso on held-out test split
    clf_lasso = LogisticRegression(penalty="l1", C=best_lasso["C"], solver="saga", max_iter=500, tol=1e-3, random_state=RANDOM_STATE)
    clf_lasso.fit(X_train, y_train)
    test_metrics = evaluate_predictions(y_test, clf_lasso.predict(X_test), clf_lasso.predict_proba(X_test))
    act_test = extract_active_linear_features(clf_lasso, sel_genes)
    best_lasso["test_accuracy"] = test_metrics["accuracy"]
    best_lasso["test_macro_f1"] = test_metrics["macro_f1"]
    best_lasso["test_log_loss"] = test_metrics["log_loss"]
    best_lasso["test_active_genes"] = len(act_test)
    tuning_summary["Lasso"] = best_lasso

    # 3. tune elastic net l1 ratio and penalty C
    c_grid_en = [0.05, 0.1, 0.2, 0.5]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_en = {"score": -1, "C": None, "l1_ratio": None, "cv_results": []}

    for c in c_grid_en:
        for l1_r in l1_ratios:
            f1_scores, losses, n_actives = [], [], []
            for cv_tr, cv_val in skf.split(X_train, y_train):
                clf = LogisticRegression(penalty="elasticnet", l1_ratio=l1_r, C=c, solver="saga", max_iter=500, tol=1e-3, random_state=RANDOM_STATE)
                clf.fit(X_train[cv_tr], y_train[cv_tr])
                preds = clf.predict(X_train[cv_val])
                probs = clf.predict_proba(X_train[cv_val])
                m = evaluate_predictions(y_train[cv_val], preds, probs)
                act = extract_active_linear_features(clf, sel_genes)
                f1_scores.append(m["macro_f1"])
                losses.append(m["log_loss"])
                n_actives.append(len(act))

            mean_f1 = np.mean(f1_scores)
            mean_loss = np.mean(losses)
            mean_act = np.mean(n_actives)
            best_en["cv_results"].append({"C": c, "l1_ratio": l1_r, "mean_f1": float(mean_f1), "mean_loss": float(mean_loss), "active_genes": float(mean_act)})

            if mean_f1 > best_en["score"] or (mean_f1 == best_en["score"] and mean_loss < best_en.get("best_loss", 999)):
                best_en["score"] = float(mean_f1)
                best_en["best_loss"] = float(mean_loss)
                best_en["C"] = c
                best_en["l1_ratio"] = l1_r
                best_en["active_genes"] = float(mean_act)

    # evaluate best elastic net on held-out test split
    clf_en = LogisticRegression(penalty="elasticnet", l1_ratio=best_en["l1_ratio"], C=best_en["C"], solver="saga", max_iter=500, tol=1e-3, random_state=RANDOM_STATE)
    clf_en.fit(X_train, y_train)
    test_metrics = evaluate_predictions(y_test, clf_en.predict(X_test), clf_en.predict_proba(X_test))
    act_test = extract_active_linear_features(clf_en, sel_genes)
    best_en["test_accuracy"] = test_metrics["accuracy"]
    best_en["test_macro_f1"] = test_metrics["macro_f1"]
    best_en["test_log_loss"] = test_metrics["log_loss"]
    best_en["test_active_genes"] = len(act_test)
    tuning_summary["ElasticNet"] = best_en

    # save tuning results
    out_file = root_dir / "tuning" / "linear_tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(tuning_summary, f, indent=2)


if __name__ == "__main__":
    tune_linear_models()
