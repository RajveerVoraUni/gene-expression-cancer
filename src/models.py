import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import RANDOM_STATE


# linear models factory
def get_linear_model(model_name: str, C: float = 0.2, l1_ratio: float = 0.5):
    # unpenalized multinomial logistic regression
    if model_name == "Softmax":
        return LogisticRegression(
            C=1e5,
            solver="lbfgs",
            max_iter=500,
            random_state=RANDOM_STATE
        )

    # ridge l2 regularized logistic regression
    elif model_name == "Ridge":
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=RANDOM_STATE
        )

    # lasso l1 sparse logistic regression via saga
    elif model_name == "Lasso":
        return LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            max_iter=500,
            tol=1e-3,
            random_state=RANDOM_STATE
        )

    # elastic net l1 and l2 logistic regression via saga
    elif model_name == "ElasticNet":
        return LogisticRegression(
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            C=C,
            solver="saga",
            max_iter=500,
            tol=1e-3,
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(f"Unknown linear model: {model_name}")


# find genes with non-zero weights across any class
def extract_active_linear_features(model: LogisticRegression, feature_names: np.ndarray, threshold: float = 1e-5):
    if not hasattr(model, "coef_"):
        return set()

    is_active = np.any(np.abs(model.coef_) > threshold, axis=0)
    return set(feature_names[is_active])


# tree models factory
def get_tree_model(model_name: str):
    # tuned random forest parameters
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=32,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # tuned xgboost parameters using histogram method
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.3,
            reg_lambda=5.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(f"Unknown tree model: {model_name}")


# get top features ranked by importance score
def extract_top_k_tree_features(model, feature_names: np.ndarray, k: int = 50):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:k]
        return set(feature_names[top_idx])

    return set()
