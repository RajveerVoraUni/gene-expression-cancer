import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from src.config import NUM_CLASSES, CLASS_NAMES


# calculate accuracy, macro f1, and multiclass log loss
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    loss = np.nan
    if y_prob is not None:
        # clip probabilities to avoid numerical errors with log(0)
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        y_prob_clipped = y_prob_clipped / y_prob_clipped.sum(axis=1, keepdims=True)
        try:
            loss = log_loss(y_true, y_prob_clipped, labels=np.arange(NUM_CLASSES))
        except Exception:
            pass

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "log_loss": float(loss) if not np.isnan(loss) else None,
        "n_samples": int(len(y_true)),
    }


# compute per-class precision, recall, f1, and confusion matrix
def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return report, cm
