"""
evaluate.py
===========
Evaluation & metric reporting for AI-IDPS.

Loads a trained model and the test split produced by preprocess.py, then
computes and logs:
  - Accuracy
  - Precision (macro + weighted)
  - Recall    (macro + weighted)
  - F1-Score  (macro + weighted)
  - Confusion Matrix (with class-name labels)

Usage:
    python evaluate.py \
        --model rf \
        --data_dir  ../models \
        --model_dir ../models
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "idps.log"), mode="a"),
    ],
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_array(data_dir: str, filename: str) -> np.ndarray:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file '{path}' not found. Run preprocess.py first."
        )
    return np.load(path, allow_pickle=False)


def _load_model(model_dir: str, model_name: str):
    path = os.path.join(model_dir, f"{model_name}_model.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'. Run train.py first."
        )
    clf = joblib.load(path)
    logger.info("Loaded model from '%s'.", path)
    return clf


def _load_label_encoder(model_dir: str):
    path = os.path.join(model_dir, "label_encoder.pkl")
    if os.path.isfile(path):
        return joblib.load(path)
    logger.warning("label_encoder.pkl not found — using numeric class labels.")
    return None


def _format_confusion_matrix(cm: np.ndarray, class_names: list) -> str:
    """Return a human-readable string representation of the confusion matrix."""
    header = "      " + "  ".join(f"{n:>10}" for n in class_names)
    rows = [header, ""]
    for i, row in enumerate(cm):
        label = f"{class_names[i]:>6}"
        values = "  ".join(f"{v:>10}" for v in row)
        rows.append(f"{label}  {values}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_name: str,
    data_dir: str,
    model_dir: str,
) -> dict:
    """
    Run full evaluation for *model_name* against the held-out test split.

    Returns
    -------
    dict of metric name → value.
    """
    # --- Load data ---
    X_test = _load_array(data_dir, "X_test.npy")
    y_test = _load_array(data_dir, "y_test.npy")
    logger.info("Test set shape: %s", X_test.shape)

    # --- Load model ---
    clf = _load_model(model_dir, model_name)

    # --- Predict ---
    y_pred = clf.predict(X_test)

    # --- Label names ---
    le = _load_label_encoder(model_dir)
    if le is not None:
        class_names = list(le.classes_)
    else:
        class_names = [str(c) for c in sorted(np.unique(y_test))]

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )

    # --- Log results ---
    separator = "=" * 60
    logger.info(separator)
    logger.info("EVALUATION RESULTS — Model: %s", model_name.upper())
    logger.info(separator)
    logger.info("  Accuracy            : %.4f", acc)
    logger.info("  Precision (macro)   : %.4f", prec_macro)
    logger.info("  Precision (weighted): %.4f", prec_weighted)
    logger.info("  Recall    (macro)   : %.4f", rec_macro)
    logger.info("  Recall    (weighted): %.4f", rec_weighted)
    logger.info("  F1-Score  (macro)   : %.4f", f1_macro)
    logger.info("  F1-Score  (weighted): %.4f", f1_weighted)
    logger.info(separator)
    logger.info("Per-class Report:\n%s", report)
    logger.info("Confusion Matrix:\n%s", _format_confusion_matrix(cm, class_names))
    logger.info(separator)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AI-IDPS model."
    )
    parser.add_argument(
        "--model",
        default="rf",
        choices=["rf", "svm"],
        help="Which model to evaluate (default: rf).",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory containing preprocessed .npy test files.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory containing trained .pkl model files.",
    )

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )
