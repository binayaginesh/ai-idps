"""
train.py
========
Training orchestration for AI-IDPS.

Loads the preprocessed NumPy arrays produced by preprocess.py and trains
both the Random Forest (primary) and SVM (secondary) models, saving each
as a .pkl file in --model_dir.

Usage:
    python train.py \
        --data_dir  ../models \
        --model_dir ../models \
        [--models rf svm]

The --data_dir is expected to contain:
    X_train.npy, y_train.npy  (output of preprocess.py)
"""

import argparse
import logging
import os
import sys
import time

import joblib
import numpy as np

# Local import
sys.path.insert(0, os.path.dirname(__file__))
from model import get_model, SUPPORTED_MODELS

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
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helper — load array
# ---------------------------------------------------------------------------

def _load_array(data_dir: str, filename: str) -> np.ndarray:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file '{path}' not found. "
            "Run preprocess.py first."
        )
    arr = np.load(path, allow_pickle=False)
    logger.info("Loaded %s — shape: %s, dtype: %s", filename, arr.shape, arr.dtype)
    return arr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_dir: str,
    **model_kwargs,
) -> None:
    """
    Instantiate, fit, and save a single model.

    Parameters
    ----------
    name        : Model identifier, e.g. 'rf' or 'svm'.
    X_train     : Scaled training features.
    y_train     : Encoded training labels.
    model_dir   : Directory to write the saved model file.
    **model_kwargs : Extra hyperparameters forwarded to get_model().
    """
    logger.info("=" * 60)
    logger.info("Training model: %s", name.upper())
    logger.info("=" * 60)

    clf = get_model(name, **model_kwargs)

    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    logger.info("Training complete in %.1f s.", elapsed)

    # Random Forest provides a free OOB accuracy estimate
    if hasattr(clf, "oob_score_"):
        logger.info("  OOB accuracy: %.4f", clf.oob_score_)

    # Save
    out_path = os.path.join(model_dir, f"{name}_model.pkl")
    joblib.dump(clf, out_path)
    logger.info("Model saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(
    data_dir: str,
    model_dir: str,
    models_to_train: list[str],
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    # Load preprocessed data
    X_train = _load_array(data_dir, "X_train.npy")
    y_train = _load_array(data_dir, "y_train.npy")

    logger.info(
        "Class distribution in training set: %s",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )

    # Load label encoder so we can log class names
    le_path = os.path.join(data_dir, "label_encoder.pkl")
    if os.path.isfile(le_path):
        le = joblib.load(le_path)
        logger.info("Label classes: %s", list(le.classes_))
    else:
        logger.warning("label_encoder.pkl not found — class names unavailable.")

    # Train requested models
    for model_name in models_to_train:
        model_name = model_name.strip().lower()
        if model_name not in SUPPORTED_MODELS:
            logger.error(
                "Unsupported model '%s'. Choose from: %s",
                model_name,
                SUPPORTED_MODELS,
            )
            continue
        try:
            train_model(model_name, X_train, y_train, model_dir)
        except Exception as exc:
            logger.exception("Failed to train model '%s': %s", model_name, exc)

    logger.info("All requested models trained and saved to '%s'.", model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for AI-IDPS."
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory containing preprocessed .npy files.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory to save trained model .pkl files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="Which models to train. Default: all.",
    )

    args = parser.parse_args()
    run_training(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        models_to_train=args.models,
    )
