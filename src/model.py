"""
model.py
========
Model factory for AI-IDPS.

Centralises all hyperparameter configuration so that train.py stays clean.
Adding a new algorithm only requires changes in this one file.

Usage (from train.py):
    from model import get_model
    rf  = get_model("rf")
    svm = get_model("svm")
"""

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logger = logging.getLogger("model")

# ---------------------------------------------------------------------------
# Supported model identifiers
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ("rf", "svm")


def get_model(name: str, **kwargs):
    """
    Return a freshly constructed, configured (but unfitted) scikit-learn
    estimator.

    Parameters
    ----------
    name : str
        One of ``'rf'`` (Random Forest) or ``'svm'`` (Support Vector Machine).
    **kwargs
        Optional keyword arguments that override the default hyperparameters.
        Pass only the parameters you want to change; defaults are preserved
        for everything else.

    Returns
    -------
    sklearn estimator

    Raises
    ------
    ValueError
        If *name* is not in SUPPORTED_MODELS.

    Examples
    --------
    >>> clf = get_model("rf", n_estimators=100)
    >>> clf = get_model("svm", C=2.0)
    """
    name = name.strip().lower()

    if name == "rf":
        defaults = dict(
            n_estimators=200,
            criterion="gini",
            max_depth=None,          # grow full trees; RF's own bagging prevents overfitting
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            oob_score=True,           # gives a free out-of-bag accuracy estimate
            n_jobs=-1,                # use all available CPU cores
            class_weight="balanced",  # handles class imbalance automatically
            random_state=42,
            verbose=0,
        )
        defaults.update(kwargs)
        model = RandomForestClassifier(**defaults)
        logger.info("RandomForestClassifier configured: %s", defaults)
        return model

    elif name == "svm":
        defaults = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,         # enable predict_proba for confidence scores
            class_weight="balanced",
            random_state=42,
            cache_size=2000,          # MB — large cache speeds up training
            verbose=False,
        )
        defaults.update(kwargs)
        model = SVC(**defaults)
        logger.info("SVC configured: %s", defaults)
        return model

    else:
        raise ValueError(
            f"Unknown model '{name}'. Supported models: {SUPPORTED_MODELS}"
        )


def list_models() -> list[str]:
    """Return the list of supported model identifiers."""
    return list(SUPPORTED_MODELS)
