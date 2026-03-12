"""
preprocess.py
=============
Full preprocessing pipeline for the CICIDS2017 dataset.

Memory-efficient design for 2M+ row datasets:
  - Feature selection (variance, correlation, mutual info) runs on a
    stratified sample of at most --fs_sample_size rows.
  - The full dataset is then reduced to only the selected columns before
    train/test split and scaling — keeping peak RAM usage manageable.

Usage:
    python preprocess.py \
        --dataset_dir ../dataset \
        --output_dir  ../models \
        [--top_k 40] \
        [--corr_threshold 0.95] \
        [--fs_sample_size 300000] \
        [--test_size 0.2] \
        [--random_state 42]

Outputs (all written to output_dir):
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    scaler.pkl
    selected_features.pkl
    label_encoder.pkl
"""

import argparse
import glob
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging setup — writes to stdout AND logs/idps.log
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
logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IRRELEVANT_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
]
LABEL_COLUMN = "Label"


# ---------------------------------------------------------------------------
# Step 1: Load dataset file-by-file (never all in RAM at once)
# ---------------------------------------------------------------------------

def find_csv_files(dataset_dir: str) -> list[str]:
    """Return sorted list of .csv paths under dataset_dir."""
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{dataset_dir}'. "
            "Place CICIDS2017 CSV files there."
        )
    return files


def load_single_csv(path: str) -> pd.DataFrame:
    """Load one CSV, strip column whitespace, and validate label exists."""
    logger.info("  Loading: %s", os.path.basename(path))
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Step 2: Basic cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns, Inf→NaN, drop NaN rows, drop duplicates."""
    # Drop metadata columns
    df = df.drop(columns=[c for c in IRRELEVANT_COLUMNS if c in df.columns],
                 errors="ignore")

    # Verify label column
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    # Inf → NaN → drop
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(
        "    Inf/NaN: %d → %d rows (dropped %d)",
        before, len(df), before - len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Step 3: Build a stratified sample across all CSV files
# ---------------------------------------------------------------------------

def build_sample(
    csv_files: list[str],
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load all CSVs one file at a time, collect a proportional random sample
    from each, and return a combined sample DataFrame.

    The per-file sample fraction = sample_size / total_rows_estimate.
    We use a fixed fraction of 10% with a hard cap so we stay well under RAM.
    """
    logger.info("=== Building stratified sample for feature selection ===")
    logger.info("Target sample size: %d", sample_size)
    frames = []
    total_loaded = 0

    for path in csv_files:
        df = load_single_csv(path)
        try:
            df = clean_dataframe(df)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        total_loaded += len(df)
        n_take = min(len(df), max(1, sample_size // len(csv_files)))
        sampled = df.sample(n=n_take, random_state=random_state)
        frames.append(sampled)
        logger.info(
            "    Sampled %d / %d rows from %s",
            n_take, len(df), os.path.basename(path),
        )
        del df

    if not frames:
        raise RuntimeError("No usable data found after cleaning all CSV files.")

    sample_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined sample: %d rows × %d columns", *sample_df.shape
    )
    return sample_df


# ---------------------------------------------------------------------------
# Step 4: Encode labels and select numeric features
# ---------------------------------------------------------------------------

def encode_labels(
    df: pd.DataFrame,
    le: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    if le is None:
        le = LabelEncoder()
        df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN].astype(str))
        logger.info("Label classes (%d): %s", len(le.classes_), list(le.classes_))
    else:
        # Transform using existing encoder (unseen → mapped to -1 via try/except)
        df[LABEL_COLUMN] = le.transform(df[LABEL_COLUMN].astype(str))
    return df, le


# ---------------------------------------------------------------------------
# Step 5: Three-stage feature selection (runs on sample only)
# ---------------------------------------------------------------------------

def select_features(
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    corr_threshold: float = 0.95,
    top_k: int = 40,
) -> list[str]:
    """
    Stage 1 — Variance Threshold
    Stage 2 — Correlation filter
    Stage 3 — Information Gain (mutual_info_classif)
    """
    # Stage 1
    logger.info("--- Stage 1: Variance Threshold ---")
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_sample)
    kept = X_sample.columns[vt.get_support()].tolist()
    logger.info("  %d → %d features", X_sample.shape[1], len(kept))
    X1 = X_sample[kept]

    # Stage 2
    logger.info("--- Stage 2: Correlation filter (|r| > %.2f) ---", corr_threshold)
    corr = X1.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    X2 = X1.drop(columns=to_drop)
    logger.info("  Dropped %d correlated → %d remaining", len(to_drop), X2.shape[1])

    # Stage 3
    logger.info("--- Stage 3: Information Gain (top_k=%d on %d rows) ---",
                top_k, len(X2))
    actual_k = min(top_k, X2.shape[1])
    mi = mutual_info_classif(X2, y_sample, random_state=42, n_jobs=-1)
    mi_s = pd.Series(mi, index=X2.columns).sort_values(ascending=False)
    selected = mi_s.head(actual_k).index.tolist()
    logger.info("  Selected (%d): %s", len(selected), selected)
    return selected


# ---------------------------------------------------------------------------
# Step 6: Full pass — load each CSV, keep only selected columns, accumulate
# ---------------------------------------------------------------------------

def build_full_dataset(
    csv_files: list[str],
    selected_features: list[str],
    le: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Second pass through all CSVs.  For each file:
      - Load & clean
      - Keep only selected_features + Label
      - Encode labels with the already-fitted LabelEncoder

    Returns X (numpy, float32), y (numpy, int32) for the whole dataset.
    Storing as float32 halves memory vs float64.
    """
    logger.info("=== Full pass: extracting selected features from all CSVs ===")
    X_parts, y_parts = [], []
    all_classes = set(le.classes_)

    for path in csv_files:
        df = load_single_csv(path)
        try:
            df = clean_dataframe(df)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        # Drop rows whose label was never seen during sample encoding
        known_mask = df[LABEL_COLUMN].astype(str).isin(all_classes)
        df = df[known_mask]

        # Encode labels
        df, _ = encode_labels(df, le=le)

        # Keep only selected numeric features + label
        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            logger.warning("  Features missing in %s (will fill 0): %s",
                           os.path.basename(path), missing)
            for col in missing:
                df[col] = 0.0

        X_parts.append(df[selected_features].values.astype(np.float32))
        y_parts.append(df[LABEL_COLUMN].values.astype(np.int32))

        rows = len(df)
        logger.info("  %s: %d rows kept", os.path.basename(path), rows)
        del df

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    logger.info("Full dataset assembled: X=%s y=%s", X_all.shape, y_all.shape)
    return X_all, y_all


# ---------------------------------------------------------------------------
# Step 7: Scale & save
# ---------------------------------------------------------------------------

def scale_and_save(
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: str,
) -> StandardScaler:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logger.info("Applied StandardScaler.")
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_dir: str,
    output_dir: str,
    top_k: int = 40,
    corr_threshold: float = 0.95,
    fs_sample_size: int = 300_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    try:
        csv_files = find_csv_files(dataset_dir)
        logger.info("Found %d CSV file(s).", len(csv_files))

        # --- Phase A: Sample-based feature selection ---
        sample_df = build_sample(csv_files, fs_sample_size, random_state)

        sample_df = sample_df.select_dtypes(
            include=[np.number, object]
        )
        # Numeric features only (exclude Label temporarily)
        X_sample = sample_df.drop(columns=[LABEL_COLUMN]).select_dtypes(include=[np.number])
        y_raw_sample = sample_df[LABEL_COLUMN]

        # Encode labels on sample → gives us the LabelEncoder
        le = LabelEncoder()
        y_sample = pd.Series(le.fit_transform(y_raw_sample.astype(str)),
                             index=y_raw_sample.index)
        logger.info("Label classes from sample (%d): %s",
                    len(le.classes_), list(le.classes_))

        selected_features = select_features(
            X_sample, y_sample,
            corr_threshold=corr_threshold,
            top_k=top_k,
        )
        del sample_df, X_sample, y_sample, y_raw_sample

        # --- Phase B: Full dataset with selected features only ---
        X_all, y_all = build_full_dataset(csv_files, selected_features, le)

        logger.info(
            "Class distribution: %s",
            dict(zip(*np.unique(y_all, return_counts=True))),
        )

        # --- Phase C: Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y_all,
        )
        logger.info(
            "Train/Test split — Train: %d | Test: %d", len(X_train), len(X_test)
        )
        del X_all, y_all

        # --- Phase D: Scale ---
        X_train_s, X_test_s, scaler = scale_and_save(X_train, X_test, output_dir)
        del X_train, X_test

        # --- Phase E: Save ---
        np.save(os.path.join(output_dir, "X_train.npy"), X_train_s)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test_s)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(selected_features, os.path.join(output_dir, "selected_features.pkl"))
        joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

        elapsed = time.time() - t0
        logger.info(
            "Preprocessing complete in %.1f s. Artifacts saved to '%s'.",
            elapsed, output_dir,
        )
        logger.info(
            "Summary — X_train: %s | X_test: %s | Classes: %d | Features: %d",
            X_train_s.shape, X_test_s.shape, len(le.classes_), len(selected_features),
        )

    except MemoryError:
        logger.exception(
            "MemoryError. Try --fs_sample_size 100000 or add more RAM."
        )
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during preprocessing.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CICIDS2017 preprocessing pipeline for AI-IDPS"
    )
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset"),
        help="Directory containing CICIDS2017 CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
        help="Directory to save processed arrays and .pkl artifacts.",
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Number of top features to keep via Information Gain.",
    )
    parser.add_argument(
        "--corr_threshold", type=float, default=0.95,
        help="Pearson correlation threshold for dropping redundant features.",
    )
    parser.add_argument(
        "--fs_sample_size", type=int, default=300_000,
        help="Rows to sample across all CSVs for feature selection (default: 300000).",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to use as the test split.",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        corr_threshold=args.corr_threshold,
        fs_sample_size=args.fs_sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
