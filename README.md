# AI-Based Intrusion Detection and Prevention System (AI-IDPS)

A production-ready, real-time Network Intrusion Detection and Prevention System that uses machine learning trained on the **CICIDS2017** dataset.

---

## Project Structure

```
AI_IDPS/
├── dataset/              ← Place CICIDS2017 CSV files here
├── models/               ← Saved .pkl and .npy artifacts (auto-generated)
├── logs/                 ← Log output (auto-generated)
├── src/
│   ├── preprocess.py     ← Phase 1: Data preprocessing & feature selection
│   ├── model.py          ← Phase 1: Model factory (RF + SVM config)
│   ├── train.py          ← Phase 1: Model training
│   ├── evaluate.py       ← Phase 1: Evaluation & metrics
│   ├── realtime_monitor.py  ← Phase 2: Real-time Snort log monitoring
│   └── blocker.py           ← Phase 3: Automated IP blocking via iptables
├── requirements.txt
└── README.md
```

---

## Phase 1 — Data Preprocessing & Model Training

### 1. Dataset Setup

Download the **CICIDS2017** dataset from the official source:

**[University of New Brunswick — CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)**

Place all CSV files inside `AI_IDPS/dataset/`:
```
AI_IDPS/dataset/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

```bash
python src/preprocess.py \
    --dataset_dir dataset \
    --output_dir  models \
    --top_k 40 \
    --corr_threshold 0.95 \
    --test_size 0.2
```

**Outputs in `models/`:**
| File | Description |
|---|---|
| `X_train.npy` | Scaled training features |
| `X_test.npy` | Scaled test features |
| `y_train.npy` | Encoded training labels |
| `y_test.npy` | Encoded test labels |
| `scaler.pkl` | Fitted `StandardScaler` |
| `selected_features.pkl` | List of selected feature names |
| `label_encoder.pkl` | `LabelEncoder` mapping class names ↔ integers |

### 4. Train Models

```bash
# Train both RF and SVM
python src/train.py --data_dir models --model_dir models --models rf svm

# Train only Random Forest (faster)
python src/train.py --data_dir models --model_dir models --models rf
```

**Outputs in `models/`:**
- `rf_model.pkl` — Random Forest (primary)
- `svm_model.pkl` — SVM (secondary)

### 5. Evaluate

```bash
# Evaluate Random Forest
python src/evaluate.py --model rf --data_dir models --model_dir models

# Evaluate SVM
python src/evaluate.py --model svm --data_dir models --model_dir models
```

Results are printed to stdout **and** appended to `logs/idps.log`.

---

## Feature Selection Pipeline

```
Raw Features (~78)
      ↓
Variance Threshold   — remove near-zero-variance features
      ↓
Correlation Filter   — remove one of each pair with |corr| > 0.95
      ↓
Information Gain     — keep top 40 features by mutual information score
      ↓
Selected Features (~40)
```

---

## Model Configuration

| Model | Key Parameters |
|---|---|
| **Random Forest** (primary) | `n_estimators=200`, `class_weight=balanced`, `oob_score=True`, `n_jobs=-1` |
| **SVM** (secondary) | `kernel=rbf`, `C=1.0`, `class_weight=balanced`, `probability=True` |

---

## Typical Expected Performance (CICIDS2017)

| Metric | Random Forest | SVM |
|---|---|---|
| Accuracy | ~99% | ~97% |
| F1-Score (weighted) | ~99% | ~97% |

> **Note**: Exact numbers depend on the subset of CSV files used and
> feature selection parameters.

---

## Phase 2 — Real-Time Detection *(coming next)*

`src/realtime_monitor.py` — monitors Snort alert logs, extracts flow features,
runs them through the trained model, and logs predictions to `logs/attacks.log`.

## Phase 3 — Automated Prevention *(coming next)*

`src/blocker.py` — blocks malicious source IPs automatically via
`iptables` when the model classifies a connection as an attack.

---

## Logging

All pipeline stages write to `logs/idps.log` in addition to stdout.
Log format: `YYYY-MM-DD HH:MM:SS [LEVEL] module — message`

---

## Requirements

- Python 3.10+
- Ubuntu Linux (for Phase 2/3 — iptables, Snort)
- Root / sudo privileges (Phase 3 only)
