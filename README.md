# Dysarthria Detection Baseline (TORGO)

This repository contains a simple baseline for dysarthria detection using the TORGO speech dataset. The goal is to establish a clear starting point using standard acoustic features and a classical machine learning model, while making the evaluation assumptions explicit.

The pipeline downloads TORGO via `kagglehub`, runs basic dataset sanity checks, extracts MFCC summary features from each audio file, and trains an RBF-kernel SVM classifier.

## 1) Overview

- **Task:** binary classification (control vs dysarthric speech)
- **Input:** TORGO `.wav` files organized by speaker/session folders
- **Output:** dataset statistics, quality scan summaries, and baseline classification metrics
- **Tools:** Python, `kagglehub`, `librosa`, NumPy, scikit-learn, matplotlib

## 2) Dataset (TORGO) and label mapping

The dataset is downloaded directly from Kaggle:

- **Kaggle dataset:** `pranaykoppula/torgo-audio`

Directory groups used in this project:

- `F_Con` – female control  
- `F_Dys` – female dysarthric  
- `M_Con` – male control  
- `M_Dys` – male dysarthric  

Binary labels used throughout the code:

- `0` = control (`F_Con`, `M_Con`)
- `1` = dysarthric (`F_Dys`, `M_Dys`)

## 3) Method

### Feature extraction

- Audio is loaded with `librosa` at 16 kHz (mono).
- For each file, 13 MFCCs are computed.
- MFCCs are summarized using the mean and standard deviation over time.
- Final feature representation: **26 dimensions per file**.

This representation is intentionally simple and commonly used as a baseline in speech classification tasks.

### Model and evaluation

- Features are standardized using `StandardScaler`.
- Classifier: `SVC` with an RBF kernel (`C=1.0`, `gamma="scale"`).
- Evaluation includes:
  - A file-level stratified train/test split (80/20).
  - File-level 5-fold cross-validation using ROC-AUC.

## 4) How to run

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install kagglehub librosa numpy scikit-learn matplotlib
```

### Run the pipeline

```bash
python3 main.py
```

Notes:
- The first run may take some time due to dataset download and feature extraction.
- `kagglehub` may require Kaggle credentials to be configured locally.
- The script prints the resolved dataset path so you can verify the download.

## 5) What the script prints

`main.py` outputs several checkpoints to make the process transparent:

- Dataset path and top-level directory contents.
- Number of `.wav` files per group (`F_Con`, `F_Dys`, `M_Con`, `M_Dys`).
- Dataset-level sanity checks:
  - speaker ID overlap across groups (to detect label mixing)
  - microphone type distribution inferred from folder names
- Random audio quality scan (default: 50 files per group):
  - duration statistics (min / median / max)
  - counts of failed loads, near-silent clips, extreme durations, and clipping
  - example file paths flagged for inspection
- Model outputs:
  - feature matrix shape
  - Accuracy, ROC-AUC, and a classification report
  - 5-fold file-level cross-validated ROC-AUC (mean and standard deviation)

## 6) Baseline results

Example baseline results from a typical run (exact values may vary):

- Feature matrix shape: ~`(17632, 26)`
- Accuracy: ~`0.93`
- ROC-AUC (held-out split): ~`0.98`
- File-level CV ROC-AUC (5-fold): ~`0.86–0.87`

These results should be treated as an **optimistic baseline**.

## 7) Limitations and future work

### Speaker leakage

The current evaluation uses a **file-level split**, meaning the same speaker can appear in both training and test sets. This can inflate performance because the model may learn speaker-specific characteristics rather than dysarthria-related patterns.

As a result, the reported accuracy and ROC-AUC do **not** reflect true speaker-independent generalization.

### Future directions

- Speaker-level or leave-one-speaker-out (LOSO) evaluation.
- Stronger controls for speaker and session confounds.
- Cross-dataset evaluation (e.g., combining TORGO with UASpeech).
- Exploring more speaker-invariant representations.

## 8) Repository structure

```text
.
├── main.py    # End-to-end baseline: download → checks → MFCCs → SVM
└── README.md
```