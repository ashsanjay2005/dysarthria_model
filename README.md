# Dysarthria Detection Baseline (TORGO)

This repository implements a baseline system for dysarthria detection using the TORGO speech dataset. The goal is to establish a clear and reproducible starting point using standard acoustic features and a classical machine learning model, while being explicit about evaluation assumptions and limitations.

The pipeline downloads TORGO via `kagglehub`, runs basic dataset sanity checks, extracts MFCC summary features from each audio file, and trains an RBF-kernel SVM classifier evaluated with a speaker-level train/test split.

## 1) Overview

- **Task:** binary classification (control vs dysarthric speech)
- **Input:** TORGO `.wav` files organized by speaker and session
- **Output:** dataset statistics, audio quality summaries, and classification metrics
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

Speaker IDs and microphone types are parsed directly from directory names.

## 3) Method

### Feature extraction

- Audio is loaded with `librosa` at 16 kHz (mono).
- For each file, 13 MFCCs are computed.
- MFCCs are summarized using the mean and standard deviation over time.
- Final feature representation: **26 dimensions per utterance**.

This feature set is intentionally simple and commonly used as a baseline in speech classification tasks.

### Model and evaluation

- Features are standardized using `StandardScaler`.
- Classifier: `SVC` with an RBF kernel (`C=1.0`, `gamma="scale"`).
- Evaluation uses a **speaker-level train/test split**:
  - Speakers are randomly split (≈80% train, 20% test).
  - All utterances from a given speaker appear only in one split.

This setup avoids speaker leakage and provides a more realistic estimate of generalization to unseen speakers.

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

`main.py` prints intermediate results to make the pipeline transparent:

- Dataset path and top-level directory structure.
- Number of `.wav` files per group (`F_Con`, `F_Dys`, `M_Con`, `M_Dys`).
- Dataset-level sanity checks:
  - speaker ID overlap across groups (to detect label mixing)
  - microphone type distribution inferred from directory names
- Random audio quality scan (default: 50 files per group):
  - duration statistics (min / median / max)
  - counts of failed loads, near-silent clips, extreme durations, and clipping
  - example file paths flagged for inspection
- Model outputs:
  - feature matrix shape
  - accuracy, ROC-AUC, and classification report on the speaker-held-out test set

## 6) Results

Representative results from a speaker-level split (exact values vary depending on the held-out speakers):

- Feature matrix shape: ~`(17k, 26)`
- Accuracy: ~`0.55–0.65`
- ROC-AUC: typically below file-level baselines

Performance is substantially lower than file-level splits, reflecting the difficulty of speaker-independent dysarthria detection with a limited number of speakers.

## 7) Limitations and future work

### Limited number of speakers

TORGO contains a small number of speakers, which leads to high variance in speaker-level evaluation. Results depend strongly on which speakers are held out.

### Future directions

- Leave-One-Speaker-Out (LOSO) evaluation for fully speaker-independent testing.
- Stronger acoustic features (e.g., delta and delta-delta MFCCs).
- Group-aware cross-validation (e.g., `GroupKFold`).
- Cross-dataset evaluation (e.g., combining TORGO with UASpeech).
- Exploration of speaker-invariant representations or domain adaptation.

## 8) Repository structure

```text
.
├── main.py    # End-to-end pipeline: download → checks → MFCCs → speaker-level SVM
└── README.md
```