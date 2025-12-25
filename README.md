# Dysarthria Detection Baseline (TORGO)

This repository implements a baseline system for dysarthria detection using the TORGO speech dataset. The goal is to provide a clear, reproducible starting point using standard acoustic features and a classical machine learning model, while being explicit about evaluation assumptions and limitations.

The pipeline downloads TORGO via `kagglehub`, runs dataset sanity checks, extracts MFCC-based features from each audio file, and trains an RBF-kernel SVM classifier. Evaluation includes a speaker-level train/test split plus LOSO (leave-one-speaker-out) analysis.

## 1) Overview

- **Task:** binary classification (control vs dysarthric speech)
- **Input:** TORGO `.wav` files organized by speaker and session
- **Output:** dataset statistics, audio quality summaries, and classification metrics
- **Tools:** Python, `kagglehub`, `librosa`, NumPy, scikit-learn

## 2) Dataset (TORGO) and label mapping

The dataset is downloaded directly from Kaggle:

- **Kaggle dataset:** `pranaykoppula/torgo-audio`

Directory groups used in this project:

- `F_Con` = female control  
- `F_Dys` = female dysarthric  
- `M_Con` = male control  
- `M_Dys` = male dysarthric  

Binary labels used throughout the code:

- `0` = control (`F_Con`, `M_Con`)
- `1` = dysarthric (`F_Dys`, `M_Dys`)

Speaker IDs and microphone types are parsed from directory names (for example: `wav_headMic_FC02S03`).

## 3) Method

### Feature extraction

- Audio is loaded with `librosa` at 16 kHz (mono).
- For each file, 13 MFCCs are computed.
- We also compute **delta** and **delta-delta** MFCCs (first and second time derivatives).
- Each coefficient is summarized using **mean** and **standard deviation** over time.
- Final feature representation: **78 dimensions per utterance**  
  `(13 MFCC + 13 delta + 13 delta-delta) * (mean + std) = 39 * 2 = 78`

Very short clips are skipped to avoid unstable MFCC and delta calculations.

### Model

- Features are standardized using `StandardScaler`.
- Classifier: `SVC` with an RBF kernel (`C=1.0`, `gamma="scale"`).

### Evaluation

This repo reports three views of performance:

1. **Speaker-level train/test split (recommended main metric)**  
   Speakers are randomly split (about 80% train, 20% test). All utterances from a speaker appear in only one split.

2. **File-level StratifiedKFold cross-validation (comparison only)**  
   Uses label-stratified folds across files. This can still include speaker leakage because folds are not grouped by speaker.

3. **LOSO (Leave-One-Speaker-Out) evaluation**  
   Holds out one speaker at a time as test data and reports per-speaker performance.

## 4) How to run

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install kagglehub librosa numpy scikit-learn
```

### Run the pipeline

```bash
python3 main.py
```

Notes:
- The first run may take time due to dataset download and feature extraction.
- `kagglehub` may require Kaggle credentials to be configured locally.
- The script prints the resolved dataset path so you can verify the download.

## 5) What the script prints

`main.py` prints intermediate results to make the pipeline transparent:

- Dataset path and top-level directory structure
- Number of `.wav` files per group (`F_Con`, `F_Dys`, `M_Con`, `M_Dys`)
- Dataset-level sanity checks:
  - speaker ID overlap across groups
  - microphone type distribution inferred from directory names
- Random audio quality scan:
  - duration statistics
  - counts of failed loads, near-silent clips, extreme durations, and clipping
  - example file paths flagged for inspection
- Model outputs:
  - feature matrix shape
  - accuracy, ROC-AUC, and classification report
  - LOSO per-speaker accuracy and summary statistics

## 6) Results

Speaker-independent evaluation is substantially harder than file-level evaluation:

- File-level baselines can exceed 90% accuracy.
- Speaker-level splits typically fall in the ~55–65% accuracy range.
- LOSO results show high variance across speakers.

These results reflect the limited number of speakers and the difficulty of dysarthria detection under speaker-independent conditions.

## 7) Limitations and future work

- Small number of speakers in TORGO leads to high variance.
- Residual confounds (microphone, session, noise) may still affect performance.

Future directions include:
- Group-aware cross-validation (e.g. `GroupKFold`)
- Stronger or learned representations
- Cross-dataset evaluation (TORGO + UASpeech)

## 8) Repository structure

```text
.
├── main.py    # End-to-end pipeline: download → checks → features → SVM → evaluation
└── README.md
```