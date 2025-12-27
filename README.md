# Dysarthria Detection and Stability Analysis (TORGO + UASpeech)

This repository implements a reproducible baseline pipeline for dysarthria detection using classical acoustic features and an RBF-kernel SVM. In addition to standard classification evaluation, it includes a **perturbation stability test** designed to assess whether model scores remain consistent under small recording variations, which is critical for longitudinal progress tracking.

The project is intentionally conservative in model choice and explicit about evaluation assumptions, limitations, and failure modes.

---

## 1) Overview

- **Task:** binary classification (control vs dysarthric speech)
- **Datasets:** TORGO and UASpeech
- **Input:** `.wav` audio files
- **Features:** MFCC + delta + delta-delta summary statistics (78-D per utterance)
- **Models:** SVM with RBF kernel
- **Extra evaluation:** perturbation-based score stability analysis

---

## 2) Datasets and labels

### TORGO

- Kaggle dataset: `pranaykoppula/torgo-audio`
- Directory groups:
  - `F_Con`, `M_Con` → control (label `0`)
  - `F_Dys`, `M_Dys` → dysarthric (label `1`)
- Speaker IDs and microphone information are parsed from directory names (for example `wav_headMic_FC02S03`).

### UASpeech

- Kaggle dataset: `aryashah2k/noise-reduced-uaspeech-dysarthria-dataset`
- Labels are inferred from folder and file path tokens:
  - control-like tokens → label `0`
  - dysarthria-like tokens → label `1`
- Speaker IDs are inferred from standardized speaker codes (for example `F02`, `M05`) with fallbacks to filename parsing when necessary.

---

## 3) Feature extraction

For each utterance:

- Audio is loaded at 16 kHz (mono)
- 13 MFCCs are computed
- First- and second-order deltas are computed
- Each coefficient is summarized using mean and standard deviation over time

Final feature vector:

- **78 dimensions per utterance**
  `(13 MFCC + 13 delta + 13 delta-delta) × (mean + std) = 39 × 2`

Very short or invalid clips are skipped to avoid unstable delta features.

Optional preprocessing used in stability tests:
- `--trim-silence`: trims leading and trailing silence
- `--peak-norm`: peak normalizes each clip

---

## 4) Models

All models use the same pipeline:

- `StandardScaler`
- `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")`

Three training regimes are supported:

1. **single_torgo** – trained on TORGO only
2. **single_uaspeech** – trained on UASpeech only
3. **multi_uaspeech+torgo** – trained on TORGO + UASpeech

---

## 5) Evaluation

### Classification evaluation

Standard evaluation in `main.py` supports:

- Speaker-level train/test splits (recommended)
- File-level cross-validation (for comparison only)
- Leave-one-speaker-out (LOSO) analysis

Speaker-level splits are enforced to avoid speaker leakage.

### Perturbation stability test

The perturbation test addresses a different question than accuracy:

**Does the model’s score remain stable when the same speech is recorded under slightly different conditions?**

This is particularly important if the model is used to track changes in speech over time rather than to make one-off diagnostic predictions.

#### Protocol

- Choose a test domain: `TORGO` or `UASPEECH`
- Hold out a small set of speakers from that domain
- Train the three models listed above
- Evaluate score stability on held-out utterances under small perturbations

#### Perturbations

- Gain: `gain_0.7`, `gain_1.3`
- Noise: `noise_0.002`, `noise_0.005`
- Optional time shift: `shift_-80`, `shift_+80`

Metric:

- `|Δ score| = |score_clean − score_perturbed|`

Lower values indicate more stable scores.

---

## 6) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install kagglehub librosa numpy scikit-learn pandas matplotlib
```

Kaggle credentials may be required for dataset download.

---

## 7) How to run

### Baseline training and evaluation

```bash
python3 main.py
```

### Perturbation stability experiments

Example runs using silence trimming, peak normalization, and time shift:

```bash
# TORGO evaluation (10 seeds)
for s in {1..10}; do
  python3 perturb_test.py \
    --test-domain TORGO \
    --seed $s \
    --trim-silence \
    --peak-norm \
    --time-shift \
    --out-csv out_torgo_v3/perturb_seed${s}.csv
done

# UASPEECH evaluation (10 seeds)
for s in {1..10}; do
  python3 perturb_test.py \
    --test-domain UASPEECH \
    --seed $s \
    --trim-silence \
    --peak-norm \
    --time-shift \
    --out-csv out_ua_v3/perturb_ua_seed${s}.csv
done
```

Aggregate and plot:

```bash
python3 aggregate_perturb_results.py
```

Outputs are written to `out_summary/`.

---

## 8) Interpreting stability results

- Additive noise is the dominant source of score instability
- Single-dataset models tend to be stable only on their own dataset
- Multi-dataset training improves cross-domain robustness at the cost of peak stability on any single dataset

For longitudinal tracking, the multi-dataset model is the most reasonable default, provided scores are interpreted relative to a personal baseline and smoothed over time.

---

## 9) Limitations and future work

- Raw SVM decision scores are not calibrated probabilities
- Noise robustness remains a limiting factor
- This system is not intended for clinical diagnosis

Future directions include learned representations, explicit calibration, domain adaptation, and user-level normalization for progress tracking.

---

## 10) Repository structure

```text
.
├── main.py                       # Baseline pipeline
├── perturb_test.py               # Perturbation stability tests
├── aggregate_perturb_results.py  # Aggregation and plotting
├── out_torgo_v3/                 # Example TORGO evaluation outputs
├── out_ua_v3/                    # Example UASPEECH evaluation outputs
└── out_summary/                  # Aggregated results and figures
```

---

This repository is intended as a technical exploration of dysarthria detection and robustness, not as a clinical tool.

---