# Dysarthria Detection, Robustness Stress Tests, and Progress Tracking (TORGO + UASpeech)

This repo is a practical, reproducible baseline for:

- **Dysarthria detection** from short speech clips
- **Robustness evaluation** using perturbation stress tests (noise, gain, shift)
- **Progress tracking** using a personal baseline and a smoothed score over time

The core idea is simple: accuracy alone is not enough if you want to use a model in the real world, especially at home. This project therefore treats **score stability** as a first class metric.

This is **not a clinical tool**. It is an engineering exploration intended for learning, portfolio demonstration, and experimentation.

---

## What this project does

### 1) A baseline dysarthria model

- **Task:** binary classification (control vs dysarthric)
- **Model:** RBF kernel SVM
- **Features:** MFCC + delta + delta delta summary stats
- **Output:** an SVM decision score (a continuous severity like signal)

Why SVM scores?

- You get a continuous value that can be used like a meter.
- For tracking, this is often more useful than a saturated probability that sits near 0 or 1.

### 2) Robustness stress testing

This repo includes experiments that answer:

> If I add small realistic noise or change volume slightly, does the model’s score stay consistent?

We run the same utterances through controlled perturbations and measure:

- **Absolute score change:** `|Δ score| = |score_clean − score_perturbed|`

Lower is better.

### 3) Progress tracking via `progress_score.py`

This repo also includes a lightweight progress tracker that:

- builds a **personal baseline** from a user’s accepted clips
- converts raw model scores into **z scores** relative to that baseline
- applies an **EMA (exponential moving average)** to reduce day to day jitter

This gives you a stable trend line, as long as recording conditions and tasks are reasonably consistent.

---

## Datasets and labels

### TORGO

- Kaggle: `pranaykoppula/torgo-audio`
- Label mapping:
  - `F_Con`, `M_Con` → control (label `0`)
  - `F_Dys`, `M_Dys` → dysarthric (label `1`)
- Speaker and mic info is inferred from directory names.

### UASpeech (noise reduced Kaggle version)

- Kaggle: `aryashah2k/noise-reduced-uaspeech-dysarthria-dataset`
- Label mapping inferred from folder tokens:
  - control like folders → label `0`
  - dysarthria like folders → label `1`
- Speaker IDs inferred from standardized speaker codes.

---

## Feature extraction

For each utterance:

1. Load audio at **16 kHz**, mono
2. Compute **13 MFCCs**
3. Compute **delta** and **delta delta**
4. Summarize each coefficient by **mean** and **std** over time

Final feature vector:

- **78 dimensions per utterance**
- `(13 MFCC + 13 delta + 13 delta-delta) × (mean + std) = 39 × 2`

Optional preprocessing flags used across experiments:

- `--trim-silence`: trims leading and trailing silence
- `--peak-norm`: peak normalizes each clip

Very short or invalid clips are skipped because deltas become unstable.

---

## Model implementation

All SVM models use this pipeline:

- `StandardScaler`
- `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")`

Training regimes:

1. `single_torgo` (TORGO only)
2. `single_uaspeech` (UASpeech only)
3. `multi_uaspeech+torgo` (TORGO + UASpeech)

The multi domain model is intended to reduce domain brittleness (studio vs laptop mic differences).

---

## Progress tracking design

### Baseline concept

For tracking, you want to compare a user to themselves, not only to the training population.

`progress_score.py` therefore maintains per user state in `progress_state.json`:

- `baseline_n`: number of accepted baseline clips
- `baseline_mean`: mean raw score across baseline clips
- `baseline_m2`: running sum of squared deviations (for variance)
- `ema_value`: the user’s current EMA of z scores

Baseline building uses a running mean and variance update (Welford style).

### Z score

For a new clip:

- `z = (raw_score − baseline_mean) / baseline_std`

This standardizes the score relative to the user.

### Z score clamping

To prevent rare outliers from dominating the trend:

- `z_clamped = clamp(z, -3, +3)`

The EMA uses the clamped z score.

### EMA smoothing

EMA (exponential moving average) is a one line smoother:

- `ema_new = alpha * z_clamped + (1 - alpha) * ema_prev`

Interpretation:

- higher alpha reacts faster but is noisier
- lower alpha is steadier but slower to reflect change

### Only update EMA on good quality audio

The tracker distinguishes:

- `gate_ok`: basic acceptance checks for reporting
- `ema_gate_ok`: stricter checks that decide whether a clip is trustworthy enough to update EMA

This prevents short, clipped, or low quality clips from moving the trend line.

### Baseline size

The tracker targets a baseline of **20+ accepted clips** (not 8).

This reduces baseline instability and makes z scores less sensitive to a few unusual samples.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install kagglehub librosa numpy scikit-learn pandas matplotlib joblib
```

Kaggle credentials may be required for dataset download.

---

## Download datasets

```bash
python3 - <<'PY'
import kagglehub
print("TORGO:", kagglehub.dataset_download("pranaykoppula/torgo-audio"))
print("UASPEECH:", kagglehub.dataset_download("aryashah2k/noise-reduced-uaspeech-dysarthria-dataset"))
PY
```

Optional convenience variables:

```bash
TORGO_PATH="/Users/<you>/.cache/kagglehub/datasets/pranaykoppula/torgo-audio/versions/1"
UA_PATH="/Users/<you>/.cache/kagglehub/datasets/aryashah2k/noise-reduced-uaspeech-dysarthria-dataset/versions/1"
```

---

## How to use this repo

### A) Train and evaluate models

`main.py` is primarily for model building and evaluation.

```bash
python3 main.py
```

Use this when you are:

- training models
- running classification evaluation
- exporting trained models (for later use by `progress_score.py`)

### B) Run robustness stress tests

Example runs (10 seeds) with silence trimming, peak normalization, and optional time shift:

```bash
# TORGO evaluation
for s in {1..10}; do
  python3 perturb_test.py \
    --test-domain TORGO \
    --seed $s \
    --trim-silence \
    --peak-norm \
    --time-shift \
    --out-csv out_torgo_v3/perturb_seed${s}.csv
done

# UASPEECH evaluation
for s in {1..10}; do
  python3 perturb_test.py \
    --test-domain UASPEECH \
    --seed $s \
    --trim-silence \
    --peak-norm \
    --time-shift \
    --out-csv out_ua_v3/perturb_ua_seed${s}.csv
done

python3 aggregate_perturb_results.py
```

Outputs are written to `out_summary/`.

### C) Track progress for a user

`progress_score.py` is the user facing runtime script.

It writes:

- `progress_scores.jsonl` (append only event log)
- `progress_state.json` (per user baseline and EMA state)

#### 1) Build a baseline (target 20+ accepted clips)

Provide multiple clips from the same user recorded in similar conditions.

```bash
python3 progress_score.py \
  --model models/multi_svm_domain_robust.joblib \
  --user test_user \
  --baseline-mode build \
  --trim-silence \
  --peak-norm \
  --inputs \
  /path/to/clip1.wav /path/to/clip2.wav /path/to/clip3.wav
```

Tip for datasets:

```bash
python3 progress_score.py \
  --model models/multi_svm_domain_robust.joblib \
  --user test_user \
  --baseline-mode build \
  --trim-silence \
  --peak-norm \
  --inputs $(find "$UA_PATH/noisereduced-uaspeech-control" -type f -name "*.wav" | head -n 25)
```

#### 2) Track new clips

```bash
python3 progress_score.py \
  --model models/multi_svm_domain_robust.joblib \
  --user test_user \
  --baseline-mode track \
  --trim-silence \
  --peak-norm \
  --inputs /path/to/new_clip.wav
```

#### 3) Inspect the latest result

```bash
tail -n 1 progress_scores.jsonl
cat progress_state.json
```

---

## Interpreting outputs

Each entry in `progress_scores.jsonl` includes:

- `raw_score`: SVM decision score
- `z_score`: standardized relative to the user baseline
- `z_score_clamped`: the value used for EMA
- `ema_z`: the smoothed trend value
- `ema_updated`: whether this clip updated the EMA
- `gate_ok` and `ema_gate_ok`: acceptance flags with reasons

Practical guidance:

- Treat `ema_z` as the trend, not a single `raw_score`.
- Expect some jitter even with trimming and normalization.
- If lots of clips fail `ema_gate_ok`, raise your recording duration and speak more consistently.

---

## Limitations and Scope

This system is **not a clinical or diagnostic tool**. It is designed as a **longitudinal progress tracker for personal, non-clinical use**. The limitations below describe the system’s intended scope and operating boundaries.

### 1. Tracking vs. Diagnosis

**Intended use:**  
This tool is built to measure **relative change over time within the same individual** (for example, “Is my speech stability improving compared to last month?”).

**Out of scope:**  
It is not capable of absolute medical diagnosis (for example, “Do I have dysarthria?”). Due to domain shift between studio-trained data and home microphones, healthy users may receive an initial positive severity score. This reflects a known **offset bias**, not a medical condition. 

### 2. The Consistency Requirement

**Constraint:**  
The system’s demonstrated stability (Δ ≈ 0.26 under simulated noise) assumes reasonably consistent recording conditions.

**User responsibility:**  
For reliable tracking, users should:
- use the same microphone and device,
- maintain similar microphone distance,
- record in broadly similar environments.

Large changes in environment (for example, moving from a quiet room to a noisy café) can introduce variance that may temporarily mask subtle improvements. This constraint is typical of most longitudinal measurement systems used outside controlled laboratory settings.

### 3. Acoustic Signal vs. Human Perception

**What the model measures:**  
Changes in acoustic feature space, specifically MFCC-derived distance to an SVM decision boundary.

**Limitation:**  
While acoustic distance often correlates with perceived speech quality, it is not a direct measure of intelligibility or articulation accuracy as judged by human listeners. A user may sound clearer to a person without a strictly linear change in the model score, or vice versa, depending on which acoustic features are affected.

### 4. Sensitivity Floor

**Design tradeoff:**  
The multi-domain model intentionally prioritizes **stability over hypersensitivity**.

**Impact:**  
To reduce false alarms caused by background noise and recording artifacts, the model may be less sensitive to extremely subtle micro-improvements than studio-grade clinical systems. Instead, it is optimized to reveal **macro-level trends** over weeks or months rather than day-to-day fluctuations.

---

## Repository structure

├── main.py
│   # Entry point for model training and evaluation.
│   # Builds SVM models (single-domain and multi-domain),
│   # performs train/test splits with speaker holdout,
│   # and exports trained models as .joblib files.
│
├── progress_score.py
│   # Runtime scoring and progress tracking script.
│   # Loads a trained model, builds a per-user baseline,
│   # computes z-scores relative to that baseline,
│   # applies EMA smoothing, and logs results.
│
├── perturb_test.py
│   # Robustness stress-testing script.
│   # Applies controlled perturbations (noise, gain, time shift)
│   # to audio and measures score stability (|Δ score|).
│
├── aggregate_perturb_results.py
│   # Aggregates perturbation test outputs across seeds.
│   # Produces summary CSVs and figures used in analysis.
│
├── models/
│   # Saved trained model artifacts (.joblib).
│   # These are loaded by progress_score.py at runtime.
│
├── out_torgo_v3/
│   # Raw perturbation experiment outputs for TORGO.
│   # One CSV per seed/run.
│
├── out_ua_v3/
│   # Raw perturbation experiment outputs for UASpeech.
│   # One CSV per seed/run.
│
├── out_summary/
│   # Aggregated results and plots derived from perturbation tests.
│   # Used for reporting and comparison between models.
│
├── progress_scores.jsonl
│   # Append-only log of per-clip scoring events.
│   # Each line corresponds to one processed audio clip.
│
├── progress_state.json
│   # Persistent per-user state for progress tracking.
│   # Stores baseline statistics and current EMA value.
│
├── utterance_summary.csv
│   # Summary statistics per utterance used in analysis.
│   # Includes raw scores, perturbation deltas, and metadata.
│
├── all_rows.csv
│   # Full experimental log across all perturbation runs.
│   # Used to compute aggregate stability metrics.
│
└── README.md
    # Project documentation and usage guide.