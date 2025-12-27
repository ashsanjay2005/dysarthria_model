import kagglehub
import librosa
import numpy as np
import os
import random
import re
import argparse
from collections import defaultdict
from typing import Callable

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# TORGO dysarthria baseline:
# - Downloads the dataset via kagglehub
# - Runs lightweight dataset checks (speaker overlap, mic distribution, random quality scan)
# - Extracts MFCC summary features per utterance
# - Trains and evaluates an SVM classifier as an initial baseline
# - Safely load a wav file at a fixed sample rate.
# - Returns (None, None) if loading fails or the file is empty.


# --- Audio normalization options ---
def normalize_audio(y: np.ndarray, sr: int, peak_norm: bool, trim_silence: bool):
    """Optional per-utterance normalization.

    - peak_norm: scales waveform so max(|y|) == 1 (with small epsilon guard)
    - trim_silence: removes leading/trailing low-energy regions using librosa.effects.trim
    """
    if y is None or len(y) == 0:
        return y

    if trim_silence:
        # top_db is a pragmatic default; expose via CLI if you want finer control later
        y, _ = librosa.effects.trim(y, top_db=30)

    if peak_norm:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / (peak + 1e-9)

    return y


def safe_load_wav(fp: str, *, peak_norm: bool = False, trim_silence: bool = False):
    """Safely load a wav file at a fixed sample rate.

    Returns (None, None) if loading fails or the file is empty.
    """
    try:
        y, sr = librosa.load(fp, sr=16000, mono=True)
        if y is None or sr is None:
            return None, None
        if len(y) == 0:
            return None, None

        y = normalize_audio(y, sr, peak_norm=peak_norm, trim_silence=trim_silence)
        if y is None or len(y) == 0:
            return None, None

        return y, sr
    except Exception:
        return None, None

# Extract MFCC-based features per utterance and summarize over time using mean and standard deviation.
# We include MFCC, delta, and delta-delta coefficients.
# Final feature dimension: 78 per utterance.
def extract_mfcc_features(
    fp: str,
    n_mfcc: int = 13,
    min_len_samples: int = 2048,
    *,
    peak_norm: bool,
    trim_silence: bool,
):
    y, sr = safe_load_wav(fp, peak_norm=peak_norm, trim_silence=trim_silence)
    if y is None or sr is None:
        return None

    # Very short clips can fail MFCC and delta computations.
    # Skip them rather than silently producing unstable features.
    if len(y) < min_len_samples:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Delta features require enough time frames. Use a smaller window on short signals.
    n_frames = mfcc.shape[1]
    width = min(9, n_frames)
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return None

    delta = librosa.feature.delta(mfcc, width=width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")

    feats = np.vstack([mfcc, delta, delta2])

    feat_mean = np.mean(feats, axis=1)
    feat_std = np.std(feats, axis=1)
    return np.concatenate([feat_mean, feat_std])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TORGO dysarthria baseline: checks, feature extraction, and SVM evaluation."
    )

    p.add_argument(
        "--dataset",
        default="pranaykoppula/torgo-audio",
        help="Kaggle dataset slug for kagglehub (default: pranaykoppula/torgo-audio).",
    )

    # Sections (opt-in). If none are provided, we default to speaker-split training only.
    p.add_argument("--list", action="store_true", help="Print dataset path and top-level structure.")
    p.add_argument("--checks", action="store_true", help="Run dataset-level confound checks (speaker/mic).")
    p.add_argument("--quality", action="store_true", help="Run random audio quality scan per group.")
    p.add_argument("--train", action="store_true", help="Train/evaluate SVM with a speaker-level split.")
    p.add_argument("--file-cv", action="store_true", help="Run file-level CV (optimistic baseline).")
    p.add_argument("--loso", action="store_true", help="Run leave-one-speaker-out evaluation.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Run list + checks + quality + train + file-cv + loso.",
    )

    # Support for UASpeech and domain-aware options
    p.add_argument(
        "--uaspeech",
        action="store_true",
        help="Include noise-reduced UASpeech dataset (aryashah2k/noise-reduced-uaspeech-dysarthria-dataset).",
    )
    p.add_argument(
        "--lodo",
        action="store_true",
        help="Run leave-one-dataset-out evaluation (train on TORGO, test on UASpeech, and vice versa if available).",
    )

    # Quality-scan knobs
    p.add_argument("--sample-per-group", type=int, default=50, help="Files per group for the quality scan.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffles/sampling.")

    # Speaker split knob
    p.add_argument("--test-speaker-frac", type=float, default=0.2, help="Fraction of speakers held out for test.")

    # Audio normalization options
    p.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading/trailing silence before feature extraction.",
    )
    p.add_argument(
        "--peak-norm",
        action="store_true",
        help="Peak-normalize each utterance before feature extraction.",
    )
    p.add_argument(
        "--per-dataset-scale",
        action="store_true",
        help="Standardize features within each dataset domain before merging (helps reduce dataset identity cues).",
    )
    p.add_argument(
        "--balance-domains",
        action="store_true",
        help="Balance training samples across dataset domains (TORGO vs UASpeech) to reduce dataset bias.",
    )

    # Output options for per-utterance scores
    p.add_argument(
        "--export-scores",
        action="store_true",
        help="Export per-utterance scores to CSV (includes speaker/domain/label/pred/score).",
    )
    p.add_argument(
        "--scores-out",
        type=str,
        default="scores.csv",
        help="Output CSV path for --export-scores.",
    )

    return p


def collect_group_files(dataset_path: str) -> dict[str, list[str]]:
    group_to_files: dict[str, list[str]] = {}
    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        group_path = os.path.join(dataset_path, group)
        files: list[str] = []
        for root, _, fs in os.walk(group_path):
            for f in fs:
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(root, f))
        group_to_files[group] = files
    return group_to_files


# --- UASpeech support ---
def collect_uaspeech_files(dataset_path: str) -> list[str]:
    """Recursively collect .wav files for the UASpeech Kaggle mirror.

    The Kaggle dataset layout can vary across mirrors. We keep this permissive
    and infer labels later from the path.
    """
    wavs: list[str] = []
    for root, _, fs in os.walk(dataset_path):
        for f in fs:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return wavs


def infer_uaspeech_label(fp: str) -> int | None:
    """Infer control/dysarthric label from the path.

    Returns:
    - 0 for control/healthy/normal
    - 1 for dysarthric/dysarthria
    - None if it can't be inferred
    """
    p = fp.lower()
    # Common directory tokens
    if any(tok in p for tok in ["control", "healthy", "normal"]):
        return 0
    if any(tok in p for tok in ["dys", "dysarth"]):
        return 1
    return None


def infer_uaspeech_speaker_id(fp: str) -> str | None:
    """Best-effort speaker id extraction for UASpeech.

    Many UASpeech file names contain a speaker token like M01/F02.
    We look for that pattern anywhere in the path, otherwise fall back
    to the first filename token.
    """
    m = re.search(r"\b([FM]\d{2})\b", fp)
    if m:
        return m.group(1)

    base = os.path.basename(fp)
    stem = os.path.splitext(base)[0]
    token = re.split(r"[_\-]", stem)[0]
    return token if token else None


def run_list(dataset_path: str):
    print("Dataset path:", dataset_path)
    items = sorted(os.listdir(dataset_path))
    print(f"Top-level items ({len(items)}):")
    for item in items:
        print(" -", item)

    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        group_path = os.path.join(dataset_path, group)
        wav_count = 0
        for root, _, fs in os.walk(group_path):
            for f in fs:
                if f.lower().endswith(".wav"):
                    wav_count += 1
        print(group, "wav files:", wav_count)


def run_checks(dataset_path: str, group_to_files: dict[str, list[str]]):
    # Dataset-level metadata checks (no audio decoding).
    # Used to verify speaker separation and inspect microphone distributions.
    print("\n=== Confound checks (dataset-level) ===")

    speaker_re = re.compile(r"wav_(?:headMic|arrayMic)_(?P<code>[FM]C?\d{2})S\d{2}")

    def parse_speaker_id_from_path(fp: str) -> str | None:
        parts = fp.split(os.sep)
        for p in parts:
            m = speaker_re.match(p)
            if m:
                return m.group("code")
        return None

    def parse_mic_from_path(fp: str) -> str | None:
        if "wav_headMic_" in fp:
            return "headMic"
        if "wav_arrayMic_" in fp:
            return "arrayMic"
        return None

    group_to_speakers: dict[str, set[str]] = {}
    group_to_mics: dict[str, dict[str, int]] = {}

    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        speakers = set()
        mic_counts = {"headMic": 0, "arrayMic": 0, "unknown": 0}

        for fp in group_to_files.get(group, []):
            spk = parse_speaker_id_from_path(fp)
            if spk is not None:
                speakers.add(spk)

            mic = parse_mic_from_path(fp)
            if mic is None:
                mic_counts["unknown"] += 1
            else:
                mic_counts[mic] += 1

        group_to_speakers[group] = speakers
        group_to_mics[group] = mic_counts

        print(f"{group}: speakers {len(speakers)} | mic counts {mic_counts}")

    all_groups = ["F_Con", "F_Dys", "M_Con", "M_Dys"]
    speaker_to_groups: dict[str, set[str]] = {}
    for g in all_groups:
        for spk in group_to_speakers[g]:
            speaker_to_groups.setdefault(spk, set()).add(g)

    multi_group = {spk: gs for spk, gs in speaker_to_groups.items() if len(gs) > 1}

    if not multi_group:
        print("OK: No speaker IDs appear in more than one group (no obvious label mixing).")
    else:
        print("WARNING: Some speaker IDs appear in multiple groups (possible label mixing):")
        for spk, gs in list(sorted(multi_group.items()))[:20]:
            print(" -", spk, "->", sorted(gs))

    f_overlap = group_to_speakers["F_Con"].intersection(group_to_speakers["F_Dys"])
    m_overlap = group_to_speakers["M_Con"].intersection(group_to_speakers["M_Dys"])
    print(f"Female Con∩Dys overlap: {len(f_overlap)}")
    if f_overlap:
        print("  examples:", sorted(list(f_overlap))[:10])
    print(f"Male Con∩Dys overlap: {len(m_overlap)}")
    if m_overlap:
        print("  examples:", sorted(list(m_overlap))[:10])

    return parse_speaker_id_from_path


def run_quality_scan(group_to_files: dict[str, list[str]], sample_per_group: int, seed: int):
    RANDOM_SEED = seed
    SAMPLE_PER_GROUP = sample_per_group
    MIN_DUR_S = 0.5
    MAX_DUR_S = 20.0
    SILENCE_RMS = 1e-3
    CLIP_FRAC = 0.001

    random.seed(RANDOM_SEED)

    print("\n=== Data quality scan (random sample) ===")

    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        wavs = group_to_files.get(group, [])
        if not wavs:
            print(f"{group}: no wav files found")
            continue

        sample = random.sample(wavs, k=min(SAMPLE_PER_GROUP, len(wavs)))

        failed_load = 0
        too_short = 0
        too_long = 0
        near_silent = 0
        clipped = 0

        durations = []
        srs = []

        flagged_examples = []

        for fp in sample:
            y_, sr_ = safe_load_wav(fp)
            if y_ is None:
                failed_load += 1
                flagged_examples.append(("failed_load", fp))
                continue

            dur = len(y_) / float(sr_)
            rms = float(np.sqrt(np.mean(y_ * y_)))
            clip_fraction = float(np.mean(np.abs(y_) >= 0.999))

            durations.append(dur)
            srs.append(sr_)

            reasons = []
            if dur < MIN_DUR_S:
                too_short += 1
                reasons.append("too_short")
            if dur > MAX_DUR_S:
                too_long += 1
                reasons.append("too_long")
            if rms < SILENCE_RMS:
                near_silent += 1
                reasons.append("near_silent")
            if clip_fraction > CLIP_FRAC:
                clipped += 1
                reasons.append("clipped")

            if reasons:
                flagged_examples.append(("+".join(reasons), fp))

        if durations:
            d_min = float(np.min(durations))
            d_med = float(np.median(durations))
            d_max = float(np.max(durations))
        else:
            d_min = d_med = d_max = float("nan")

        sr_unique = sorted(set(srs))

        print(
            f"{group}: sampled {len(sample)} | failed {failed_load} | short {too_short} | long {too_long} | "
            f"silent {near_silent} | clipped {clipped} | dur(min/med/max) {d_min:.2f}/{d_med:.2f}/{d_max:.2f} | sr {sr_unique}"
        )

        if flagged_examples:
            print("  flagged examples:")
            for reason, fp in flagged_examples[:5]:
                print("   -", reason, "::", fp)


def build_feature_table(
    group_to_files: dict[str, list[str]],
    parse_speaker_id_from_path: Callable[[str], str | None],
    *,
    include_uaspeech: bool,
    trim_silence: bool,
    peak_norm: bool,
):
    """Build X/y plus group labels for speaker- and dataset-aware evaluation."""

    # TORGO: group directory implies label
    label_map = {"F_Con": 0, "M_Con": 0, "F_Dys": 1, "M_Dys": 1}

    X_list, y_list, spk_list, dom_list = [], [], [], []

    # TORGO domain
    for group, files in group_to_files.items():
        label = label_map[group]
        for fp in files:
            spk = parse_speaker_id_from_path(fp)
            feats = extract_mfcc_features(fp, peak_norm=peak_norm, trim_silence=trim_silence)
            if feats is None or spk is None:
                continue
            X_list.append(feats)
            y_list.append(label)
            spk_list.append(f"TORGO::{spk}")
            dom_list.append("TORGO")

    # Optional UASpeech domain
    if include_uaspeech:
        ua_path = kagglehub.dataset_download("aryashah2k/noise-reduced-uaspeech-dysarthria-dataset")
        ua_files = collect_uaspeech_files(ua_path)

        skipped_no_label = 0
        for fp in ua_files:
            label = infer_uaspeech_label(fp)
            if label is None:
                skipped_no_label += 1
                continue

            spk = infer_uaspeech_speaker_id(fp)
            feats = extract_mfcc_features(fp, peak_norm=peak_norm, trim_silence=trim_silence)
            if feats is None or spk is None:
                continue

            X_list.append(feats)
            y_list.append(label)
            spk_list.append(f"UASPEECH::{spk}")
            dom_list.append("UASPEECH")

        if skipped_no_label > 0:
            print(f"[UASpeech] Skipped {skipped_no_label} files with no label inferred from path.")

    X = np.array(X_list)
    y = np.array(y_list)
    speakers = np.array(spk_list)
    domains = np.array(dom_list)
    return X, y, speakers, domains


# --- Domain-aware scaling and balancing helpers ---
def fit_transform_per_domain(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dom_train: np.ndarray,
    dom_test: np.ndarray,
):
    """Scale features separately within each dataset domain.

    This reduces easy "dataset identity" cues when mixing corpora.
    """
    scalers: dict[str, StandardScaler] = {}

    X_train_out = np.zeros_like(X_train)
    X_test_out = np.zeros_like(X_test)

    for dom in sorted(set(dom_train)):
        tr_mask = dom_train == dom
        te_mask = dom_test == dom
        sc = StandardScaler()
        X_train_out[tr_mask] = sc.fit_transform(X_train[tr_mask])
        if np.any(te_mask):
            X_test_out[te_mask] = sc.transform(X_test[te_mask])
        scalers[dom] = sc

    # If the test set contains a domain not present in training, fall back to global scaling.
    unseen = [d for d in set(dom_test) if d not in scalers]
    if unseen:
        sc = StandardScaler()
        X_train_out = sc.fit_transform(X_train)
        X_test_out = sc.transform(X_test)

    return X_train_out, X_test_out


def balance_domains_indices(domains: np.ndarray, seed: int) -> np.ndarray:
    """Return a subset of indices that equalizes sample counts across domains."""
    rng = random.Random(seed)
    dom_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(domains):
        dom_to_idx[d].append(i)

    if len(dom_to_idx) <= 1:
        return np.arange(len(domains))

    min_n = min(len(v) for v in dom_to_idx.values())
    keep: list[int] = []
    for d, idxs in dom_to_idx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        keep.extend(idxs[:min_n])

    keep = sorted(keep)
    return np.array(keep, dtype=int)


# --- Helper: Export per-utterance scores to CSV ---
def export_scores_csv(
    out_path: str,
    *,
    speakers: np.ndarray,
    domains: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
):
    """Write per-utterance predictions/scores to a CSV for downstream analysis."""
    import csv

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "domain", "y_true", "y_pred", "score"])
        for spk, dom, yt, yp, sc in zip(speakers, domains, y_true, y_pred, y_score):
            w.writerow([spk, dom, int(yt), int(yp), float(sc)])

    print(f"Wrote scores CSV: {out_path}")




def run_speaker_split_train(
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    domains: np.ndarray,
    *,
    seed: int,
    test_speaker_frac: float,
    per_dataset_scale: bool,
    balance_domains: bool,
    export_scores: bool,
    scores_out: str,
):
    print("\n=== SVM Training (MFCC features) ===")
    print("\n=== Speaker-level split ===")

    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speakers):
        speaker_to_indices[spk].append(i)

    unique_speakers = list(speaker_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_speakers)

    split_idx = int((1.0 - test_speaker_frac) * len(unique_speakers))
    train_speakers = set(unique_speakers[:split_idx])
    test_speakers = set(unique_speakers[split_idx:])

    train_idx = [i for spk in train_speakers for i in speaker_to_indices[spk]]
    test_idx = [i for spk in test_speakers for i in speaker_to_indices[spk]]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    dom_train, dom_test = domains[train_idx], domains[test_idx]

    print("Feature matrix shape:", X.shape)
    print(f"Train speakers: {len(train_speakers)} | Test speakers: {len(test_speakers)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    if balance_domains:
        keep = balance_domains_indices(dom_train, seed=seed)
        X_train, y_train, dom_train = X_train[keep], y_train[keep], dom_train[keep]

    if per_dataset_scale:
        X_train, X_test = fit_transform_per_domain(X_train, X_test, dom_train, dom_test)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_scores = svm.decision_function(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_scores))
    print(classification_report(y_test, y_pred))

    if export_scores:
        export_scores_csv(
            scores_out,
            speakers=speakers[test_idx],
            domains=domains[test_idx],
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_scores,
        )


def run_file_level_cv(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    *,
    seed: int,
    per_dataset_scale: bool,
    balance_domains: bool,
):
    print("\n=== File-level CV (StratifiedKFold) ===")
    print("Note: This CV is file-level, so it may still include speaker leakage.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    cv_acc = []
    cv_auc = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        dom_tr, dom_te = domains[tr_idx], domains[te_idx]

        if balance_domains:
            keep = balance_domains_indices(dom_tr, seed=seed)
            X_tr, y_tr, dom_tr = X_tr[keep], y_tr[keep], dom_tr[keep]

        if per_dataset_scale:
            X_tr, X_te = fit_transform_per_domain(X_tr, X_te, dom_tr, dom_te)
        else:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)


        m = SVC(kernel="rbf", C=1.0, gamma="scale")
        m.fit(X_tr, y_tr)

        p = m.predict(X_te)
        s = m.decision_function(X_te)

        cv_acc.append(accuracy_score(y_te, p))
        cv_auc.append(roc_auc_score(y_te, s))

    print(f"CV Accuracy: {np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
    print(f"CV ROC-AUC: {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}")


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    domains: np.ndarray,
    *,
    seed: int,
    per_dataset_scale: bool,
    balance_domains: bool,
):
    # Leave-One-Speaker-Out (LOSO) evaluation.
    # Each run holds out one speaker entirely and trains on all remaining speakers.
    print("\n=== LOSO Evaluation (Speaker-Independent) ===")

    loso_accs = []
    loso_aucs = []

    for test_spk in sorted(set(speakers)):
        tr_idx = [i for i, spk in enumerate(speakers) if spk != test_spk]
        te_idx = [i for i, spk in enumerate(speakers) if spk == test_spk]

        if not te_idx:
            continue

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        dom_tr, dom_te = domains[tr_idx], domains[te_idx]

        if balance_domains:
            keep = balance_domains_indices(dom_tr, seed=seed)
            X_tr, y_tr, dom_tr = X_tr[keep], y_tr[keep], dom_tr[keep]

        if per_dataset_scale:
            X_tr, X_te = fit_transform_per_domain(X_tr, X_te, dom_tr, dom_te)
        else:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)


        m = SVC(kernel="rbf", C=1.0, gamma="scale")
        m.fit(X_tr, y_tr)

        p = m.predict(X_te)
        s = m.decision_function(X_te)

        acc = accuracy_score(y_te, p)
        loso_accs.append(acc)

        auc = None
        if len(set(y_te)) == 2:
            auc = roc_auc_score(y_te, s)
            loso_aucs.append(auc)

        if auc is None:
            print(f"Speaker {test_spk}: samples={len(te_idx)} | acc={acc:.3f} | auc=NA")
        else:
            print(f"Speaker {test_spk}: samples={len(te_idx)} | acc={acc:.3f} | auc={auc:.3f}")

    print(f"LOSO mean accuracy: {np.mean(loso_accs):.3f} ± {np.std(loso_accs):.3f}")
    if loso_aucs:
        print(f"LOSO mean ROC-AUC: {np.mean(loso_aucs):.3f} ± {np.std(loso_aucs):.3f}")


def run_lodo(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    *,
    per_dataset_scale: bool,
    balance_domains: bool,
    seed: int,
):
    print("\n=== Leave-One-Dataset-Out (LODO) Evaluation ===")

    doms = sorted(set(domains))
    if len(doms) < 2:
        print("Only one dataset domain present; LODO requires at least two.")
        return

    for test_dom in doms:
        tr_idx = np.where(domains != test_dom)[0]
        te_idx = np.where(domains == test_dom)[0]

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        dom_tr, dom_te = domains[tr_idx], domains[te_idx]

        if balance_domains:
            keep = balance_domains_indices(dom_tr, seed=seed)
            X_tr, y_tr, dom_tr = X_tr[keep], y_tr[keep], dom_tr[keep]

        if per_dataset_scale:
            X_tr, X_te = fit_transform_per_domain(X_tr, X_te, dom_tr, dom_te)
        else:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)


        m = SVC(kernel="rbf", C=1.0, gamma="scale")
        m.fit(X_tr, y_tr)

        p = m.predict(X_te)
        s = m.decision_function(X_te)

        acc = accuracy_score(y_te, p)
        auc = roc_auc_score(y_te, s) if len(set(y_te)) == 2 else None

        if auc is None:
            print(f"Test domain {test_dom}: samples={len(te_idx)} | acc={acc:.3f} | auc=NA")
        else:
            print(f"Test domain {test_dom}: samples={len(te_idx)} | acc={acc:.3f} | auc={auc:.3f}")


def main():
    args = build_arg_parser().parse_args()

    # If the user didn't request any section, default to training only.
    requested_any = any(
        [
            args.list,
            args.checks,
            args.quality,
            args.train,
            args.file_cv,
            args.loso,
            args.all,
        ]
    )
    if not requested_any:
        args.train = True

    if args.all:
        args.list = True
        args.checks = True
        args.quality = True
        args.train = True
        args.file_cv = True
        args.loso = True


    dataset_path = kagglehub.dataset_download(args.dataset)

    group_to_files = collect_group_files(dataset_path)

    if args.list:
        run_list(dataset_path)

    # We need speaker parsing for training/LOSO. If checks are off, reuse the same parser logic.
    if args.checks:
        parse_speaker_id_from_path = run_checks(dataset_path, group_to_files)
    else:
        speaker_re = re.compile(r"wav_(?:headMic|arrayMic)_(?P<code>[FM]C?\d{2})S\d{2}")

        def parse_speaker_id_from_path(fp: str) -> str | None:
            parts = fp.split(os.sep)
            for p in parts:
                m = speaker_re.match(p)
                if m:
                    return m.group("code")
            return None

    if args.quality:
        run_quality_scan(group_to_files, sample_per_group=args.sample_per_group, seed=args.seed)

    needs_features = args.train or args.file_cv or args.loso or args.lodo
    if needs_features:
        X, y, speakers, domains = build_feature_table(
            group_to_files,
            parse_speaker_id_from_path,
            include_uaspeech=args.uaspeech,
            trim_silence=args.trim_silence,
            peak_norm=args.peak_norm,
        )

    if args.train:
        run_speaker_split_train(
            X,
            y,
            speakers,
            domains,
            seed=args.seed,
            test_speaker_frac=args.test_speaker_frac,
            per_dataset_scale=args.per_dataset_scale,
            balance_domains=args.balance_domains,
            export_scores=args.export_scores,
            scores_out=args.scores_out,
        )

    if args.file_cv:
        run_file_level_cv(
            X,
            y,
            domains,
            seed=args.seed,
            per_dataset_scale=args.per_dataset_scale,
            balance_domains=args.balance_domains,
        )

    if args.loso:
        run_loso(
            X,
            y,
            speakers,
            domains,
            seed=args.seed,
            per_dataset_scale=args.per_dataset_scale,
            balance_domains=args.balance_domains,
        )

    if getattr(args, "lodo", False):
        run_lodo(
            X,
            y,
            domains,
            per_dataset_scale=args.per_dataset_scale,
            balance_domains=args.balance_domains,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()