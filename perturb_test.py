#!/usr/bin/env python3
"""
Perturbation stability test for dysarthria score outputs.

Trains three SVM models using the same MFCC(+delta+delta2) summary feature pipeline:
  1) Single-domain model: TORGO only
  2) Single-domain model: UASPEECH only
  3) Multi-domain model: UASPEECH + TORGO

Then evaluates score stability on a held-out test domain (default: TORGO) under small,
in-memory audio perturbations. This is intended to probe score robustness for
longitudinal tracking use cases, not diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import zlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Set, Dict, List, Tuple

import kagglehub
import librosa
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TORGO_SLUG = "pranaykoppula/torgo-audio"
UASPEECH_SLUG = "aryashah2k/noise-reduced-uaspeech-dysarthria-dataset"
TORGO_GROUPS = ("F_Con", "F_Dys", "M_Con", "M_Dys")

# In-process cache to avoid repeated KaggleHub version/network checks within one run.
_DATASET_PATH_CACHE: dict[str, str] = {}

def _dataset_download_cached(slug: str, cache_key: str, *, retries: int = 3, sleep_s: float = 2.0) -> str:
    """Download (or resolve) a KaggleHub dataset path once per process.

    KaggleHub may perform a network version check even when data is already cached locally.
    This helper memoizes the resolved path and retries transient network failures.
    """
    if cache_key in _DATASET_PATH_CACHE:
        return _DATASET_PATH_CACHE[cache_key]

    last_err: Exception | None = None
    for i in range(max(1, int(retries))):
        try:
            path = kagglehub.dataset_download(slug)
            _DATASET_PATH_CACHE[cache_key] = path
            return path
        except Exception as e:
            last_err = e
            # Brief backoff for transient HTTPS timeouts.
            if i < retries - 1:
                time.sleep(float(sleep_s) * (i + 1))

    assert last_err is not None
    raise last_err

# Speaker-code heuristics for TORGO
# Controls are typically FCxx / MCxx, dysarthric are typically Fxx / Mxx.
# We still compute speaker labels from data, but these help with sanity prints.

def _split_csv_list(s: Optional[str]) -> Set[str]:
    if not s:
        return set()
    return {tok.strip() for tok in s.split(",") if tok.strip()}

def pick_holdout_speakers(
    all_files: List['LabeledFile'],
    *,
    seed: int,
    explicit: Optional[Set[str]] = None,
    n_per_class: int = 2,
) -> Set[str]:
    """Pick held-out speakers for the test domain.

    If `explicit` is provided and non-empty, use it.
    Otherwise, sample `n_per_class` speakers per class (control=0, dys=1), based on
    per-speaker majority label in `all_files`.

    This prevents leakage (held-out speakers are removed from training) and avoids
    degenerate test sets with only one class.
    """
    if explicit:
        return set(explicit)

    # speaker -> list of labels
    spk_to_labels: Dict[str, List[int]] = {}
    for lf in all_files:
        spk_to_labels.setdefault(lf.speaker, []).append(int(lf.y_true))

    # Determine each speaker's label by majority vote.
    spk_to_label: Dict[str, int] = {}
    for spk, ys in spk_to_labels.items():
        ones = int(sum(ys))
        zeros = len(ys) - ones
        spk_to_label[spk] = 1 if ones > zeros else 0

    controls = [s for s, lab in spk_to_label.items() if lab == 0]
    dys = [s for s, lab in spk_to_label.items() if lab == 1]

    rng = random.Random(seed)
    rng.shuffle(controls)
    rng.shuffle(dys)

    n_c = min(n_per_class, len(controls))
    n_d = min(n_per_class, len(dys))
    holdout = set(controls[:n_c] + dys[:n_d])

    # If one side is empty, fall back to just sampling overall speakers.
    if not holdout:
        all_spk = list(spk_to_label.keys())
        rng.shuffle(all_spk)
        holdout = set(all_spk[: min(4, len(all_spk))])

    return holdout

def filter_by_speakers(files: List['LabeledFile'], speakers: Set[str], keep: bool) -> List['LabeledFile']:
    """If keep=True, keep only files whose speaker is in `speakers`. If keep=False, drop them."""
    if not speakers:
        return files
    if keep:
        return [f for f in files if f.speaker in speakers]
    return [f for f in files if f.speaker not in speakers]

def speaker_class_balance(files: List['LabeledFile']) -> Tuple[int, int, int]:
    """Return (n_files, n_control, n_dys)."""
    n = len(files)
    n_d = int(sum(int(f.y_true) for f in files))
    n_c = n - n_d
    return n, n_c, n_d


@dataclass(frozen=True)
class LabeledFile:
    fp: str
    speaker: str
    domain: str
    y_true: int


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train single-domain (UASPEECH) and multi-domain (UASPEECH+TORGO) SVMs, "
            "then measure score stability on a test domain under small audio perturbations."
        )
    )
    p.add_argument(
        "--single",
        type=str,
        default="uaspeech",
        choices=["uaspeech"],
        help="Training domain for the single-domain model (fixed to UASPEECH).",
    )
    p.add_argument(
        "--multi",
        type=str,
        default="both",
        choices=["both"],
        help="Training domains for the multi-domain model (fixed to UASPEECH + TORGO).",
    )
    p.add_argument(
        "--test-domain",
        type=str,
        default="TORGO",
        choices=["TORGO", "UASPEECH"],
        help="Domain to evaluate stability on (default: TORGO).",
    )
    p.add_argument(
        "--n-files",
        type=int,
        default=500,
        help="Maximum number of test-domain files to evaluate (random sample).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling/perturbations.")

    p.add_argument(
        "--holdout-speakers",
        type=str,
        default=None,
        help=(
            "Comma-separated speaker IDs to hold out for testing within the chosen test domain. "
            "If omitted, the script will sample holdout speakers stratified by class."
        ),
    )
    p.add_argument(
        "--holdout-n-per-class",
        type=int,
        default=2,
        help=(
            "If --holdout-speakers is not provided, sample this many speakers per class "
            "(control and dysarthric) from the test domain for holdout testing."
        ),
    )

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
        "--time-shift",
        action="store_true",
        help="Include a small +/- 80 sample time-shift perturbation.",
    )

    p.add_argument(
        "--train-limit-uaspeech",
        type=int,
        default=None,
        help="Optional cap on number of UASPEECH training files (random sample).",
    )
    p.add_argument(
        "--train-limit-torgo",
        type=int,
        default=None,
        help="Optional cap on number of TORGO training files for the multi-domain model.",
    )

    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("out/perturb_results.csv"),
        help="Output CSV path for per-utterance perturbation results.",
    )
    return p


# --- Audio IO and features (mirrors the baseline pipeline in main.py) ---
def normalize_audio(y: np.ndarray, sr: int, peak_norm: bool, trim_silence: bool) -> np.ndarray:
    if y is None or len(y) == 0:
        return y
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
    if peak_norm:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / (peak + 1e-9)
    return y


def safe_load_wav(fp: str, *, peak_norm: bool, trim_silence: bool) -> tuple[np.ndarray | None, int | None]:
    try:
        y, sr = librosa.load(fp, sr=16000, mono=True)
        if y is None or sr is None or len(y) == 0:
            return None, None
        y = normalize_audio(y, sr, peak_norm=peak_norm, trim_silence=trim_silence)
        if y is None or len(y) == 0:
            return None, None
        return y.astype(np.float32, copy=False), int(sr)
    except Exception:
        return None, None


def extract_features_from_audio(
    y: np.ndarray, sr: int, *, n_mfcc: int = 13, min_len_samples: int = 2048
) -> np.ndarray | None:
    """
    13 MFCC + delta + delta-delta, summarized by mean/std -> 78-D.
    Matches the feature logic used in `main.py`.
    """
    if y is None or len(y) < min_len_samples:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames)
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return None

    delta = librosa.feature.delta(mfcc, width=width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")
    feats = np.vstack([mfcc, delta, delta2])  # (39, T)
    feat_mean = np.mean(feats, axis=1)
    feat_std = np.std(feats, axis=1)
    out = np.concatenate([feat_mean, feat_std])  # (78,)
    if out.shape[0] != 78:
        return None
    return out.astype(np.float32, copy=False)


def extract_features_from_path(
    fp: str, *, peak_norm: bool, trim_silence: bool
) -> np.ndarray | None:
    y, sr = safe_load_wav(fp, peak_norm=peak_norm, trim_silence=trim_silence)
    if y is None or sr is None:
        return None
    return extract_features_from_audio(y, sr)


# --- Dataset collection and label/speaker inference (consistent with main.py) ---
def collect_torgo_files() -> dict[str, list[str]]:
    dataset_path = _dataset_download_cached(TORGO_SLUG, "TORGO")
    group_to_files: dict[str, list[str]] = {}
    for group in TORGO_GROUPS:
        group_path = os.path.join(dataset_path, group)
        files: list[str] = []
        for root, _, fs in os.walk(group_path):
            for f in fs:
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(root, f))
        group_to_files[group] = files
    return group_to_files


def collect_uaspeech_files() -> list[str]:
    dataset_path = _dataset_download_cached(UASPEECH_SLUG, "UASPEECH")
    wavs: list[str] = []
    for root, _, fs in os.walk(dataset_path):
        for f in fs:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return wavs


def infer_uaspeech_label(fp: str) -> int | None:
    p = fp.lower()
    if any(tok in p for tok in ["control", "healthy", "normal"]):
        return 0
    if any(tok in p for tok in ["dys", "dysarth"]):
        return 1
    return None


def infer_uaspeech_speaker(fp: str) -> str | None:
    m = re.search(r"\b([FM]\d{2})\b", fp)
    if m:
        return m.group(1)
    base = os.path.basename(fp)
    stem = os.path.splitext(base)[0]
    token = re.split(r"[_\-]", stem)[0]
    return token if token else None


def infer_torgo_speaker(fp: str) -> str | None:
    speaker_re = re.compile(r"wav_(?:headMic|arrayMic)_(?P<code>[FM]C?\d{2})S\d{2}")
    for part in fp.split(os.sep):
        m = speaker_re.match(part)
        if m:
            return m.group("code")
    return None


def iter_labeled_files(domain: str) -> Iterator[LabeledFile]:
    if domain == "TORGO":
        group_to_files = collect_torgo_files()
        label_map = {"F_Con": 0, "M_Con": 0, "F_Dys": 1, "M_Dys": 1}
        for group, files in group_to_files.items():
            y_true = label_map[group]
            for fp in files:
                spk = infer_torgo_speaker(fp)
                if spk is None:
                    continue
                yield LabeledFile(fp=fp, speaker=spk, domain="TORGO", y_true=y_true)
        return

    if domain == "UASPEECH":
        for fp in collect_uaspeech_files():
            y_true = infer_uaspeech_label(fp)
            spk = infer_uaspeech_speaker(fp)
            if y_true is None or spk is None:
                continue
            yield LabeledFile(fp=fp, speaker=spk, domain="UASPEECH", y_true=int(y_true))
        return

    raise ValueError(f"Unknown domain: {domain}")


def sample_files(files: list[LabeledFile], n: int | None, seed: int) -> list[LabeledFile]:
    if n is None or n <= 0 or n >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, k=n)


def build_training_table(
    domains: list[str],
    *,
    exclude_speakers_by_domain: Optional[Dict[str, Set[str]]] = None,
    peak_norm: bool,
    trim_silence: bool,
    seed: int,
    train_limit_uaspeech: int | None,
    train_limit_torgo: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    labeled: list[LabeledFile] = []
    for dom in domains:
        dom_files = list(iter_labeled_files(dom))
        # Leakage guard: exclude held-out speakers for this domain (if configured)
        if exclude_speakers_by_domain and dom in exclude_speakers_by_domain:
            hold = exclude_speakers_by_domain[dom]
            before = len(dom_files)
            dom_files = filter_by_speakers(dom_files, hold, keep=False)
            removed = before - len(dom_files)
            if removed > 0:
                print(f"  [Leakage Guard] Excluded {removed} {dom} files from held-out speakers: {sorted(list(hold))}")
        if dom == "UASPEECH":
            dom_files = sample_files(dom_files, train_limit_uaspeech, seed)
        if dom == "TORGO":
            dom_files = sample_files(dom_files, train_limit_torgo, seed)
        labeled.extend(dom_files)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for lf in labeled:
        feats = extract_features_from_path(lf.fp, peak_norm=peak_norm, trim_silence=trim_silence)
        if feats is None:
            continue
        X_list.append(feats)
        y_list.append(int(lf.y_true))

    if not X_list:
        raise RuntimeError(f"No features extracted for training domains: {domains}")

    X = np.vstack(X_list).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=int)
    return X, y


def train_svm(X: np.ndarray, y: np.ndarray, *, seed: int) -> Pipeline:
    # class_weight='balanced' is important here to reduce degenerate predictions when mixing domains.
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=seed,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", clf)])
    pipe.fit(X, y)
    return pipe


# --- Perturbations ---
def clamp_audio(y: np.ndarray) -> np.ndarray:
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


def make_perturbations(include_time_shift: bool) -> list[tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]]:
    perts: list[tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]] = []

    def gain(factor: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
        def _f(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            _ = rng  # unused, kept for a uniform signature
            return clamp_audio(y * factor)

        return _f

    def noise(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
        def _f(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            eps = rng.normal(loc=0.0, scale=sigma, size=y.shape).astype(np.float32, copy=False)
            return clamp_audio(y + eps)

        return _f

    perts.append((f"gain_{0.7:g}", gain(0.7)))
    perts.append((f"gain_{1.3:g}", gain(1.3)))
    perts.append((f"noise_{0.002:g}", noise(0.002)))
    perts.append((f"noise_{0.005:g}", noise(0.005)))

    if include_time_shift:
        def shift(samples: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
            def _f(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
                _ = rng  # unused
                return clamp_audio(np.roll(y, samples))

            return _f

        perts.append(("shift_-80", shift(-80)))
        perts.append(("shift_+80", shift(80)))

    return perts


@dataclass(frozen=True)
class ResultRow:
    model_name: str
    domain: str
    speaker: str
    file: str
    y_true: int
    perturb: str
    score_clean: float
    score_pert: float
    delta: float


def decision_score(model: Pipeline, feats: np.ndarray) -> float:
    # SVC decision_function returns shape (n_samples,)
    s = model.decision_function(feats.reshape(1, -1))
    return float(np.asarray(s).ravel()[0])


def stable_perturb_seed(fp: str, perturb_name: str, seed: int) -> int:
    # Stable 32-bit seed derived from file path and perturbation name.
    h_fp = zlib.crc32(fp.encode("utf-8")) & 0xFFFFFFFF
    h_p = zlib.crc32(perturb_name.encode("utf-8")) & 0xFFFFFFFF
    return (seed ^ h_fp ^ h_p) & 0xFFFFFFFF


def evaluate_stability(
    *,
    models: dict[str, Pipeline],
    test_files: list[LabeledFile],
    perturbations: list[tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]],
    peak_norm: bool,
    trim_silence: bool,
    seed: int,
) -> tuple[list[ResultRow], dict[str, list[float]]]:
    rows: list[ResultRow] = []
    clean_scores: dict[str, list[float]] = {name: [] for name in models.keys()}

    for lf in test_files:
        y, sr = safe_load_wav(lf.fp, peak_norm=peak_norm, trim_silence=trim_silence)
        if y is None or sr is None:
            continue

        feats_clean = extract_features_from_audio(y, sr)
        if feats_clean is None:
            continue

        scores_clean_by_model = {name: decision_score(m, feats_clean) for name, m in models.items()}
        for name, sc in scores_clean_by_model.items():
            clean_scores[name].append(float(sc))

        for pert_name, fn in perturbations:
            # Independent RNG per (file, perturbation) for reproducibility and independence.
            pert_rng = np.random.default_rng(stable_perturb_seed(lf.fp, pert_name, seed))
            y_pert = fn(y, pert_rng)
            feats_pert = extract_features_from_audio(y_pert, sr)
            if feats_pert is None:
                continue
            for model_name, model in models.items():
                sc = scores_clean_by_model[model_name]
                sp = decision_score(model, feats_pert)
                d = abs(sc - sp)
                rows.append(
                    ResultRow(
                        model_name=model_name,
                        domain=lf.domain,
                        speaker=lf.speaker,
                        file=lf.fp,
                        y_true=int(lf.y_true),
                        perturb=pert_name,
                        score_clean=float(sc),
                        score_pert=float(sp),
                        delta=float(d),
                    )
                )

    return rows, clean_scores


def summarize_results(rows: list[ResultRow]) -> str:
    if not rows:
        return "No results to summarize (empty rows)."

    # Mean/median delta across utterances, per model and perturbation
    by_model: dict[str, list[ResultRow]] = {}
    for r in rows:
        by_model.setdefault(r.model_name, []).append(r)

    lines: list[str] = []
    for model_name in sorted(by_model.keys()):
        model_rows = by_model[model_name]
        lines.append(f"\n[{model_name}]")

        # Utterance-level deltas
        pert_to_deltas: dict[str, list[float]] = {}
        for r in model_rows:
            pert_to_deltas.setdefault(r.perturb, []).append(r.delta)

        lines.append("Utterance-level |delta| by perturbation:")
        for pert in sorted(pert_to_deltas.keys()):
            deltas = np.asarray(pert_to_deltas[pert], dtype=float)
            lines.append(
                f"  - {pert}: n={deltas.size} mean={float(np.mean(deltas)):.4f} median={float(np.median(deltas)):.4f}"
            )

        # Speaker-level means, then overall aggregation across speakers
        lines.append("Speaker-level mean(|delta|) aggregated across speakers:")
        for pert in sorted(pert_to_deltas.keys()):
            spk_to_vals: dict[str, list[float]] = {}
            for r in model_rows:
                if r.perturb != pert:
                    continue
                spk_to_vals.setdefault(r.speaker, []).append(r.delta)

            spk_means = np.asarray([np.mean(v) for v in spk_to_vals.values()], dtype=float)
            if spk_means.size == 0:
                continue
            lines.append(
                f"  - {pert}: speakers={spk_means.size} mean={float(np.mean(spk_means)):.4f} median={float(np.median(spk_means)):.4f}"
            )

    return "\n".join(lines).lstrip()


def write_results_csv(out_path: Path, rows: list[ResultRow]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_name",
                "domain",
                "speaker",
                "file",
                "y_true",
                "perturb",
                "score_clean",
                "score_pert",
                "delta",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.model_name,
                    r.domain,
                    r.speaker,
                    r.file,
                    r.y_true,
                    r.perturb,
                    f"{r.score_clean:.8f}",
                    f"{r.score_pert:.8f}",
                    f"{r.delta:.8f}",
                ]
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    # Training
    print("=== Training models ===")
    print(f"Audio options: trim_silence={bool(args.trim_silence)} peak_norm={bool(args.peak_norm)}")

    print("\n=== Preparing speaker holdout ===")
    test_all = list(iter_labeled_files(args.test_domain))
    if not test_all:
        raise RuntimeError(f"No files found for test domain: {args.test_domain}")

    explicit_holdout = _split_csv_list(args.holdout_speakers)
    holdout_speakers = pick_holdout_speakers(
        test_all,
        seed=args.seed,
        explicit=explicit_holdout if explicit_holdout else None,
        n_per_class=int(args.holdout_n_per_class),
    )

    test_pool = filter_by_speakers(test_all, holdout_speakers, keep=True)
    n_total, n_c, n_d = speaker_class_balance(test_pool)
    print(f"Test domain: {args.test_domain}")
    print(f"Holdout speakers: {sorted(list(holdout_speakers))}")
    print(f"Holdout test pool: files={n_total} | control={n_c} dysarthric={n_d}")

    if n_total == 0:
        raise RuntimeError("Holdout selection produced an empty test pool. Provide --holdout-speakers explicitly.")

    # --- Single-domain model: UASPEECH only ---
    exclude_single_uaspeech: Dict[str, Set[str]] = {}
    if args.test_domain == "UASPEECH":
        exclude_single_uaspeech["UASPEECH"] = holdout_speakers

    X_uas_only, y_uas_only = build_training_table(
        ["UASPEECH"],
        exclude_speakers_by_domain=exclude_single_uaspeech if exclude_single_uaspeech else None,
        peak_norm=args.peak_norm,
        trim_silence=args.trim_silence,
        seed=args.seed,
        train_limit_uaspeech=args.train_limit_uaspeech,
        train_limit_torgo=None,
    )
    print(f"Single-domain training (UASPEECH): X={X_uas_only.shape} y_pos={int(y_uas_only.sum())}")
    uaspeech_only_model = train_svm(X_uas_only, y_uas_only, seed=args.seed)

    # --- Single-domain model: TORGO only ---
    exclude_single_torgo: Dict[str, Set[str]] = {}
    if args.test_domain == "TORGO":
        exclude_single_torgo["TORGO"] = holdout_speakers

    X_torgo_only, y_torgo_only = build_training_table(
        ["TORGO"],
        exclude_speakers_by_domain=exclude_single_torgo if exclude_single_torgo else None,
        peak_norm=args.peak_norm,
        trim_silence=args.trim_silence,
        seed=args.seed,
        train_limit_uaspeech=None,
        train_limit_torgo=args.train_limit_torgo,
    )
    print(f"Single-domain training (TORGO): X={X_torgo_only.shape} y_pos={int(y_torgo_only.sum())}")
    torgo_only_model = train_svm(X_torgo_only, y_torgo_only, seed=args.seed)

    exclude_multi: Dict[str, Set[str]] = {}
    if args.test_domain == "TORGO":
        exclude_multi["TORGO"] = holdout_speakers
    if args.test_domain == "UASPEECH":
        exclude_multi["UASPEECH"] = holdout_speakers

    X_multi, y_multi = build_training_table(
        ["UASPEECH", "TORGO"],
        exclude_speakers_by_domain=exclude_multi if exclude_multi else None,
        peak_norm=args.peak_norm,
        trim_silence=args.trim_silence,
        seed=args.seed,
        train_limit_uaspeech=args.train_limit_uaspeech,
        train_limit_torgo=args.train_limit_torgo,
    )
    print(f"Multi-domain training (UASPEECH+TORGO): X={X_multi.shape} y_pos={int(y_multi.sum())}")
    multi_model = train_svm(X_multi, y_multi, seed=args.seed)

    print("\n=== Collecting test files (speaker-held-out) ===")
    rng = random.Random(args.seed)
    if args.n_files > 0 and args.n_files < len(test_pool):
        test_sample = rng.sample(test_pool, k=int(args.n_files))
    else:
        test_sample = list(test_pool)

    speakers_in_sample = sorted({f.speaker for f in test_sample})
    n_s, n_c_s, n_d_s = speaker_class_balance(test_sample)
    print(f"Sampled test files: files={n_s} | control={n_c_s} dysarthric={n_d_s}")
    print(f"Speakers in sample ({len(speakers_in_sample)}): {speakers_in_sample}")

    perturbations = make_perturbations(include_time_shift=bool(args.time_shift))
    print(f"Perturbations: {', '.join([name for name, _ in perturbations])}")

    # Evaluation
    print("\n=== Evaluating stability ===")
    rows, clean_scores = evaluate_stability(
        models={
            "single_torgo": torgo_only_model,
            "single_uaspeech": uaspeech_only_model,
            "multi_uaspeech+torgo": multi_model,
        },
        test_files=test_sample,
        perturbations=perturbations,
        peak_norm=args.peak_norm,
        trim_silence=args.trim_silence,
        seed=args.seed,
    )
    write_results_csv(args.out_csv, rows)

    # Clean-score dispersion is a secondary statistic; it is meaningful only when enough files succeeded.
    def score_std(xs: list[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        return float(np.std(np.asarray(xs, dtype=float), ddof=1))

    processed: dict[str, int] = {}
    for r in rows:
        processed.setdefault(r.file, int(r.y_true))
    n_processed = len(processed)
    n_dys = int(sum(processed.values()))
    n_con = n_processed - n_dys

    print("\n=== Summary ===")
    print(f"Results rows: {len(rows)} (written to {args.out_csv})")
    print(f"Processed files: {n_processed} | control={n_con} dysarthric={n_dys}")
    print(
        "Clean score std | "
        f"single_torgo: {score_std(clean_scores.get('single_torgo', [])):.4f} | "
        f"single_uaspeech: {score_std(clean_scores.get('single_uaspeech', [])):.4f} | "
        f"multi: {score_std(clean_scores.get('multi_uaspeech+torgo', [])):.4f}"
    )
    print()
    print(summarize_results(rows))


if __name__ == "__main__":
    main()
