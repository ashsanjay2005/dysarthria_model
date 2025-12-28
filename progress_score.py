#!/usr/bin/env python3
"""Progress score layer for dysarthria tracking.

This script sits on top of a trained sklearn model (typically an SVM decision-function scorer)
and turns raw model scores into a more usable longitudinal progress signal.

Key ideas:
- Quality gating: reject clips that are likely to be silence, clipped, or very noisy.
- User baseline: express scores relative to a per-user baseline (z-score style).
- Smoothing: apply an exponential moving average (EMA) to reduce session-to-session jitter.

This is not a clinical tool. It is intended for technical experimentation and trend tracking.
"""

# -----------------------------------------------------------------------------
# HOW TO USE THIS SCRIPT
# -----------------------------------------------------------------------------
# This file wraps a trained sklearn model (usually an SVM) and produces a
# longitudinal “trend” signal per user.
#
# It supports two modes:
#
# 1) Baseline build mode (collect a personal reference)
#    Use this first. The script will accept or reject clips based on audio
#    quality. Accepted clips update the baseline mean and std for that user.
#
#    Example (build baseline from many clips):
#      python3 progress_score.py \
#        --model models/multi_svm_domain_robust.joblib \
#        --user my_user \
#        --baseline-mode build \
#        --trim-silence \
#        --peak-norm \
#        --inputs /path/to/clip1.wav /path/to/clip2.wav ...
#
#    You need a minimum number of accepted clips before tracking works.
#    Default is 20 (see --baseline-min).
#
# 2) Track mode (score new clips over time)
#    After the baseline is built, run track mode on each new recording.
#    The script will produce:
#      - raw_score: model decision score
#      - z_score: baseline-relative score
#      - z_score_clamped: clamped to [-z-clip, +z-clip] before smoothing
#      - ema_z: smoothed trend over time
#
#    Example (track one new clip):
#      python3 progress_score.py \
#        --model models/multi_svm_domain_robust.joblib \
#        --user my_user \
#        --baseline-mode track \
#        --trim-silence \
#        --peak-norm \
#        --inputs /path/to/new_clip.wav
#
# Outputs:
#   - progress_state.json: saved per-user baseline + EMA state (this changes)
#   - progress_scores.jsonl: appended log of every processed clip
#
# Tip: not every accepted clip should update the EMA.
# EMA is only updated when “ema_gate_ok” is true (stricter quality thresholds).
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import librosa
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "librosa is required for progress_score.py. Install with: pip install librosa"
    ) from e


# ----------------------------
# Audio utilities
# ----------------------------

def load_audio(
    path: str,
    *,
    sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    if y is None or len(y) == 0:
        raise ValueError("empty audio")
    return y.astype(np.float32), sr_out


def peak_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak < eps:
        return y
    return (y / peak).astype(np.float32)


def trim_silence(y: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    # librosa.effects.trim uses an energy threshold; it is fast and reliable for leading/trailing silence.
    y2, _ = librosa.effects.trim(y, top_db=top_db)
    return y2.astype(np.float32) if y2 is not None and len(y2) > 0 else y


def quality_metrics(y: np.ndarray, sr: int) -> Dict[str, float]:
    # Basic, fast metrics. These are not perfect, but they are cheap and useful.
    eps = 1e-8
    rms = float(np.sqrt(np.mean(np.square(y)) + eps))

    # Fraction of samples that are near full-scale (clipping proxy).
    clip_frac = float(np.mean(np.abs(y) >= 0.98))

    # Speech activity proxy: fraction of frames above a small energy threshold.
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    if frame_len <= 0 or hop <= 0 or len(y) < frame_len:
        speech_frac = 0.0
    else:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=0) + eps)
        thr = max(1e-4, 0.25 * np.median(frame_rms))
        speech_frac = float(np.mean(frame_rms > thr))

    # A rough "noise" proxy: low speech_frac with non-trivial rms.
    noise_proxy = float((1.0 - speech_frac) * rms)

    return {
        "rms": rms,
        "clip_frac": clip_frac,
        "speech_frac": speech_frac,
        "noise_proxy": noise_proxy,
        "duration_s": float(len(y) / max(sr, 1)),
    }



@dataclass
class GateConfig:
    min_duration_s: float = 0.8
    min_rms: float = 0.008
    max_clip_frac: float = 0.01
    min_speech_frac: float = 0.15


def passes_gate(m: Dict[str, float], g: GateConfig) -> Tuple[bool, List[str]]:
    """Primary accept/reject gate for clips.

    Returns (ok, reasons). Reasons is empty when ok is True.
    """

    reasons: List[str] = []
    if m["duration_s"] < g.min_duration_s:
        reasons.append(f"too_short<{g.min_duration_s}s")
    if m["rms"] < g.min_rms:
        reasons.append(f"too_quiet(rms<{g.min_rms})")
    if m["clip_frac"] > g.max_clip_frac:
        reasons.append(f"clipping(>{g.max_clip_frac})")
    if m["speech_frac"] < g.min_speech_frac:
        reasons.append(f"low_speech_frac<{g.min_speech_frac}")

    return (len(reasons) == 0), reasons


@dataclass
class EmaQualityConfig:
    """Stricter thresholds for when a clip is allowed to move the EMA trend."""

    min_duration_s: float = 1.0
    min_rms: float = 0.01
    max_clip_frac: float = 0.01
    min_speech_frac: float = 0.30
    max_noise_proxy: float = 0.06


def ema_quality_ok(m: Dict[str, float], q: EmaQualityConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if m["duration_s"] < q.min_duration_s:
        reasons.append(f"ema_too_short<{q.min_duration_s}s")
    if m["rms"] < q.min_rms:
        reasons.append(f"ema_too_quiet(rms<{q.min_rms})")
    if m["clip_frac"] > q.max_clip_frac:
        reasons.append(f"ema_clipping(>{q.max_clip_frac})")
    if m["speech_frac"] < q.min_speech_frac:
        reasons.append(f"ema_low_speech_frac<{q.min_speech_frac}")
    if m["noise_proxy"] > q.max_noise_proxy:
        reasons.append(f"ema_too_noisy(>{q.max_noise_proxy})")
    return (len(reasons) == 0), reasons


# ----------------------------
# Feature extraction (MFCC + deltas summarized)
# ----------------------------

def mfcc_summary_78(y: np.ndarray, sr: int) -> np.ndarray:
    # 13 MFCC + 13 delta + 13 delta2, then mean/std across time => 78 dims.
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, d1, d2])  # (39, T)
    mu = np.mean(feats, axis=1)
    sd = np.std(feats, axis=1)
    x = np.concatenate([mu, sd], axis=0).astype(np.float32)  # (78,)
    if x.shape[0] != 78:
        raise ValueError(f"unexpected feature dim: {x.shape}")
    return x


# ----------------------------
# Baseline and smoothing
# ----------------------------


def ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None or math.isnan(prev):
        return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)


@dataclass
class UserState:
    # Baseline for z-scoring
    baseline_n: int = 0
    baseline_mean: float = 0.0
    baseline_m2: float = 0.0  # for running variance

    # Smoothed progress
    ema_value: Optional[float] = None


def user_state_std(s: UserState) -> float:
    if s.baseline_n < 2:
        return 0.0
    var = s.baseline_m2 / (s.baseline_n - 1)
    return float(math.sqrt(max(var, 0.0)))


def user_state_update_baseline(s: UserState, x: float) -> None:
    # Welford update
    s.baseline_n += 1
    delta = x - s.baseline_mean
    s.baseline_mean += delta / s.baseline_n
    delta2 = x - s.baseline_mean
    s.baseline_m2 += delta * delta2


def z_score(x: float, mean: float, std: float, eps: float = 1e-6) -> float:
    return float((x - mean) / (std + eps))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ----------------------------
# Model scoring
# ----------------------------


# Compatibility wrapper for joblib loading of PerDomainScaledSVM models
class PerDomainScaledSVM:
    """Compatibility wrapper for joblib loading.

    NOTE: This class exists so that joblib can unpickle a model saved from a script
    where PerDomainScaledSVM was defined in the __main__ module.

    It supports optional per-domain scaling at inference time. If no domains are
    provided, it falls back to the global scaler.
    """

    def __init__(
        self,
        svm: Any,
        scalers: Dict[str, StandardScaler],
        global_scaler: StandardScaler,
        feature_dim: int,
    ):
        self.svm = svm
        self.scalers = scalers
        self.global_scaler = global_scaler
        self.feature_dim = feature_dim

    def _scale(self, X: np.ndarray, domains: Optional[np.ndarray]):
        if domains is None:
            return self.global_scaler.transform(X)

        X_out = np.zeros_like(X)
        used_any = False
        for dom in sorted(set(domains)):
            mask = domains == dom
            sc = self.scalers.get(dom)
            if sc is None:
                X_out[mask] = self.global_scaler.transform(X[mask])
            else:
                X_out[mask] = sc.transform(X[mask])
                used_any = True

        if not used_any:
            X_out = self.global_scaler.transform(X)

        return X_out

    def decision_function(self, X: np.ndarray, domains: Optional[np.ndarray] = None):
        Xs = self._scale(X, domains)
        return self.svm.decision_function(Xs)

    def predict(self, X: np.ndarray, domains: Optional[np.ndarray] = None):
        Xs = self._scale(X, domains)
        return self.svm.predict(Xs)


def load_model(model_path: str):
    model = joblib.load(model_path)
    return model


def decision_score(model: Any, X: np.ndarray) -> np.ndarray:
    # Support common sklearn patterns.
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        # Fallback: logit(prob) as a pseudo-score
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    raise TypeError("Model must expose decision_function or predict_proba")


# ----------------------------
# Persistence
# ----------------------------


def load_state(path: Path) -> Dict[str, UserState]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: Dict[str, UserState] = {}
    for user_id, d in data.items():
        out[user_id] = UserState(
            baseline_n=int(d.get("baseline_n", 0)),
            baseline_mean=float(d.get("baseline_mean", 0.0)),
            baseline_m2=float(d.get("baseline_m2", 0.0)),
            ema_value=d.get("ema_value", None),
        )
    return out


def save_state(path: Path, state: Dict[str, UserState]) -> None:
    serial = {k: asdict(v) for k, v in state.items()}
    path.write_text(json.dumps(serial, indent=2, sort_keys=True))


# ----------------------------
# Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute a longitudinal progress score from a trained model.")

    p.add_argument("--model", required=True, help="Path to a joblib-saved sklearn model or pipeline.")
    p.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help="Audio file paths (one or more). You can pass a glob by quoting it in your shell.",
    )

    p.add_argument("--user", required=True, help="User identifier for baseline tracking.")
    p.add_argument(
        "--state",
        default="progress_state.json",
        help="Path to JSON file storing per-user baseline and EMA state.",
    )

    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--trim-silence", action="store_true")
    p.add_argument("--peak-norm", action="store_true")

    p.add_argument(
        "--baseline-mode",
        choices=["build", "track"],
        default="track",
        help=(
            "build: accepted clips update the baseline only (no progress output). "
            "track: accepted clips produce z-score and EMA using the existing baseline."
        ),
    )

    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor. Higher reacts faster, lower is smoother.",
    )

    p.add_argument(
        "--baseline-min",
        type=int,
        default=20,
        help="Minimum accepted baseline clips required before producing progress z-scores.",
    )

    p.add_argument(
        "--z-clip",
        type=float,
        default=3.0,
        help="Clamp z-scores to [-z-clip, +z-clip] before updating EMA.",
    )

    # EMA update quality thresholds (stricter than the gate)
    p.add_argument("--ema-min-duration", type=float, default=1.0)
    p.add_argument("--ema-min-rms", type=float, default=0.01)
    p.add_argument("--ema-max-clip-frac", type=float, default=0.01)
    p.add_argument("--ema-min-speech-frac", type=float, default=0.30)
    p.add_argument("--ema-max-noise-proxy", type=float, default=0.06)

    # Quality gate thresholds
    p.add_argument("--min-duration", type=float, default=0.8)
    p.add_argument("--min-rms", type=float, default=0.008)
    p.add_argument("--max-clip-frac", type=float, default=0.01)
    p.add_argument("--min-speech-frac", type=float, default=0.15)

    p.add_argument(
        "--out",
        default="progress_scores.jsonl",
        help="Write one JSON record per input clip to this file.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(args.model)

    state_path = Path(args.state)
    state = load_state(state_path)
    user_state = state.get(args.user, UserState())

    gate = GateConfig(
        min_duration_s=float(args.min_duration),
        min_rms=float(args.min_rms),
        max_clip_frac=float(args.max_clip_frac),
        min_speech_frac=float(args.min_speech_frac),
    )

    ema_q = EmaQualityConfig(
        min_duration_s=float(args.ema_min_duration),
        min_rms=float(args.ema_min_rms),
        max_clip_frac=float(args.ema_max_clip_frac),
        min_speech_frac=float(args.ema_min_speech_frac),
        max_noise_proxy=float(args.ema_max_noise_proxy),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for inp in args.inputs:
        path = str(inp)
        rec: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "user": args.user,
            "path": path,
            "baseline_mode": args.baseline_mode,
        }

        try:
            y, sr = load_audio(path, sr=args.sr)
            if args.trim_silence:
                y = trim_silence(y, sr)
            if args.peak_norm:
                y = peak_normalize(y)

            m = quality_metrics(y, sr)
            ok, reasons = passes_gate(m, gate)
            # First gate: decides whether this clip is usable at all.
            # Second gate (EMA gate): decides whether this clip is high-quality
            # enough to update the EMA trend (ema_value).
            rec.update(m)
            rec["gate_ok"] = ok
            rec["gate_reasons"] = reasons

            if not ok:
                rec["status"] = "rejected"
                records.append(rec)
                continue

            x = mfcc_summary_78(y, sr)
            X = x.reshape(1, -1)
            raw = float(decision_score(model, X)[0])
            rec["raw_score"] = raw

            # Baseline build or track
            if args.baseline_mode == "build":
                user_state_update_baseline(user_state, raw)
                rec["status"] = "baseline_updated"
                rec["baseline_n"] = user_state.baseline_n
                rec["baseline_min"] = int(args.baseline_min)
                rec["baseline_mean"] = user_state.baseline_mean
                rec["baseline_std"] = user_state_std(user_state)
                records.append(rec)
                continue

            # track
            mean = user_state.baseline_mean
            std = user_state_std(user_state)
            if user_state.baseline_n < int(args.baseline_min):
                # Not enough baseline to make z-scores reliable
                rec["status"] = "need_more_baseline"
                rec["baseline_n"] = user_state.baseline_n
                rec["baseline_min"] = int(args.baseline_min)
                rec["baseline_mean"] = mean
                rec["baseline_std"] = std
                records.append(rec)
                continue
            # Convert raw model score into a baseline-relative value (z-score).
            # Then clamp to avoid a single outlier dominating the EMA.
            z = z_score(raw, mean, std)
            zc = clamp(z, -float(args.z_clip), float(args.z_clip))

            # Only allow high-quality clips to move the EMA trend.
            ema_ok, ema_reasons = ema_quality_ok(m, ema_q)
            rec["ema_gate_ok"] = ema_ok
            rec["ema_gate_reasons"] = ema_reasons
            rec["z_score"] = z
            rec["z_score_clamped"] = zc

            if ema_ok:
                user_state.ema_value = ema(user_state.ema_value, zc, float(args.ema_alpha))
                rec["ema_updated"] = True
            else:
                rec["ema_updated"] = False

            rec["status"] = "ok"
            rec["baseline_n"] = user_state.baseline_n
            rec["baseline_mean"] = mean
            rec["baseline_std"] = std
            rec["ema_z"] = user_state.ema_value

            records.append(rec)

        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)
            records.append(rec)

    # Persist updated user state so future runs continue the same baseline and EMA.
    # This is why you should not delete progress_state.json if you want continuity.
    state[args.user] = user_state
    save_state(state_path, state)

    # Write outputs
    with out_path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Console summary
    accepted = sum(1 for r in records if r.get("status") in {"ok", "baseline_updated"})
    rejected = sum(1 for r in records if r.get("status") == "rejected")
    errors = sum(1 for r in records if r.get("status") == "error")

    print("=== Progress score run ===")
    print(f"inputs: {len(records)} | accepted: {accepted} | rejected: {rejected} | errors: {errors}")
    print(f"state: {state_path}")
    print(f"out:   {out_path}")


if __name__ == "__main__":
    main()