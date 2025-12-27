#!/usr/bin/env python3
"""
Post-hoc analysis for exported per-utterance model scores.

This script does not train models or extract features. It only analyzes the CSV
produced by `main.py --export-scores`, which is expected to include:
  - speaker
  - domain
  - y_true (0 = control, 1 = dysarthric)
  - y_pred
  - score (continuous; higher = more dysarthria-like)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


REQUIRED_COLUMNS = ("speaker", "domain", "y_true", "y_pred", "score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze exported per-utterance model scores as a continuous scale "
            "(distributions, AUC, effect size, speaker trajectories, reliability, domain checks)."
        )
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "/Users/ashsanjay/Documents/PersonalProjects/dysarthria_model/out/scores_both.csv"
        ),
        help="Path to CSV exported by main.py --export-scores.",
    )
    parser.add_argument(
        "--min-utterances",
        type=int,
        default=30,
        help="Minimum utterances per speaker for test–retest reliability.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting halves.")
    parser.add_argument(
        "--top-speakers",
        type=int,
        default=8,
        help="Number of speakers (by utterance count) to plot trajectories for.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def load_scores_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["speaker"] = df["speaker"].astype(str)
    df["domain"] = df["domain"].astype(str)

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["y_true", "score", "speaker", "domain"]).copy()
    df["y_true"] = df["y_true"].astype(int)

    invalid_labels = sorted(set(df["y_true"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Expected y_true in {{0,1}}; found labels: {invalid_labels}")

    df["class"] = np.where(df["y_true"] == 1, "dysarthric", "control")
    return df


def roc_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Rank-based AUC equivalent to the Mann–Whitney U statistic.
    Uses average ranks for ties.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined with only one class present.")

    ranks = stats.rankdata(scores, method="average")
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    Cohen's d between two groups: (mean1 - mean0) / pooled_sd.
    Uses sample standard deviations (ddof=1).
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    if x0.size < 2 or x1.size < 2:
        return float("nan")

    m0 = float(np.mean(x0))
    m1 = float(np.mean(x1))
    s0 = float(np.std(x0, ddof=1))
    s1 = float(np.std(x1, ddof=1))

    n0 = x0.size
    n1 = x1.size
    pooled = np.sqrt(((n0 - 1) * s0 * s0 + (n1 - 1) * s1 * s1) / float(n0 + n1 - 2))
    if pooled == 0.0:
        return float("nan")
    return float((m1 - m0) / pooled)


def score_summary_by_class(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("class", observed=True)["score"]
        .agg(["count", "mean", "median", "std"])
        .sort_index()
    )


def ensure_figures_dir(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


def plot_score_distribution(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="score",
        hue="class",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
    )
    sns.kdeplot(data=df, x="score", hue="class", common_norm=False, linewidth=2)
    plt.title("Score distribution by class")
    plt.xlabel("SVM decision score (higher = more dysarthria-like)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def add_speaker_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def zscore_series(s: pd.Series) -> pd.Series:
        values = s.to_numpy(dtype=float)
        std = float(np.std(values, ddof=0))
        if std == 0.0:
            return pd.Series(np.zeros_like(values), index=s.index)
        return (s - float(np.mean(values))) / std

    df["speaker_z"] = df.groupby("speaker", observed=True)["score"].transform(zscore_series)
    df["utterance_index"] = df.groupby("speaker", observed=True).cumcount() + 1
    return df


def plot_speaker_trajectories(df: pd.DataFrame, top_speakers: int, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    counts = df["speaker"].value_counts()
    selected = counts.head(top_speakers).index.tolist()
    sub = df[df["speaker"].isin(selected)].copy()
    if sub.empty:
        raise ValueError("No speakers available for trajectory plotting.")

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=sub,
        x="utterance_index",
        y="speaker_z",
        hue="speaker",
        estimator=None,
        linewidth=1.8,
        alpha=0.9,
    )
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.title(f"Within-speaker z-scored score trajectories (top {len(selected)} speakers)")
    plt.xlabel("Utterance index (order in CSV within speaker)")
    plt.ylabel("Within-speaker z-score of decision score")
    plt.legend(title="speaker", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def test_retest_reliability(
    df: pd.DataFrame, min_utterances: int, seed: int
) -> tuple[float, float, int]:
    """
    For each speaker with >= min_utterances:
      - randomly split their utterances into two halves
      - compute mean score in each half
    Across speakers:
      - Pearson r and Spearman rho between half means.
    """
    rng = np.random.default_rng(seed)

    half_a: list[float] = []
    half_b: list[float] = []

    for speaker, g in df.groupby("speaker", observed=True):
        scores = g["score"].to_numpy(dtype=float)
        if scores.size < min_utterances:
            continue

        perm = rng.permutation(scores.size)
        mid = scores.size // 2
        a = scores[perm[:mid]]
        b = scores[perm[mid:]]
        if a.size == 0 or b.size == 0:
            continue

        half_a.append(float(np.mean(a)))
        half_b.append(float(np.mean(b)))

    n = len(half_a)
    if n < 2:
        return float("nan"), float("nan"), n

    pearson_r, _ = stats.pearsonr(half_a, half_b)
    spearman_r, _ = stats.spearmanr(half_a, half_b)
    return float(pearson_r), float(spearman_r), n


def domain_breakdown(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean_by_domain = df.groupby("domain", observed=True)["score"].mean().sort_values(ascending=False)
    mean_by_domain_class = (
        df.groupby(["domain", "class"], observed=True)["score"]
        .mean()
        .sort_values(ascending=False)
    )
    return mean_by_domain, mean_by_domain_class


def plot_score_by_domain(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    g = sns.displot(
        data=df,
        x="score",
        hue="class",
        col="domain",
        kind="kde",
        fill=True,
        common_norm=False,
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("SVM decision score", "Density")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle("Score distribution by domain", y=1.04)
    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=200)
    plt.close(g.fig)


def format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main() -> None:
    args = parse_args()
    ensure_figures_dir(args.figures_dir)

    df = load_scores_csv(args.scores)
    n = len(df)
    n_pos = int((df["y_true"] == 1).sum())
    n_neg = int((df["y_true"] == 0).sum())

    auc = roc_auc_rank(df["y_true"].to_numpy(), df["score"].to_numpy())

    summary = score_summary_by_class(df)
    d = cohen_d(
        df.loc[df["y_true"] == 0, "score"].to_numpy(),
        df.loc[df["y_true"] == 1, "score"].to_numpy(),
    )

    df_z = add_speaker_zscores(df)
    pearson_r, spearman_r, n_speakers_rel = test_retest_reliability(
        df, min_utterances=args.min_utterances, seed=args.seed
    )

    mean_by_domain, mean_by_domain_class = domain_breakdown(df)

    # Figures
    plot_score_distribution(df, args.figures_dir / "score_distribution.png")
    plot_speaker_trajectories(
        df_z, top_speakers=args.top_speakers, out_path=args.figures_dir / "speaker_trajectories.png"
    )
    plot_score_by_domain(df, args.figures_dir / "score_by_domain.png")

    # Report
    print("=== Score Analysis Report ===")
    print(f"Scores file: {args.scores}")
    print(f"Dataset size: {n} utterances")
    print(
        f"Class balance: control={n_neg} ({format_pct(n_neg / n)}) | "
        f"dysarthric={n_pos} ({format_pct(n_pos / n)})"
    )
    print()

    print("Ranking quality")
    print(f"ROC-AUC (score vs y_true): {auc:.4f}")
    print()

    print("Effect size (score)")
    with pd.option_context("display.max_columns", 20, "display.width", 120):
        print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"Cohen's d (dysarthric - control): {d:.4f}")
    print()

    print("Test–retest reliability (speaker means across random halves)")
    print(f"Speakers included (>= {args.min_utterances} utterances): {n_speakers_rel}")
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman ρ: {spearman_r:.4f}")
    print()

    print("Domain shift sanity check (mean score)")
    print("Mean score per domain:")
    print(mean_by_domain.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("Mean score per (domain, class):")
    print(mean_by_domain_class.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    print("Saved figures:")
    print(f"- {args.figures_dir / 'score_distribution.png'}")
    print(f"- {args.figures_dir / 'speaker_trajectories.png'}")
    print(f"- {args.figures_dir / 'score_by_domain.png'}")


if __name__ == "__main__":
    main()

