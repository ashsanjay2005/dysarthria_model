#!/usr/bin/env python3
"""
Aggregate perturbation stability CSV outputs across multiple seeds and summarize model stability.

Expected inputs:
  - TORGO test-domain runs:   out_torgo_v3/perturb_seed*.csv (preferred) OR oyour own runs csv files
  - UASPEECH test-domain runs: out_ua_v3/perturb_ua_seed*.csv (preferred) OR your own runs csv files

Outputs:
  - out_summary/all_rows.csv
  - out_summary/utterance_summary.csv
  - out_summary/speaker_summary.csv
  - out_summary/figures/*.png

Run:
  python3 aggregate_perturb_results.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PERTURB_ORDER = [
    "gain_0.7",
    "gain_1.3",
    "noise_0.002",
    "noise_0.005",
    "shift_-80",
    "shift_+80",
]

DEFAULT_SOURCES = [
    # Prefer the latest rerun folders (v3) if present.
    ("TORGO", Path("out_torgo_v3"), "perturb_seed*.csv"),
    ("UASPEECH", Path("out_ua_v3"), "perturb_ua_seed*.csv"),

]

REQUIRED_COLUMNS = [
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


def parse_seed_from_filename(path: Path) -> int | None:
    m = re.search(r"seed_?(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1))


def find_csvs() -> list[tuple[str, int, Path]]:
    """
    Returns: list of (test_domain, seed, csv_path)
    """
    found: list[tuple[str, int, Path]] = []
    for test_domain, folder, pattern in DEFAULT_SOURCES:
        if not folder.exists():
            continue
        for p in sorted(folder.glob(pattern)):
            seed = parse_seed_from_filename(p)
            if seed is None:
                continue
            found.append((test_domain, seed, p))
    return found


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (found {list(df.columns)})")
    return df


def load_all(csvs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for test_domain, seed, path in csvs:
        df = load_one_csv(path).copy()
        df["test_domain"] = test_domain
        df["seed"] = seed
        df["source_path"] = str(path)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")
    for col in ["score_clean", "score_pert", "delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["model_name", "test_domain", "perturb", "speaker", "file", "domain"]:
        df[col] = df[col].astype(str)
    df = df.dropna(subset=["seed", "delta", "model_name", "test_domain", "perturb"])
    df["seed"] = df["seed"].astype(int)
    return df


def utterance_level_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A) Utterance-level stability summary:
      Group by: test_domain, model_name, perturb, seed
      Compute: mean_delta, median_delta, n
      Then aggregate across seeds.
    """
    per_seed = (
        df.groupby(["test_domain", "model_name", "perturb", "seed"], observed=True)["delta"]
        .agg(mean_delta="mean", median_delta="median", n="count")
        .reset_index()
    )

    across_seeds = (
        per_seed.groupby(["test_domain", "model_name", "perturb"], observed=True)
        .agg(
            mean_of_means=("mean_delta", "mean"),
            std_of_means=("mean_delta", "std"),
            mean_of_medians=("median_delta", "mean"),
            std_of_medians=("median_delta", "std"),
            seeds=("seed", "nunique"),
            n_total=("n", "sum"),
        )
        .reset_index()
    )

    # If only one seed, std will be NaN; treat as 0 for plotting.
    across_seeds["std_of_means"] = across_seeds["std_of_means"].fillna(0.0)
    across_seeds["std_of_medians"] = across_seeds["std_of_medians"].fillna(0.0)
    return per_seed, across_seeds


def speaker_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    B) Speaker-level stability summary:
      - speaker_mean_delta per (test_domain, model_name, perturb, seed, speaker)
      - mean_speaker_delta per (test_domain, model_name, perturb, seed)
      - aggregate across seeds per (test_domain, model_name, perturb)
    """
    per_speaker_seed = (
        df.groupby(["test_domain", "model_name", "perturb", "seed", "speaker"], observed=True)["delta"]
        .mean()
        .rename("speaker_mean_delta")
        .reset_index()
    )

    per_seed = (
        per_speaker_seed.groupby(["test_domain", "model_name", "perturb", "seed"], observed=True)[
            "speaker_mean_delta"
        ]
        .mean()
        .rename("mean_speaker_delta")
        .reset_index()
    )

    across_seeds = (
        per_seed.groupby(["test_domain", "model_name", "perturb"], observed=True)
        .agg(
            mean_of_seed_means=("mean_speaker_delta", "mean"),
            std_of_seed_means=("mean_speaker_delta", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    across_seeds["std_of_seed_means"] = across_seeds["std_of_seed_means"].fillna(0.0)
    return across_seeds


def ordered_perturbations(present: Iterable[str]) -> list[str]:
    present_set = set(present)
    ordered = [p for p in PERTURB_ORDER if p in present_set]
    # Add any unknown perturbations at the end (sorted) for robustness.
    extras = sorted([p for p in present_set if p not in set(PERTURB_ORDER)])
    return ordered + extras


def plot_grouped_bars(
    summary: pd.DataFrame,
    *,
    test_domain: str,
    y_col: str,
    err_col: str,
    title: str,
    out_path: Path,
) -> None:
    sub = summary[summary["test_domain"] == test_domain].copy()
    if sub.empty:
        return

    perturbations = ordered_perturbations(sub["perturb"].unique())
    model_names = sorted(sub["model_name"].unique())

    # Matrix-like access: values[model][perturb] -> (y, err)
    pivot_y = sub.pivot_table(index="perturb", columns="model_name", values=y_col, aggfunc="first")
    pivot_e = sub.pivot_table(index="perturb", columns="model_name", values=err_col, aggfunc="first")
    pivot_y = pivot_y.reindex(index=perturbations, columns=model_names).fillna(0.0)
    pivot_e = pivot_e.reindex(index=perturbations, columns=model_names).fillna(0.0)

    x = np.arange(len(perturbations))
    width = 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(perturbations)), 5.5))
    for i, model in enumerate(model_names):
        y = pivot_y[model].to_numpy(dtype=float)
        e = pivot_e[model].to_numpy(dtype=float)
        ax.bar(
            x + (i - (len(model_names) - 1) / 2.0) * width,
            y,
            width=width,
            yerr=e,
            capsize=3,
            label=model,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Mean |Δ score|")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    if len(model_names) > 1:
        ax.legend(title="model_name", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overall_stability(
    per_seed: pd.DataFrame, *, test_domain: str, out_path: Path
) -> None:
    """
    Plot overall stability per model:
      overall_mean = mean(mean_delta over all perturbations and seeds)
    Uses the utterance-level per-seed table: (test_domain, model_name, perturb, seed, mean_delta).
    """
    sub = per_seed[per_seed["test_domain"] == test_domain].copy()
    if sub.empty:
        return

    overall = (
        sub.groupby(["test_domain", "model_name"], observed=True)["mean_delta"]
        .mean()
        .rename("overall_mean")
        .reset_index()
        .sort_values("overall_mean", ascending=True)
    )

    models = overall["model_name"].tolist()
    vals = overall["overall_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(models, vals, alpha=0.9)
    ax.set_title(f"{test_domain}: Overall mean |Δ score| (averaged over perturbations and seeds)")
    ax.set_xlabel("model_name")
    ax.set_ylabel("Overall mean |Δ score|")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def console_report(utter_summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    print("=== Perturbation Stability Aggregation Report ===")
    if utter_summary.empty:
        print("No summary available (empty input).")
        return

    for test_domain in sorted(utter_summary["test_domain"].unique()):
        print(f"\n[{test_domain}]")
        sub = utter_summary[utter_summary["test_domain"] == test_domain].copy()
        models = sorted(sub["model_name"].unique())

        # Top 3 perturbations by mean_of_means per model
        for model in models:
            msub = sub[sub["model_name"] == model].sort_values("mean_of_means", ascending=False)
            top3 = msub.head(3)[["perturb", "mean_of_means"]]
            items = ", ".join([f"{r.perturb}={r.mean_of_means:.4f}" for r in top3.itertuples(index=False)])
            print(f"- Top perturbations (highest mean |Δ|) for {model}: {items}")

        # Overall mean per model (based on per-seed mean_delta)
        o = (
            per_seed[per_seed["test_domain"] == test_domain]
            .groupby("model_name", observed=True)["mean_delta"]
            .mean()
            .sort_values()
        )
        for model, val in o.items():
            print(f"- Overall mean |Δ| for {model}: {float(val):.4f}")

        # Noise stability comparisons
        for noise in ["noise_0.005", "noise_0.002"]:
            nsub = sub[sub["perturb"] == noise]
            if nsub.empty or len(models) < 2:
                continue
            vals = {m: float(nsub[nsub["model_name"] == m]["mean_of_means"].mean()) for m in models}
            best = min(vals.items(), key=lambda kv: kv[1])[0]
            print(f"- More stable on {noise}: {best} (lower mean |Δ|)")


def main() -> None:
    csvs = find_csvs()
    if not csvs:
        raise SystemExit(
            "No CSV files found.\n"
            "Expected at least one of:\n"
            "  - out_torgo_v3/perturb_seed*.csv\n"
            "  - out_ua_v3/perturb_ua_seed*.csv\n"
            "  - out_v2/perturb_seed*.csv\n"
            "  - out_ua_v2/perturb_ua_seed*.csv\n"
            "  - out/perturb_seed*.csv\n"
            "  - out_ua/perturb_ua_seed*.csv"
        )

    print(f"Found {len(csvs)} CSVs:")
    for test_domain, seed, p in csvs:
        print(f"- {test_domain} seed={seed}: {p}")

    df = load_all(csvs)
    df = coerce_types(df)
    if df.empty:
        raise SystemExit("Loaded inputs but no usable rows after type coercion/filtering.")

    out_dir = Path("out_summary")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # A) Utterance summaries
    utter_per_seed, utter_across = utterance_level_summary(df)

    # B) Speaker summaries
    speaker_across = speaker_level_summary(df)

    # Save raw + summaries
    df.to_csv(out_dir / "all_rows.csv", index=False)
    utter_across.to_csv(out_dir / "utterance_summary.csv", index=False)
    speaker_across.to_csv(out_dir / "speaker_summary.csv", index=False)

    # Plots (per domain; skip if missing)
    for test_domain in sorted(df["test_domain"].unique()):
        plot_grouped_bars(
            utter_across,
            test_domain=test_domain,
            y_col="mean_of_means",
            err_col="std_of_means",
            title=f"{test_domain}: Utterance-level mean |Δ score| (±1 SD across seeds)",
            out_path=fig_dir / f"utterance_bar_{test_domain}.png",
        )
        plot_grouped_bars(
            speaker_across,
            test_domain=test_domain,
            y_col="mean_of_seed_means",
            err_col="std_of_seed_means",
            title=f"{test_domain}: Speaker-level mean |Δ score| (±1 SD across seeds)",
            out_path=fig_dir / f"speaker_bar_{test_domain}.png",
        )
        plot_overall_stability(
            utter_per_seed,
            test_domain=test_domain,
            out_path=fig_dir / f"overall_stability_{test_domain}.png",
        )

    console_report(utter_across, utter_per_seed)
    print("\nWrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
