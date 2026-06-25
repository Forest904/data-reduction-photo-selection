"""Generate report figures from saved experiment result CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=Path("experiments/results"), type=Path)
    parser.add_argument("--output", default=Path("experiments/figures"), type=Path)
    args = parser.parse_args()

    try:
        frame = _load_results(args.results)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    frame = _canonical_or_all(frame)
    successful = frame[frame["status"] == "success"].copy()
    if successful.empty:
        _write_placeholder(args.output / "utility.png", "No successful result rows")
        _write_placeholder(args.output / "runtime.png", "No successful result rows")
        _write_placeholder(args.output / "memory.png", "No successful result rows")
        _write_placeholder(args.output / "scalability.png", "No successful result rows")
        _write_placeholder(
            args.output / "budget_sensitivity.png",
            "No successful result rows",
        )
        _write_placeholder(
            args.output / "holdout_utility.png",
            "No successful result rows",
        )
    else:
        _plot_utility_by_method(
            _experiment_or_all(successful, {"small_exact_comparison", "scalability"}),
            args.output / "utility.png",
        )
        _plot_metric_by_method(
            _experiment_or_all(successful, {"small_exact_comparison", "scalability"}),
            "runtime_seconds",
            "Runtime by Method",
            "Runtime seconds",
            args.output / "runtime.png",
        )
        _plot_metric_by_method(
            _experiment_or_all(successful, {"small_exact_comparison", "scalability"}),
            "peak_memory_mb",
            "Peak Memory by Method",
            "Peak memory MB",
            args.output / "memory.png",
        )
        _plot_scalability(
            _experiment_or_all(successful, {"scalability"}),
            args.output / "scalability.png",
        )
        _plot_budget_sensitivity(
            _experiment_or_all(successful, {"budget_sensitivity"}),
            args.output / "budget_sensitivity.png",
        )
        _plot_holdout_utility(successful, args.output / "holdout_utility.png")

    print(f"Wrote figures to {args.output}")
    return 0


def _load_results(results_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(results_dir.rglob("results.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No results.csv files found under {results_dir}")
    frames = [pd.read_csv(path) for path in csv_paths]
    frame = pd.concat(frames, ignore_index=True)
    for column in (
        "dataset_size",
        "train_query_count",
        "eval_query_count",
        "budget",
        "cosine_proxy_utility_eval",
        "jaccard_precision_utility_eval",
        "runtime_seconds",
        "peak_memory_mb",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _canonical_or_all(frame: pd.DataFrame) -> pd.DataFrame:
    if "batch_id" not in frame:
        return frame
    canonical = frame[frame["batch_id"].astype(str).str.endswith("_canonical")]
    return canonical.copy() if not canonical.empty else frame


def _experiment_or_all(
    frame: pd.DataFrame,
    experiment_names: set[str],
) -> pd.DataFrame:
    if "experiment_name" not in frame:
        return frame
    filtered = frame[frame["experiment_name"].isin(experiment_names)].copy()
    return filtered if not filtered.empty else frame


def _plot_metric_by_method(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    data = frame.dropna(subset=[metric])
    if data.empty:
        _write_placeholder(output_path, f"No data for {metric}")
        return
    grouped = (
        data.groupby(["method", "dataset_size"], as_index=False)[metric]
        .mean()
        .sort_values(["method", "dataset_size"])
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    for method, method_frame in grouped.groupby("method"):
        axis.plot(
            method_frame["dataset_size"],
            method_frame[metric],
            marker="o",
            label=str(method),
        )
    axis.set_title(title)
    axis.set_xlabel("Dataset size (photos)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_utility_by_method(frame: pd.DataFrame, output_path: Path) -> None:
    data = frame.dropna(subset=["cosine_proxy_utility_eval"])
    if data.empty:
        _write_placeholder(output_path, "No utility data")
        return

    exact = data[data.get("experiment_name") == "small_exact_comparison"].copy()
    scalable = data[data.get("experiment_name") == "scalability"].copy()
    if exact.empty or scalable.empty:
        _plot_metric_by_method(
            data,
            "cosine_proxy_utility_eval",
            "Utility by Method",
            "Cosine proxy utility",
            output_path,
        )
        return

    exact_grouped = (
        exact.groupby(["method", "budget"], as_index=False)["cosine_proxy_utility_eval"]
        .mean()
        .sort_values(["method", "budget"])
    )
    scalable_grouped = (
        scalable.groupby(["method", "dataset_size"], as_index=False)[
            "cosine_proxy_utility_eval"
        ]
        .mean()
        .sort_values(["method", "dataset_size"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, method_frame in exact_grouped.groupby("method"):
        axes[0].plot(
            method_frame["budget"],
            method_frame["cosine_proxy_utility_eval"],
            marker="o",
            label=str(method),
        )
    for method, method_frame in scalable_grouped.groupby("method"):
        axes[1].plot(
            method_frame["dataset_size"],
            method_frame["cosine_proxy_utility_eval"],
            marker="o",
            label=str(method),
        )

    axes[0].set_title("Exact-Scale Utility", fontsize=13)
    axes[0].set_xlabel("Budget", fontsize=11)
    axes[0].set_ylabel("Cosine proxy utility", fontsize=11)
    axes[0].set_xticks(sorted(exact_grouped["budget"].dropna().unique()))
    axes[1].set_title("Scalable Utility", fontsize=13)
    axes[1].set_xlabel("Dataset size (photos)", fontsize=11)
    axes[1].set_ylabel("Cosine proxy utility", fontsize=11)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)
        axis.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scalability(frame: pd.DataFrame, output_path: Path) -> None:
    data = frame.dropna(subset=["runtime_seconds", "peak_memory_mb"])
    if data.empty:
        _write_placeholder(output_path, "No scalability data")
        return
    grouped = (
        data.groupby(["method", "dataset_size"], as_index=False)[
            ["runtime_seconds", "peak_memory_mb"]
        ]
        .mean()
        .sort_values(["method", "dataset_size"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, method_frame in grouped.groupby("method"):
        axes[0].plot(
            method_frame["dataset_size"],
            method_frame["runtime_seconds"],
            marker="o",
            label=str(method),
        )
        axes[1].plot(
            method_frame["dataset_size"],
            method_frame["peak_memory_mb"],
            marker="o",
            label=str(method),
        )
    axes[0].set_title("Runtime Scaling", fontsize=13)
    axes[0].set_xlabel("Dataset size (photos)", fontsize=11)
    axes[0].set_ylabel("Runtime seconds", fontsize=11)
    axes[1].set_title("Memory Scaling", fontsize=13)
    axes[1].set_xlabel("Dataset size (photos)", fontsize=11)
    axes[1].set_ylabel("Peak memory MB", fontsize=11)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)
        axis.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_budget_sensitivity(frame: pd.DataFrame, output_path: Path) -> None:
    data = frame.dropna(subset=["budget", "cosine_proxy_utility_eval"])
    if data.empty:
        _write_placeholder(output_path, "No budget sensitivity data")
        return
    grouped = (
        data.groupby(["method", "budget"], as_index=False)["cosine_proxy_utility_eval"]
        .mean()
        .sort_values(["method", "budget"])
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    for method, method_frame in grouped.groupby("method"):
        axis.plot(
            method_frame["budget"],
            method_frame["cosine_proxy_utility_eval"],
            marker="o",
            label=str(method),
        )
    axis.set_title("Budget Sensitivity")
    axis.set_xlabel("Budget")
    axis.set_ylabel("Cosine proxy utility")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_holdout_utility(frame: pd.DataFrame, output_path: Path) -> None:
    data = frame[
        (frame.get("experiment_name") == "query_holdout")
        & (frame.get("evaluation_scope") == "holdout")
    ].copy()
    data = data.dropna(subset=["budget", "cosine_proxy_utility_eval"])
    if data.empty:
        _write_placeholder(output_path, "No query holdout data")
        return

    grouped = (
        data.groupby(["method", "budget"], as_index=False)["cosine_proxy_utility_eval"]
        .mean()
        .sort_values(["method", "budget"])
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    for method, method_frame in grouped.groupby("method"):
        axis.plot(
            method_frame["budget"],
            method_frame["cosine_proxy_utility_eval"],
            marker="o",
            label=str(method),
        )
    axis.set_title("Query Holdout Utility")
    axis.set_xlabel("Budget")
    axis.set_ylabel("Held-out cosine proxy utility")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_placeholder(output_path: Path, message: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 3))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
