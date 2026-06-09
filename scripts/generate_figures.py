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
    else:
        _plot_metric_by_method(
            successful,
            "cosine_proxy_utility_eval",
            "Utility by Method",
            "Cosine proxy utility",
            args.output / "utility.png",
        )
        _plot_metric_by_method(
            successful,
            "runtime_seconds",
            "Runtime by Method",
            "Runtime seconds",
            args.output / "runtime.png",
        )
        _plot_metric_by_method(
            successful,
            "peak_memory_mb",
            "Peak Memory by Method",
            "Peak memory MB",
            args.output / "memory.png",
        )
        _plot_scalability(successful, args.output / "scalability.png")
        _plot_budget_sensitivity(successful, args.output / "budget_sensitivity.png")

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
        "budget",
        "cosine_proxy_utility_eval",
        "jaccard_precision_utility_eval",
        "runtime_seconds",
        "peak_memory_mb",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
    axes[0].set_title("Runtime Scaling")
    axes[0].set_xlabel("Dataset size (photos)")
    axes[0].set_ylabel("Runtime seconds")
    axes[1].set_title("Memory Scaling")
    axes[1].set_xlabel("Dataset size (photos)")
    axes[1].set_ylabel("Peak memory MB")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
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


def _write_placeholder(output_path: Path, message: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 3))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
