#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot token throughput by TP and model size.")
    parser.add_argument(
        "--summary",
        default="/Volumes/980Pro/Tabular-Condension/vllm-bench/results/remote/summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/Volumes/980Pro/Tabular-Condension/vllm-bench/results/plots",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--metric",
        default="total_token_throughput",
        choices=["total_token_throughput", "output_token_throughput", "request_throughput"],
    )
    return parser.parse_args()


def model_size_b(model_name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if not match:
        raise ValueError(f"Cannot parse parameter size from model name: {model_name}")
    return float(match.group(1))


def model_variant(model_name: str) -> str:
    if "-FP8" in model_name:
        return "fp8"
    if "-Base" in model_name:
        return "base"
    return "default"


def load_latest_rows(summary_path: Path, concurrency: int) -> pd.DataFrame:
    rows = list(csv.DictReader(summary_path.open()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["status"] == "ok"].copy()
    df = df[df["concurrency"].astype(int) == concurrency].copy()
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"])
    df["ngpu"] = df["ngpu"].astype(int)
    df["size_b"] = df["model"].map(model_size_b)
    df["variant"] = df["model"].map(model_variant)
    df["metric_value"] = df["total_token_throughput"].astype(float)
    df["mean_ttft_ms"] = df["mean_ttft_ms"].astype(float)
    grouped = (
        df.groupby(["model", "ngpu"], as_index=False)
        .agg(
            size_b=("size_b", "first"),
            variant=("variant", "first"),
            metric_value=("metric_value", "mean"),
            metric_std=("metric_value", "std"),
            mean_ttft_ms=("mean_ttft_ms", "mean"),
            repeats=("metric_value", "size"),
            ts=("ts", "max"),
        )
        .fillna({"metric_std": 0.0})
    )
    return grouped.sort_values(["ngpu", "size_b", "model"]).reset_index(drop=True)


def plot_one_tp(df: pd.DataFrame, tp: int, metric: str, concurrency: int, out_dir: Path) -> None:
    subset = df[df["ngpu"] == tp].copy()
    if subset.empty:
        return

    colors = {"default": "#1f77b4", "base": "#ff7f0e", "fp8": "#2ca02c"}
    markers = {"default": "o", "base": "s", "fp8": "^"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, group in subset.groupby("variant"):
        ax.errorbar(
            group["size_b"],
            group["metric_value"],
            yerr=group["metric_std"],
            fmt=markers.get(variant, "o"),
            ms=8,
            color=colors.get(variant, "#444444"),
            label=variant,
            capsize=4,
            linestyle="none",
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['model']} (n={int(row['repeats'])})",
                (row["size_b"], row["metric_value"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters, log scale)")
    ax.set_ylabel(f"{metric} @ cc={concurrency} (tok/s)")
    ax.set_title(f"TP={tp}: Model Size vs Token Throughput")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="variant")
    fig.tight_layout()
    out_path = out_dir / f"tp{tp}_cc{concurrency}_{metric}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_latest_rows(summary_path, args.concurrency)
    if df.empty:
        raise SystemExit("No matching successful rows found for plotting.")

    plot_csv = out_dir / f"plot_data_cc{args.concurrency}.csv"
    df.to_csv(plot_csv, index=False)

    for tp in sorted(df["ngpu"].unique()):
        plot_one_tp(df, int(tp), args.metric, args.concurrency, out_dir)

    print(f"Wrote plot data: {plot_csv}")
    for tp in sorted(df["ngpu"].unique()):
        print(f"Wrote plot: {out_dir / f'tp{int(tp)}_cc{args.concurrency}_{args.metric}.png'}")


if __name__ == "__main__":
    main()
