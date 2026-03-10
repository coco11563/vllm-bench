#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize sequential vLLM benchmark results.")
    parser.add_argument("--summary", default="/data/benchmarks/vllm_qwen35_seq/results/summary.csv")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick_best(rows: list[dict], group_keys: list[str], metric: str) -> list[dict]:
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row[k] for k in group_keys)
        score = to_float(row.get(metric))
        if score is None:
            continue
        if key not in grouped or score > to_float(grouped[key].get(metric)):
            grouped[key] = row
    return [grouped[key] for key in sorted(grouped)]


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_path)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]

    best_by_model_gpu = pick_best(ok_rows, ["model", "ngpu"], "total_token_throughput")
    best_by_model = pick_best(ok_rows, ["model"], "total_token_throughput")

    write_csv(
        out_dir / "best_by_model_gpu.csv",
        best_by_model_gpu,
        fieldnames=list(best_by_model_gpu[0].keys()) if best_by_model_gpu else [],
    )
    write_csv(
        out_dir / "best_by_model.csv",
        best_by_model,
        fieldnames=list(best_by_model[0].keys()) if best_by_model else [],
    )

    group_stats = defaultdict(lambda: {"ok": 0, "failed": 0})
    for row in rows:
        group = row.get("model_group", "unknown")
        if row.get("status") == "ok":
            group_stats[group]["ok"] += 1
        else:
            group_stats[group]["failed"] += 1

    report_lines = []
    report_lines.append("# vLLM Sequential Benchmark Summary")
    report_lines.append("")
    report_lines.append(f"- summary: `{summary_path}`")
    report_lines.append(f"- total rows: {len(rows)}")
    report_lines.append(f"- ok rows: {len(ok_rows)}")
    report_lines.append(f"- failed rows: {len(failed_rows)}")
    report_lines.append("")
    report_lines.append("## Group Status")
    report_lines.append("")
    for group in sorted(group_stats):
        stats = group_stats[group]
        report_lines.append(f"- {group}: ok={stats['ok']} failed={stats['failed']}")
    report_lines.append("")
    report_lines.append("## Best Total Token Throughput By Model")
    report_lines.append("")
    for row in best_by_model:
        report_lines.append(
            "- {model}: tp={ngpu} cc={concurrency} total_tps={total} tokens_per_gpu={per_gpu} "
            "mean_ttft_ms={ttft}".format(
                model=row["model"],
                ngpu=row["ngpu"],
                concurrency=row["concurrency"],
                total=row.get("total_token_throughput", ""),
                per_gpu=row.get("tokens_per_gpu", ""),
                ttft=row.get("mean_ttft_ms", ""),
            )
        )

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {out_dir / 'best_by_model_gpu.csv'}")
    print(f"Wrote {out_dir / 'best_by_model.csv'}")
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
