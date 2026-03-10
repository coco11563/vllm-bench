#!/usr/bin/env python3
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


METRIC_PATTERNS = {
    "request_throughput": r"Request throughput.*?([0-9.]+)",
    "output_token_throughput": r"Output token throughput.*?([0-9.]+)",
    "total_token_throughput": r"Total token throughput.*?([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT.*?([0-9.]+)",
    "p50_ttft_ms": r"P50 TTFT.*?([0-9.]+)",
    "p95_ttft_ms": r"P95 TTFT.*?([0-9.]+)",
    "mean_itl_ms": r"Mean ITL.*?([0-9.]+)",
    "p50_itl_ms": r"P50 ITL.*?([0-9.]+)",
    "p95_itl_ms": r"P95 ITL.*?([0-9.]+)",
}

STARTUP_PATTERNS = {
    "model_loading_seconds": r"Model loading took .*? and ([0-9.]+) seconds",
    "engine_init_seconds": r"init engine .*? took ([0-9.]+) seconds",
    "compile_seconds": r"torch\.compile and initial profiling run took ([0-9.]+) s",
}

CSV_FIELDS = [
    "run_id",
    "ts",
    "repeat_idx",
    "model",
    "model_group",
    "ngpu",
    "visible_gpus",
    "port",
    "concurrency",
    "num_prompts",
    "input_len",
    "output_len",
    "status",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "tokens_per_gpu",
    "output_tokens_per_gpu",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "mean_itl_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "deploy_seconds",
    "model_loading_seconds",
    "engine_init_seconds",
    "compile_seconds",
    "bench_seconds",
    "failure_reason",
    "result_log",
    "result_err",
    "server_stdout",
    "server_stderr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential vLLM serve benchmark runner.")
    parser.add_argument("--model-root", default="/data/model_repo")
    parser.add_argument("--output-root", default="/data/benchmarks/vllm_qwen35_seq")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8200)
    parser.add_argument("--gpus", default="1,2,4,8")
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--min-prompts", type=int, default=64)
    parser.add_argument("--prompts-per-concurrency", type=int, default=2)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--extra-serve-arg", action="append", default=[])
    parser.add_argument("--model", action="append", default=[], help="Restrict to specific model name(s).")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume-ok", action="store_true", help="Skip benchmark cases already marked ok.")
    parser.add_argument("--resume-any", action="store_true", help="Skip benchmark cases already recorded with any status.")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def infer_model_group(model_name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if not match:
        return "unknown"
    size_b = float(match.group(1))
    if size_b <= 10:
        return "small"
    if size_b <= 40:
        return "medium"
    return "large"


def discover_models(model_root: Path, requested: Iterable[str]) -> list[str]:
    requested_set = {item.strip() for item in requested if item.strip()}
    models = []
    for entry in sorted(model_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("models--"):
            continue
        if not (entry / "config.json").exists():
            continue
        if requested_set and entry.name not in requested_set:
            continue
        models.append(entry.name)
    if requested_set:
        missing = sorted(requested_set - set(models))
        if missing:
            raise SystemExit(f"Requested models not found under {model_root}: {', '.join(missing)}")
    return models


def ensure_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    logs_root = output_root / "logs"
    results_root = output_root / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    return output_root, logs_root, results_root


def get_gpu_count() -> int:
    proc = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def get_gpu_memory_mib() -> int:
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n 1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def wait_for_server(
    host: str,
    port: int,
    timeout_s: int,
    proc: subprocess.Popen | None = None,
) -> bool:
    deadline = time.time() + timeout_s
    urls = [
        f"http://{host}:{port}/health",
        f"http://{host}:{port}/v1/models",
    ]
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        for url in urls:
            try:
                with urlopen(url, timeout=3) as response:
                    if response.status < 500:
                        return True
            except URLError:
                pass
            except Exception:
                pass
        time.sleep(2)
    return False


def stop_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def tail_text(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    return text[-max_chars:].strip()


def parse_bench_metrics(log_path: Path) -> dict[str, float | None]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    metrics = {}
    for field, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        metrics[field] = float(match.group(1)) if match else None
    return metrics


def parse_startup_metrics(log_path: Path) -> dict[str, float | None]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    metrics = {}
    for field, pattern in STARTUP_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        metrics[field] = float(match.group(1)) if match else None
    return metrics


def load_existing_cases(summary_path: Path, ok_only: bool) -> set[tuple[str, int, int, int]]:
    done = set()
    if not summary_path.exists():
        return done
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if ok_only and row.get("status") != "ok":
                continue
            repeat_idx = int(row.get("repeat_idx") or 1)
            done.add((row["model"], int(row["ngpu"]), int(row["concurrency"]), repeat_idx))
    return done


def estimate_model_weight_gib(model_name: str, ngpu: int) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if not match:
        return None
    total_params_b = float(match.group(1))
    bytes_per_param = 1.0 if "FP8" in model_name.upper() else 2.0
    total_bytes = total_params_b * 1_000_000_000 * bytes_per_param
    return total_bytes / ngpu / (1024**3)


def obvious_failure_reason(model_name: str, ngpu: int, gpu_memory_mib: int) -> str | None:
    weight_gib = estimate_model_weight_gib(model_name, ngpu)
    gpu_memory_gib = gpu_memory_mib / 1024.0
    if weight_gib is not None and weight_gib > gpu_memory_gib:
        return (
            "precheck_impossible_weight_memory"
            f" | estimated_weight_gib={weight_gib:.1f}"
            f" | gpu_memory_gib={gpu_memory_gib:.1f}"
        )

    if model_name == "Qwen3.5-35B-A3B-FP8" and ngpu == 8:
        return "precheck_incompatible_fp8_tp8_partition | block_k=128 | input_size_per_partition=64"

    return None


def append_row(summary_path: Path, row: dict[str, object]) -> None:
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def build_env(visible_gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_gpus
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return env


def record_failure(
    summary_path: Path,
    common: dict[str, object],
    concurrency: int,
    num_prompts: int,
    reason: str,
    server_stdout: Path,
    server_stderr: Path,
    result_log: Path | None,
    result_err: Path | None,
    deploy_seconds: float,
    startup_metrics: dict[str, float | None] | None = None,
    bench_seconds: float = 0.0,
) -> None:
    row = dict(common)
    startup_metrics = startup_metrics or {}
    row.update(
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "concurrency": concurrency,
            "num_prompts": num_prompts,
            "status": "failed",
            "failure_reason": reason,
            "deploy_seconds": round(deploy_seconds, 3),
            "model_loading_seconds": startup_metrics.get("model_loading_seconds"),
            "engine_init_seconds": startup_metrics.get("engine_init_seconds"),
            "compile_seconds": startup_metrics.get("compile_seconds"),
            "bench_seconds": round(bench_seconds, 3),
            "server_stdout": str(server_stdout),
            "server_stderr": str(server_stderr),
            "result_log": str(result_log) if result_log else "",
            "result_err": str(result_err) if result_err else "",
        }
    )
    append_row(summary_path, row)


def main() -> int:
    args = parse_args()
    model_root = Path(args.model_root)
    output_root, logs_root, results_root = ensure_dirs(Path(args.output_root))
    summary_path = results_root / "summary.csv"

    ngpu_list = parse_int_list(args.gpus)
    cc_list = parse_int_list(args.concurrency)
    models = discover_models(model_root, args.model)
    total_gpus = get_gpu_count()
    gpu_memory_mib = get_gpu_memory_mib()
    done_cases = load_existing_cases(summary_path, ok_only=not args.resume_any) if (args.resume_ok or args.resume_any) else set()

    print(
        f"Discovered models={len(models)} total_gpus={total_gpus} min_gpu_memory_mib={gpu_memory_mib}",
        flush=True,
    )
    print(f"Models: {', '.join(models)}", flush=True)

    port = args.base_port

    for repeat_idx in range(1, args.repeats + 1):
        for model in models:
            for ngpu in ngpu_list:
                visible = ",".join(str(idx) for idx in range(ngpu))
                run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{model}_tp{ngpu}_r{repeat_idx}"
                common = {
                    "run_id": run_id,
                    "repeat_idx": repeat_idx,
                    "model": model,
                    "model_group": infer_model_group(model),
                    "ngpu": ngpu,
                    "visible_gpus": visible,
                    "port": port,
                    "input_len": args.input_len,
                    "output_len": args.output_len,
                }

                if ngpu > total_gpus:
                    for cc in cc_list:
                        num_prompts = max(args.min_prompts, cc * args.prompts_per_concurrency)
                        record_failure(
                            summary_path,
                            common,
                            cc,
                            num_prompts,
                            f"requested_ngpu>{total_gpus}",
                            Path(""),
                            Path(""),
                            None,
                            None,
                            deploy_seconds=0.0,
                        )
                    port += 1
                    continue

                if all((model, ngpu, cc, repeat_idx) in done_cases for cc in cc_list):
                    print(f"SKIP OK {model} tp={ngpu} repeat={repeat_idx}", flush=True)
                    port += 1
                    continue

                skip_reason = obvious_failure_reason(model, ngpu, gpu_memory_mib)
                if skip_reason:
                    print(f"SKIP PRECHECK {model} tp={ngpu} repeat={repeat_idx}: {skip_reason}", flush=True)
                    for cc in cc_list:
                        if (model, ngpu, cc, repeat_idx) in done_cases:
                            continue
                        num_prompts = max(args.min_prompts, cc * args.prompts_per_concurrency)
                        record_failure(
                            summary_path,
                            common,
                            cc,
                            num_prompts,
                            skip_reason,
                            Path(""),
                            Path(""),
                            None,
                            None,
                            deploy_seconds=0.0,
                        )
                    port += 1
                    continue

                run_log_dir = logs_root / run_id
                run_result_dir = results_root / run_id
                run_log_dir.mkdir(parents=True, exist_ok=True)
                run_result_dir.mkdir(parents=True, exist_ok=True)

                server_stdout = run_log_dir / "serve_stdout.log"
                server_stderr = run_log_dir / "serve_stderr.log"

                serve_cmd = [
                    "vllm",
                    "serve",
                    str(model_root / model),
                    "--host",
                    args.host,
                    "--port",
                    str(port),
                    "--tensor-parallel-size",
                    str(ngpu),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--dtype",
                    args.dtype,
                ]
                for extra_arg in args.extra_serve_arg:
                    serve_cmd.extend(extra_arg.split())

                server_proc = None
                deploy_started = time.time()
                print(f"START {model} tp={ngpu} repeat={repeat_idx} port={port}", flush=True)
                try:
                    with server_stdout.open("w") as out_handle, server_stderr.open("w") as err_handle:
                        server_proc = subprocess.Popen(
                            serve_cmd,
                            stdout=out_handle,
                            stderr=err_handle,
                            env=build_env(visible),
                            start_new_session=True,
                        )

                    ready = wait_for_server(args.host, port, args.server_timeout, server_proc)
                    deploy_seconds = time.time() - deploy_started
                    startup_metrics = parse_startup_metrics(server_stdout)
                    if not ready or server_proc.poll() is not None:
                        reason_parts = ["server_not_ready"]
                        stderr_tail = tail_text(server_stderr)
                        stdout_tail = tail_text(server_stdout)
                        if server_proc.poll() is not None:
                            reason_parts.append(f"exit_code={server_proc.returncode}")
                        if stderr_tail:
                            reason_parts.append(f"stderr_tail={stderr_tail}")
                        elif stdout_tail:
                            reason_parts.append(f"stdout_tail={stdout_tail}")
                        reason = " | ".join(reason_parts)
                        print(f"FAIL START {model} tp={ngpu} repeat={repeat_idx}: {reason}", flush=True)
                        for cc in cc_list:
                            if (model, ngpu, cc, repeat_idx) in done_cases:
                                continue
                            num_prompts = max(args.min_prompts, cc * args.prompts_per_concurrency)
                            record_failure(
                                summary_path,
                                common,
                                cc,
                                num_prompts,
                                reason,
                                server_stdout,
                                server_stderr,
                                None,
                                None,
                                deploy_seconds=deploy_seconds,
                                startup_metrics=startup_metrics,
                            )
                        port += 1
                        stop_process_tree(server_proc)
                        continue

                    print(
                        f"READY {model} tp={ngpu} repeat={repeat_idx} deploy={deploy_seconds:.1f}s",
                        flush=True,
                    )

                    for cc in cc_list:
                        if (model, ngpu, cc, repeat_idx) in done_cases:
                            print(f"  SKIP OK cc={cc} repeat={repeat_idx}", flush=True)
                            continue

                        num_prompts = max(args.min_prompts, cc * args.prompts_per_concurrency)
                        result_log = run_result_dir / f"cc{cc}_serve.log"
                        result_err = run_result_dir / f"cc{cc}_serve.err"
                        bench_cmd = [
                            "vllm",
                            "bench",
                            "serve",
                            "--backend",
                            "openai",
                            "--base-url",
                            f"http://{args.host}:{port}",
                            "--endpoint",
                            "/v1/completions",
                            "--model",
                            str(model_root / model),
                            "--dataset-name",
                            "random",
                            "--num-prompts",
                            str(num_prompts),
                            "--random-input-len",
                            str(args.input_len),
                            "--random-output-len",
                            str(args.output_len),
                            "--max-concurrency",
                            str(cc),
                        ]

                        print(f"  BENCH cc={cc} repeat={repeat_idx} prompts={num_prompts}", flush=True)
                        bench_started = time.time()
                        with result_log.open("w") as log_handle, result_err.open("w") as err_handle:
                            bench_proc = subprocess.run(
                                bench_cmd,
                                stdout=log_handle,
                                stderr=err_handle,
                                env=build_env(visible),
                                text=True,
                            )
                        bench_seconds = time.time() - bench_started

                        if bench_proc.returncode != 0:
                            reason = f"bench_exit_code={bench_proc.returncode}"
                            err_tail = tail_text(result_err)
                            if err_tail:
                                reason = f"{reason} | err_tail={err_tail}"
                            print(f"  FAIL cc={cc}: {reason}", flush=True)
                            record_failure(
                                summary_path,
                                common,
                                cc,
                                num_prompts,
                                reason,
                                server_stdout,
                                server_stderr,
                                result_log,
                                result_err,
                                deploy_seconds=deploy_seconds,
                                startup_metrics=startup_metrics,
                                bench_seconds=bench_seconds,
                            )
                            break

                        metrics = parse_bench_metrics(result_log)
                        row = dict(common)
                        row.update(metrics)
                        row.update(startup_metrics)
                        row.update(
                            {
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                "concurrency": cc,
                                "num_prompts": num_prompts,
                                "status": "ok",
                                "deploy_seconds": round(deploy_seconds, 3),
                                "bench_seconds": round(bench_seconds, 3),
                                "failure_reason": "",
                                "server_stdout": str(server_stdout),
                                "server_stderr": str(server_stderr),
                                "result_log": str(result_log),
                                "result_err": str(result_err),
                            }
                        )
                        total_tps = row.get("total_token_throughput")
                        output_tps = row.get("output_token_throughput")
                        row["tokens_per_gpu"] = round(total_tps / ngpu, 3) if total_tps else ""
                        row["output_tokens_per_gpu"] = round(output_tps / ngpu, 3) if output_tps else ""
                        append_row(summary_path, row)
                        print(
                            "  OK cc={cc} repeat={repeat} total_tps={total} ttft={ttft}".format(
                                cc=cc,
                                repeat=repeat_idx,
                                total=row.get("total_token_throughput"),
                                ttft=row.get("mean_ttft_ms"),
                            ),
                            flush=True,
                        )
                finally:
                    stop_process_tree(server_proc)
                    port += 1

    print(f"Done. Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
