"""Sequential paired CPU experiments. Raw prompts, tokens and paths stay in reports/."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TUNABLES = {"threads", "threads_batch", "batch", "persistent_threadpool", "prefill_last_only", "cpu_weight_repacking"}
METRICS = ("load_s", "prefill_s", "ttft_s", "decode_average_tps", "elapsed_s", "sampled_peak_working_set_bytes")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Never mix failed or partial requests into successful timing aggregates."""
    result: dict[str, Any] = {}
    for run in runs:
        key = f"{run['case']}/{run['profile']}"
        group = result.setdefault(key, {"samples": 0, "failures": 0, "metrics": {}})
        if not run["ok"] or run.get("diagnostic", False):
            group["failures"] += 1
            continue
        group["samples"] += 1
        for metric in METRICS:
            value = run["metrics"].get(metric)
            if value is not None:
                group["metrics"].setdefault(metric, []).append(value)
    for group in result.values():
        group["metrics"] = {
            key: {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
            for key, values in group["metrics"].items()
        }
    return result


def run_once(exe: Path, config: Path, case: dict[str, Any], folder: Path, timeout: float) -> dict[str, Any]:
    folder.mkdir()
    prompt = case.get("prefix", "") + case.get("repeat_text", "") * case.get("repeat", 0) + case["prompt"]
    prompt_path = folder / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    record = folder / "run.jsonl"
    command = [str(exe), "--config", str(config), "--prompt-file", str(prompt_path),
               "--record", str(record), "--max-tokens", str(case["max_tokens"]),
               "--seed", str(case.get("seed", 42)), "--timeout", str(timeout), "--quiet-metrics"]
    started = time.perf_counter()
    with (folder / "stdout.txt").open("wb") as output, (folder / "stderr.txt").open("wb") as error:
        # Communicate with timeout, and reap our own process on failure or user cancellation.
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=error)
        try:
            code = process.wait(timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = -1
        except BaseException:
            process.kill()
            process.wait()
            raise
    events = []
    if record.exists():
        for line in record.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    end = next((event for event in reversed(events) if event.get("event") == "end"), {})
    start = next((event for event in events if event.get("event") == "start"), {})
    metrics = end.get("metrics", {})
    expected_reason = case.get("end_reason", "max_tokens")
    diagnostic = bool(start.get("profiling_diagnostic") or start.get("logits_diagnostic"))
    ok = code == 0 and metrics.get("end_reason") == expected_reason and not diagnostic
    if expected_reason == "max_tokens":
        ok = ok and metrics.get("output_tokens") == case["max_tokens"]
    output_text = (folder / "stdout.txt").read_text(encoding="utf-8")
    answer_ok = None
    if "expected_answer" in case:
        answer_ok = output_text.split("</think>")[-1].strip() == case["expected_answer"]
        ok = ok and answer_ok
    return {"ok": ok, "diagnostic": diagnostic, "exit_code": code, "wall_s": time.perf_counter() - started,
            "metrics": metrics, "effective_config": start.get("config"), "identity": end.get("identity"),
            "token_ids": end.get("token_ids", []), "answer_ok": answer_ok,
            "prompt_sha256": digest(prompt_path), "record": str(record)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--profiles", type=Path, default=ROOT / "benchmarks/cpu-profiles.json")
    parser.add_argument("--workloads", type=Path, default=ROOT / "benchmarks/workloads.json")
    parser.add_argument("--cases", nargs="+", help="Subset of workload names")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, help="New output directory; existing results are never overwritten")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    profiles = read_json(args.profiles)
    workloads = read_json(args.workloads)
    if args.cases:
        if set(args.cases) - {case["name"] for case in workloads}:
            parser.error("Unknown workload name")
        workloads = [case for case in workloads if case["name"] in args.cases]
    if not profiles or not workloads:
        parser.error("At least one profile and workload are required")
    for entries in (profiles, workloads):
        names = [entry["name"] for entry in entries]
        if len(names) != len(set(names)) or any(not name or not all(c.isalnum() or c in "-_" for c in name) for name in names):
            parser.error("Names must be unique and contain only letters, digits, hyphens or underscores")
    base = read_json(args.config)
    model = (ROOT / base["model"]).resolve()
    output = (args.output or ROOT / "reports" / f"cpu-benchmark-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = []
    for profile in profiles:
        changes = profile.get("settings", {})
        if set(changes) - TUNABLES:
            parser.error("CPU profiles may change only execution controls; keep model/context/sampling fixed")
        config = {**base, **changes}
        config["model"] = str(model)
        path = output / f"{profile['name']}.json"
        save_json(path, config)
        exe = (ROOT / profile.get("exe", "build/bin/Release/wllm.exe")).resolve()
        prepared.append((profile["name"], exe, path))
    summary: dict[str, Any] = {
        "schema_version": 1, "model_sha256": digest(model), "base_config": base,
        "profiles": profiles, "workloads": workloads, "repeats": args.repeats,
        "executable_sha256": {name: digest(exe) for name, exe, _ in prepared},
        "order": "forward/reverse alternation by repeat, sequential processes",
        "cache": "OS file cache uncontrolled; fresh process per request, no prefix reuse",
        "acceptance_policy": {"latency_regression_fraction": 0.05, "throughput_regression_fraction": 0.05,
                              "minimum_improvement_fraction": 0.05,
                              "note": "Screening only. Token equality and task quality checked separately; no automatic promotion."},
        "warmups": [], "runs": [],
    }
    save_json(output / "summary.json", summary)
    if not args.no_warmup:
        for name, exe, config in prepared:
            warmup = {"prompt": "Hello", "max_tokens": 2}
            result = run_once(exe, config, warmup, output / f"warmup-{name}", args.timeout)
            summary["warmups"].append({"profile": name, **result})
            save_json(output / "summary.json", summary)
            if not result["ok"]:
                raise SystemExit(f"Warmup failed: {name}; see {output}")
    for repeat in range(args.repeats):
        order = prepared if repeat % 2 == 0 else list(reversed(prepared))
        for case in workloads:
            paired = []
            for name, exe, config in order:
                folder = output / f"r{repeat + 1}-{case['name']}-{name}"
                result = run_once(exe, config, case, folder, args.timeout)
                result.update(profile=name, case=case["name"], repeat=repeat + 1)
                summary["runs"].append(result)
                paired.append(result)
                summary["aggregates"] = summarize(summary["runs"])
                save_json(output / "summary.json", summary)
                metrics = result["metrics"]
                print(f"r{repeat + 1} {case['name']}/{name}: ok={result['ok']} "
                      f"pp={metrics.get('prefill_s')}s ttft={metrics.get('ttft_s')}s "
                      f"decode={metrics.get('decode_average_tps')} tok/s", flush=True)
            reference = next(run for run in paired if run["profile"] == prepared[0][0])
            for result in paired:
                result["tokens_equal_reference"] = (
                    result["token_ids"] == reference["token_ids"] if result["ok"] and reference["ok"] else None
                )
            save_json(output / "summary.json", summary)
    print(f"Report: {output / 'summary.json'}", flush=True)
    if any(not run["ok"] for run in summary["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
