"""Fail-closed CPU/Vulkan numerical screening before any performance comparison."""
from __future__ import annotations

from array import array
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from benchmark import ROOT, digest, read_json, render_prompt, save_json


def compare_logits(reference: array, candidate: array) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError("Logits vocabulary size mismatch")
    if not all(math.isfinite(value) for values in (reference, candidate) for value in values):
        raise ValueError("Non-finite logits")
    max_a, max_b = max(reference), max(candidate)
    log_a = max_a + math.log(sum(math.exp(x - max_a) for x in reference))
    log_b = max_b + math.log(sum(math.exp(x - max_b) for x in candidate))
    kl = max(0.0, sum(math.exp(x - log_a) * ((x - log_a) - (y - log_b)) for x, y in zip(reference, candidate)))
    return {"vocabulary_size": len(reference), "max_abs_delta": max(abs(x-y) for x, y in zip(reference, candidate)),
            "logits_close": all(math.isclose(x, y, abs_tol=1e-3, rel_tol=1e-4) for x, y in zip(reference, candidate)),
            "softmax_kl": kl, "kl_passed": kl <= 1e-6}


def execute(exe: Path, config: Path, prompt: str, folder: Path, layers: int, disable_f16: bool = False, context: int = 16384) -> tuple[dict, array, bytes]:
    folder.mkdir()
    prompt_file = folder / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    record, logits = folder / "run.jsonl", folder / "logits.f32"
    backend_log = ROOT / "reports/backend.log"
    offset = backend_log.stat().st_size if backend_log.exists() else 0
    command = [str(exe), "--config", str(config), "--prompt-file", str(prompt_file), "--gpu-layers", str(layers),
               "--threads", "4", "--threads-batch", "4", "--batch", "128", "--context", str(context),
               "--seed", "42", "--max-tokens", "32", "--timeout", "900", "--quiet-metrics",
               "--record", str(record), "--logits-file", str(logits), "--memory-diagnostic"]
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("GGML_VK_", "GGML_VULKAN_"))}
    if layers and disable_f16:
        environment["GGML_VK_DISABLE_F16"] = "1"
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=environment)
        try:
            code = process.wait(timeout=930)
        except BaseException:
            process.kill()
            process.wait()
            raise
    with backend_log.open("rb") as stream:
        stream.seek(offset)
        (folder / "backend.log").write_bytes(stream.read())
    events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    start, end = events[0], events[-1]
    if code or end["metrics"]["end_reason"] != "max_tokens" or len(end["token_ids"]) != 32:
        raise RuntimeError(f"Incomplete numerical diagnostic: {folder}")
    if start["config"]["gpu_layers"] != layers or start["config"]["context"] != context:
        raise RuntimeError("Effective placement/context mismatch")
    if layers:
        buffers = end["identity"]["allocation_observations"][0]["buffers"]
        if not any("Vulkan" in row["buffer_type"] and row["model_bytes"] > 0 for row in buffers):
            raise RuntimeError("No measured Vulkan model allocation; CPU fallback is not a GPU test")
    return {"config": start["config"], "metrics": end["metrics"], "identity": end["identity"],
            "backend_environment": {key: value for key, value in environment.items() if key.startswith(("GGML_VK_", "GGML_VULKAN_"))},
            "token_ids": end["token_ids"]}, array("f", logits.read_bytes()), (folder / "stdout.txt").read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/vulkan-probe/bin/Release/wllm-vulkan.exe")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--gpu-layers", type=int, default=2)
    parser.add_argument("--context", type=int, help="Default: context in the supplied config; no hidden 4K override")
    parser.add_argument("--disable-f16", action="store_true", help="Independent higher-compute-precision candidate; KV remains F16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gpu_layers < 1:
        parser.error("GPU layer count must be explicit and positive")
    output = (args.output or ROOT / "reports" / f"vulkan-correctness-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    context = config.get("context", 16384) if args.context is None else args.context
    if not 512 <= context <= 262144:
        parser.error("context must be 512..262144")
    model = (ROOT / config["model"]).resolve()
    model_stat = model.stat()
    config_hash = digest(args.config)
    report = {"status": "running", "exe_sha256": digest(args.exe), "model_sha256": digest(model),
              "context": context, "gpu_layers": args.gpu_layers,
              "disable_f16_compute": args.disable_f16,
              "policy": {"atol": 1e-3, "rtol": 1e-4, "max_kl": 1e-6,
                         "scope": "existing CPU ISA numerical threshold, fixed before GPU tests; not relaxed after results",
                         "timing_eligible": False}, "cases": []}
    save_json(output / "results.json", report)
    workloads = read_json(ROOT / "benchmarks/representative-workloads.json")
    cases = [("short", "你好，请简要解释质数。"), ("multi-batch", "word " * 260 + "What is 1+1?"),
             ("long", render_prompt(next(row for row in workloads if row["name"] == "long-prefill-fixed")))]
    try:
        for name, prompt in cases:
            values = [execute(args.exe.resolve(), args.config.resolve(), prompt, output / f"{name}-{backend}", layers, args.disable_f16, context)
                      for backend, layers in (("cpu", 0), ("vulkan", args.gpu_layers))]
            current = model.stat()
            if (current.st_size, current.st_mtime_ns) != (model_stat.st_size, model_stat.st_mtime_ns) or digest(args.exe) != report["exe_sha256"] or digest(args.config) != config_hash:
                raise RuntimeError("Executable/model/config changed during numerical comparison")
            if values[0][0]["metrics"]["input_tokens"] != values[1][0]["metrics"]["input_tokens"]:
                raise RuntimeError("CPU/GPU tokenization mismatch")
            row = {"name": name, **compare_logits(values[0][1], values[1][1]),
                   "tokens_equal": values[0][0]["token_ids"] == values[1][0]["token_ids"],
                   "output_equal": values[0][2] == values[1][2],
                   "input_tokens": values[0][0]["metrics"]["input_tokens"],
                   "runs": {backend: value[0] for backend, value in zip(("cpu", "vulkan"), values)}}
            row["numeric_passed"] = row["logits_close"] and row["kl_passed"]
            report["cases"].append(row)
            save_json(output / "results.json", report)
            print(json.dumps({key: value for key, value in row.items() if key != "runs"}), flush=True)
            if not row["numeric_passed"]:
                report["status"] = "numeric_rejected"
                save_json(output / "results.json", report)
                raise SystemExit(f"Numerical gate failed; no performance comparison: {output}")
        report["status"] = "numeric_passed_quality_and_lifecycle_pending"
        save_json(output / "results.json", report)
    except Exception as error:
        report.update(status="execution_failed", error=str(error))
        save_json(output / "results.json", report)
        raise


if __name__ == "__main__":
    main()
