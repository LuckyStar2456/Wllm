"""Observe bounded CPU memory diagnostics, never mix their timings into performance samples."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from benchmark import ROOT, digest, read_json, render_prompt, save_json
from memory_calibration import CONFIG_KEYS, config_signature, validate_binding
from resource_ledger import capture_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/Release/wllm.exe")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--workloads", type=Path, default=ROOT / "benchmarks/representative-workloads.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / "reports" / f"memory-calibration-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    native = capture_native(args.exe, args.config, output / "native.json")
    if native["config"]["backend"] != "CPU" or native["config"].get("gpu_layers", 0) != 0:
        parser.error("This calibration observes CPU only; GPU needs separate driver/device accounting")
    config = read_json(args.config)
    config["model"] = str((ROOT / config["model"]).resolve())
    model = Path(config["model"])
    model_stat = model.stat()
    calibration = {"schema_version": 1, "status": "incomplete", "scope": "memory diagnostics only; no timing/quality acceptance",
        "binding": {"executable_sha256": digest(args.exe), "model_sha256": digest(model),
                    "workloads_sha256": digest(args.workloads), "config_signature": config_signature(native["config"]),
                    "memory_config": {key: native["config"][key] for key in CONFIG_KEYS},
                    "context": native["config"]["context"], "llama_commit": native["llama_commit"],
                    "cpu_build": native["cpu_build"], "cpu_brand": native["cpu"]["brand"],
                    "physical_total_bytes": native["system"]["physical_total_bytes"]}, "samples": []}
    if native["config"]["cpu_weight_repacking"]:
        parser.error("Calibration requires accepted CPU layout; repacking must be off")
    save_json(output / "config.json", config)
    cases = {case["name"]: case for case in read_json(args.workloads)}
    selections = [("chat", "chat-explanation", 32), ("code", "code-generation", 128), ("long", "long-prefill-fixed", 32)]
    for category, name, tokens in selections:
        case = cases[name]
        folder = output / name
        folder.mkdir()
        prompt = folder / "prompt.txt"
        prompt.write_text(render_prompt(case), encoding="utf-8")
        record = folder / "run.jsonl"
        command = [str(args.exe.resolve()), "--config", str(output / "config.json"), "--prompt-file", str(prompt),
                   "--record", str(record), "--seed", "42", "--max-tokens", str(tokens), "--timeout", "900",
                   "--quiet-metrics", "--memory-diagnostic"]
        with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=930)
            except BaseException:
                process.kill(); process.wait()
                raise
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        start = next(row for row in events if row["event"] == "start")
        end = next(row for row in reversed(events) if row["event"] == "end")
        metrics = end["metrics"]
        observations = end["identity"].get("allocation_observations", [])
        if code or metrics["end_reason"] not in {"max_tokens", "eos"} or len(observations) != 3 or not start.get("memory_diagnostic"):
            raise RuntimeError(f"Calibration diagnostic failed: {name}; preserve {output}")
        row = {"category": category, "name": name, "input_tokens": metrics["input_tokens"],
               "output_tokens": metrics["output_tokens"], "end_reason": metrics["end_reason"],
               "config_signature": config_signature(start["config"]), "prompt_sha256": digest(prompt),
               "process_peak_private_bytes": metrics["process_peak_private_bytes"],
               "process_peak_working_set_bytes": metrics["process_peak_working_set_bytes"], "allocations": observations}
        calibration["samples"].append(row)
        save_json(output / "calibration.json", calibration)
        print(json.dumps({key: row[key] for key in ("name", "input_tokens", "output_tokens", "process_peak_private_bytes", "process_peak_working_set_bytes")}), flush=True)
    current = model.stat()
    if (current.st_size, current.st_mtime_ns) != (model_stat.st_size, model_stat.st_mtime_ns) or digest(args.exe) != calibration["binding"]["executable_sha256"]:
        raise RuntimeError("Artifacts changed during calibration")
    peak = max(row["process_peak_private_bytes"] for row in calibration["samples"])
    calibration.update(status="calibrated", peak_private_bytes=peak, margin_bytes=max(256*1024**2, (peak+3)//4))
    validate_binding(calibration, calibration["binding"]["executable_sha256"], calibration["binding"]["model_sha256"], calibration["binding"]["workloads_sha256"])
    save_json(output / "calibration.json", calibration)
    print(f"Calibration: {output / 'calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
