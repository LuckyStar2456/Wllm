"""Check authored workload token lengths using the current template; no weight inference or performance claim."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from benchmark import ROOT, digest, read_json, render_prompt, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/Release/wllm.exe")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--workloads", type=Path, default=ROOT / "benchmarks/representative-workloads.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = read_json(args.workloads)
    names = [case["name"] for case in cases]
    if not names or len(names) != len(set(names)) or any(not name or not all(c.isalnum() or c in "-_" for c in name) for name in names):
        parser.error("Workload names must be nonempty, unique and contain only letters, digits, hyphens or underscores")
    output = (args.output or ROOT / "reports" / f"workload-inspection-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    config["model"] = str((ROOT / config["model"]).resolve())
    profile = output / "config.json"
    save_json(profile, config)
    summary = {"scope": "vocabulary/template inspection only; no generation, task quality or performance acceptance",
               "executable_sha256": digest(args.exe), "workloads_sha256": digest(args.workloads),
               "config": config, "cases": []}
    for case in cases:
        folder = output / case["name"]
        folder.mkdir()
        prompt = folder / "prompt.txt"
        prompt.write_text(render_prompt(case), encoding="utf-8")
        inspected = folder / "inspect.json"
        with (folder / "stderr.txt").open("wb") as error:
            subprocess.run([str(args.exe.resolve()), "--config", str(profile), "--prompt-file", str(prompt),
                "--inspect", str(inspected), "--record", str(folder / "run.jsonl"), "--quiet-metrics"],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=error, timeout=60, check=True)
        data = read_json(inspected)
        tokens = len(data["token_ids"])
        row = {"name": case["name"], "category": case.get("category"), "source": case.get("source"),
               "input_tokens": tokens, "context": config["context"], "max_output_tokens": case["max_tokens"],
               "fits_full_output_budget": tokens + case["max_tokens"] <= config["context"],
               "prompt_sha256": digest(prompt), "quality_rubric": case.get("quality_rubric", []), "quality_status": "not_evaluated"}
        summary["cases"].append(row)
        save_json(output / "summary.json", summary)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    if not all(case["fits_full_output_budget"] for case in summary["cases"]):
        raise SystemExit("Some workloads exceed the fixed context plus output budget; settings were not reduced")
    print(f"Report: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
