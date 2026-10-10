"""Compare a bounded greedy diagnostic with the unmodified upstream simple example."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/reference"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
    config.update(temperature=0.0, presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.0)
    profile = OUT / "greedy-diagnostic.json"
    profile.write_text(json.dumps(config), encoding="utf-8")
    exe = ROOT / "build/bin/Release/wllm.exe"
    reference = ROOT / "build/bin/Release/llama-simple.exe"
    prompt = "What is 1+1? Answer with one digit."
    record = OUT / f"run-{time.time_ns()}.jsonl"
    inspect = OUT / "prompt.json"
    subprocess.run([str(exe), "--config", str(profile), "--prompt", prompt, "--inspect", str(inspect), "--record", str(record)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
    rendered = json.loads(inspect.read_text(encoding="utf-8"))["rendered_prompt"]
    with (OUT / "wllm.stderr.txt").open("wb") as log:
        ours = subprocess.run([str(exe), "--config", str(profile), "--prompt", prompt, "--max-tokens", "16", "--record", str(record)],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=log, check=True, timeout=120)
    with (OUT / "upstream.stderr.txt").open("wb") as log:
        other = subprocess.run([str(reference), "-m", config["model"], "-ngl", "0", "-n", "16", rendered],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=log, check=True, timeout=120)
    actual = ours.stdout.decode("utf-8").replace("\r\n", "\n")
    expected_full = other.stdout.decode("utf-8").replace("\r\n", "\n")
    if not expected_full.startswith(rendered):
        raise AssertionError("Reference prompt echo did not match inspected template")
    expected = expected_full[len(rendered):]
    if expected.endswith("\n"):
        expected = expected[:-1]  # upstream simple adds one terminal newline
    result = {
        "passed": actual == expected, "tokens_requested": 16, "prompt": prompt,
        "wllm_output": actual, "upstream_output": expected,
        "profile": "greedy, no penalties, diagnostic only; not a default-settings performance comparison",
        "upstream": "unmodified examples/simple at dependencies.json commit",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise AssertionError("Greedy diagnostic differs; inspect reports/reference/result.json")
    print("Greedy 16-token output equals unmodified upstream reference")


if __name__ == "__main__":
    main()
