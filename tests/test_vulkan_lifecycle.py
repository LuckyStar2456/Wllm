"""On-device experimental Vulkan cancellation and fail-closed entry checks; no timing acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def events(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gpu = ROOT / "build/vulkan-probe/bin/Release/wllm-vulkan.exe"
    cpu = ROOT / "build/bin/Release/wllm.exe"
    config = ROOT / "configs/local.json"
    report = {"scope": "lifecycle and explicit rejection only; numerical candidate remains closed", "checks": {}}
    rejected = subprocess.run([str(cpu), "--config", str(config), "--gpu-layers", "2", "--prompt", "test"],
                              cwd=ROOT, capture_output=True, timeout=20)
    assert rejected.returncode == 1 and b"not compiled" in rejected.stderr
    report["checks"]["cpu_build_rejects_gpu_no_fallback"] = True
    rejected = subprocess.run([str(gpu), "--config", str(config), "--gpu-layers", "1024",
                               "--resources", str(output / "invalid.json")], cwd=ROOT, capture_output=True, timeout=20)
    assert rejected.returncode == 1 and b"exceed this model" in rejected.stderr and not (output / "invalid.json").exists()
    report["checks"]["layer_count_cannot_be_implicitly_clamped"] = True
    for name in ("timeout", "cancel"):
        record = output / f"{name}.jsonl"
        cancel_file = output / "cancel.signal"
        command = [str(gpu), "--config", str(config), "--gpu-layers", "2", "--prompt", "你好，请简要解释质数。",
                   "--max-tokens", "128", "--seed", "42", "--record", str(record), "--quiet-metrics", "--memory-diagnostic"]
        command += ["--timeout", "0.01"] if name == "timeout" else ["--cancel-file", str(cancel_file), "--timeout", "60"]
        with (output / f"{name}.stdout.txt").open("wb") as stdout, (output / f"{name}.stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            try:
                if name == "cancel":
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline and process.poll() is None:
                        rows = events(record)
                        if rows and rows[-1].get("metrics", {}).get("output_tokens", 0) > 0:
                            cancel_file.touch()
                            break
                        time.sleep(0.1)
                    else:
                        raise AssertionError("GPU did not reach generation for cancellation check")
                code = process.wait(timeout=70)
            except BaseException:
                process.kill()
                process.wait()
                raise
        end = events(record)[-1]
        reason, expected = ("timeout", 124) if name == "timeout" else ("cancelled", 130)
        assert code == expected and end["event"] == "end" and end["metrics"]["end_reason"] == reason
        if name == "cancel":
            assert 0 < end["metrics"]["output_tokens"] < 128
            assert end["identity"]["allocation_observations"][-1]["phase"] == "before_context_release"
        report["checks"][f"{name}_exits_and_records_end"] = True
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
