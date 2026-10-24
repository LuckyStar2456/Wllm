"""Low-memory CPU operator screening; never substitutes for model inference acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TENSORS = ("blk.0.ffn_gate.weight", "blk.0.ssm_out.weight", "blk.3.attn_q.weight")
COLUMNS = (1, 2, 3, 7, 8, 128)


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def paired_summary(case: dict) -> dict:
    """Ratios use the same round; preserve each phase rather than combining throughput."""
    result = {}
    for name in case["profiles"]:
        if name == "upstream":
            continue
        ratios = [row["seconds"][name] / row["seconds"]["upstream"] for row in case["rounds"]]
        # Paired 5% screening boundary is fixed before running. This is no CI,
        # end-to-end quality result, or automatic adoption criterion.
        result[name] = {
            "kind": case["profiles"][name].get("kind", "operator_profile"),
            "candidate_applied": (case["pair_dispatch"] == "custom_pair"
                                  if name in {"pair_shared_unpack", "same_schedule_scalar"} else True),
            "median_latency_ratio": statistics.median(ratios),
            "min_latency_ratio": min(ratios), "max_latency_ratio": max(ratios),
            "rounds_better_by_5pct": sum(ratio <= 0.95 for ratio in ratios),
            "rounds_worse_by_5pct": sum(ratio >= 1.05 for ratio in ratios),
            "rounds": len(ratios),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/Release/wllm-cpu-kernel-bench.exe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 5 or args.repeats % 2 == 0 or args.repeats > 99:
        parser.error("Use an odd repeat count from 5 to 99")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    model = (ROOT / config["model"]).resolve()
    exe = args.exe.resolve()
    before = model.stat()
    output = (args.output or ROOT / "reports" / f"cpu-kernels-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    with exe.open("rb") as stream:
        executable_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    executable_before = exe.stat()
    summary = {
        "schema_version": 1, "status": "running", "executable_sha256": executable_hash,
        "scope": "real selected GGUF tensors with synthetic activations; isolated warm CPU operators",
        "inference_executed": False, "model_quality_evaluated": False,
        "engine_defaults_changed": False, "context_allocated": False,
        "gate": "finite, byte-equal full operator output before timing; fresh 1GiB system reserve per round",
        "minimum_screening_improvement_fraction": 0.05,
        "order": "fixed tensor/column cases; each case alternates all profiles forward/reverse by round",
        "repeats": args.repeats, "cases": [],
        "weight_identity_scope": "selected tensor SHA256; file size/mtime checked throughout, not a full-file digest",
    }
    save(output / "summary.json", summary)
    try:
        for tensor in TENSORS:
            for columns in COLUMNS:
                current = model.stat()
                if (current.st_size, current.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise RuntimeError("GGUF file changed during the experiment")
                label = f"{tensor}-n{columns}"
                destination = output / f"{label}.json"
                with (output / f"{label}.stdout.txt").open("wb") as stdout, (output / f"{label}.stderr.txt").open("wb") as stderr:
                    process = subprocess.run([
                        str(exe), "--model", str(model), "--tensor", tensor, "--columns", str(columns),
                        "--threads", "4", "--thread-sweep", "--repeats", str(args.repeats),
                        "--output", str(destination),
                    ], cwd=ROOT, stdout=stdout, stderr=stderr, timeout=180, check=False)
                result = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {}
                current, executable_current = model.stat(), exe.stat()
                if (current.st_size, current.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise RuntimeError("GGUF file changed during an operator case")
                if (executable_current.st_size, executable_current.st_mtime_ns) != (executable_before.st_size, executable_before.st_mtime_ns):
                    raise RuntimeError("CPU experiment executable changed during screening")
                row = {"tensor": tensor, "columns": columns, "exit_code": process.returncode, "report": str(destination),
                       "status": result.get("status", "missing_report")}
                summary["cases"].append(row)
                if process.returncode or result.get("status") != "passed":
                    raise RuntimeError(f"Operator case failed: {label}: {result.get('error', '')}")
                case = result["cases"][0]
                row.update(tensor_sha256=result["tensor"]["sha256"], measurements=case,
                           paired_comparisons=paired_summary(case))
                save(output / "summary.json", summary)
                print(f"{label}: byte-equal, paired rounds={args.repeats}", flush=True)
        summary["status"] = "operator_screening_completed"
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        raise
    finally:
        save(output / "summary.json", summary)


if __name__ == "__main__":
    main()
