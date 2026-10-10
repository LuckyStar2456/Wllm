"""Compare CPU instruction/layout paths with full-vocabulary numerical checks."""
from array import array
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/bin/Release/wllm.exe")
    parser.add_argument("--candidate-repacking", action="store_true")
    args = parser.parse_args()
    output = ROOT / "reports" / f"cpu-build-correctness-{time.time_ns()}"
    output.mkdir()
    # Quantized integer products are unchanged; float reduction order may differ by ISA.
    report = {"atol": 1e-3, "rtol": 1e-4, "max_kl": 1e-6,
              "candidate_repacking": args.candidate_repacking, "results": []}
    prompts = ["你好，请简要解释质数。", "word " * 260 + "What is 1+1?"]
    for index, prompt in enumerate(prompts):
        values = []
        for name, exe in [("baseline", args.baseline.resolve()), ("candidate", args.candidate.resolve())]:
            record, logits_file = output / f"{index}-{name}.jsonl", output / f"{index}-{name}.f32"
            with (output / f"{index}-{name}.stderr.txt").open("wb") as log:
                result = subprocess.run([
                    str(exe), "--config", str(ROOT / "configs/local.json"), "--prompt", prompt,
                    "--threads", "4", "--threads-batch", "4", "--batch", "128",
                    "--persistent-threadpool", "off", "--prefill-last-only", "off",
                    "--seed", "42", "--max-tokens", "32", "--timeout", "300", "--quiet-metrics",
                    "--record", str(record), "--logits-file", str(logits_file),
                    *(["--cpu-weight-repacking", "on"] if name == "candidate" and args.candidate_repacking else []),
                ], cwd=ROOT, stdout=subprocess.PIPE, stderr=log, timeout=330, check=True)
            end = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
            if end["metrics"]["end_reason"] != "max_tokens" or len(end["token_ids"]) != 32:
                raise AssertionError("CPU build comparison did not complete 32 tokens")
            values.append((array("f", logits_file.read_bytes()), end, result.stdout))
        a, b = values[0][0], values[1][0]
        if len(a) != len(b) or not all(math.isfinite(v) for v in [*a, *b]):
            raise AssertionError("Invalid vocabulary logits")
        close = all(math.isclose(x, y, rel_tol=report["rtol"], abs_tol=report["atol"]) for x, y in zip(a, b))
        # Compute stable softmax KL before any top-k, temperature or penalties.
        max_a, max_b = max(a), max(b)
        log_z_a = max_a + math.log(sum(math.exp(x - max_a) for x in a))
        log_z_b = max_b + math.log(sum(math.exp(x - max_b) for x in b))
        kl = max(0.0, sum(math.exp(x - log_z_a) * ((x - log_z_a) - (y - log_z_b)) for x, y in zip(a, b)))
        equal = values[0][1]["token_ids"] == values[1][1]["token_ids"]
        row = {"prompt_index": index, "max_abs_logit_delta": max(abs(x - y) for x, y in zip(a, b)),
               "logits_close": close, "softmax_kl": kl, "tokens_equal": equal,
               "output_equal": values[0][2] == values[1][2], "vocabulary_size": len(a)}
        report["results"].append(row)
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(row, flush=True)
        if not close or kl > report["max_kl"]:
            raise AssertionError(f"CPU numerical gate failed: {output}")
        # Equality is a diagnostic; quality/complete-answer checks remain separate.
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
