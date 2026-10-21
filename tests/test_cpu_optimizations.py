"""Model-backed regression of CPU execution changes at unchanged model/sampling/context."""
from array import array
import json
import math
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"


class CpuOptimizationTests(unittest.TestCase):
    def test_logits_tokens_and_recurrent_state(self):
        output = ROOT / "reports" / f"cpu-correctness-{time.time_ns()}"
        output.mkdir()
        # Fix acceptance tolerance before evaluating the changed execution paths.
        report = {"absolute_tolerance": 1e-5, "relative_tolerance": 1e-5, "results": []}
        prompts = ["你好，请简要解释质数。", "word " * 260 + "What is 1+1?"]
        try:
            for index, prompt in enumerate(prompts):
                reference = None
                for name, pool, last, batch_threads in [
                    ("legacy", "off", "off", "4"), ("pool", "on", "off", "4"),
                    ("last", "off", "on", "4"), ("combined", "on", "on", "4"),
                    ("split-threads", "on", "on", "2"),
                ]:
                    record = output / f"{index}-{name}.jsonl"
                    logits_file = output / f"{index}-{name}.f32"
                    with (output / f"{index}-{name}.stderr.txt").open("wb") as log:
                        process = subprocess.run([
                            str(EXE), "--config", str(ROOT / "configs/local.json"), "--prompt", prompt,
                            "--threads", "4", "--threads-batch", batch_threads,
                            "--batch", "128", "--persistent-threadpool", pool, "--prefill-last-only", last,
                            "--seed", "42", "--max-tokens", "16", "--timeout", "240", "--quiet-metrics",
                            "--record", str(record), "--logits-file", str(logits_file),
                        ], cwd=ROOT, stdout=subprocess.PIPE, stderr=log, timeout=270, check=True)
                    end = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
                    logits = array("f", logits_file.read_bytes())
                    self.assertTrue(all(math.isfinite(value) for value in logits))
                    if reference is None:
                        reference = (logits, end["token_ids"], process.stdout)
                    self.assertEqual(len(logits), len(reference[0]))
                    delta = max(abs(a - b) for a, b in zip(logits, reference[0]))
                    close = all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(logits, reference[0]))
                    equal = end["token_ids"] == reference[1]
                    report["results"].append({"prompt_index": index, "profile": name, "vocabulary_size": len(logits),
                                              "max_abs_logit_delta": delta, "logits_close": close, "tokens_equal": equal})
                    self.assertTrue(close, (name, delta))
                    self.assertTrue(equal, name)
                    self.assertEqual(process.stdout, reference[2])
                    self.assertEqual(end["metrics"]["end_reason"], "max_tokens")
                    count = end["metrics"]["prefill_batches"]
                    self.assertEqual(end["metrics"]["prefill_logits_batches"], 1 if last == "on" else count)
                    if index == 1:
                        self.assertGreaterEqual(count, 3)
        finally:
            (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"CPU correctness report: {output}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
