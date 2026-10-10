"""On-device checks that diagnostics preserve output and report actual work."""
from array import array
import json
import math
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_profile_timeout_keeps_partial_record(self):
        folder = ROOT / "reports" / f"profile-timeout-{time.time_ns()}"
        folder.mkdir()
        record = folder / "run.jsonl"
        with (folder / "stderr.txt").open("wb") as error:
            process = subprocess.run([
                str(ROOT / "build/bin/Release/wllm.exe"), "--config", "configs/local.json",
                "--prompt", "Explain prime numbers.", "--profile-ops", "--max-tokens", "128",
                "--timeout", "3", "--record", str(record), "--quiet-metrics",
            ], cwd=ROOT, stdout=subprocess.PIPE, stderr=error, timeout=40)
        self.assertEqual(process.returncode, 124)
        end = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(end["metrics"]["end_reason"], "timeout")
        self.assertTrue(end["profile"]["diagnostic"])
        self.assertTrue(end["profile"]["operator_callbacks"])
        self.assertTrue(all(row["seconds"] >= 0 for row in end["profile"]["stages"]))
        print(f"Profile timeout check: {folder}", flush=True)

    def test_resident_profile_resets_and_records_failure(self):
        folder = ROOT / "reports" / f"profile-session-{time.time_ns()}"
        folder.mkdir()
        record = folder / "run.jsonl"
        command = [str(ROOT / "build/bin/Release/wllm.exe"), "--config", "configs/local.json",
                   "--interactive", "--profile-ops", "--context", "512", "--max-tokens", "2",
                   "--seed", "42", "--timeout", "60", "--quiet-metrics", "--record", str(record)]
        with (folder / "stderr.txt").open("wb") as error:
            process = subprocess.run(command, input=("Hello\nHello\n" + "word " * 600 + "\n").encode(),
                                     cwd=ROOT, stdout=subprocess.PIPE, stderr=error, timeout=150)
        self.assertEqual(process.returncode, 1)
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        ends = [event for event in events if event["event"] == "end"]
        self.assertEqual(len(ends), 3)
        self.assertEqual(ends[0]["token_ids"], ends[1]["token_ids"])
        for event in ends[:2]:
            stages = {row["name"]: row for row in event["profile"]["stages"]}
            self.assertEqual(stages["sample"]["calls"], 2)
            self.assertEqual(stages["decode_process"]["calls"], 1)
        self.assertEqual(ends[-1]["metrics"]["end_reason"], "error")
        self.assertEqual(ends[-1]["profile"]["stages"], [])
        self.assertEqual(ends[-1]["profile"]["operators"], [])
        self.assertEqual(events[-1]["event"], "session_end")
        print(f"Profile session check: {folder}", flush=True)

    def test_diagnostic_equivalence(self):
        folder = ROOT / "reports" / f"profile-check-{time.time_ns()}"
        folder.mkdir()
        baseline = None
        report = {"absolute_tolerance": 1e-5, "relative_tolerance": 1e-5, "runs": []}
        try:
            for mode in ("plain", "profile", "profile-ops"):
                record, logits_path = folder / f"{mode}.jsonl", folder / f"{mode}.f32"
                command = [str(ROOT / "build/bin/Release/wllm.exe"), "--config", "configs/local.json",
                           "--prompt", "Explain why the sky is blue in simple terms.", "--seed", "42",
                           "--max-tokens", "16", "--timeout", "180", "--quiet-metrics",
                           "--record", str(record), "--logits-file", str(logits_path)]
                if mode != "plain":
                    command.append(f"--{mode}")
                with (folder / f"{mode}.stderr.txt").open("wb") as error:
                    process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                             stderr=error, timeout=210, check=True)
                events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
                end = events[-1]
                logits = array("f", logits_path.read_bytes())
                self.assertTrue(all(math.isfinite(value) for value in logits))
                if baseline is None:
                    baseline = (logits, end["token_ids"], process.stdout)
                self.assertEqual(len(logits), len(baseline[0]))
                delta = max(abs(a - b) for a, b in zip(logits, baseline[0]))
                close = all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                            for a, b in zip(logits, baseline[0]))
                report["runs"].append({"mode": mode, "max_abs_delta": delta, "logits_close": close,
                                       "tokens_equal": end["token_ids"] == baseline[1], "profile": end["profile"]})
                self.assertTrue(close)
                self.assertEqual(end["token_ids"], baseline[1])
                self.assertEqual(process.stdout, baseline[2])
                self.assertEqual(end["metrics"]["end_reason"], "max_tokens")
                self.assertEqual(events[0]["profiling_diagnostic"], mode != "plain")
                if mode == "plain":
                    self.assertIsNone(end["profile"])
                    continue
                stages = {row["name"]: row for row in end["profile"]["stages"]}
                self.assertEqual(stages["decode_process"]["calls"], 15)
                self.assertEqual(stages["sample"]["calls"], 16)
                self.assertEqual(stages["render_and_write"]["calls"], 16)
                self.assertEqual(stages["prefill_process"]["calls"], end["metrics"]["prefill_batches"])
                self.assertTrue(all(row["seconds"] > 0 for row in stages.values()))
                if mode == "profile-ops":
                    nodes = end["profile"]["operators"]
                    self.assertTrue(any(row["name"].startswith("decode/MUL_MAT/") for row in nodes))
                    self.assertTrue(any(row["name"].startswith("prefill/MUL_MAT/") for row in nodes))
                    self.assertLessEqual(sum(row["seconds"] for row in nodes),
                                         stages["decode_process"]["seconds"] + stages["prefill_process"]["seconds"])
                else:
                    self.assertEqual(end["profile"]["operators"], [])
        finally:
            (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Profile check: {folder}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
