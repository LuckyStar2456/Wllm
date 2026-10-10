"""Verify retained weights with isolated request state and explicit release."""
import json
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"


class SessionTests(unittest.TestCase):
    def test_resident_model_preserves_independent_requests(self):
        output = ROOT / "reports" / f"session-{time.time_ns()}"
        output.mkdir()
        prompts = ["你好，请简要解释质数。", "What is 1+1? Answer with one digit.", "你好，请简要解释质数。"]
        common = [str(EXE), "--config", str(ROOT / "configs/local.json"), "--seed", "42",
                  "--max-tokens", "32", "--timeout", "180", "--quiet-metrics"]
        summary = {"prompts": prompts, "rounds": []}
        # Three repetitions; alternate group order to reduce cache/order bias.
        for repeat in range(3):
            groups = {}
            for mode in (["isolated", "session"] if repeat % 2 == 0 else ["session", "isolated"]):
                ends = []
                if mode == "session":
                    record = output / f"{repeat}-session.jsonl"
                    with (output / f"{repeat}-session.stderr.txt").open("wb") as log:
                        run = subprocess.run(common + ["--interactive", "--record", str(record)],
                                             input=("\n".join(prompts) + "\n/exit\n").encode("utf-8"),
                                             cwd=ROOT, stdout=subprocess.PIPE, stderr=log, timeout=600, check=True)
                    events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
                    ends = [event for event in events if event["event"] == "end"]
                    self.assertEqual(events[-1]["event"], "session_end")
                    self.assertGreaterEqual(events[-1]["model_release_s"], 0)
                    self.assertEqual(len(ends), 3)
                    self.assertEqual([event["metrics"]["model_reused"] for event in ends], [False, True, True])
                    self.assertEqual(len({event["request_id"] for event in ends}), 3)
                    self.assertNotIn(b"[Wllm", run.stdout)
                else:
                    for index, prompt in enumerate(prompts):
                        record = output / f"{repeat}-{index}-isolated.jsonl"
                        with (output / f"{repeat}-{index}-isolated.stderr.txt").open("wb") as log:
                            subprocess.run(common + ["--prompt", prompt, "--record", str(record)], cwd=ROOT,
                                           stdout=subprocess.DEVNULL, stderr=log, timeout=210, check=True)
                        ends.append(json.loads(record.read_text(encoding="utf-8").splitlines()[-1]))
                groups[mode] = ends
            for a, b in zip(groups["isolated"], groups["session"]):
                self.assertEqual(a["token_ids"], b["token_ids"], "Request state leaked across prompts")
                self.assertEqual(a["metrics"]["input_tokens"], b["metrics"]["input_tokens"])
                self.assertEqual(b["metrics"]["input_reused"], 0)
                self.assertEqual(b["metrics"]["output_tokens"], 32)
            summary["rounds"].append({mode: [event["metrics"] for event in ends] for mode, ends in groups.items()})
            (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Session round {repeat + 1}: independent token sequences match", flush=True)
        print(f"Session report: {output}", flush=True)

    def test_failure_releases_session(self):
        output = ROOT / "reports" / f"session-error-{time.time_ns()}"
        output.mkdir()
        record = output / "run.jsonl"
        process = subprocess.run([
            str(EXE), "--config", str(ROOT / "configs/local.json"), "--interactive", "--context", "512",
            "--max-tokens", "1", "--record", str(record), "--quiet-metrics",
        ], input=("Hello\n" + "word " * 600 + "\n").encode(), cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        self.assertEqual(process.returncode, 1)
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        ends = [event for event in events if event["event"] == "end"]
        self.assertEqual(len(ends), 2)
        self.assertEqual(ends[-1]["metrics"]["end_reason"], "error")
        self.assertTrue(ends[-1]["metrics"]["model_reused"])
        self.assertEqual(events[-1]["event"], "session_end")

    def test_cancel_releases_resident_model(self):
        output = ROOT / "reports" / f"session-cancel-{time.time_ns()}"
        output.mkdir()
        record, marker = output / "run.jsonl", output / "cancel.flag"
        with (output / "stderr.txt").open("wb") as log:
            process = subprocess.Popen([
                str(EXE), "--config", str(ROOT / "configs/local.json"), "--interactive",
                "--max-tokens", "8", "--record", str(record), "--cancel-file", str(marker), "--quiet-metrics",
            ], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                process.stdin.write(b"Hello\nExplain prime numbers.\n")
                process.stdin.flush()
                deadline = time.monotonic() + 90
                while process.poll() is None and time.monotonic() < deadline:
                    events = []
                    if record.exists():
                        for line in record.read_text(encoding="utf-8").splitlines():
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    starts = [event for event in events if event["event"] == "start"]
                    if len(starts) == 2 and any(
                        event.get("request_id") == starts[-1]["request_id"] and
                        event.get("metrics", {}).get("output_tokens", 0) >= 2 for event in events
                    ):
                        marker.write_text("cancel", encoding="utf-8")
                        break
                    time.sleep(0.1)
                self.assertTrue(marker.exists(), "Second request did not reach generation")
                self.assertEqual(process.wait(timeout=30), 130)
            finally:
                process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        ends = [event for event in events if event["event"] == "end"]
        self.assertEqual(ends[-1]["metrics"]["end_reason"], "cancelled")
        self.assertTrue(ends[-1]["metrics"]["model_reused"])
        self.assertEqual(events[-1]["event"], "session_end")


if __name__ == "__main__":
    unittest.main(verbosity=2)
