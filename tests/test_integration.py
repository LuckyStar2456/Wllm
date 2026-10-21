"""Bounded on-device acceptance checks; run only after preparing the local model."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"
REPORTS = ROOT / "reports/acceptance"
CONFIG = ROOT / "configs/local.json"


def events(path: Path) -> list[dict]:
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may be between write and flush.
            continue
    return result


class PrototypeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        REPORTS.mkdir(parents=True, exist_ok=True)
        previous = REPORTS / "results.json"
        cls.results: dict[str, dict] = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}

    @classmethod
    def tearDownClass(cls) -> None:
        (REPORTS / "results.json").write_text(json.dumps(cls.results, indent=2), encoding="utf-8")

    def run_case(self, name: str, args: list[str], *, expected: int = 0) -> tuple[str, dict]:
        record = REPORTS / f"{name}-{time.time_ns()}.jsonl"
        with (REPORTS / f"{name}.out.txt").open("wb") as output, (REPORTS / f"{name}.metrics.txt").open("wb") as metrics:
            process = subprocess.run(
                [str(EXE), "--config", str(CONFIG), "--record", str(record), *args],
                cwd=ROOT, stdout=output, stderr=metrics, timeout=300, check=False,
            )
        data = events(record)
        self.assertEqual(process.returncode, expected, (REPORTS / f"{name}.metrics.txt").read_text(encoding="utf-8")[-1500:])
        self.assertEqual(data[-1]["event"], "end")
        summary = data[-1]
        self.results[name] = {"exit_code": process.returncode, "record": str(record), "metrics": summary["metrics"]}
        self.assertTrue(any(item["event"] == "sample" for item in data))
        self.assertIsNotNone(summary["metrics"]["working_set_bytes"])
        self.assertEqual(len(summary["token_times_s"]), summary["metrics"]["output_tokens"])
        if len(summary["token_times_s"]) >= 2:
            times = summary["token_times_s"]
            self.assertAlmostEqual(summary["metrics"]["decode_average_tps"], (len(times) - 1) / (times[-1] - times[0]))
        return (REPORTS / f"{name}.out.txt").read_text(encoding="utf-8"), summary

    def test_01_template_and_unicode(self) -> None:
        inspection = REPORTS / "inspect.json"
        prompt = "你好，请计算 1+1。"
        _, summary = self.run_case("template", ["--prompt", prompt, "--inspect", str(inspection)])
        data = json.loads(inspection.read_text(encoding="utf-8"))
        self.assertEqual(data["rendered_prompt"], f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        self.assertEqual(data["config"]["context"], 16384)
        self.assertEqual(summary["metrics"]["end_reason"], "inspected")

    def test_02_default_thinking_completion(self) -> None:
        output, summary = self.run_case("completion", ["--prompt", "What is 1+1? Answer with one digit.", "--seed", "42", "--max-tokens", "512", "--timeout", "240"])
        self.assertEqual(summary["metrics"]["end_reason"], "eos")
        self.assertIn("</think>", output)
        self.assertEqual(output.split("</think>")[-1].strip(), "2")
        self.assertNotIn("[Wllm", output)
        self.assertGreater(summary["metrics"]["prefill_tps"], 0)
        self.assertEqual(summary["metrics"]["input_processed"], summary["metrics"]["input_tokens"])

    def test_03_single_token(self) -> None:
        _, summary = self.run_case("one-token", ["--prompt", "Hello", "--max-tokens", "1"])
        self.assertEqual(summary["metrics"]["output_tokens"], 1)
        self.assertIsNone(summary["metrics"]["decode_average_tps"])
        self.assertEqual(summary["metrics"]["end_reason"], "max_tokens")

    def test_04_context_and_memory_failures(self) -> None:
        _, summary = self.run_case("context-full", ["--context", "512", "--prompt", "hello " * 600], expected=1)
        self.assertIn("no generation room", summary["metrics"]["error"])
        self.assertEqual(summary["metrics"]["output_tokens"], 0)
        _, summary = self.run_case("memory-budget", ["--context", "262144", "--prompt", "Hello"], expected=1)
        self.assertIn("No automatic", summary["metrics"]["error"])
        self.assertEqual(summary["metrics"]["output_tokens"], 0)

    def test_05_timeout(self) -> None:
        _, summary = self.run_case("timeout", ["--prompt", "Hello", "--timeout", "0.001"], expected=124)
        self.assertEqual(summary["metrics"]["end_reason"], "timeout")

    def test_06_cancel_during_generation(self) -> None:
        marker = REPORTS / f"cancel-{time.time_ns()}.flag"
        record = REPORTS / f"cancel-{time.time_ns()}.jsonl"
        with (REPORTS / "cancel.out.txt").open("wb") as output, (REPORTS / "cancel.metrics.txt").open("wb") as metrics:
            process = subprocess.Popen(
                [str(EXE), "--config", str(CONFIG), "--record", str(record), "--cancel-file", str(marker), "--prompt", "Explain prime numbers.", "--timeout", "90"],
                cwd=ROOT, stdout=output, stderr=metrics,
            )
            try:
                deadline = time.monotonic() + 80
                while time.monotonic() < deadline and process.poll() is None:
                    if any(item.get("metrics", {}).get("output_tokens", 0) >= 2 for item in events(record)):
                        marker.write_text("cancel", encoding="utf-8")
                        break
                    time.sleep(0.1)
                self.assertTrue(marker.exists(), "Did not reach generation for cancellation test")
                self.assertEqual(process.wait(timeout=30), 130)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        summary = events(record)[-1]
        self.assertEqual(summary["metrics"]["end_reason"], "cancelled")
        self.assertGreater(summary["metrics"]["output_tokens"], 0)
        self.results["cancel"] = {"exit_code": 130, "record": str(record), "metrics": summary["metrics"]}

    def test_07_multiple_input_batches(self) -> None:
        _, summary = self.run_case("input-batches", ["--prompt", "word " * 160 + "What is 1+1?", "--max-tokens", "1", "--timeout", "120"])
        self.assertGreater(summary["metrics"]["input_tokens"], 128)
        samples = events(Path(self.results["input-batches"]["record"]))
        self.assertTrue(any(0 < item.get("metrics", {}).get("input_processed", 0) < summary["metrics"]["input_tokens"] for item in samples))

    def test_08_generation_context_boundary(self) -> None:
        probe = REPORTS / "boundary-probe.json"
        self.run_case("boundary-probe", ["--prompt", "word " * 10, "--inspect", str(probe)])
        overhead = len(json.loads(probe.read_text(encoding="utf-8"))["token_ids"]) - 10
        prompt = "word " * (511 - overhead)
        _, summary = self.run_case("generation-boundary", ["--context", "512", "--prompt", prompt, "--max-tokens", "4", "--timeout", "180"])
        self.assertEqual(summary["metrics"]["input_tokens"], 511)
        self.assertEqual(summary["metrics"]["output_tokens"], 1)
        self.assertEqual(summary["metrics"]["end_reason"], "context_limit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
