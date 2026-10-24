"""Bounded native CPU kernel arithmetic/dispatch and experiment-entry regression."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm-cpu-kernel-bench.exe"


class CpuKernelTests(unittest.TestCase):
    def invoke(self, directory, arguments, env=None):
        destination = Path(directory) / "result.json"
        result = subprocess.run([str(EXE), "--output", str(destination), *arguments],
                                cwd=ROOT, capture_output=True, timeout=45, env=env, check=False)
        return result, json.loads(destination.read_text(encoding="utf-8"))

    def test_arithmetic_and_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            process, report = self.invoke(directory, ["--self-test"])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 126)
        self.assertFalse(report["candidate_integrated_into_engine"])
        for case in report["cases"]:
            expected = "custom_pair" if 2 <= case["activation_columns"] < 8 else "unchanged_upstream"
            self.assertEqual(case["pair_dispatch"], expected)
            self.assertLess(case["planned_payload_bytes"], 256 * 1024**2)
            self.assertTrue(all(row["all_outputs_byte_equal"] for row in case["profiles"].values()))

    def test_reject_non_default_tiled_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            process, report = self.invoke(directory, ["--self-test"], {**os.environ, "GGML_CPU_TILED_MM": "0"})
        self.assertEqual(process.returncode, 1)
        self.assertIn("Non-default tiled", report["error"])
        self.assertEqual(report["cases"], [])

    def test_reject_invalid_controls_before_allocation(self):
        for argument, value in (("--columns", "130"), ("--threads", "9"), ("--repeats", "3trailing")):
            with self.subTest(argument=argument), tempfile.TemporaryDirectory() as directory:
                process, report = self.invoke(directory, ["--self-test", argument, value])
            self.assertEqual(process.returncode, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["cases"], [])

    def test_never_overwrite_report(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            destination.write_text('{"sentinel": true}', encoding="utf-8")
            process, report = self.invoke(directory, ["--self-test"])
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report, {"sentinel": True})


if __name__ == "__main__":
    unittest.main()
