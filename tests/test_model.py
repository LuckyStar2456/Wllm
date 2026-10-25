"""Real-model release checks; WLLM_TEST_MODEL supplies the local GGUF."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()
MODEL = os.environ.get("WLLM_TEST_MODEL")


@unittest.skipUnless(MODEL, "Set WLLM_TEST_MODEL to run real-model checks")
class ModelTests(unittest.TestCase):
    def test_tui_model_lifecycle_cancel_recovery_and_export(self):
        from test_console import Console

        with tempfile.TemporaryDirectory(prefix="wllm-ui-model-") as folder:
            console = Console([str(EXE), "--tui", "--model", MODEL, "--max-tokens", "128", "--timeout", "180"], folder, columns=200)
            try:
                console.expect("wllm>")
                console.send("/load", "Model weights loaded", timeout=180)
                offset = len(console.output)
                console.process.write("Explain prime numbers in detail.\r")
                console.expect("[Wllm] CPU", offset=offset)
                console.process.write("\x03")
                console.expect("[Wllm cancelled]", offset=offset, timeout=180)
                console.send("/set max_tokens 4", "Setting applies")
                console.send("Hello", "[Wllm finished]", timeout=180)
                console.send("/export result.json", "Result exported")
                result = json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["metrics"]["output_tokens"], 4)
                console.send("/unload", "Model weights released")
                console.send("/exit", "Goodbye")
                self.assertEqual({p.name for p in Path(folder).iterdir()}, {"result.json"})
            finally:
                console.close()

    def test_generation_decision_and_independent_resident_requests(self):
        with tempfile.TemporaryDirectory(prefix="wllm-model-") as folder:
            cwd = Path(folder)
            common = [str(EXE), "--model", MODEL, "--seed", "42", "--max-tokens", "4", "--timeout", "180", "--quiet-metrics"]
            isolated = cwd / "isolated.jsonl"
            first = subprocess.run(common + ["--prompt", "Hello", "--record", str(isolated)],
                                   cwd=cwd, capture_output=True, timeout=210)
            self.assertEqual(first.returncode, 0, first.stderr.decode("utf-8"))
            self.assertTrue(first.stdout)
            baseline = [json.loads(line) for line in isolated.read_text().splitlines() if json.loads(line)["event"] == "end"][0]
            record = cwd / "resident.jsonl"
            run = subprocess.run(common + ["--interactive", "--record", str(record)], input=b"Hello\nWhat is 1+1?\nHello\n/exit\n",
                                 cwd=cwd, capture_output=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8"))
            events = [json.loads(line) for line in record.read_text().splitlines()]
            ends = [event for event in events if event["event"] == "end"]
            self.assertEqual(len(ends), 3)
            self.assertEqual([e["metrics"]["model_reused"] for e in ends], [False, True, True])
            self.assertEqual(ends[0]["token_ids"], baseline["token_ids"])
            self.assertEqual(ends[2]["token_ids"], baseline["token_ids"])
            self.assertTrue(all(e["metrics"]["input_reused"] == 0 for e in ends))
            self.assertEqual(events[-1]["event"], "session_end")
            decision = subprocess.run(common + ["--mode", "candidate_score", "--candidates", str(ROOT / "configs/light-switch.candidates.json"), "--prompt", "打开灯"],
                                      cwd=cwd, capture_output=True, timeout=210)
            self.assertEqual(decision.returncode, 0, decision.stderr.decode("utf-8"))
            response = json.loads(decision.stdout)
            self.assertIn(response["selected_candidate_id"], ("turn_on", "turn_off", "query", "unknown", None))
            self.assertEqual(response["metrics"]["output_tokens"], 0)
            system_one = subprocess.run(common + ["--mode", "system_one", "--request-file", str(ROOT / "configs/system-one.example.json")],
                                        cwd=cwd, capture_output=True, timeout=210)
            self.assertEqual(system_one.returncode, 0, system_one.stderr.decode("utf-8"))
            answer = json.loads(system_one.stdout)
            self.assertIn(answer["answers"]["action"]["choice"], ("turn_on", "turn_off", "clarify"))
            self.assertEqual(answer["usage"]["output_tokens"], 0)
            self.assertEqual({p.name for p in cwd.iterdir()}, {"isolated.jsonl", "resident.jsonl"})


if __name__ == "__main__":
    unittest.main()
