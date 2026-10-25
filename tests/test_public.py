"""Model-free checks of the distributed command-line contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()


class PublicCliTests(unittest.TestCase):
    def invoke(self, *args: str, body: str | None = None, env: dict | None = None):
        with tempfile.TemporaryDirectory(prefix="wllm-public-") as folder:
            cwd = Path(folder)
            result = subprocess.run(
                [str(EXE), *args], input=body, cwd=cwd, env=env,
                text=True, encoding="utf-8", capture_output=True, timeout=30,
            )
            self.assertEqual(list(cwd.iterdir()), [], "Default invocation left application state")
            self.assertNotIn("\x1b", result.stdout)
            self.assertNotIn("\x1b", result.stderr)
            return result

    def test_help_and_version_from_empty_directory(self):
        help_result = self.invoke("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--tui", help_result.stdout)
        self.assertEqual(self.invoke("--version").returncode, 0)
        self.assertIn("0.4.0", self.invoke("--version").stdout)

    def test_no_implicit_local_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            (root / "configs/local.json").write_text("invalid", encoding="utf-8")
            run = subprocess.run([str(EXE), "--prompt", "hello", "--quiet-metrics"], cwd=root,
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(run.returncode, 1)
            self.assertIn("GGUF file not found", run.stderr)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["configs"])

    def test_invalid_input_and_timeout_are_validated(self):
        for args in [("--context", "1"), ("--thinking", "maybe"), ("--mode", "other"),
                     ("--top-p", "0"), ("--top-k", "1.5"), ("--timeout", "-1")]:
            with self.subTest(args=args):
                run = self.invoke(*args)
                self.assertEqual(run.returncode, 1)
                self.assertIn("Wllm:", run.stderr)

    def test_tui_cannot_pollute_redirected_output(self):
        run = self.invoke("--tui")
        self.assertEqual(run.returncode, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("requires console", run.stderr)

    def test_system_one_json_and_error_contract(self):
        request = {"state": "中文状态", "questions": {"action": {
            "type": "choice", "instructions": "固定选项", "criteria": {"only": None}}}}
        run = self.invoke("--mode", "system_one", body=json.dumps(request, ensure_ascii=False))
        self.assertEqual(run.returncode, 0, run.stderr)
        response = json.loads(run.stdout)
        self.assertEqual(response["answers"]["action"]["choice"], "only")
        self.assertEqual(response["usage"]["output_tokens"], 0)
        bad = self.invoke("--mode", "system_one", body='{"state":"x","ask":{}}')
        self.assertEqual(bad.returncode, 1)
        self.assertEqual(json.loads(bad.stdout)["error"]["type"], "invalid_request")

    def test_unicode_paths_explicit_records_and_no_color(self):
        with tempfile.TemporaryDirectory(prefix="wllm-中文 空格-") as folder:
            root = Path(folder)
            config = root / "参数.json"
            config.write_text(json.dumps({"record": str(root / "结果.jsonl")}), encoding="utf-8")
            request = {"state": "x", "questions": {"fixed": {
                "type": "choice", "instructions": "one", "criteria": {"one": None}}}}
            run = subprocess.run([str(EXE), "--config", str(config), "--mode", "system_one"],
                                 input=json.dumps(request), cwd=root, text=True, encoding="utf-8",
                                 capture_output=True, timeout=30, env={**os.environ, "NO_COLOR": "1"})
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual({p.name for p in root.iterdir()}, {"参数.json", "结果.jsonl"})
            events = [json.loads(line) for line in (root / "结果.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["event"], "system_one_end")


if __name__ == "__main__":
    unittest.main()
