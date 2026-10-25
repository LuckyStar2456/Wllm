"""Architecture/precision/template acceptance using explicitly supplied local GGUFs.

WLLM_TEST_MODELS points to a JSON array of {path, context, architecture} records.
Weights and machine-specific manifests are kept outside source control.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()
MANIFEST = os.environ.get("WLLM_TEST_MODELS")


@unittest.skipUnless(MANIFEST, "Set WLLM_TEST_MODELS to a local model manifest")
class GenericModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = json.loads(Path(MANIFEST).read_text(encoding="utf-8-sig"))
        cls.summaries = []

    def run_request(self, model, folder, *args, body=None, expected=0):
        result = subprocess.run(
            [str(EXE), "--model", model["path"], "--context", str(model["context"]),
             "--max-tokens", "8", "--seed", "42", "--timeout", "180", "--quiet-metrics", *args],
            input=body, cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=240,
        )
        self.assertEqual(result.returncode, expected, result.stderr)
        self.assertNotIn("\x1b", result.stdout)
        return result

    def test_architectures_precisions_generation_and_request_isolation(self):
        for model in self.models:
            with self.subTest(model=model["path"]), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                resource = folder / "resource.json"
                self.run_request(model, folder, "--resources", str(resource))
                profile = json.loads(resource.read_text(encoding="utf-8"))["model"]
                self.assertEqual(profile["architecture"], model["architecture"])
                self.assertGreater(profile["parameter_count"], 0)
                self.assertGreater(profile["context_plan_bytes"], 0)
                self.assertIn("no-allocation", profile["planning_method"])
                record = folder / "one.jsonl"
                isolated = self.run_request(model, folder, "--prompt", "Hello", "--record", str(record))
                self.assertTrue(isolated.stdout)
                events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
                baseline = next(e for e in events if e["event"] == "end")
                self.assertEqual(baseline["metrics"]["output_tokens"], 8)
                resident = folder / "resident.jsonl"
                self.run_request(model, folder, "--interactive", "--record", str(resident), body="Hello\nWhat is 1+1?\nHello\n/exit\n")
                events = [json.loads(line) for line in resident.read_text(encoding="utf-8").splitlines()]
                ends = [e for e in events if e["event"] == "end"]
                self.assertEqual([e["metrics"]["model_reused"] for e in ends], [False, True, True])
                self.assertEqual(ends[0]["token_ids"], baseline["token_ids"])
                self.assertEqual(ends[2]["token_ids"], baseline["token_ids"])
                self.assertEqual(events[-1]["event"], "session_end")
                self.summaries.append({"name": profile["name"], "architecture": profile["architecture"],
                                       "quantization": profile["quantization"], "context": model["context"],
                                       "weight_bytes": profile["weight_bytes"], "independent_requests": True})

    def test_reflex_and_all_system_one_question_types(self):
        body = json.dumps({"model": "local-selection", "state": {"light": "off", "request": "turn on"}, "questions": {
            "action": {"type": "choice", "instructions": "Select the requested action", "criteria": {"on": "turn on", "off": "turn off"}},
            "needed": {"type": "noul", "instructions": "The light should be turned on"},
            "level": {"type": "score", "instructions": "How clearly is the action specified?", "criteria": ["unclear", "clear"]},
        }})
        for model in self.models:
            with self.subTest(model=model["path"]), tempfile.TemporaryDirectory() as folder:
                result = self.run_request(model, folder, "--mode", "system_one", body=body)
                response = json.loads(result.stdout)
                self.assertEqual(response["_wllm"]["requested_model"], "local-selection")
                self.assertEqual(response["_wllm"]["model_identity"]["architecture"], model["architecture"])
                self.assertIn(response["answers"]["action"]["choice"], ("on", "off"))
                self.assertGreaterEqual(response["answers"]["needed"]["noul"], 0)
                self.assertLessEqual(response["answers"]["needed"]["noul"], 1)
                self.assertGreaterEqual(response["answers"]["level"]["score"], 0)
                self.assertLessEqual(response["answers"]["level"]["score"], 1)
                self.assertEqual(response["usage"]["output_tokens"], 0)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_raw_completion_missing_template_and_explicit_template_override(self):
        model = min(self.models, key=lambda m: Path(m["path"]).stat().st_size)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            inspect = folder / "inspect.json"
            self.run_request(model, folder, "--prompt", "Hello", "--inspect", str(inspect))
            template = json.loads(inspect.read_text(encoding="utf-8"))["template"]
            plain = folder / "base-model.gguf"
            shutil.copyfile(model["path"], plain)
            key = b"tokenizer.chat_template"
            with plain.open("r+b") as stream:
                header = stream.read(16 * 1024 * 1024)
                at = header.index(struct.pack("<Q", len(key)) + key) + 8
                stream.seek(at)
                stream.write(b"wllm.test.no_template".ljust(len(key), b"_"))
            raw_model = {**model, "path": str(plain)}
            error = self.run_request(raw_model, folder, "--prompt", "Hello", expected=1)
            self.assertIn("--raw", error.stderr)
            raw_inspect = folder / "raw.json"
            self.run_request(raw_model, folder, "--raw", "--prompt", "Hello", "--inspect", str(raw_inspect))
            self.assertEqual(json.loads(raw_inspect.read_text(encoding="utf-8"))["rendered_prompt"], "Hello")
            raw = self.run_request(raw_model, folder, "--raw", "--prompt", "Hello")
            self.assertTrue(raw.stdout)
            override = folder / "chat.jinja"
            override.write_text(template, encoding="utf-8")
            self.assertTrue(self.run_request(raw_model, folder, "--chat-template-file", str(override), "--prompt", "Hello").stdout)
            # Exercise the native tokenizer/template BOS contract when the
            # supplied model has an explicit add_bos boolean metadata field.
            bos_key = b"tokenizer.ggml.add_bos_token"
            field = struct.pack("<Q", len(bos_key)) + bos_key + struct.pack("<I", 7)
            if field in header:
                with plain.open("r+b") as stream:
                    stream.seek(header.index(field) + len(field))
                    stream.write(b"\x01")
                bos_id_key = b"tokenizer.ggml.bos_token_id"
                id_field = struct.pack("<Q", len(bos_id_key)) + bos_id_key + struct.pack("<I", 4)
                bos_id = struct.unpack_from("<I", header, header.index(id_field) + len(id_field))[0]
                override.write_text("{{ bos_token }}{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}", encoding="utf-8")
                bos_inspect = folder / "bos.json"
                self.run_request(raw_model, folder, "--chat-template-file", str(override), "--prompt", "Hello", "--inspect", str(bos_inspect))
                tokens = json.loads(bos_inspect.read_text(encoding="utf-8"))["token_ids"]
                self.assertEqual(tokens[0], bos_id)
                self.assertNotEqual(tokens[1], bos_id, "Template and tokenizer duplicated BOS")

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get("WLLM_TEST_RESULT")
        if destination:
            Path(destination).write_text(json.dumps(cls.summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
