"""Bounded CPU prefix-state acceptance, not an admitted performance benchmark."""
from __future__ import annotations

from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"


@unittest.skipUnless(os.name == "nt" and EXE.exists(), "Windows executable required")
class PrefixCacheEntryTests(unittest.TestCase):
    def test_cache_needs_an_explicit_resident_session(self):
        run = subprocess.run([str(EXE), "--prefix-cache-mib", "128", "--prompt", "Hello"],
                             cwd=ROOT, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 1)
        self.assertIn(b"requires --interactive or system_one", run.stderr)

    def test_unvalidated_gpu_cache_combination_is_rejected(self):
        run = subprocess.run([str(EXE), "--prefix-cache-mib", "128", "--gpu-layers", "1"],
                             cwd=ROOT, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 1)
        self.assertIn(b"requires CPU execution", run.stderr)

    def test_resource_probe_accounts_budget_and_refuses_single_request_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resources.json"
            subprocess.run([str(EXE), "--config", "configs/local.json", "--prefix-cache-mib", "128",
                            "--resources", str(output)], cwd=ROOT, capture_output=True, check=True, timeout=30)
            data = json.loads(output.read_text(encoding="utf-8"))
            ledger = data["model"]["resource_ledger"]
            self.assertEqual(ledger["components"]["prefix_cache_budget_bytes"], 128*1024**2)
            self.assertIsNone(ledger["components"]["prefix_cache_bytes"])
            self.assertIn("prefix_cache_requires_separate_session_calibration", ledger["performance_rejection_reasons"])
            self.assertFalse(ledger["performance_preflight_eligible"])

    def test_diagnostics_cannot_overwrite_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([str(EXE), "--interactive", "--logits-directory", directory],
                                 cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 1)
            self.assertIn(b"refusing to overwrite diagnostics", run.stderr)


@unittest.skipUnless(os.name == "nt" and EXE.exists(), "Windows model executable required")
class PrefixCacheAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "reports" / f"prefix-cache-{time.time_ns()}"
        cls.folder.mkdir()
        with EXE.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        cls.results = {"exe_sha256": digest, "scope": "bounded CPU correctness; diagnostic timings are not performance adoption"}

    @classmethod
    def tearDownClass(cls):
        (cls.folder / "results.json").write_text(json.dumps(cls.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prefix cache evidence: {cls.folder}", flush=True)

    def invoke(self, name, prompts, *, budget=0, system="", mode="generate", extra=(), expected=0, diagnostics=True):
        folder = self.folder / name
        folder.mkdir()
        record = folder / "run.jsonl"
        command = [str(EXE), "--config", "configs/local.json", "--interactive", "--quiet-metrics",
                   "--seed", "42", "--max-tokens", "4", "--timeout", "180", "--record", str(record),
                   "--prefix-cache-mib", str(budget)]
        if diagnostics:
            command += ["--logits-directory", str(folder / "logits")]
        if system:
            command += ["--system", system]
        if mode != "generate":
            command += ["--mode", mode]
        if mode == "candidate_score":
            command += ["--candidates", "configs/light-switch.candidates.json"]
        command += list(extra)
        with (folder / "stdout.txt").open("wb") as output, (folder / "stderr.txt").open("wb") as error:
            run = subprocess.run(command, cwd=ROOT, input=("\n".join(prompts)+"\n/exit\n").encode("utf-8"),
                                 stdout=output, stderr=error, timeout=600, check=False)
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()] if record.exists() else []
        ends = [event for event in events if event["event"] == "end"]
        self.results[name] = {"exit_code": run.returncode, "ends": ends, "folder": str(folder)}
        self.assertEqual(run.returncode, expected, (folder / "stderr.txt").read_text(encoding="utf-8")[-2000:])
        for end in ends:
            metrics = end["metrics"]
            if metrics["end_reason"] in {"max_tokens", "eos", "selected", "abstained"}:
                self.assertEqual(metrics["input_processed"]+metrics["input_reused"], metrics["input_tokens"])
            self.assertEqual(end["identity"]["resource_ledger"]["components"]["prefix_cache_budget_bytes"], budget*1024**2)
        return folder, ends, events

    def compare_logits(self, reference, candidate):
        a, b = array("f", reference.read_bytes()), array("f", candidate.read_bytes())
        self.assertEqual(len(a), 248320)
        self.assertEqual(len(a), len(b))
        self.assertTrue(all(math.isfinite(v) for v in a) and all(math.isfinite(v) for v in b))
        diff = max(abs(x-y) for x, y in zip(a, b))
        within = all(abs(x-y) <= 1e-3+1e-4*abs(x) for x, y in zip(a, b))
        ma, mb = max(a), max(b)
        za = sum(math.exp(x-ma) for x in a)
        zb = sum(math.exp(y-mb) for y in b)
        kl = sum(math.exp(x-ma)/za*((x-ma-math.log(za))-(y-mb-math.log(zb))) for x, y in zip(a, b))
        self.assertTrue(within, f"full-vocabulary tolerance failed: {diff}")
        self.assertLessEqual(kl, 1e-6)
        return {"max_absolute_error": diff, "kl": kl, "byte_equal": reference.read_bytes() == candidate.read_bytes()}

    def test_01_generation_state_restore_and_suffix_isolation(self):
        system = "The fixed reference value is 7319. "*20
        prompts = ["State the reference value.", "State the reference value.", "What is the reference value plus one?"]
        baseline, a, _ = self.invoke("generation-off", prompts, system=system)
        cached, b, events = self.invoke("generation-on", prompts, system=system, budget=128)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        self.assertEqual(b[0]["metrics"]["input_reused"], 0)
        self.assertGreaterEqual(b[1]["metrics"]["input_reused"], 128)
        comparisons = []
        for i, (left, right) in enumerate(zip(a, b)):
            self.assertEqual(left["token_ids"], right["token_ids"], "Generated tokens/state changed")
            comparisons.append(self.compare_logits(baseline / "logits" / f"{i}.f32", cached / "logits" / f"{i}.f32"))
        self.assertEqual((baseline / "stdout.txt").read_bytes(), (cached / "stdout.txt").read_bytes())
        self.assertGreater(events[-1]["prefix_cache_released_bytes"], 0)
        self.results["generation_logit_comparisons"] = comparisons

    def test_02_reflex_logits_and_small_budget(self):
        prompts = ["请把客厅的灯打开", "请把客厅的灯打开", "请把客厅的灯关闭"]
        baseline, a, _ = self.invoke("reflex-off", prompts, mode="candidate_score")
        cached, b, _ = self.invoke("reflex-on", prompts, mode="candidate_score", budget=128)
        self.assertGreaterEqual(b[1]["metrics"]["input_reused"], 128)
        comparisons = []
        for i, (left, right) in enumerate(zip(a, b)):
            comparisons.append(self.compare_logits(baseline / "logits" / f"{i}.f32", cached / "logits" / f"{i}.f32"))
            self.assertEqual(left["decision"]["selected_candidate_id"], right["decision"]["selected_candidate_id"])
            self.assertEqual(right["metrics"]["output_tokens"], 0)
            self.assertIsNone(right["metrics"]["decode_average_tps"])
        self.results["reflex_logit_comparisons"] = comparisons
        _, small, _ = self.invoke("reflex-budget", prompts[:1], mode="candidate_score", budget=1)
        self.assertEqual(small[0]["metrics"]["prefix_cache_status"], "snapshot_exceeds_budget")
        self.assertEqual(small[0]["metrics"]["input_reused"], 0)

    def test_03_failed_parent_invalidates_system_one_cache(self):
        body = json.loads((ROOT / "configs/system-one.example.json").read_text(encoding="utf-8"))
        value = json.dumps(body, ensure_ascii=False)
        folder, ends, _ = self.invoke("systemone-parent", [value, value, "{}", value], mode="system_one",
                                     budget=128, expected=1, diagnostics=False)
        outputs = [json.loads(line) for line in (folder / "stdout.txt").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(outputs), 4)
        self.assertGreater(ends[1]["metrics"]["input_reused"], 0)
        self.assertEqual(ends[2]["metrics"]["input_reused"], 0, "Failed parent published a checkpoint")
        self.assertEqual(outputs[0]["answers"], outputs[1]["answers"])
        self.assertEqual(outputs[0]["answers"], outputs[3]["answers"])
        for output, end in zip((outputs[0], outputs[1], outputs[3]), ends):
            self.assertEqual(output["usage"]["input_tokens"], end["metrics"]["input_processed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
