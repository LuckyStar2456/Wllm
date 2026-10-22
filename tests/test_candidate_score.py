"""Bounded CPU acceptance for the explicit candidate-score branch; no speedup claim."""
from array import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"
TASK = ROOT / "configs/light-switch.candidates.json"


class CandidateScoreAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "reports" / f"candidate-score-{time.time_ns()}"
        cls.folder.mkdir()
        cls.results = {}
        with EXE.open("rb") as stream:
            cls.results["evidence"] = {"exe_sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                       "scope": "bounded CPU acceptance; not a performance or general quality benchmark"}
        cls.inspection_path = cls.folder / "inspect.json"
        cls.invoke("inspect", ["--inspect", str(cls.inspection_path)], prompt="请把客厅的灯打开")
        cls.inspection = json.loads(cls.inspection_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        (cls.folder / "results.json").write_text(json.dumps(cls.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Candidate-score evidence: {cls.folder}", flush=True)

    @classmethod
    def invoke(cls, name, args=(), *, prompt="请把客厅的灯打开", expected=0, mode=True, input_text=None):
        record = cls.folder / f"{name}.jsonl"
        command = [str(EXE), "--config", "configs/local.json", "--record", str(record), "--quiet-metrics", "--timeout", "90"]
        if mode:
            command += ["--mode", "candidate_score", "--candidates", str(TASK)]
        if input_text is None:
            command += ["--prompt", prompt]
        command += list(args)
        run = subprocess.run(command, cwd=ROOT, input=None if input_text is None else input_text.encode(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
        (cls.folder / f"{name}.stdout.txt").write_bytes(run.stdout)
        (cls.folder / f"{name}.stderr.txt").write_bytes(run.stderr)
        if run.returncode != expected:
            raise AssertionError(f"{name}: {run.returncode}, expected {expected}: {run.stderr.decode('utf-8')[-1800:]}")
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        ends = [e for e in events if e["event"] == "end"]
        if not ends:
            raise AssertionError(f"{name}: missing end record")
        outputs = [json.loads(line) for line in run.stdout.decode("utf-8").splitlines() if line.strip()] if mode else []
        cls.results[name] = {"exit_code": run.returncode, "record": str(record), "ends": ends, "outputs": outputs}
        return outputs, ends, events

    def test_01_template_boundary_and_mapping(self):
        data = self.inspection
        self.assertTrue(data["rendered_prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(data["config"]["context"], 16384)
        self.assertTrue(data["config"]["thinking"])
        self.assertFalse(data["config"]["effective_thinking"])
        self.assertFalse(data["config"]["sampling_applied"])
        self.assertEqual(len(set(data["candidate_token_ids"])), 4)
        self.assertEqual(data["candidate_task"]["fallback"], "disabled")

    def test_02_raw_logits_and_no_generation(self):
        logits = self.folder / "score.f32"
        outputs, ends, _ = self.invoke("scores", ["--logits-file", str(logits), "--profile"])
        result, end = outputs[0], ends[0]
        values = array("f", logits.read_bytes())
        rows = result["scores"]
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_candidate_id"], "turn_on")
        self.assertEqual(result, end["decision"])
        self.assertEqual(result["metrics"]["output_tokens"], 0)
        self.assertIsNone(result["metrics"]["ttft_s"])
        self.assertIsNone(result["metrics"]["decode_average_tps"])
        self.assertEqual(end["token_ids"], [])
        self.assertFalse(end["identity"]["generation_loop_entered"])
        self.assertGreaterEqual(result["metrics"]["decision_s"], result["metrics"]["prefill_s"])
        self.assertTrue(all(r["logit"] == values[r["token_id"]] for r in rows))
        maximum = max(r["logit"] for r in rows)
        denominator = sum(math.exp(r["logit"]-maximum) for r in rows)
        for row in rows:
            self.assertAlmostEqual(row["candidate_normalized_score"], math.exp(row["logit"]-maximum)/denominator)
        stages = {r["name"]: r for r in end["profile"]["stages"]}
        self.assertGreater(stages["prefill_process"]["calls"], 0)
        for name in ("sample", "decode_process", "render_and_write"):
            self.assertEqual(stages.get(name, {}).get("calls", 0), 0)

        reference_logits = self.folder / "reference.f32"
        _, references, _ = self.invoke("reference", ["--system", self.inspection["effective_system"],
            "--thinking", "off", "--temperature", "0", "--max-tokens", "1", "--logits-file", str(reference_logits)],
            prompt=self.inspection["effective_user"], mode=False)
        self.assertEqual(logits.read_bytes(), reference_logits.read_bytes(), "Answer prefix changed the model calculation")
        self.assertEqual(references[0]["metrics"]["output_tokens"], 1)

        alternate, _, _ = self.invoke("sampling-ignored", ["--thinking", "off", "--temperature", "0", "--seed", "123"])
        self.assertEqual(alternate[0]["scores"], rows)

    def test_03_invalid_inputs_are_rejected(self):
        base = json.loads(TASK.read_text(encoding="utf-8"))
        cases = []
        multi = json.loads(json.dumps(base)); multi["candidates"][0]["label"] = "severalwordstogether12345"
        cases.append(("multi-token", multi))
        special = json.loads(json.dumps(base)); special["candidates"][0]["label"] = "<|im_end|>"
        cases.append(("special-label", special))
        for name, spec in cases:
            task = self.folder / f"{name}.json"
            task.write_text(json.dumps(spec), encoding="utf-8")
            outputs, ends, _ = self.invoke(name, ["--candidates", str(task)], expected=1)
            self.assertEqual(outputs[0]["status"], "error")
            self.assertIsNone(outputs[0]["selected_candidate_id"])
            self.assertEqual(ends[0]["metrics"]["input_processed"], 0)
        for args in (["--mode", "candidate_score"], ["--candidates", str(TASK)],
                     ["--mode", "candidate_score", "--candidates", str(TASK), "--gpu-layers", "2"]):
            run = subprocess.run([str(EXE), *args], cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(run.stdout, b"")

    def test_04_resident_requests_and_unknown(self):
        prompts = ["请把客厅的灯打开", "请关掉客厅的灯", "请把客厅的灯打开", "这是关于蛋糕配方的问题，与灯无关。"]
        outputs, ends, events = self.invoke("resident", ["--interactive"],
                                           input_text="\n".join(prompts)+"\n/exit\n")
        self.assertEqual(len(outputs), 4)
        self.assertEqual([r["top_candidate_id"] for r in outputs], ["turn_on", "turn_off", "turn_on", "unknown"])
        self.assertEqual(outputs[-1]["status"], "abstained")
        self.assertIsNone(outputs[-1]["selected_candidate_id"])
        self.assertEqual(outputs[0]["scores"], outputs[2]["scores"], "Request states leaked")
        self.assertEqual([e["metrics"]["model_reused"] for e in ends], [False, True, True, True])
        self.assertTrue(all(e["metrics"]["output_tokens"] == 0 for e in ends))
        self.assertEqual(events[-1]["event"], "session_end")

    def test_05_timeout_has_no_selected_decision(self):
        outputs, ends, _ = self.invoke("timeout", ["--timeout", "0.001"], expected=124)
        self.assertEqual(outputs[0]["status"], "timeout")
        self.assertIsNone(outputs[0]["selected_candidate_id"])
        self.assertEqual(ends[0]["metrics"]["output_tokens"], 0)

    def test_06_cooperative_cancel_in_prefill(self):
        record, marker = self.folder / "cancel.jsonl", self.folder / "cancel.flag"
        command = [str(EXE), "--config", "configs/local.json", "--mode", "candidate_score", "--candidates", str(TASK),
                   "--prompt", "background "*1200+"\n请开灯", "--record", str(record), "--cancel-file", str(marker),
                   "--quiet-metrics", "--timeout", "45"]
        with (self.folder / "cancel.stdout.txt").open("wb") as out, (self.folder / "cancel.stderr.txt").open("wb") as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            cancelled_in_prefill = False
            try:
                deadline = time.monotonic()+40
                while time.monotonic()<deadline and process.poll() is None:
                    lines = record.read_text(encoding="utf-8").splitlines() if record.exists() else []
                    events = []
                    for line in lines:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Writer may be between write and flush.
                    if any(e.get("metrics", {}).get("input_processed", 0)>=128 for e in events):
                        cancelled_in_prefill = True
                        marker.write_text("cancel", encoding="utf-8")
                        break
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    marker.write_text("cancel", encoding="utf-8")
                process.wait(timeout=60)
        self.assertTrue(cancelled_in_prefill)
        self.assertEqual(process.returncode, 130)
        result = json.loads((self.folder / "cancel.stdout.txt").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["metrics"]["output_tokens"], 0)
        self.results["cancel"] = {"exit_code": process.returncode, "decision": result}

    def test_07_label_remap_and_option_order(self):
        task = json.loads(TASK.read_text(encoding="utf-8"))
        task["candidates"][0]["label"], task["candidates"][1]["label"] = "B", "A"
        task["candidates"].reverse()
        path = self.folder / "remapped.json"
        path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        outputs, _, _ = self.invoke("label-remap", ["--candidates", str(path)], prompt="Please turn the living room light on.")
        self.assertEqual(outputs[0]["selected_candidate_id"], "turn_on")
        best = max(outputs[0]["scores"], key=lambda r: r["logit"])
        self.assertEqual(best["label"], "B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
