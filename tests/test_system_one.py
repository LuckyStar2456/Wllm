"""Local System One text-wire acceptance; no external API or Jev weights used."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/bin/Release/wllm.exe"


def choice_request(state="灯现在处于开启状态。"):
    return {"model": "wllm-local", "state": {"text": state}, "questions": {
        "light": {"type": "choice", "instructions": "根据状态判断灯目前是开还是关。",
                  "criteria": {"on": "灯开启", "off": "灯关闭"}}}}


class SystemOneAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "reports" / f"system-one-{time.time_ns()}"
        cls.folder.mkdir()
        with EXE.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        cls.results = {"evidence": {"exe_sha256": digest, "scope": "bounded protocol/CPU tests; no Jev quality or performance equivalence"}}

    @classmethod
    def tearDownClass(cls):
        (cls.folder / "results.json").write_text(json.dumps(cls.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"System One evidence: {cls.folder}", flush=True)

    def invoke(self, name, request=None, *, raw=None, args=(), expected=0, interactive=False):
        record = self.folder / f"{name}.jsonl"
        command = [str(EXE), "--config", "configs/local.json", "--mode", "system_one",
                   "--record", str(record), "--quiet-metrics", "--timeout", "180"]
        data = None
        if interactive:
            command += ["--interactive"]
            data = raw.encode("utf-8")
        elif raw is not None:
            data = raw.encode("utf-8")  # Noninteractive stdin supports pretty JSON.
        else:
            path = self.folder / f"{name}.request.json"
            path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            command += ["--request-file", str(path)]
        command += list(args)
        run = subprocess.run(command, cwd=ROOT, input=data, capture_output=True, timeout=240, check=False)
        (self.folder / f"{name}.stdout.txt").write_bytes(run.stdout)
        (self.folder / f"{name}.stderr.txt").write_bytes(run.stderr)
        self.assertEqual(run.returncode, expected, run.stderr.decode("utf-8")[-1600:])
        outputs = [json.loads(line) for line in run.stdout.decode("utf-8").splitlines() if line.strip()]
        self.assertTrue(outputs, "Protocol response missing")
        events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        self.results[name] = {"exit_code": run.returncode, "outputs": outputs, "record": str(record)}
        return outputs, events

    def test_01_mixed_question_types_and_formulas(self):
        body = choice_request()
        body["model"] = "jev-latest"  # Wire alias; response must identify the actual local model.
        body["questions"]["is_on"] = {"type": "noul", "instructions": {"question": "灯现在开启了吗？"},
                                              "criteria": {"true": ["状态明确说明灯开启"], "false": "状态明确说明灯关闭"}}
        body["questions"]["level"] = {"type": "score", "instructions": "根据灯的状态选择对应等级。",
                                               "criteria": [{"state": "灯关闭"}, {"state": "灯开启"}]}
        outputs, events = self.invoke("mixed", body)
        result = outputs[0]
        self.assertEqual(result["model"], "wllm-qwen3.5-q4_k_m")
        self.assertEqual(result["_wllm"]["requested_model"], "jev-latest")
        self.assertNotIn("error", result)
        self.assertEqual(set(result["answers"]), {"light", "is_on", "level"})
        answer = result["answers"]["light"]
        self.assertEqual(answer["choice"], "on")
        self.assertEqual(set(answer["probabilities"]), {"on", "off"})
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1)
        self.assertAlmostEqual(answer["confidence"], 2*max(answer["probabilities"].values())-1)
        self.assertGreater(result["answers"]["is_on"]["noul"], .5)
        self.assertNotIn("confidence", result["answers"]["is_on"])
        score = result["answers"]["level"]
        self.assertGreater(score["score"], .5)
        self.assertEqual(score["legend"], {"0": {"state": "灯关闭"}, "1": {"state": "灯开启"}})
        self.assertAlmostEqual(score["score"], score["probabilities"]["1"])
        best = int(max(score["probabilities"], key=score["probabilities"].get))
        spread = sum(p*abs(int(k)-best) for k, p in score["probabilities"].items())
        self.assertAlmostEqual(score["confidence"], max(0, 1-spread/.5))
        leaves = [e for e in events if e["event"] == "end"]
        self.assertEqual(len(leaves), 3)
        self.assertEqual([e["metrics"]["model_reused"] for e in leaves], [False, True, True])
        self.assertEqual(result["usage"]["input_tokens"], sum(e["metrics"]["input_processed"] for e in leaves))
        self.assertEqual(result["usage"]["output_tokens"], 0)
        self.assertTrue(all(e["metrics"]["output_tokens"] == 0 for e in leaves))
        self.assertTrue(all(e["metrics"]["decode_average_tps"] is None for e in leaves))
        self.assertEqual(events[-1]["event"], "session_end")

    def test_02_question_names_do_not_change_model_input(self):
        body = choice_request()
        first, _ = self.invoke("independent-a", body)
        renamed = copy.deepcopy(body)
        renamed["questions"]["unrelated_conflicting_name"] = renamed["questions"].pop("light")
        # A constant companion question does not enter the target question's context.
        renamed["questions"]["companion"] = {"type": "choice", "instructions": "Do not affect the other question.", "criteria": {"only": None}}
        second, _ = self.invoke("independent-b", renamed)
        self.assertEqual(first[0]["answers"]["light"], second[0]["answers"]["unrelated_conflicting_name"])
        self.assertEqual(second[0]["answers"]["companion"]["choice"], "only")

    def test_03_invalid_request_is_atomic(self):
        body = choice_request()
        body["questions"]["bad"] = {"type": "box", "instructions": "locate"}
        outputs, events = self.invoke("invalid-type", body, expected=1)
        self.assertEqual(outputs[0]["answers"], {})
        self.assertEqual(outputs[0]["error"]["type"], "invalid_request")
        self.assertFalse(any(e["event"] == "start" for e in events))
        for name, request in (("missing-state", {"questions": body["questions"]}),
                              ("ask-rejected", {"state": "on", "ask": choice_request()["questions"]}),
                              ("unknown-model", {**choice_request(), "model": "unavailable-model"})):
            outputs, _ = self.invoke(name, request, expected=1)
            self.assertEqual(outputs[0]["answers"], {})
            self.assertEqual(outputs[0]["usage"]["output_tokens"], 0)

    def test_04_json_stdin(self):
        request = {"state": [], "questions": {"one": {"type": "choice", "instructions": "Pick one.", "criteria": {"only": None}}}}
        outputs, events = self.invoke("stdin-json", raw=json.dumps(request, indent=2))
        self.assertEqual(outputs[0]["answers"]["one"], {"type": "choice", "choice": "only", "probabilities": {"only": 1.0}, "confidence": 1.0})
        self.assertEqual(outputs[0]["usage"], {"input_tokens": 0, "output_tokens": 0})
        self.assertFalse(any(e["event"] == "start" for e in events), "Constant choice should not pretend to evaluate a model")

    def test_05_interactive_recovers_invalid_json_and_rebuilds_state(self):
        on = choice_request()
        off = choice_request("灯现在处于关闭状态。")
        raw = "not-json\n"+json.dumps(on, ensure_ascii=False)+"\n"+json.dumps(off, ensure_ascii=False)+"\n/exit\n"
        outputs, events = self.invoke("interactive", raw=raw, interactive=True, expected=1)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0]["error"]["type"], "invalid_request")
        self.assertEqual([r["answers"]["light"]["choice"] for r in outputs[1:]], ["on", "off"])
        leaves = [e for e in events if e["event"] == "end"]
        self.assertEqual([e["metrics"]["model_reused"] for e in leaves], [False, True])
        self.assertEqual(events[-1]["event"], "session_end")

    def test_06_parent_budget_clears_partial_answers_and_cancels(self):
        request = choice_request()
        request["questions"] = {"constant": {"type": "choice", "instructions": "Pick one.", "criteria": {"only": None}},
                                **request["questions"]}
        outputs, _ = self.invoke("parent-timeout", request, args=["--timeout", "0.01"], expected=124)
        result = outputs[0]
        self.assertEqual(result["error"]["type"], "timeout")
        self.assertEqual(result["answers"], {})
        self.assertTrue(result["_wllm"]["questions"]["constant"]["constant_choice"])
        marker = self.folder / "cancel.flag"
        marker.write_text("cancel", encoding="utf-8")
        outputs, _ = self.invoke("parent-cancel", choice_request(), args=["--cancel-file", str(marker)], expected=130)
        self.assertEqual(outputs[0]["error"]["type"], "cancelled")
        self.assertEqual(outputs[0]["answers"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
