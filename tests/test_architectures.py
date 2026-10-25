"""Small random-weight GGUFs exercise architecture contracts and actual graphs.

These are structural fixtures, not trained models or language-quality results.
No model download is performed. Tensor layouts target the pinned backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
from gguf import GGUFWriter  # noqa: E402

EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()


def byte_vocabulary():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {byte: chr(byte) for byte in visible}
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + sum(value >= 256 for value in map(ord, mapping.values())))
    return [mapping[byte] for byte in range(256)] + ["<s>", "</s>", "<unk>", "AA"]


def make_fixture(folder, kind):
    arch = {"shared_moe": "qwen2moe", "mla_moe": "deepseek2", "swa": "gemma2", "disabled_swa": "phi3"}.get(kind, "llama")
    path = Path(folder) / (kind + ".gguf")
    writer = GGUFWriter(path, arch)
    writer.add_name("Wllm random structural fixture " + kind)
    writer.add_context_length(4096)
    writer.add_embedding_length(32)
    writer.add_block_count(2)
    writer.add_feed_forward_length(64)
    writer.add_head_count(4)
    kv = 4 if kind == "mha" else 1 if kind in ("mqa", "mla_moe") else 2
    writer.add_head_count_kv(kv)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(8)
    writer.add_file_type(0)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    vocab = byte_vocabulary() + (["</assistant>"] if kind == "template_stop" else [])
    writer.add_token_list(vocab)
    writer.add_token_types([1] * 256 + [3, 3, 2, 1] + ([1] if kind == "template_stop" else []))
    writer.add_token_merges(["A A"])
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(257)
    writer.add_unk_token_id(258)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    if kind == "template_stop":
        # Exercise the pinned native analyser's literal-stop contract, including
        # a normal vocabulary token that is not classified as EOG.
        writer.add_chat_template("{# laguna_glm_thinking #}{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] + '</' + message['role'] + '>\n' }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}")
    else:
        writer.add_chat_template("{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
    experts = kind in ("moe", "shared_moe", "mla_moe")
    if experts:
        writer.add_expert_count(4)
        writer.add_expert_used_count(2)
    if kind in ("shared_moe", "mla_moe"):
        writer.add_expert_feed_forward_length(24)
        writer.add_expert_shared_feed_forward_length(24)
    if kind == "mla_moe":
        writer.add_q_lora_rank(8)
        writer.add_kv_lora_rank(8)
        writer.add_key_length(16)
        writer.add_value_length(16)
        writer.add_key_length_mla(16)
        writer.add_value_length_mla(16)
        writer.add_expert_shared_count(1)
        writer.add_leading_dense_block_count(1)
    if kind in ("swa", "disabled_swa"):
        writer.add_sliding_window(32)
    rng = np.random.default_rng(6219)

    def tensor(name, shape, norm=False):
        data = np.ones(shape, np.float32) if norm else rng.normal(0, 0.02, shape).astype(np.float32)
        writer.add_tensor(name + ".weight", data)

    embedding = 1 + rng.normal(0, 0.005, (len(vocab), 32)).astype(np.float32)
    embedding[65] = 1.5  # ASCII A; keep fixture generation valid UTF-8.
    writer.add_tensor("token_embd.weight", embedding)
    tensor("output_norm", (32,), True)
    if kind != "swa":
        output = np.zeros((len(vocab), 32), np.float32)
        output[65], output[66] = 0.5, 0.25
        if kind == "template_stop":
            output[260] = 1
        writer.add_tensor("output.weight", output)
    for layer in range(2):
        prefix = f"blk.{layer}."
        tensor(prefix + "attn_norm", (32,), True)
        tensor(prefix + "ffn_norm", (32,), True)
        if kind == "mla_moe":
            tensor(prefix + "attn_q_a_norm", (8,), True)
            tensor(prefix + "attn_kv_a_norm", (8,), True)
            tensor(prefix + "attn_q_a", (8, 32))
            tensor(prefix + "attn_q_b", (64, 8))
            tensor(prefix + "attn_kv_a_mqa", (16, 32))
            tensor(prefix + "attn_k_b", (4, 8, 8))
            tensor(prefix + "attn_v_b", (4, 16, 8))
            tensor(prefix + "attn_output", (32, 64))
        else:
            tensor(prefix + "attn_q", (32, 32))
            tensor(prefix + "attn_k", (kv * 8, 32))
            tensor(prefix + "attn_v", (kv * 8, 32))
            tensor(prefix + "attn_output", (32, 32))
        routed = experts and not (kind == "mla_moe" and layer == 0)
        if routed:
            width = 24 if kind in ("shared_moe", "mla_moe") else 64
            tensor(prefix + "ffn_gate_inp", (4, 32))
            tensor(prefix + "ffn_gate_exps", (4, width, 32))
            tensor(prefix + "ffn_up_exps", (4, width, 32))
            tensor(prefix + "ffn_down_exps", (4, 32, width))
            if kind in ("shared_moe", "mla_moe"):
                tensor(prefix + "ffn_gate_shexp", (24, 32))
                tensor(prefix + "ffn_up_shexp", (24, 32))
                tensor(prefix + "ffn_down_shexp", (32, 24))
                if kind == "shared_moe":
                    tensor(prefix + "ffn_gate_inp_shexp", (32,))
        else:
            if kind != "disabled_swa":
                tensor(prefix + "ffn_gate", (64, 32))
            tensor(prefix + "ffn_up", (128 if kind == "disabled_swa" else 64, 32))
            tensor(prefix + "ffn_down", (32, 64))
        if kind == "swa":
            tensor(prefix + "post_attention_norm", (32,), True)
            tensor(prefix + "post_ffw_norm", (32,), True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="wllm-architecture-")
        cls.fixtures = {kind: make_fixture(cls.directory.name, kind) for kind in
                        ["mha", "mqa", "gqa", "moe", "shared_moe", "mla_moe", "swa", "disabled_swa", "template_stop"]}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def invoke(self, kind, *args, body=None, code=0):
        result = subprocess.run([str(EXE), "--model", str(self.fixtures[kind]), "--context", "4096",
                                 "--temperature", "0", "--max-tokens", "4", "--seed", "42", "--quiet-metrics", *args],
                                input=body, cwd=self.directory.name, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, code, result.stderr.decode("utf-8", errors="replace"))
        return result

    def test_native_geometry_graphs_and_prefill_chunk_equivalence(self):
        expected = {"mha": "mha", "mqa": "mqa", "gqa": "gqa", "moe": "gqa",
                    "shared_moe": "gqa", "mla_moe": "mla", "swa": "gqa"}
        for kind, topology in expected.items():
            with self.subTest(kind=kind):
                report = Path(self.directory.name) / f"{kind}.json"
                self.invoke(kind, "--resources", str(report))
                model = json.loads(report.read_text(encoding="utf-8"))["model"]
                contract = model["architecture_contract"]
                self.assertEqual(contract["attention_kinds"], [topology])
                self.assertEqual(contract["moe_layers"], 1 if kind == "mla_moe" else 2 if "moe" in kind else 0)
                self.assertEqual(contract["shared_expert_layers"], 1 if kind == "mla_moe" else 2 if kind == "shared_moe" else 0)
                self.assertGreater(model["context_plan_bytes"], 0)
                if kind == "swa":
                    self.assertEqual(contract["sliding_window"], 32)
                    self.assertGreater(contract["sliding_layers"], 0)
                values = []
                outputs = []
                for batch in [1, 128]:
                    logits = Path(self.directory.name) / f"{kind}-{batch}.bin"
                    outputs.append(self.invoke(kind, "--prompt", "Architecture boundary " * 5, "--batch", str(batch), "--logits-file", str(logits)).stdout)
                    values.append(np.fromfile(logits, dtype="<f4"))
                self.assertTrue(outputs[0])
                self.assertEqual(outputs[0], outputs[1])
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(values[0], values[1], rtol=1e-4, atol=1e-4)

    def test_system_one_runs_on_each_architecture_without_model_name_rules(self):
        request = json.dumps({"state": "Switch on", "questions": {"action": {"type": "choice",
                             "instructions": "Choose action", "criteria": {"on": "switch on", "off": "switch off"}}}}).encode()
        for kind in self.fixtures:
            if kind == "disabled_swa":
                continue
            with self.subTest(kind=kind):
                result = json.loads(self.invoke(kind, "--mode", "system_one", body=request).stdout)
                self.assertIn(result["answers"]["action"]["choice"], ("on", "off"))
                self.assertEqual(result["usage"]["output_tokens"], 0)
                self.assertIn("architecture_contract", result["_wllm"]["model_identity"])

    def test_backend_semantic_changes_are_refused(self):
        result = self.invoke("disabled_swa", "--prompt", "Hello", code=1)
        self.assertIn(b"sliding-window semantics", result.stderr)

    def test_native_template_stops_apply_to_generation_but_not_raw_completion(self):
        report = Path(self.directory.name) / "native-stops.json"
        self.invoke("template_stop", "--prompt", "Hello", "--inspect", str(report))
        self.assertIn("</assistant>", json.loads(report.read_text(encoding="utf-8"))["additional_stops"])
        record = Path(self.directory.name) / "template-stops.jsonl"
        result = self.invoke("template_stop", "--prompt", "Hello", "--record", str(record))
        self.assertEqual(result.stdout, b"")
        end = next(json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if json.loads(line)["event"] == "end")
        self.assertEqual(end["metrics"]["end_reason"], "stop")
        self.assertEqual(end["metrics"]["output_tokens"], 1)
        raw = self.invoke("template_stop", "--raw", "--prompt", "Hello")
        self.assertEqual(raw.stdout, b"</assistant>" * 4)

    def test_moe_graph_executes_native_expert_operators(self):
        for kind in ("moe", "shared_moe", "mla_moe"):
            with self.subTest(kind=kind):
                record = Path(self.directory.name) / (kind + "-operators.jsonl")
                self.invoke(kind, "--prompt", "Operator coverage", "--profile-ops", "--record", str(record))
                events = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
                end = next(event for event in events if event["event"] == "end")
                names = [row["name"] for row in end["profile"]["operators"] if row["calls"]]
                self.assertTrue(any("/MUL_MAT_ID/" in name for name in names), names)

    def test_resident_requests_reset_state_for_all_structures(self):
        for kind in self.fixtures:
            if kind == "disabled_swa":
                continue
            with self.subTest(kind=kind):
                logits = Path(self.directory.name) / (kind + "-resident")
                self.invoke(kind, "--interactive", "--logits-directory", str(logits),
                            body=b"Boundary A\nOther request with different length\nBoundary A\n/exit\n")
                files = sorted(logits.iterdir())
                self.assertEqual(len(files), 3)
                np.testing.assert_array_equal(np.fromfile(files[0], dtype="<f4"), np.fromfile(files[2], dtype="<f4"))

    def test_shared_http_engine_handles_each_structure(self):
        import test_service as http_test

        previous_model = http_test.MODEL
        try:
            for kind in ("mha", "mqa", "gqa", "moe", "shared_moe", "mla_moe", "swa", "template_stop"):
                with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                    http_test.MODEL = str(self.fixtures[kind])
                    server = http_test.Service(folder, auth=False)
                    try:
                        status, answer, _ = server.request("/v1/chat/completions", {
                            "model": "local-test", "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 4, "temperature": 0})
                        self.assertEqual(status, 200, answer)
                        if kind == "template_stop":
                            self.assertEqual(answer["choices"][0]["message"]["content"], "")
                            self.assertEqual(answer["choices"][0]["finish_reason"], "stop")
                        else:
                            self.assertTrue(answer["choices"][0]["message"]["content"])
                        status, answer, _ = server.request("/v1/systemone", {
                            "state": "Switch on", "questions": {"action": {"type": "choice",
                            "instructions": "Choose", "criteria": {"on": "switch on", "off": "switch off"}}}})
                        self.assertEqual(status, 200, answer)
                        self.assertIn("architecture_contract", answer["_wllm"]["model_identity"])
                    finally:
                        server.close()
                    self.assertEqual(list(Path(folder).iterdir()), [])
        finally:
            http_test.MODEL = previous_model


if __name__ == "__main__":
    unittest.main()
