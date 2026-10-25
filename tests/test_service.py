"""HTTP acceptance against an explicitly supplied local GGUF; no remote model API."""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()
MODEL = os.environ.get("WLLM_TEST_SERVICE_MODEL")
TOKEN = "local-http-test-token"


class Service:
    def __init__(self, folder, kind="all", *, timeout=60, auth=True, port=None, context=4096, record=None):
        if port is None:
            with socket.socket() as handle:
                handle.bind(("127.0.0.1", 0))
                port = handle.getsockname()[1]
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.auth = auth
        self.logs = []
        env = dict(os.environ, WLLM_HTTP_TEST_KEY=TOKEN)
        args = [str(EXE), "--serve", "--service", kind, "--host", "127.0.0.1", "--port", str(port),
                "--model", MODEL, "--model-alias", "local-test", "--context", str(context), "--max-tokens", "8",
                "--seed", "42", "--timeout", str(timeout)]
        if auth:
            args += ["--api-key-env", "WLLM_HTTP_TEST_KEY"]
        if record:
            args += ["--record", str(record)]
        self.process = subprocess.Popen(args, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.readers = [threading.Thread(target=self._read, args=(stream,), daemon=True)
                        for stream in [self.process.stdout, self.process.stderr]]
        for reader in self.readers:
            reader.start()
        try:
            deadline = time.monotonic() + 60
            while True:
                if self.process.poll() is not None:
                    raise AssertionError(b"".join(self.logs).decode("utf-8", errors="replace"))
                try:
                    status, _, _ = self.request("/health", timeout=0.5)
                    if status == 200:
                        break
                except (OSError, TimeoutError, http.client.HTTPException):
                    pass
                if time.monotonic() > deadline:
                    raise AssertionError("Service did not become ready")
                time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def _read(self, stream):
        while True:
            piece = stream.read(4096)
            if not piece:
                return
            self.logs.append(piece)

    def request(self, path, body=None, *, auth=None, timeout=60):
        headers = {"Content-Type": "application/json"}
        if self.auth if auth is None else auth:
            headers["Authorization"] = "Bearer " + TOKEN
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=payload, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            if response.headers.get_content_type() == "text/event-stream":
                value = data.decode("utf-8")
            else:
                value = json.loads(data)
            return response.status, value, response.headers

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()  # Only the child created by this test.
        self.process.wait(timeout=20)
        for reader in self.readers:
            reader.join(timeout=3)
        self.process.stdout.close()
        self.process.stderr.close()


@unittest.skipUnless(MODEL, "Set WLLM_TEST_SERVICE_MODEL to a local GGUF")
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="wllm-http-")
        cls.service = Service(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        try:
            if list(Path(cls.directory.name).iterdir()):
                raise AssertionError("Service created implicit application state")
        finally:
            cls.directory.cleanup()

    def chat(self, **extra):
        return {"model": "local-test", "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8, "seed": 42, **extra}

    def test_auth_discovery_and_errors(self):
        service = self.service
        self.assertEqual(service.request("/health", auth=False)[0], 401)
        self.assertEqual(service.request("/health")[1]["parallel_inference"], 1)
        self.assertEqual(service.request("/v1/models")[1]["data"][0]["id"], "local-test")
        for body in [self.chat(messages=[]), self.chat(seed=1.5), self.chat(tools=[])]:
            status, value, _ = service.request("/v1/chat/completions", body)
            self.assertEqual(status, 400, value)
            self.assertIn("error", value)
        self.assertEqual(service.request("/v1/chat/completions", self.chat(model="other"))[0], 404)
        self.assertEqual(service.request("/v1/systemone", {"state": "x", "ask": {}})[0], 400)
        self.assertEqual(service.request("/missing")[0], 404)

    def test_openai_python_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            self.skipTest("Optional openai Python client is not installed")
        with OpenAI(base_url=self.service.base + "/v1", api_key=TOKEN, max_retries=0, timeout=30) as client:
            self.assertEqual(client.models.list().data[0].id, "local-test")
            answer = client.chat.completions.create(**self.chat())
            self.assertTrue(answer.choices[0].message.content)
            stream = client.chat.completions.create(**self.chat(stream=True, stream_options={"include_usage": True}))
            content = ""
            usage = None
            for event in stream:
                if event.choices:
                    content += event.choices[0].delta.content or ""
                if event.usage:
                    usage = event.usage
            self.assertEqual(content, answer.choices[0].message.content)
            self.assertEqual(usage.total_tokens, answer.usage.total_tokens)
            raw = client.completions.create(model="local-test", prompt="Hello", max_tokens=4)
            self.assertTrue(raw.choices[0].text)

    def test_generation_streaming_stop_and_independent_history(self):
        service = self.service
        status, first, _ = service.request("/v1/chat/completions", self.chat())
        self.assertEqual(status, 200, first)
        content = first["choices"][0]["message"]["content"]
        self.assertTrue(content)
        self.assertEqual(first["usage"]["completion_tokens"], 8)
        history = [{"role": "system", "content": "Answer briefly"}, {"role": "user", "content": "One"},
                   {"role": "assistant", "content": "Two"}, {"role": "user", "content": "Three"}]
        self.assertEqual(service.request("/v1/chat/completions", self.chat(messages=history))[0], 200)
        again = service.request("/v1/chat/completions", self.chat())[1]
        self.assertEqual(again["choices"], first["choices"])
        status, stream, headers = service.request("/v1/chat/completions", self.chat(stream=True, stream_options={"include_usage": True}))
        self.assertEqual(status, 200, stream)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))
        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        combined = "".join(row["choices"][0]["delta"].get("content", "") for row in events if row["choices"])
        self.assertEqual(combined, content)
        self.assertEqual(events[-1]["usage"], first["usage"])
        status, stopped, _ = service.request("/v1/chat/completions", self.chat(stop=content[:2]))
        self.assertEqual(status, 200, stopped)
        self.assertEqual(stopped["choices"][0]["message"]["content"], "")
        self.assertEqual(stopped["choices"][0]["finish_reason"], "stop")
        status, raw, _ = service.request("/v1/completions", {"model": "local-test", "prompt": "The capital of France is", "max_tokens": 4})
        self.assertEqual(status, 200, raw)
        self.assertIn("text", raw["choices"][0])

    def test_systemone_options_and_question_types(self):
        request = {"model": "jev-latest", "state": {"request": "turn the light on"}, "questions": {
            "action": {"type": "choice", "instructions": "Choose action", "criteria": {"on": "turn on", "off": "turn off"}},
            "needed": {"type": "noul", "instructions": "Should the light be on?"},
            "level": {"type": "score", "instructions": "Clarity", "criteria": ["unclear", "clear"]}}}
        responses = []
        for path in ["/v1/systemone"]:
            status, value, _ = self.service.request(path, request)
            self.assertEqual(status, 200, value)
            self.assertEqual(value["usage"]["output_tokens"], 0)
            self.assertEqual(value["_wllm"]["requested_model"], "jev-latest")
            self.assertNotIn("model_path", value["_wllm"])
            self.assertTrue(all(q["metrics"]["model_reused"] for q in value["_wllm"]["questions"].values()))
            responses.append(value["answers"])
        self.assertEqual(set(responses[0]), {"action", "needed", "level"})
        self.assertEqual(self.service.request("/v1/reflex", request)[0], 404)

    def test_busy_disconnect_and_recovery(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.service.port, timeout=20)
        connection.request("POST", "/v1/chat/completions", json.dumps(self.chat(stream=True, max_tokens=1024)),
                           {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"data: ", response.readline())
        status, value, headers = self.service.request("/v1/chat/completions", self.chat())
        self.assertEqual(status, 429, value)
        self.assertEqual(headers["Retry-After"], "1")
        response.close()
        connection.close()
        deadline = time.monotonic() + 15
        while self.service.request("/health")[1]["busy"]:
            self.assertLess(time.monotonic(), deadline, "Disconnected request retained the inference slot")
            time.sleep(0.05)
        self.assertEqual(self.service.request("/v1/chat/completions", self.chat())[0], 200)

    def test_service_type_port_conflict_and_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            for kind, disabled in [("openai", "/v1/systemone"), ("systemone", "/v1/chat/completions")]:
                service = Service(folder, kind, auth=False)
                try:
                    self.assertEqual(service.request(disabled, {})[0], 404)
                    conflict = subprocess.run([str(EXE), "--serve", "--port", str(service.port), "--model", MODEL, "--context", "4096"],
                                              cwd=folder, capture_output=True, timeout=20)
                    self.assertEqual(conflict.returncode, 1)
                    self.assertIn(b"Cannot bind", conflict.stderr)
                finally:
                    service.close()
            service = Service(folder, "openai", timeout=0.001, auth=False)
            try:
                self.assertEqual(service.request("/v1/chat/completions", self.chat())[0], 504)
            finally:
                service.close()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_console_ctrl_c_stops_service(self):
        from test_console import Console

        with socket.socket() as handle:
            handle.bind(("127.0.0.1", 0))
            port = handle.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            console = Console([str(EXE), "--serve", "--service", "all", "--port", str(port),
                               "--model", MODEL, "--context", "4096"], folder)
            try:
                console.expect("auth=off", timeout=60)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                console.process.write("\x03")
                deadline = time.monotonic() + 10
                while console.process.isalive():
                    self.assertLess(time.monotonic(), deadline, "Ctrl+C did not stop the service")
                    time.sleep(0.05)
                self.assertEqual(list(Path(folder).iterdir()), [])
            finally:
                console.close()


if __name__ == "__main__":
    unittest.main()
