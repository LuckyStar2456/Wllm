"""Exercise the real Windows console input/output path through ConPTY."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from winpty import Backend, PtyProcess

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("WLLM_EXE", ROOT / "build/public/bin/Release/wllm.exe")).resolve()


class Console:
    def __init__(self, command, cwd, *, columns=80, env=None):
        self.process = PtyProcess.spawn(command, cwd=str(cwd), env=env or dict(os.environ),
                                        dimensions=(28, columns), backend=Backend.ConPTY)
        self.output = ""
        self.condition = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                part = self.process.read(4096)
                with self.condition:
                    self.output += part
                    self.condition.notify_all()
        except (EOFError, OSError):
            pass

    def expect(self, text, *, offset=0, timeout=20):
        deadline = time.monotonic() + timeout
        with self.condition:
            while text not in self.output[offset:]:
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Missing {text!r}; output tail: {self.output[-1800:]!r}")
                self.condition.wait(min(0.1, max(0, deadline-time.monotonic())))
        return len(self.output)

    def send(self, command, expected, *, timeout=20):
        offset = len(self.output)
        self.process.write(command + "\r")
        return self.expect(expected, offset=offset, timeout=timeout)

    def close(self):
        if self.process.isalive():
            self.process.close(force=True)
        self.reader.join(timeout=3)
        self.process.fileobj.close()
        self.process._server.close()


class ConsoleTests(unittest.TestCase):
    def test_banner_editor_commands_resize_paste_and_no_state(self):
        with tempfile.TemporaryDirectory(prefix="wllm-console-") as folder:
            color_env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
            console = Console([str(EXE), "--tui"], folder, env=color_env)
            try:
                console.expect("wllm>")
                self.assertIn("\x1b[38;2;64;158;255m", console.output)
                self.assertIn("\x1b[38;2;37;90;145m", console.output)
                self.assertNotIn("\ufffd", console.output)
                self.assertNotIn("\x1b[?1049h", console.output)
                console.send("/help", "Ctrl+U clears input")
                console.send("/set threads 2", "Setting applies")
                console.send("/config", '"threads": 2')
                console.send("/mode generateX\x7f", "Mode: generate")
                console.send("/set context 1", "context must be")
                console.send("/mode system_one", "Mode: system_one")
                request = '{\n"state":"中文状态",\n"questions":{"a":{"type":"choice","instructions":"选", "criteria":{"固定":null}}}\n}'
                console.send("\x1b[200~" + request + "\x1b[201~", '"answers"')
                console.send("/mode generate", "Mode: generate")
                console.process.setwinsize(18, 38)
                console.send("/info", "backend")
                console.process.setwinsize(32, 100)
                console.send("/info", "backend")
                # Recall the previous input and submit it again.
                console.send("\x1b[A", "backend")
                console.send("/reset", "last result cleared")
                console.send("/export result.json", "Run a request before exporting")
                console.send("/exit", "Goodbye")
                self.assertEqual(list(Path(folder).iterdir()), [])
            finally:
                console.close()

    def test_cmd_and_powershell_hosts_restore_code_pages(self):
        for shell in ("cmd", "powershell"):
            with self.subTest(shell=shell), tempfile.TemporaryDirectory() as folder:
                if shell == "cmd":
                    command = ["cmd.exe", "/d", "/c", str(EXE), "--tui", "--ascii"]
                else:
                    native = 'using System; using System.Runtime.InteropServices; public class ConsoleNative { [DllImport("kernel32.dll")] public static extern uint GetConsoleCP(); [DllImport("kernel32.dll")] public static extern uint GetConsoleOutputCP(); [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int n); [DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint mode); }'
                    script = f"Add-Type -TypeDefinition '{native}'; function State {{ [uint32]$i=0; [uint32]$o=0; [ConsoleNative]::GetConsoleMode([ConsoleNative]::GetStdHandle(-10),[ref]$i) > $null; [ConsoleNative]::GetConsoleMode([ConsoleNative]::GetStdHandle(-11),[ref]$o) > $null; return ('{{0}}:{{1}}:{{2}}:{{3}}' -f [ConsoleNative]::GetConsoleCP(),[ConsoleNative]::GetConsoleOutputCP(),$i,$o) }}; $before=State; & '{EXE}' --tui --ascii; Write-Output ('RESTORED='+($before -eq (State)))"
                    command = ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", script]
                console = Console(command, folder, columns=48, env={**os.environ, "NO_COLOR": "1"})
                try:
                    console.expect("wllm>")
                    self.assertNotIn("\x1b[38;2", console.output)
                    self.assertNotIn("\ufffd", console.output)
                    console.send("/help", "Ctrl+U clears input")
                    console.send("/exit", "Goodbye")
                    if shell == "powershell":
                        console.expect("RESTORED=True")

                    self.assertEqual(list(Path(folder).iterdir()), [])
                finally:
                    console.close()


if __name__ == "__main__":
    unittest.main()
