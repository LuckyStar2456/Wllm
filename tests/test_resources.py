"""Resource probes must not invent capacity, merge distinct GPUs or execute the model."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resource_ledger", ROOT / "scripts/resource_ledger.py")
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)
EXE = Path(os.environ.get("WLLM_TEST_EXE", str(ROOT / "build/bin/Release/wllm.exe")))


class DeviceIdentityTests(unittest.TestCase):
    def test_repeated_uuid_is_one_device(self):
        text = "\n".join(f"GPU{i}:\n deviceName = Iris Xe\n deviceUUID = 8680499a-0100-0000-0002-000000000000" for i in range(3))
        result = ledger.parse_vulkan_summary(text)
        self.assertEqual(result["raw_device_count"], 3)
        self.assertEqual(result["unique_device_count"], 1)
        self.assertEqual(result["devices"][0]["enumeration_indices"], [0, 1, 2])

    def test_name_or_zero_uuid_is_not_identity(self):
        result = ledger.parse_vulkan_summary("GPU0:\n deviceName = Same GPU\n deviceUUID = 00000000-0000-0000-0000-000000000000\nGPU1:\n deviceName = Same GPU")
        self.assertEqual(result["unique_device_count"], 2)

    def test_two_identical_pci_models_are_not_merged(self):
        dxgi = [{"software": False, "vendor_id": 0x8086, "device_id": 0x9a49, "adapter_luid": "one"},
                {"software": False, "vendor_id": 0x8086, "device_id": 0x9a49, "adapter_luid": "two"}]
        devices = [{"vendorID": "0x8086", "deviceID": "0x9a49"}]
        self.assertIsNone(ledger.link_vulkan_devices(dxgi, devices)[0]["dxgi_adapter_luid"])
        self.assertIsNone(ledger.link_vulkan_devices(dxgi[:1], devices)[0]["dxgi_adapter_luid"], "PCI model IDs are not physical identity")

    def test_uuid_groups_distinct_valid_luid_aliases(self):
        text = "\n".join(f"GPU{i}:\n deviceUUID = 8680499a-0100-0000-0002-000000000000\n deviceLUID = {value}-00000000\n deviceLUIDValid = true"
                         for i, value in enumerate(["9e040100", "64fd0100", "eadb0100"]))
        parsed = ledger.parse_vulkan_summary(text)
        dxgi = [{"software": False, "adapter_luid": luid} for luid in ["00000000:0001049e", "00000000:0001fd64", "00000000:0001dbea"]]
        linked = ledger.link_vulkan_devices(dxgi, parsed["devices"])
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["dxgi_adapter_luids"], [entry["adapter_luid"] for entry in dxgi])
        self.assertFalse(linked[0]["model_execution_verified"])

    def test_conflicting_luid_across_uuids_stays_unresolved(self):
        devices = [{"adapter_luids": ["same"]}, {"adapter_luids": ["same"]}]
        linked = ledger.link_vulkan_devices([{"software": False, "adapter_luid": "same"}], devices)
        self.assertTrue(all(row["cross_api_match"] == "unresolved" for row in linked))


@unittest.skipUnless(os.name == "nt" and EXE.exists(), "Windows candidate executable required")
class NativeProbeTests(unittest.TestCase):
    def test_system_only_probe_needs_no_model_or_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "system.json"
            subprocess.run([str(EXE), "--resources", str(output)], input="", capture_output=True,
                           text=True, check=True, timeout=30, cwd=directory)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(report["model"])
            self.assertGreater(report["system"]["physical_total_bytes"], 0)
            self.assertLessEqual(report["system"]["commit_total_bytes"], report["system"]["commit_limit_bytes"])
            self.assertFalse((Path(directory) / "reports").exists(), "Probe must not start backend or metric writer")
            for device in report["dxgi"]["devices"]:
                self.assertFalse(device["capacity_additive_to_ram"])
                self.assertFalse(device["model_execution_verified"])
            result = subprocess.run([str(EXE), "--resources", str(output)], capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, "Existing evidence must not be overwritten")

    def test_probe_cannot_accidentally_run_a_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reject.json"
            result = subprocess.run([str(EXE), "--resources", str(output), "--prompt", "Hello"],
                                    capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_model_budget_rejects_unavailable_reserve_without_loading(self):
        config = ROOT / "configs/local.json"
        if not config.exists():
            self.skipTest("Prepared local config required")
        with tempfile.TemporaryDirectory() as directory:
            settings = json.loads(config.read_text(encoding="utf-8"))
            settings["reserve_mib"] = 1024 * 1024  # Deliberately impossible, no allocation.
            profile = Path(directory) / "config.json"
            profile.write_text(json.dumps(settings), encoding="utf-8")
            output = Path(directory) / "budget.json"
            subprocess.run([str(EXE), "--config", str(profile), "--resources", str(output)],
                           capture_output=True, check=True, timeout=45, cwd=ROOT)
            report = json.loads(output.read_text(encoding="utf-8"))
            model = report["model"]
            budget = model["resource_ledger"]
            self.assertFalse(budget["performance_preflight_eligible"])
            self.assertIn("available_physical_below_conservative_resident_budget", budget["performance_rejection_reasons"])
            self.assertEqual(model["kv_estimate_bytes"], 128 * 1024**2)
            self.assertEqual(model["recurrent_estimate_bytes"], int(50.25 * 1024**2))
            self.assertIsNone(budget["components"]["mapped_file_resident_bytes"])
            self.assertFalse(budget["peaks_calibrated"])

    def test_benchmark_rejects_before_hash_warmup_or_generation(self):
        config = ROOT / "configs/local.json"
        if not config.exists():
            self.skipTest("Prepared local config required")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            settings = json.loads(config.read_text(encoding="utf-8"))
            settings["reserve_mib"] = 1024 * 1024
            profile = folder / "config.json"
            profile.write_text(json.dumps(settings), encoding="utf-8")
            profiles = folder / "profiles.json"
            profiles.write_text(json.dumps([{"name": "candidate", "exe": str(EXE.resolve()), "settings": {}}]), encoding="utf-8")
            output = folder / "benchmark"
            result = subprocess.run(["python", str(ROOT / "scripts/benchmark.py"), "--config", str(profile),
                "--profiles", str(profiles), "--cases", "short", "--repeats", "1", "--output", str(output)],
                cwd=ROOT, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "resource_rejected")
            self.assertIsNone(summary["model_sha256"])
            self.assertEqual(summary["warmups"], [])
            self.assertEqual(summary["runs"], [])
            self.assertEqual(summary["aggregates"], {})
            self.assertFalse(list(output.glob("warmup-*")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
