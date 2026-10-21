"""Calibration must preserve artifact/config identity and reject unsafe or partial evidence."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("memory_calibration", ROOT / "scripts/memory_calibration.py")
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)
MIB = 1024**2


def fixture():
    config = {key: False for key in memory.CONFIG_KEYS}
    config.update(context=4096, threads=4, threads_batch=4, batch=128, backend="CPU",
                  cache_k="f16", cache_v="f16", recurrent_state="f32")
    signature = memory.config_signature(config)
    system = {"physical_total_bytes": 16*1024**3, "physical_available_bytes": 9*1024**3,
              "commit_headroom_bytes": 10*1024**3}
    native = {"llama_commit": "pin", "cpu_build": "avx2", "cpu": {"brand": "CPU"}, "config": config,
              "system": system, "model": {"weight_bytes": 5*1024**3, "resource_ledger": {
                  "components": {"system_reserve_bytes": 1024**3}, "system_at_admission": system,
                  "startup_physical_budget_bytes": 10*1024**3}}}
    calibration = {"schema_version": 1, "status": "calibrated", "peak_private_bytes": 700*MIB,
                   "margin_bytes": 256*MIB, "binding": {"executable_sha256": "exe", "model_sha256": "model",
                       "workloads_sha256": "workloads", "config_signature": signature, "context": 4096,
                       "memory_config": config,
                       "llama_commit": "pin", "cpu_build": "avx2", "cpu_brand": "CPU", "physical_total_bytes": 16*1024**3},
                   "samples": [{"category": category, "process_peak_private_bytes": peak*MIB, "input_tokens": tokens,
                                "output_tokens": 32, "end_reason": "max_tokens", "config_signature": signature}
                               for category, peak, tokens in [("chat", 500, 35), ("code", 600, 71), ("long", 700, 3280)]]}
    return native, calibration


class CalibrationTests(unittest.TestCase):
    def test_exact_artifacts_and_complete_evidence(self):
        _, calibration = fixture()
        memory.validate_binding(calibration, "exe", "model", "workloads")

    def test_changed_executable_model_or_manifest_rejected(self):
        _, calibration = fixture()
        for values in [("other", "model", "workloads"), ("exe", "other", "workloads"), ("exe", "model", "other")]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                memory.validate_binding(calibration, *values)

    def test_partial_failed_or_short_coverage_rejected(self):
        for change in [lambda c: c.update(status="incomplete"), lambda c: c["samples"].pop(),
                       lambda c: c["samples"][2].update(input_tokens=100),
                       lambda c: c['binding'].update(context=512),
                       lambda c: c["samples"][0].update(output_tokens=float('nan')),
                       lambda c: c["samples"][1].update(end_reason="timeout"),
                       lambda c: c["samples"][0].update(output_tokens=1)]:
            _, calibration = fixture()
            change(calibration)
            with self.assertRaises(ValueError):
                memory.validate_binding(calibration, "exe", "model", "workloads")

    def test_peaks_margin_and_memory_config_cannot_be_weakened(self):
        for change in [lambda c: c.update(peak_private_bytes=1), lambda c: c.update(margin_bytes=0),
                       lambda c: c.update(margin_bytes=float('nan')), lambda c: c.update(margin_bytes=True),
                       lambda c: c["samples"][0].update(process_peak_private_bytes=-1),
                       lambda c: c["samples"][0].update(config_signature="different")]:
            _, calibration = fixture()
            change(calibration)
            with self.assertRaises(ValueError):
                memory.validate_binding(calibration, "exe", "model", "workloads")

    def test_budget_includes_full_model_private_peak_margin_and_reserve(self):
        native, calibration = fixture()
        result = memory.assess_calibrated(native, calibration)
        self.assertEqual(result["calibrated_residency_budget_bytes"], 6*1024**3 + 956*MIB)
        self.assertEqual(result["calibrated_additional_commit_budget_bytes"], 1024**3 + 956*MIB)
        self.assertTrue(result["performance_preflight_eligible"])
        self.assertNotIn("calibration_applied", native["model"]["resource_ledger"], "Preserve original evidence")

    def test_memory_headroom_still_checked_after_calibration(self):
        native, calibration = fixture()
        native["model"]["resource_ledger"]["system_at_admission"] = {
            "physical_available_bytes": 1024**3, "commit_headroom_bytes": 1024**3}
        result = memory.assess_calibrated(native, calibration)
        self.assertFalse(result["performance_preflight_eligible"])
        self.assertEqual(len(result["performance_rejection_reasons"]), 2)

    def test_cpu_calibration_cannot_authorize_gpu(self):
        native, calibration = fixture()
        native["config"] = {**native["config"], "gpu_layers": 2}
        with self.assertRaises(ValueError):
            memory.assess_calibrated(native, calibration)
        calibration["binding"]["memory_config"] = {**calibration["binding"]["memory_config"], "backend": "Vulkan"}
        with self.assertRaises(ValueError):
            memory.validate_binding(calibration, "exe", "model", "workloads")

    def test_changed_context_batch_state_build_or_hardware_rejected(self):
        native, calibration = fixture()
        for changes in [("config", "context", 8192), ("config", "batch", 64), ("config", "cache_k", "q8_0"),
                        ("cpu", "brand", "other"), ("system", "physical_total_bytes", 32*1024**3)]:
            changed = copy.deepcopy(native)
            changed[changes[0]][changes[1]] = changes[2]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                memory.assess_calibrated(changed, calibration)
        for key in ("llama_commit", "cpu_build"):
            changed = copy.deepcopy(native)
            changed[key] = "other"
            with self.assertRaises(ValueError):
                memory.assess_calibrated(changed, calibration)


if __name__ == "__main__":
    unittest.main(verbosity=2)
