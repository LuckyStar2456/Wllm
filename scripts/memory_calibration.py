"""Bind observed CPU allocation envelopes to exact artifacts; this never changes inference settings."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

MIB = 1024**2
CONFIG_KEYS = ("context", "threads", "threads_batch", "batch", "persistent_threadpool", "prefill_last_only",
               "cpu_weight_repacking", "backend", "cache_k", "cache_v", "recurrent_state", "flash_attention", "context_shift")


def config_signature(config: dict[str, Any]) -> str:
    values = {key: config[key] for key in CONFIG_KEYS}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_binding(calibration: dict[str, Any], executable_hash: str, model_hash: str, workloads_hash: str) -> None:
    if calibration.get("schema_version") != 1 or calibration.get("status") != "calibrated":
        raise ValueError("Incomplete/unsupported memory calibration")
    binding = calibration["binding"]
    bound_config = binding["memory_config"]
    if config_signature(bound_config) != binding["config_signature"] or bound_config["context"] != binding["context"]:
        raise ValueError("Calibration configuration fingerprint is inconsistent")
    context = binding["context"]
    if not isinstance(context, int) or isinstance(context, bool) or not 512 <= context <= 262144:
        raise ValueError("Invalid calibration context")
    for key, expected in (("executable_sha256", executable_hash), ("model_sha256", model_hash), ("workloads_sha256", workloads_hash)):
        if binding.get(key) != expected:
            raise ValueError(f"Memory calibration identity mismatch: {key}")
    samples = calibration.get("samples", [])
    if len(samples) < 3 or {row["category"] for row in samples} != {"chat", "code", "long"}:
        raise ValueError("Calibration requires completed chat/code/long diagnostics")
    peaks = [row["process_peak_private_bytes"] for row in samples]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in peaks):
        raise ValueError("Invalid measured process commit peak")
    peak = calibration["peak_private_bytes"]
    if not isinstance(peak, int) or isinstance(peak, bool) or peak != max(peaks):
        raise ValueError("Calibration peak does not match its samples")
    margin = calibration["margin_bytes"]
    if not isinstance(margin, int) or isinstance(margin, bool) or margin < max(256*MIB, (max(peaks)+3)//4):
        raise ValueError("Calibration margin below fixed policy")
    for row in samples:
        if any(not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] <= 0 for key in ("input_tokens", "output_tokens")):
            raise ValueError("Invalid calibration token counts")
        if row.get("end_reason") not in {"max_tokens", "eos"} or row.get("output_tokens", 0) < 2:
            raise ValueError("Partial/failed calibration sample")
        if row.get("config_signature") != binding["config_signature"]:
            raise ValueError("Calibration mixes memory configurations")
    if max(row["input_tokens"] for row in samples) < binding["context"] * 0.7:
        raise ValueError("Calibration lacks long-prefill coverage")


def assess_calibrated(native: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """A full-file-residency estimate plus measured process commit and fixed safety margin."""
    binding = calibration["binding"]
    if native["llama_commit"] != binding["llama_commit"] or native["cpu_build"] != binding["cpu_build"]:
        raise ValueError("Calibration backend/build mismatch")
    if config_signature(native["config"]) != binding["config_signature"]:
        raise ValueError("Calibration memory configuration mismatch")
    if native["cpu"]["brand"] != binding["cpu_brand"] or native["system"]["physical_total_bytes"] != binding["physical_total_bytes"]:
        raise ValueError("Calibration hardware mismatch")
    if native["config"]["cpu_weight_repacking"]:
        raise ValueError("Unaccepted weight repacking cannot use the calibrated performance gate")
    result = copy.deepcopy(native["model"]["resource_ledger"])
    reserve = result["components"]["system_reserve_bytes"]
    peak, margin = calibration["peak_private_bytes"], calibration["margin_bytes"]
    required = native["model"]["weight_bytes"] + peak + margin + reserve
    commit = peak + margin + reserve
    reasons = []
    system = result["system_at_admission"]
    if required > system["physical_available_bytes"]:
        reasons.append("available_physical_below_calibrated_residency_budget")
    if commit > system["commit_headroom_bytes"]:
        reasons.append("system_commit_headroom_below_calibrated_budget")
    result.update(calibration_applied=True, calibration_scope="bounded CPU diagnostics, exact artifacts/config/workload manifest",
                  uncalibrated_startup_physical_budget_bytes=result["startup_physical_budget_bytes"],
                  calibrated_residency_budget_bytes=required, calibrated_additional_commit_budget_bytes=commit,
                  measured_process_peak_private_bytes=peak, calibration_margin_bytes=margin,
                  performance_preflight_eligible=not reasons, performance_rejection_reasons=reasons)
    return result
