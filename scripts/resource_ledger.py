"""Capture a dated Windows resource ledger without loading model weights or changing settings."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_vulkan_summary(text: str) -> dict[str, Any]:
    """Deduplicate by nonzero UUID; a display name is never a physical-device identity."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        match = re.fullmatch(r"\s*GPU(\d+):\s*", line)
        if match:
            current = {"enumeration_index": int(match[1])}
            records.append(current)
        elif current is not None:
            match = re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$", line)
            if match and match[1] in {"apiVersion", "driverVersion", "vendorID", "deviceID", "deviceType", "deviceName",
                                     "deviceUUID", "driverUUID", "driverName", "driverInfo", "deviceLUID", "deviceLUIDValid"}:
                current[match[1]] = match[2]
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        uuid = str(record.get("deviceUUID", "")).lower()
        valid_uuid = re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid)
        identity = uuid if valid_uuid and uuid.replace("-", "").strip("0") else f"unresolved-index-{record['enumeration_index']}"
        group = grouped.setdefault(identity, {**record, "identity": identity, "enumeration_indices": [], "adapter_luids": []})
        group["enumeration_indices"].append(record["enumeration_index"])
        if record.get("deviceLUIDValid") == "true":
            encoded = str(record.get("deviceLUID", "")).replace("-", "")
            if re.fullmatch(r"[0-9a-fA-F]{16}", encoded):
                raw = bytes.fromhex(encoded)
                luid = f"{int.from_bytes(raw[4:], 'little'):08x}:{int.from_bytes(raw[:4], 'little'):08x}"
                if luid not in group["adapter_luids"]:
                    group["adapter_luids"].append(luid)
    return {"raw_device_count": len(records), "unique_device_count": len(grouped),
            "devices": list(grouped.values()), "deduplication": "nonzero deviceUUID; unresolved records remain separate"}


def link_vulkan_devices(dxgi: list[dict[str, Any]], vulkan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link valid Vulkan device LUIDs to DXGI interfaces; PCI model IDs alone are insufficient."""
    result = []
    for device in vulkan:
        item = dict(device)
        luids = device.get("adapter_luids", [])
        candidates = [entry for entry in dxgi if not entry["software"] and entry["adapter_luid"] in luids
                      and sum(entry["adapter_luid"] in other.get("adapter_luids", []) for other in vulkan) == 1]
        item["dxgi_adapter_luids"] = [entry["adapter_luid"] for entry in candidates]
        item["dxgi_adapter_luid"] = candidates[0]["adapter_luid"] if len(candidates) == 1 else None
        item["cross_api_match"] = "valid Vulkan LUID matched to DXGI interfaces; grouped by Vulkan UUID" if candidates else "unresolved"
        item["model_execution_verified"] = False
        result.append(item)
    return result


def capture_native(exe: Path, config: Path, destination: Path) -> dict[str, Any]:
    subprocess.run([str(exe.resolve()), "--config", str(config.resolve()), "--resources", str(destination.resolve())],
                   cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", timeout=45)
    return json.loads(destination.read_text(encoding="utf-8"))


def collect_vulkan() -> dict[str, Any]:
    tool = shutil.which("vulkaninfo")
    if not tool:
        return {"status": "tool_not_on_path", "devices": [], "model_execution_verified": False}
    try:
        # Full output includes VkPhysicalDeviceIDProperties.deviceLUID, omitted by --summary.
        process = subprocess.run([tool], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "probe_failed", "error": str(error), "devices": [], "model_execution_verified": False}
    return {"status": "enumerated" if process.returncode == 0 else "probe_failed",
            "tool": tool, "exit_code": process.returncode, "warnings": process.stderr.strip(),
            "model_execution_verified": False, **parse_vulkan_summary(process.stdout)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/Release/wllm.exe")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local.json")
    parser.add_argument("--output", type=Path, help="New JSON path; existing reports are never replaced")
    args = parser.parse_args()
    output = (args.output or ROOT / "reports" / f"resource-ledger-{time.time_ns()}.json").resolve()
    if output.exists():
        parser.error("Output exists; choose a new report path")
    output.parent.mkdir(parents=True, exist_ok=True)
    native_path = output.with_suffix(".native.json")
    if native_path.exists():
        parser.error("Native report path exists; choose a new report path")
    report = capture_native(args.exe, args.config, native_path)
    report["native_report"] = str(native_path)
    report["vulkan"] = collect_vulkan()
    report["vulkan"]["devices"] = link_vulkan_devices(report["dxgi"]["devices"], report["vulkan"]["devices"])
    report["dxgi"]["physical_identity_scope"] = "Raw adapter interfaces, not additive physical GPUs; group aliases using Vulkan UUID plus valid LUID"
    report["tools_on_path"] = {name: shutil.which(name) for name in ("vulkaninfo", "glslc", "sycl-ls", "icx")}
    model = Path(report["config"]["model"]).resolve()
    disk = shutil.disk_usage(model.parent)
    report["storage"] = {"model_volume": model.anchor, "total_bytes": disk.total,
                         "free_bytes": disk.free, "source": "GetDiskFreeSpaceEx via shutil.disk_usage",
                         "measured_bandwidth_bytes_s": None, "measured_iops": None}
    report["limitations"] = [
        "Resource snapshot only; no numerical, model execution, throughput or state-recovery acceptance.",
        "Shared-system GPU capacity is a limit within RAM and must not be added to physical RAM.",
        "DXGI budget/usage belongs to the probe process, not global free GPU memory.",
        "Missing PATH tools do not prove an SDK/runtime is absent; this CPU executable cannot validate GPU execution.",
        "Workspace/startup estimates are conservative, uncalibrated allowances; mapped size is not resident usage.",
        "Memory bandwidth, physical storage throughput, IOPS and actual GPU model kernels remain unmeasured.",
    ]
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    ledger = report["model"]["resource_ledger"]
    print(json.dumps({"report": str(output), "performance_preflight_eligible": ledger["performance_preflight_eligible"],
                      "reasons": ledger["performance_rejection_reasons"],
                      "vulkan_raw_devices": report["vulkan"].get("raw_device_count"),
                      "vulkan_unique_devices": report["vulkan"].get("unique_device_count")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
