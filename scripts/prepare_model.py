"""Make a text-only, byte-preserving view of the known Ollama Qwen3.5 GGUF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

EXPECTED_SHA = "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_name(name: str) -> str | None:
    if name.startswith(("v.", "mm.", "mtp.")):
        return None
    if name.endswith(".ssm_dt"):
        return name + ".bias"
    return name


def prepare(source: Path, output: Path, report: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    print("Hashing original model (read-only)...", flush=True)
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SHA:
        raise ValueError("This adapter is validated only for the recorded Qwen3.5 blob SHA256")
    reader = GGUFReader(source)
    for key in ("qwen35.ssm.v_head_reordered", "qwen35.rope.mrope_interleaved"):
        if reader.fields[key].contents() is not True:
            raise ValueError(f"Unsupported layout: {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.gguf")
    if temporary.exists():
        raise FileExistsError(f"Incomplete output already exists: {temporary}")
    writer = GGUFWriter(temporary, "qwen35")
    changes: dict[str, object] = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key.startswith("qwen35.vision."):
            changes[key] = "removed from text-only container"
            continue
        value = field.contents()
        if key == "qwen35.attention.head_count_kv":
            value = max(value)
            writer.add_uint32(key, value)
            changes[key] = value
        elif key == "qwen35.rope.dimension_sections":
            value = list(value) + [0]
            writer.add_key_value(key, value, GGUFValueType.ARRAY, GGUFValueType.INT32)
            changes[key] = value
        else:
            writer.add_key_value(key, value, field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_string("wllm.source_sha256", source_hash)
    kept = [(tensor, text_name(tensor.name)) for tensor in reader.tensors if text_name(tensor.name)]
    skipped = [tensor.name for tensor in reader.tensors if text_name(tensor.name) is None]
    for tensor, name in kept:
        writer.add_tensor_info(name, tensor.data.shape, tensor.data.dtype, tensor.n_bytes, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    hashes: dict[str, str] = {}
    try:
        for index, (tensor, name) in enumerate(kept):
            hashes[name] = hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()
            writer.write_tensor_data(tensor.data)
            if index % 50 == 0:
                print(f"Copied {index + 1}/{len(kept)} text tensors", flush=True)
    finally:
        writer.close()
    verified = GGUFReader(temporary)
    if len(verified.tensors) != len(kept):
        raise ValueError("Tensor count changed")
    for tensor, (original, name) in zip(verified.tensors, kept):
        if tensor.name != name or tensor.tensor_type != original.tensor_type or tensor.shape.tolist() != original.shape.tolist():
            raise ValueError(f"Tensor metadata mismatch: {name}")
        if hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest() != hashes[name]:
            raise ValueError(f"Tensor bytes changed: {name}")
    # Release all mappings before the Windows rename.
    del tensor, verified, reader, kept
    import gc
    gc.collect()
    temporary.rename(output)
    output_hash = sha256_file(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "source": str(source), "source_sha256": source_hash,
        "output": str(output), "output_sha256": output_hash,
        "metadata_changes": changes, "tensor_renames": "*.ssm_dt -> *.ssm_dt.bias",
        "skipped_nontext_tensors": skipped, "verified_text_tensors": len(hashes),
        "all_text_tensor_bytes_equal": True, "tensor_sha256": hashes,
        "basis": "https://github.com/ollama/ollama/blob/v0.35.1/llama/compat/llama-ollama-compat.cpp",
    }, indent=2), encoding="utf-8")
    print(f"Verified {len(hashes)} unchanged text tensors. Model: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama/models"))
    parser.add_argument("--source", type=Path, default=model_root / "blobs" / f"sha256-{EXPECTED_SHA}")
    parser.add_argument("--output", type=Path, default=ROOT / "models/qwen3.5-9b-q4_k_m-text.gguf")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/model-adaptation.json")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
