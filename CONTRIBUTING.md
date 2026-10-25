# Contributing

Build with `scripts/build.ps1`, then run the model-free checks:

```powershell
$env:WLLM_EXE = (Resolve-Path build/public/bin/Release/wllm.exe).Path
py -3.11 -m unittest discover -s tests -p test_public.py -v
```

Terminal tests use Windows ConPTY and `pywinpty==2.0.15`. Install the test dependency into an isolated Python environment, then run `test_console.py` in the same way. Real-model checks accept an explicit model path through `WLLM_TEST_MODEL`.

Keep prompts and model contexts isolated across requests. Changes to precision, sampling, context size or inference behavior should be explicit. Include the configuration, model and checks used when reporting results. Exclude machine-specific configurations, model weights, logs and personal notes from commits.

Bug reports should include Wllm version, Windows version, CPU, terminal/shell, command or redacted configuration, and steps to reproduce.

Build a Windows distribution with `scripts/package.ps1`. The package contains an explicit file list, dependency notices and a SHA256 manifest. Git tags matching `v*` run the Windows build/check/package workflow and publish its assets as a GitHub Release.

After a release's ZIP and checksum assets are available, run `scripts/update_scoop.ps1` to refresh `bucket/wllm.json` from that release. Commit the resulting version/hash so Scoop clients can update from the repository. The manifest also includes standard Scoop checkver/autoupdate metadata.

For cross-model checks, set `WLLM_TEST_MODELS` to a local UTF-8 JSON array containing `path`, `context` and expected `architecture` for each GGUF, then run `test_generic_models.py`. `WLLM_EXE` selects the built executable. Optional `WLLM_TEST_RESULT` writes a local result summary. Supply models from different architectures and precisions; weights and local manifests stay outside commits. The suite covers native metadata/plans, generation, resident-request isolation, all three System One question types, raw completion and explicit template overrides.

`--thinking` is passed to the selected template; it does not add reasoning capabilities to a model. Model support should be reported with exact model files and settings, separately from the backend's available architecture implementations.

HTTP acceptance uses `WLLM_TEST_SERVICE_MODEL` and `test_service.py`. It starts only local child services with a supplied GGUF, checks route selection, auth, JSON/SSE, history isolation, stop sequences, System One options, backpressure, disconnect recovery, timeout and port conflicts, then releases the children. Use a small model for this suite because service-type checks start a second local process.

The optional OpenAI Python client check uses the locally installed `openai` package (verified with 2.41.0) and talks only to the test service on 127.0.0.1. No cloud API credentials or remote inference are used.
