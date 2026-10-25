# Changelog

## 0.4.0

- Generic GGUF autoregressive text-model entry point using backend capabilities, without Qwen3.5 or Q4_K_M allowlists.
- Native architecture, tokenizer, precision, context, expert and split-file metadata; architecture-aware backend memory planning.
- Model-native chat templates, explicit Jinja overrides and raw text completion for base models.
- Model-native assistant-content boundaries and tokenizer-validated labels for Reflex/System One.
- Actual model identities in results and TUI information; caller-provided System One aliases are recorded separately.
- Cross-model checks with Qwen2.5 Q8_0 and SmolLM2 Q4_K_M/F16, plus the Qwen3.5 regression baseline.

## 0.3.0

- Pure terminal interface with a blue block-character W-LLM banner and shaded depth.
- Editable Unicode input, multiline paste, process-local input history and terminal scrollback.
- Model selection, explicit loading/release, mode selection, settings and result export.
- Independent CPU text generation and Reflex candidate/System One decisions.
- Opt-in recording and configuration; default execution creates no application state files.
- Windows x64 distribution, reproducible build scripts and terminal/CLI checks.
