# Changelog

## 0.5.0

- HTTP service with selectable `openai`, `systemone` or `all` routes and explicit host/port/model alias.
- OpenAI-compatible text Chat Completions and Completions, including SSE, usage, request-supplied history and stop sequences.
- One System One decision API at `/v1/systemone`: business options in `criteria`, structured answers, and the existing choice/noul/score semantics.
- Unified the public decision feature name as System One; removed the duplicate Reflex API name and endpoint.
- Shared resident weights, isolated request contexts, one inference slot with 429 backpressure, disconnect cancellation and request deadlines.
- Optional environment-supplied Bearer authentication; no implicit request or session files.

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
