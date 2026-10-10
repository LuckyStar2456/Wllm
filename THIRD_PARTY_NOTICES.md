# Third-party dependencies

- llama.cpp / ggml, MIT: https://github.com/ggml-org/llama.cpp . Revision is pinned in `dependencies.json`; full upstream notices remain in `vendor/llama.cpp/LICENSE` and `vendor/llama.cpp/licenses/`.
- nlohmann/json, MIT, provided by the same pinned tree: `vendor/llama.cpp/vendor/nlohmann/json.hpp`.
- The offline text-container normalization follows the documented format differences in Ollama v0.35.1: https://github.com/ollama/ollama/blob/v0.35.1/llama/compat/llama-ollama-compat.cpp . No Ollama server is required at runtime.
- The existing model is supplied by the user. Its original license remains in the Ollama blob store. Preparing a text container does not change its license or ownership.

When redistributing a build, include the upstream licenses and notices from the exact pinned dependency tree. Model files are excluded from source control.
