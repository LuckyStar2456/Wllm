# Third-party notices

- **llama.cpp / ggml** — MIT, https://github.com/ggml-org/llama.cpp . Exact revision: `dependencies.json`. Binary distributions include its license and the notices from the pinned `licenses/` directory.
- **nlohmann/json** — MIT, distributed in the pinned llama.cpp tree. Its full license is included in `licenses/upstream/LICENSE-jsonhpp` in the Windows package.
- **cpp-httplib** — MIT, distributed in the pinned llama.cpp tree; provides the local HTTP transport. Its full license is included in `licenses/vendor/cpp-httplib/LICENSE` in the Windows package.
- **Ollama format reference** — the optional offline container adapter follows the format differences in https://github.com/ollama/ollama/blob/v0.35.1/llama/compat/llama-ollama-compat.cpp . It copies retained text tensor bytes into a separate container.

Model weights are supplied separately by the user and retain their own licenses. Wllm binary packages contain no model weights.
