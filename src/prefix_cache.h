#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace wllm {
// A process-local, single-entry checkpoint. No generated tokens or sampler state.
// Publication is performed only after a successful request and record write.
struct PrefixCache {
    std::string key;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> state;

    size_t bytes() const { return state.size() + tokens.size()*sizeof(int32_t) + key.size(); }
    void clear() { PrefixCache empty; swap(empty); }
    void swap(PrefixCache & other) noexcept {
        key.swap(other.key); tokens.swap(other.tokens); state.swap(other.state);
    }
    size_t match(const std::string & identity, const std::vector<int32_t> & prompt, int batch) const {
        // Preserve the uncached batch boundaries, and always recompute a suffix
        // so the current request obtains fresh final logits from the backend.
        if (state.empty() || key != identity || batch < 1 || tokens.empty() ||
                tokens.size()%size_t(batch) || tokens.size() >= prompt.size()) return 0;
        return std::equal(tokens.begin(),tokens.end(),prompt.begin()) ? tokens.size() : 0;
    }
    static size_t checkpoint(size_t prompt_tokens, int batch) {
        return prompt_tokens && batch > 0 ? (prompt_tokens-1)/size_t(batch)*size_t(batch) : 0;
    }
};
}
