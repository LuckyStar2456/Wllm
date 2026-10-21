#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace wllm {
using json = nlohmann::ordered_json;

struct Config {
    std::string model, record = "reports/runs.jsonl";
    int context = 4096, threads = 4, threads_batch = 0, batch = 128, max_tokens = -1;
    bool persistent_threadpool = false, prefill_last_only = false;
    bool cpu_weight_repacking = false;
    double temperature = 1, top_p = .95, min_p = 0, repeat_penalty = 1;
    double presence_penalty = 1.5, frequency_penalty = 0;
    int top_k = 20, repeat_last_n = 64, refresh_ms = 500;
    uint32_t seed = UINT32_MAX;
    bool thinking = true;
    double timeout_s = 0, reserve_mib = 1024;

    static Config parse(const json & j) {
        Config c;
        const auto known = json::parse(R"(["model","record","context","threads","threads_batch","batch","persistent_threadpool","prefill_last_only","cpu_weight_repacking","max_tokens","temperature","top_p","min_p","repeat_penalty","presence_penalty","frequency_penalty","top_k","repeat_last_n","refresh_ms","seed","thinking","timeout_s","reserve_mib"])");
        if (!j.is_object()) throw std::runtime_error("Configuration must be an object");
        for (const auto * key : {"context","threads","threads_batch","batch","max_tokens","top_k","repeat_last_n","refresh_ms","seed"}) {
            if (j.contains(key) && !j.at(key).is_number_integer()) throw std::runtime_error(std::string("Expected integer: ") + key);
            if (j.contains(key)) {
                const auto v = j.at(key).get<int64_t>();
                if (std::string(key)=="seed" ? (v<0 || v>UINT32_MAX) : (v<INT32_MIN || v>INT32_MAX))
                    throw std::runtime_error(std::string("Integer out of range: ") + key);
            }
        }
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (std::find(known.begin(), known.end(), it.key()) == known.end())
                throw std::runtime_error("Unknown config key: " + it.key());
        }
#define READ_FIELD(k) if (j.contains(#k)) j.at(#k).get_to(c.k)
        READ_FIELD(model); READ_FIELD(record); READ_FIELD(context); READ_FIELD(threads);
        READ_FIELD(threads_batch); READ_FIELD(persistent_threadpool); READ_FIELD(prefill_last_only);
        READ_FIELD(cpu_weight_repacking);
        READ_FIELD(batch); READ_FIELD(max_tokens); READ_FIELD(temperature); READ_FIELD(top_p);
        READ_FIELD(min_p); READ_FIELD(repeat_penalty); READ_FIELD(presence_penalty);
        READ_FIELD(frequency_penalty); READ_FIELD(top_k); READ_FIELD(repeat_last_n);
        READ_FIELD(refresh_ms); READ_FIELD(seed); READ_FIELD(thinking); READ_FIELD(timeout_s);
        READ_FIELD(reserve_mib);
#undef READ_FIELD
        c.validate();
        return c;
    }
    void validate() const {
        if (context < 512 || context > 262144) throw std::runtime_error("context must be 512..262144 (default 4096); no automatic shrinking");
        if (threads < 1 || threads > 256 || threads_batch < 0 || threads_batch > 256 || batch < 1 || batch > context) throw std::runtime_error("Invalid threads or batch");
        if (max_tokens != -1 && max_tokens < 1) throw std::runtime_error("max_tokens must be -1 or positive");
        if (!std::isfinite(temperature) || temperature < 0 || !std::isfinite(top_p) || top_p <= 0 || top_p > 1 ||
            !std::isfinite(min_p) || min_p < 0 || min_p > 1 || top_k < 0 ||
            !std::isfinite(repeat_penalty) || repeat_penalty <= 0 || repeat_last_n < -1 ||
            !std::isfinite(presence_penalty) || !std::isfinite(frequency_penalty)) throw std::runtime_error("Invalid sampling parameters");
        if (refresh_ms < 100 || refresh_ms > 10000 || !std::isfinite(timeout_s) || timeout_s < 0 ||
            !std::isfinite(reserve_mib) || reserve_mib < 0) throw std::runtime_error("Invalid refresh/timeout/reserve");
    }
    int effective_threads_batch() const { return threads_batch == 0 ? threads : threads_batch; }
    json dump() const {
        return {{"model", model}, {"context", context}, {"threads", threads}, {"batch", batch},
            {"threads_batch", effective_threads_batch()}, {"persistent_threadpool", persistent_threadpool},
            {"prefill_last_only", prefill_last_only},
            {"max_tokens", max_tokens}, {"temperature", temperature}, {"top_k", top_k}, {"top_p", top_p},
            {"min_p", min_p}, {"repeat_penalty", repeat_penalty}, {"repeat_last_n", repeat_last_n},
            {"presence_penalty", presence_penalty}, {"frequency_penalty", frequency_penalty},
            {"seed", seed}, {"thinking", thinking}, {"refresh_ms", refresh_ms}, {"timeout_s", timeout_s},
            {"reserve_mib", reserve_mib}, {"record", record}, {"backend", "CPU"},
            {"cache_k", "f16"}, {"cache_v", "f16"}, {"recurrent_state", "f32"}, {"flash_attention", false}, {"context_shift", false},
            {"cpu_weight_repacking", cpu_weight_repacking}};
    }
};

// Times are monotonic seconds relative to the start of this request.
struct Metrics {
    std::string stage = "starting", reason, error;
    double load_start = -1, load_end = -1, prefill_start = -1, prefill_end = -1, ended = -1;
    int input_total = 0, input_done = 0, prefill_batches = 0, prefill_logits_batches = 0;
    std::vector<double> token_times;
    std::vector<int32_t> token_ids;
    uint64_t working_set = 0, private_bytes = 0, available = 0, peak_working_set = 0;
    uint64_t commit_total = 0, commit_limit = 0, page_faults = 0;
    uint64_t io_read_bytes = 0, io_write_bytes = 0;
    bool system_commit_available = false, io_available = false;
    bool memory_available = false;
    bool model_reused = false;
    std::optional<double> average_tps() const {
        if (token_times.size() < 2 || token_times.back() <= token_times.front()) return {};
        return (token_times.size() - 1) / (token_times.back() - token_times.front());
    }
    std::optional<double> window_tps(double now) const {
        if (token_times.empty() || now <= token_times.front()) return {};
        const double start = std::max(token_times.front(), now - 2.0);
        const auto begin = std::upper_bound(token_times.begin(), token_times.end(), start);
        const auto end = std::upper_bound(begin, token_times.end(), now);
        return std::distance(begin, end) / (now - start);
    }
    json snapshot(double now) const {
        if (ended >= 0) now = ended;
        auto num = [](std::optional<double> v) -> json { return v ? json(*v) : json(nullptr); };
        const double pp = prefill_start < 0 ? 0 : (prefill_end >= 0 ? prefill_end : now) - prefill_start;
        return {{"stage", stage}, {"end_reason", reason}, {"error", error}, {"elapsed_s", now},
            {"model_reused", model_reused},
            {"load_s", load_start < 0 ? json(nullptr) : json((load_end >= 0 ? load_end : now) - load_start)},
            {"input_tokens", input_total}, {"input_processed", input_done}, {"input_reused", 0},
            {"prefill_batches", prefill_batches}, {"prefill_logits_batches", prefill_logits_batches},
            {"prefill_s", prefill_start < 0 ? json(nullptr) : json(pp)},
            {"prefill_tps", pp > 0 && input_done > 0 ? json(input_done / pp) : json(nullptr)},
            {"ttft_s", token_times.empty() ? json(nullptr) : json(token_times.front())},
            {"output_tokens", token_times.size()}, {"decode_average_tps", num(average_tps())},
            {"decode_window_tps", num(window_tps(now))},
            {"decode_window_s", token_times.empty() ? 0.0 : std::min(2.0, now - token_times.front())},
            {"generation_s", token_times.empty() ? json(nullptr) : json(now - token_times.front())},
            {"working_set_bytes", memory_available ? json(working_set) : json(nullptr)},
            {"private_bytes", memory_available ? json(private_bytes) : json(nullptr)},
            {"system_available_bytes", memory_available ? json(available) : json(nullptr)},
            {"sampled_peak_working_set_bytes", memory_available ? json(peak_working_set) : json(nullptr)},
            {"system_commit_total_bytes", system_commit_available ? json(commit_total) : json(nullptr)},
            {"system_commit_limit_bytes", system_commit_available ? json(commit_limit) : json(nullptr)},
            {"process_page_fault_count", memory_available ? json(page_faults) : json(nullptr)},
            {"process_io_read_bytes", io_available ? json(io_read_bytes) : json(nullptr)},
            {"process_io_write_bytes", io_available ? json(io_write_bytes) : json(nullptr)}};
    }
};

inline void check_context(int input, int context) {
    if (input >= context) throw std::runtime_error("Input leaves no generation room in configured context; change --context explicitly or shorten input");
}
} // namespace wllm
