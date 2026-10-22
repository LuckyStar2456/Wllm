#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <io.h>
#include <intrin.h>
#include "core.h"
#include "candidate_score.h"
#include "system_one.h"
#include "resources.h"
#include "profile.h"
#include "terminal.h"
#include "llama.h"
#include "llama-ext.h" // WIP allocation query API, bound to the exact dependencies.json commit.
#include "gguf.h"
#include "ggml-cpu.h"
#include "chat.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

using namespace wllm;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
#ifndef WLLM_VULKAN
#define WLLM_VULKAN 0
#endif
static std::atomic<bool> interrupted{false};
static void check_cpu_target() {
    int features[4]{};
    __cpuid(features, 1);
    const uint32_t avx_os = (1u << 27) | (1u << 28);
    if ((uint32_t(features[2]) & avx_os) != avx_os || (_xgetbv(0) & 6) != 6)
        throw std::runtime_error("CPU/OS AVX state support is required");
    __cpuidex(features, 7, 0);
    if ((uint32_t(features[1]) & (1u << 5)) == 0) throw std::runtime_error("CPU AVX2 support is required");
    if (WLLM_AVX512) {
        const uint32_t ebx = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
        const uint32_t ecx = (1u << 1) | (1u << 11);
        if ((uint32_t(features[1]) & ebx) != ebx || (uint32_t(features[2]) & ecx) != ecx || (_xgetbv(0) & 0xe6) != 0xe6)
            throw std::runtime_error("This build requires AVX512 F/DQ/CD/BW/VL/VBMI/VNNI and OS support; rebuild with -CpuTarget avx2");
    }
}
static BOOL WINAPI console_handler(DWORD kind) {
    if (kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT) { interrupted = true; return TRUE; }
    return FALSE;
}

static std::string read_text(const std::string & path) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::string s((std::istreambuf_iterator<char>(f)), {});
    if (s.compare(0, 3, "\xef\xbb\xbf") == 0) s.erase(0, 3);
    return s;
}

struct State {
    Terminal terminal;
    Config config;
    Metrics metrics;
    Profile profile;
    json identity = json::object();
    json decision = nullptr;
    Clock::time_point start = Clock::now();
    std::mutex mutex;
    std::condition_variable wake;
    bool finished = false;
    bool memory_diagnostic = false;
    std::atomic<bool> timed_out{false};
    std::string cancel_file;
    double now() const { return std::chrono::duration<double>(Clock::now() - start).count(); }
    bool cancelled() {
        if (config.timeout_s > 0 && now() >= config.timeout_s) timed_out = true;
        return interrupted.load() || timed_out.load();
    }
    void sample_memory() {
        PROCESS_MEMORY_COUNTERS_EX p{};
        p.cb = sizeof(p);
        MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&p), sizeof(p)) && GlobalMemoryStatusEx(&m)) {
            metrics.memory_available = true;
            metrics.working_set = p.WorkingSetSize;
            metrics.private_bytes = p.PrivateUsage;
            metrics.available = m.ullAvailPhys;
            metrics.peak_working_set = std::max(metrics.peak_working_set, metrics.working_set);
            metrics.process_peak_working_set = p.PeakWorkingSetSize;
            metrics.process_peak_private = p.PeakPagefileUsage;
            metrics.page_faults = p.PageFaultCount;
        }
        PERFORMANCE_INFORMATION commit{}; commit.cb = sizeof(commit);
        if (GetPerformanceInfo(&commit, sizeof(commit))) {
            metrics.system_commit_available = true;
            metrics.commit_total = uint64_t(commit.CommitTotal) * commit.PageSize;
            metrics.commit_limit = uint64_t(commit.CommitLimit) * commit.PageSize;
        }
        IO_COUNTERS io{};
        if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
            metrics.io_available = true;
            metrics.io_read_bytes = io.ReadTransferCount; metrics.io_write_bytes = io.WriteTransferCount;
        }
    }
};

// Query allocations only at lifecycle boundaries, never from the monitor thread.
static void observe_allocations(State & s, llama_context * context, const char * phase) {
    if (!s.memory_diagnostic) return;
    json buffers = json::array();
    uint64_t model = 0, state = 0, compute = 0;
    for (const auto & [type, usage] : llama_get_memory_breakdown(context)) {
        buffers.push_back({{"buffer_type",ggml_backend_buft_name(type)}, {"model_bytes",usage.model},
            {"context_bytes",usage.context}, {"compute_bytes",usage.compute}});
        model += usage.model; state += usage.context; compute += usage.compute;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.sample_memory();
    s.identity["allocation_observations"].push_back({{"phase",phase}, {"buffers",buffers},
        {"model_buffer_bytes",model}, {"context_buffer_bytes",state}, {"compute_buffer_bytes",compute},
        {"process_private_bytes",s.metrics.private_bytes}, {"process_peak_private_bytes",s.metrics.process_peak_private},
        {"process_working_set_bytes",s.metrics.working_set}, {"process_peak_working_set_bytes",s.metrics.process_peak_working_set},
        {"note","backend buffer allocation; mapped model bytes are not measured residency; process peaks are lifetime cumulative"}});
}

class FinalAllocationObservation {
    State & state;
    llama_context * context;
public:
    FinalAllocationObservation(State & s, llama_context * ctx) : state(s), context(ctx) {}
    ~FinalAllocationObservation() {
        // Diagnostic reporting must not prevent normal error/cancellation cleanup.
        try { observe_allocations(state, context, "before_context_release"); } catch (...) {}
    }
};

static std::string value(const json & j, const char * key, double divisor = 1) {
    if (j.at(key).is_null()) return "--";
    std::ostringstream s; s << std::fixed << std::setprecision(2) << j.at(key).get<double>() / divisor; return s.str();
}

static void display(const json & j, bool final = false) {
    std::ostringstream s;
    s << "[Wllm " << j.at("stage").get<std::string>() << "] elapsed=" << value(j,"elapsed_s")
      << "s load=" << value(j,"load_s") << "s input=" << j.at("input_processed") << '/' << j.at("input_tokens")
      << " pp=" << value(j,"prefill_tps") << "tok/s TTFT=" << value(j,"ttft_s")
      << "s output=" << j.at("output_tokens") << " live=" << value(j,"decode_window_tps")
      << "tok/s avg=" << value(j,"decode_average_tps") << "tok/s RAM=" << value(j,"working_set_bytes",1048576)
      << "MiB private=" << value(j,"private_bytes",1048576) << "MiB free=" << value(j,"system_available_bytes",1048576)
      << "MiB peak=" << value(j,"sampled_peak_working_set_bytes",1048576) << "MiB";
    s << " reused=" << j.at("model_reused");
    if (!j.at("decision_s").is_null()) s << " decision=" << value(j,"decision_s") << "s";
    if (final) s << " reason=" << j.at("end_reason").get<std::string>();
    // Separate stderr lines work both on a terminal and when stdout is redirected.
    std::cerr << s.str() << '\n';
}

static void observer(State & s, std::ofstream & records, const std::string & id, bool quiet) {
    std::unique_lock<std::mutex> lock(s.mutex);
    while (!s.finished) {
        if (!s.cancel_file.empty()) {
            std::error_code error;
            if (fs::exists(fs::u8path(s.cancel_file), error)) interrupted = true;
        }
        s.cancelled();
        s.sample_memory();
        const auto snapshot = s.metrics.snapshot(s.now());
        lock.unlock();
        if (!quiet) {
            if (s.terminal.active()) s.terminal.render(snapshot, s.config.context);
            else display(snapshot);
        }
        if (records) { records << json({{"event","sample"},{"request_id",id},{"metrics",snapshot}}).dump() << '\n'; records.flush(); }
        lock.lock();
        s.wake.wait_for(lock, std::chrono::milliseconds(s.config.refresh_ms), [&] { return s.finished; });
    }
}

static bool abort_callback(void * data) { return static_cast<State *>(data)->cancelled(); }
static bool load_progress(float, void * data) { return !abort_callback(data); }
static void llama_log(ggml_log_level level, const char * text, void * data) {
    // Keep backend diagnostics separate from generated text and live metrics.
    if (data && level != GGML_LOG_LEVEL_DEBUG) { std::fputs(text, static_cast<FILE *>(data)); std::fflush(static_cast<FILE *>(data)); }
}

static uint64_t meta_u32(const gguf_context * meta, const std::string & name) {
    const auto key = gguf_find_key(meta, name.c_str());
    if (key < 0 || gguf_get_kv_type(meta, key) != GGUF_TYPE_UINT32) throw std::runtime_error("Missing uint32 metadata: " + name);
    return gguf_get_val_u32(meta, key);
}

static void preflight(State & s, bool enforce = true) {
    const auto & c = s.config;
    if (c.model.empty() || !fs::is_regular_file(fs::u8path(c.model))) throw std::runtime_error("GGUF file not found; set model in config or use --model");
    gguf_init_params p{}; p.no_alloc = true;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(c.model.c_str(), p), gguf_free);
    if (!meta) throw std::runtime_error("Cannot read GGUF metadata");
    const auto arch_key = gguf_find_key(meta.get(), "general.architecture");
    if (arch_key < 0 || gguf_get_kv_type(meta.get(), arch_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(meta.get(), arch_key)) != "qwen35")
        throw std::runtime_error("Prototype supports qwen35 text GGUF only");
    if (meta_u32(meta.get(), "general.file_type") != LLAMA_FTYPE_MOSTLY_Q4_K_M)
        throw std::runtime_error("Prototype baseline requires Q4_K_M; other precisions are not yet validated");
    const uint64_t train = meta_u32(meta.get(), "qwen35.context_length");
    if (uint64_t(c.context) > train) throw std::runtime_error("Requested context exceeds model training context");
    const auto heads_key = gguf_find_key(meta.get(), "qwen35.attention.head_count_kv");
    const auto interval = meta_u32(meta.get(), "qwen35.full_attention_interval");
    const auto layers = meta_u32(meta.get(), "qwen35.block_count");
    if (uint32_t(c.gpu_layers) > layers + 1) throw std::runtime_error("Requested GPU layers exceed this model including its output layer; refusing implicit placement change");
    if (!interval || layers % interval || gguf_find_key(meta.get(), "qwen35.attention.recurrent_layers") >= 0)
        throw std::runtime_error("Resource estimate requires the validated qwen35 periodic layer layout");
    uint64_t heads = 0;
    if (heads_key < 0) throw std::runtime_error("Missing KV head metadata");
    if (gguf_get_kv_type(meta.get(), heads_key) == GGUF_TYPE_ARRAY && gguf_get_arr_type(meta.get(), heads_key) == GGUF_TYPE_UINT32) {
        const auto * a = static_cast<const uint32_t *>(gguf_get_arr_data(meta.get(), heads_key));
        for (size_t i=0; i<gguf_get_arr_n(meta.get(),heads_key); ++i) heads += a[i];
    } else {
        heads = meta_u32(meta.get(), "qwen35.attention.head_count_kv") *
            (layers / interval);
    }
    const uint64_t kv = uint64_t(c.context) * heads * 2 * (meta_u32(meta.get(),"qwen35.attention.key_length") + meta_u32(meta.get(),"qwen35.attention.value_length"));
    const uint64_t weights = fs::file_size(fs::u8path(c.model));
    // Match pinned qwen35 single-sequence hybrid memory (F32 conv + F32 recurrent).
    // Explicit recurrent-layer overrides require a separate estimate rather than guessing.
    const uint64_t recurrent_layers = layers - layers / interval;
    const uint64_t conv = meta_u32(meta.get(), "qwen35.ssm.conv_kernel");
    const uint64_t inner = meta_u32(meta.get(), "qwen35.ssm.inner_size");
    const uint64_t state = meta_u32(meta.get(), "qwen35.ssm.state_size");
    const uint64_t groups = meta_u32(meta.get(), "qwen35.ssm.group_count");
    if (!conv) throw std::runtime_error("Invalid recurrent convolution size");
    const uint64_t recurrent = recurrent_layers * 4 * ((conv - 1) * (inner + 2 * groups * state) + state * inner);
    // A conservative admission estimate, not a measured runtime allocation.
    const uint64_t workspace = 3072ULL << 20;
    const uint64_t estimated = weights + kv + recurrent + workspace;
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) throw std::runtime_error("Cannot read system memory budget");
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.identity = {{"architecture","qwen35"},{"quantization","Q4_K_M"},{"weight_bytes",weights},
            {"kv_estimate_bytes",kv},{"recurrent_estimate_bytes",recurrent},{"working_budget_estimate_bytes",estimated},{"physical_total_bytes",m.ullTotalPhys},
            {"physical_available_at_admission",m.ullAvailPhys},{"model_context",train},
            {"llama_commit",WLLM_LLAMA_COMMIT},{"pid",GetCurrentProcessId()}};
        s.identity["cpu_build"] = WLLM_CPU_TARGET;
        const auto source_key = gguf_find_key(meta.get(), "wllm.source_sha256");
        if (source_key >= 0 && gguf_get_kv_type(meta.get(), source_key) == GGUF_TYPE_STRING)
            s.identity["original_weight_sha256"] = gguf_get_val_str(meta.get(), source_key);
    }
    const uint64_t reserve = uint64_t(c.reserve_mib * 1048576);
    const auto resources = system_resources();
    const uint64_t headroom = resources.at("commit_headroom_bytes").get<uint64_t>();
    const uint64_t additional_copy = c.cpu_weight_repacking ? weights : 0;
    const uint64_t physical_required = estimated + additional_copy + reserve;
    const uint64_t commit_required = kv + recurrent + workspace + additional_copy + reserve;
    std::vector<std::string> reasons;
    if (c.gpu_layers) reasons.push_back("experimental_vulkan_memory_requires_separate_calibration");
    if (physical_required > resources.at("physical_available_bytes").get<uint64_t>()) reasons.push_back("available_physical_below_conservative_resident_budget");
    if (commit_required > headroom) reasons.push_back("system_commit_headroom_below_estimate");
    if (c.cpu_weight_repacking) reasons.push_back("experimental_repacking_not_numerically_accepted");
    s.identity["resource_ledger"] = {
        {"estimate_scope", "pinned CPU qwen35, one sequence, no prefix/draft/rollback cache"},
        {"source", "GGUF metadata + pinned llama-hparams.cpp/llama-model.cpp; workspace is a conservative allowance"},
        {"system_at_admission", resources},
        {"components", {{"mapped_file_bytes", weights}, {"mapped_file_resident_bytes", nullptr},
            {"attention_kv_f16_estimate_bytes", kv}, {"recurrent_f32_estimate_bytes", recurrent},
            {"workspace_allowance_bytes", workspace}, {"workspace_measured_bytes", nullptr},
            {"extra_weight_copy_estimate_bytes", additional_copy}, {"prefix_cache_bytes", 0},
            {"draft_bytes", 0}, {"rollback_snapshot_bytes", 0}, {"system_reserve_bytes", reserve}}},
        {"startup_physical_budget_bytes", physical_required}, {"steady_physical_budget_bytes", physical_required},
        {"additional_commit_budget_bytes", commit_required}, {"peaks_calibrated", false},
        {"performance_preflight_eligible", reasons.empty()}, {"performance_rejection_reasons", reasons},
        {"inference_policy", "performance screening is stricter than ordinary inference; no automatic setting changes"}};
    if (c.gpu_layers) s.identity["resource_ledger"]["estimate_scope"]="CPU reference estimate only; Vulkan driver/device peaks are not calibrated";
    if (!enforce) return;
    if (estimated + reserve > m.ullTotalPhys || kv + recurrent + (512ULL << 20) + reserve > m.ullAvailPageFile || m.ullAvailPhys < reserve) {
        std::ostringstream error;
        error << "Insufficient memory budget: estimated weights+KV+recurrent+workspace=" << estimated / 1048576
              << " MiB, reserve=" << reserve / 1048576 << " MiB, physical total=" << m.ullTotalPhys / 1048576
              << " MiB, available=" << m.ullAvailPhys / 1048576 << " MiB; requested context=" << c.context
              << ". No automatic context/precision reduction; close applications or explicitly choose --context.";
        throw std::runtime_error(error.str());
    }
    if (estimated + reserve > m.ullAvailPhys) std::cerr << "[Wllm] Available RAM is below the working-budget estimate; paging may occur. No settings were changed.\n";
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,true,true);
    if (n >= 0) throw std::runtime_error("Unexpected empty tokenization");
    std::vector<llama_token> tokens(-n);
    n = llama_tokenize(vocab,text.data(),int(text.size()),tokens.data(),int(tokens.size()),true,true);
    if (n < 0) throw std::runtime_error("Tokenizer failed");
    tokens.resize(n); return tokens;
}

// Own one CPU pool for the request; detach before freeing it or the context.
class CpuPool {
    llama_context * context;
    ggml_threadpool_t pool = nullptr;
public:
    CpuPool(llama_context * ctx, const Config & config) : context(ctx) {
        if (!config.persistent_threadpool) return;
        auto params = ggml_threadpool_params_default(std::max(config.threads, config.effective_threads_batch()));
        pool = ggml_threadpool_new(&params);
        if (!pool) throw std::runtime_error("CPU thread pool allocation failed");
        llama_attach_threadpool(context, pool, pool);
    }
    ~CpuPool() {
        if (pool) { llama_detach_threadpool(context); ggml_threadpool_free(pool); }
    }
    CpuPool(const CpuPool &) = delete;
    CpuPool & operator=(const CpuPool &) = delete;
};

static void set_batch(llama_batch_ext * batch, const llama_token * tokens, int count, int position, bool output = true) {
    llama_batch_ext_clear(batch);
    for (int i=0;i<count;++i) {
        const int idx=llama_batch_ext_add_token(batch,0,tokens[i]);
        llama_pos pos=position+i;
        if (idx < 0 || !llama_batch_ext_set_pos(batch,idx,&pos)) throw std::runtime_error("Invalid token batch");
    }
    if (output && !llama_batch_ext_set_output_logits(batch,count-1,true)) throw std::runtime_error("Cannot request output logits");
}

static std::vector<int32_t> candidate_tokens(const CandidateTask & task, const llama_vocab * vocab,
        const std::string & prefix, const std::vector<llama_token> & prefix_tokens) {
    std::vector<int32_t> result;
    std::set<int32_t> seen;
    for (const auto & c:task.candidates) {
        const auto combined=tokenize(vocab,prefix+c.label);
        if (combined.size()!=prefix_tokens.size()+1 || !std::equal(prefix_tokens.begin(),prefix_tokens.end(),combined.begin()))
            throw std::runtime_error("Label must be one token at the unchanged answer boundary: " + c.id);
        const auto token=combined.back();
        const auto attr=llama_vocab_get_attr(vocab,token);
        if ((attr & (LLAMA_TOKEN_ATTR_CONTROL|LLAMA_TOKEN_ATTR_UNKNOWN|LLAMA_TOKEN_ATTR_UNUSED|LLAMA_TOKEN_ATTR_USER_DEFINED)) ||
            llama_vocab_is_eog(vocab,token) || !seen.insert(token).second)
            throw std::runtime_error("Label is special or maps to a duplicate token: " + c.id);
        std::vector<char> piece(c.label.size()+32);
        const int n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,false);
        if (n<0 || std::string(piece.data(),size_t(n))!=c.label)
            throw std::runtime_error("Label bytes do not match the answer token: " + c.id);
        result.push_back(token);
    }
    return result;
}

static void assign_system_one_labels(CandidateTask & task, const llama_vocab * vocab) {
    const std::string boundary="<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const auto prefix=tokenize(vocab,boundary);
    std::vector<std::string> pool;
    const std::string simple="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (char c:simple) pool.emplace_back(1,c);
    // Large choice sets use other short ordinary ASCII tokens. Every mapping is
    // checked again at the complete rendered answer prefix before any inference.
    std::set<std::string> considered;
    size_t next=0;
    auto add=[&](const std::string & label) {
        if (next>=task.candidates.size() || !considered.insert(label).second) return;
        CandidateTask probe; probe.candidates.push_back({"probe",label,"probe"});
        try {
            candidate_tokens(probe,vocab,boundary,prefix);
            task.candidates[next++].label=label;
        } catch (const std::runtime_error &) {} // This token spelling is not a valid label at this boundary.
    };
    for (const auto & label:pool) add(label);
    for (llama_token token=0;next<task.candidates.size() && token<llama_vocab_n_tokens(vocab);++token) {
        char piece[32];
        const int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,false);
        if (n<1 || n>16) continue;
        const std::string label(piece,size_t(n));
        if (std::all_of(label.begin(),label.end(),[](unsigned char ch) {
            return (ch>='A' && ch<='Z') || (ch>='a' && ch<='z') || (ch>='0' && ch<='9');
        })) add(label);
    }
    if (next!=task.candidates.size()) throw std::runtime_error("Not enough validated single-token labels for this model");
}

static std::string run(State & s, const std::string & prompt, const std::string & system, const std::string & inspect_path,
        const std::string & logits_path, ModelPtr * resident, const CandidateTask * task) {
    ModelPtr transient(nullptr, llama_model_free);
    auto & model = resident ? *resident : transient;
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.model_reused = bool(model); }
    preflight(s);
    const auto & c=s.config;
    if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.stage="loading"; s.metrics.load_start=s.now(); }
    llama_model_params mp=llama_model_default_params();
    mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP; mp.progress_callback=load_progress; mp.progress_callback_user_data=&s;
    ggml_backend_dev_t devices[2] = {nullptr,nullptr};
    if (c.gpu_layers) {
        for (size_t i=0;i<ggml_backend_dev_count();++i) {
            auto device = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "Vulkan") continue;
            if (devices[0]) throw std::runtime_error("Validation requires exactly one Vulkan device; refusing ambiguous placement");
            devices[0]=device;
        }
        if (!devices[0]) throw std::runtime_error("Requested Vulkan placement has no device; refusing CPU fallback");
        mp.devices=devices; mp.n_gpu_layers=c.gpu_layers;
        std::lock_guard<std::mutex> lock(s.mutex);
        s.identity["gpu_device"]={{"name",ggml_backend_dev_name(devices[0])},{"description",ggml_backend_dev_description(devices[0])}};
    }
    mp.use_extra_bufts=c.cpu_weight_repacking;
    mp.vocab_only=!inspect_path.empty();
    if (!model) model.reset(llama_model_load_from_file(c.model.c_str(),mp));
    if (!model) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        throw std::runtime_error("Model load failed; see reports/backend.log");
    }
    const llama_vocab * vocab=llama_model_get_vocab(model.get());
    CandidateTask mapped_task;
    if (task && task->automatic_labels) {
        mapped_task=*task; assign_system_one_labels(mapped_task,vocab); task=&mapped_task;
    }
    const char * template_src=llama_model_chat_template(model.get(),nullptr);
    if (!template_src || !*template_src) throw std::runtime_error("GGUF has no chat template; refusing to guess one");
    auto templates=common_chat_templates_init(model.get(),"");
    common_chat_templates_inputs inputs;
    const bool thinking=!task && c.thinking;
    inputs.enable_thinking=thinking;
    inputs.chat_template_kwargs["enable_thinking"]=thinking ? "true" : "false";
    const std::string effective_system=task?task->system_prompt(system):system;
    const std::string effective_prompt=task?task->user_prompt(prompt):prompt;
    if (!effective_system.empty()) { common_chat_msg m; m.role="system"; m.content=effective_system; inputs.messages.push_back(m); }
    common_chat_msg user; user.role="user"; user.content=effective_prompt; inputs.messages.push_back(user);
    const auto chat=common_chat_templates_apply(templates.get(),inputs);
    const auto tokens=tokenize(vocab,chat.prompt);
    check_context(int(tokens.size()),c.context,!task);
    std::vector<int32_t> labels;
    if (task) {
        const std::string boundary="<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (chat.prompt.size()<boundary.size() || chat.prompt.compare(chat.prompt.size()-boundary.size(),boundary.size(),boundary)!=0)
            throw std::runtime_error("Unverified non-thinking answer boundary; candidate_score requires the recorded Qwen3.5 template");
        labels=candidate_tokens(*task,vocab,chat.prompt,tokens);
    }
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.metrics.input_total=int(tokens.size());
        s.identity["template_source"]="GGUF tokenizer.chat_template via upstream Jinja";
        s.identity["thinking_prompt_suffix"]=chat.generation_prompt;
        if (task) {
            s.identity["candidate_task"]=task->manifest();
            s.identity["candidate_token_ids"]=labels;
            s.identity["generation_loop_entered"]=false;
        }
    }
    if (!inspect_path.empty()) {
        std::ofstream out(fs::u8path(inspect_path),std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write inspect file");
        json inspected={{"rendered_prompt",chat.prompt},{"token_ids",tokens},{"template",template_src},{"config",c.dump()}};
        if (task) {
            inspected["candidate_task"]=task->manifest(); inspected["candidate_token_ids"]=labels;
            inspected["effective_system"]=effective_system; inspected["effective_user"]=effective_prompt;
        }
        out << inspected.dump(2);
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.load_end=s.now(); }
        return "inspected";
    }
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=c.context; cp.n_batch=c.batch; cp.n_ubatch=c.batch; cp.n_seq_max=1;
    cp.n_threads=c.threads; cp.n_threads_batch=c.effective_threads_batch();
    cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=c.gpu_layers>0; cp.op_offload=false; cp.no_perf=false;
    cp.abort_callback=abort_callback; cp.abort_callback_data=&s;
    if (s.profile.operators) { cp.cb_eval=Profile::callback; cp.cb_eval_user_data=&s.profile; }
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
    if (!ctx) throw std::runtime_error("Context allocation failed; requested context was not reduced");
    if (llama_n_ctx(ctx.get()) != uint32_t(c.context)) throw std::runtime_error("Backend changed context size; refusing implicit change");
    observe_allocations(s, ctx.get(), "context_ready");
    FinalAllocationObservation final_observation(s, ctx.get());
    CpuPool pool(ctx.get(), c);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.metrics.load_end=s.now(); s.metrics.stage="prefill"; s.metrics.prefill_start=s.now();
        s.sample_memory();
    }
    std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(nullptr,llama_sampler_free);
    if (!task) {
        auto sp=llama_sampler_chain_default_params();
        sampler.reset(llama_sampler_chain_init(sp));
        llama_sampler_chain_add(sampler.get(),llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),
            c.repeat_last_n < 0 ? c.context : c.repeat_last_n,float(c.repeat_penalty),float(c.frequency_penalty),float(c.presence_penalty)));
        if (c.temperature==0) llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
        else {
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_top_k(c.top_k));
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_top_p(float(c.top_p),1));
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_min_p(float(c.min_p),1));
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_temp(float(c.temperature)));
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_dist(c.seed));
        }
        for (auto t:tokens) llama_sampler_accept(sampler.get(),t);
    }
    std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);
    int pos=0;
    while(pos<int(tokens.size())) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        int n=std::min(c.batch,int(tokens.size())-pos);
        // Intermediate batches update all attention/recurrent state but need no vocabulary projection.
        const bool output = !c.prefill_last_only || pos+n==int(tokens.size());
        set_batch(batch.get(),tokens.data()+pos,n,pos,output);
        s.profile.phase="prefill";
        int rc;
        {
            Profile::Span timer(s.profile,"prefill_process");
            rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
            if (s.profile.enabled) llama_synchronize(ctx.get());
        }
        if (rc!=0) { if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled"; throw std::runtime_error("Prefill failed: " + std::to_string(rc)); }
        pos+=n;
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.input_done=pos;
          ++s.metrics.prefill_batches; if (output) ++s.metrics.prefill_logits_batches; }
    }
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.prefill_end=s.now(); s.metrics.stage=task?"selecting":"generating"; }
    observe_allocations(s, ctx.get(), "prefill_complete");
    if (!logits_path.empty()) {
        // Diagnostic only: full, unsampled vocabulary logits for this fixed input prefix.
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) throw std::runtime_error("No final prefill logits for diagnostic");
        std::ofstream out(fs::u8path(logits_path), std::ios::binary);
        out.write(reinterpret_cast<const char *>(logits), std::streamsize(llama_vocab_n_tokens(vocab)) * sizeof(float));
        if (!out) throw std::runtime_error("Cannot write logits diagnostic");
    }
    if (task) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.decision_start=s.now(); }
        auto decision=select_candidate(*task,labels,llama_get_logits_ith(ctx.get(),-1),llama_vocab_n_tokens(vocab));
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        const auto reason=decision.at("status").get<std::string>();
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.decision_ready=s.now(); s.decision=std::move(decision); }
        return reason;
    }
    int generated=0;
    while(true) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        llama_token token;
        { Profile::Span timer(s.profile,"sample"); token=llama_sampler_sample(sampler.get(),ctx.get(),-1); }
        if (llama_vocab_is_eog(vocab,token)) return "eos";
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.metrics.token_times.push_back(s.now()); s.metrics.token_ids.push_back(token);
        }
        ++generated;
        {
            Profile::Span timer(s.profile,"render_and_write");
            char token_buffer[256]; int n=llama_token_to_piece(vocab,token,token_buffer,sizeof(token_buffer),0,true);
            std::string piece;
            if (n>=0) piece.assign(token_buffer,n);
            else { piece.resize(-n); n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true); if(n<0) throw std::runtime_error("Token rendering failed"); piece.resize(n); }
            s.terminal.write(piece);
            if (!std::cout) throw std::runtime_error("Output stream failed");
        }
        if (c.max_tokens>0 && generated>=c.max_tokens) return "max_tokens";
        if (int(tokens.size())+generated>=c.context) return "context_limit";
        set_batch(batch.get(),&token,1,pos++);
        s.profile.phase="decode";
        int rc;
        {
            Profile::Span timer(s.profile,"decode_process");
            rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
            if (s.profile.enabled) llama_synchronize(ctx.get());
        }
        if (rc!=0) { if(s.cancelled()) return s.timed_out ? "timeout" : "cancelled"; throw std::runtime_error("Decode failed: " + std::to_string(rc)); }
    }
}

static void usage() {
    std::cout << "Wllm 0.2.0 - local Qwen3.5 Q4_K_M text inference\n"
        "  wllm --config configs/local.json --prompt \"Hello\"\n"
        "  --model PATH --context N (default 16384) --threads N --batch N\n"
        "  --threads-batch N (0 = threads) --persistent-threadpool on|off\n"
        "  --prefill-last-only on|off (skip unused intermediate logits)\n"
        "  --cpu-weight-repacking on|off (experimental CPU weight layout)\n"
        "  --prompt TEXT | --prompt-file UTF8_FILE (otherwise read one line)\n"
        "  --interactive (retain weights between independent prompts; /exit quits)\n"
        "  --system TEXT --max-tokens N (-1 = until EOS/context) --seed N\n"
        "  --temperature N --thinking on|off --timeout SECONDS\n"
        "  --record PATH --cancel-file PATH --inspect PATH --quiet-metrics\n"
        "  --logits-file PATH (diagnostic: final prefill logits, float32 little-endian)\n"
        "  --mode generate|candidate_score --candidates UTF8_JSON (CPU; returns a JSON decision)\n"
        "  --mode system_one [--request-file UTF8_JSON] (state + questions; stdin/interactive JSON also supported)\n"
        "  --profile (phase timings) --profile-ops (intrusive operator diagnostic)\n"
        "  --resources PATH (write resource/CPU budget report; no inference or prompt read)\n"
        "  --memory-diagnostic (allocation/OS peaks; excluded from performance benchmarks)\n"
        "  --gpu-layers N (experimental Vulkan validation build only; default 0)\n"
        "Metrics: stderr + JSONL; generated text: stdout. Ctrl+C cancels.\n";
}

class BackendRuntime {
    std::unique_ptr<FILE,decltype(&std::fclose)> log{nullptr,std::fclose};
public:
    BackendRuntime() {
        fs::create_directories("reports");
        log.reset(std::fopen("reports/backend.log","a"));
        if (!log) throw std::runtime_error("Cannot open backend log");
        llama_log_set(llama_log,log.get());
        llama_backend_init();
    }
    ~BackendRuntime() { llama_backend_free(); llama_log_set(nullptr,nullptr); }
};

static int execute_request(const Config & config, const std::string & prompt, const std::string & system,
        const std::string & inspect, const std::string & logits_file, const std::string & cancel_file,
        bool quiet, bool profile, bool profile_ops, bool memory_diagnostic, ModelPtr * resident, const CandidateTask * task,
        bool emit_decision=true, json * decision_out=nullptr) {
    State state; state.config=config; state.cancel_file=cancel_file;
    state.memory_diagnostic=memory_diagnostic;
    state.profile.enabled=profile || profile_ops; state.profile.operators=profile_ops;
    if(prompt.empty()) throw std::runtime_error("Prompt is empty");
    if(!inspect.empty() && !logits_file.empty()) throw std::runtime_error("--inspect cannot be combined with --logits-file");
    if(!cancel_file.empty() && fs::exists(fs::u8path(cancel_file))) throw std::runtime_error("Cancel file already exists; use a new path");
    state.start=Clock::now();
    fs::create_directories("reports");
    if(!state.config.record.empty() && fs::u8path(state.config.record).has_parent_path()) fs::create_directories(fs::u8path(state.config.record).parent_path());
    std::ofstream record(fs::u8path(state.config.record),std::ios::app|std::ios::binary);
    if(!record) throw std::runtime_error("Cannot open metric record file");
    const std::string id=std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())+"-"+std::to_string(GetCurrentProcessId());
    record << json({{"event","start"},{"schema_version",1},{"request_id",id},{"config",state.config.dump()},
        {"candidate_task",task?task->manifest():json(nullptr)},
        {"logits_diagnostic",!logits_file.empty()},
        {"memory_diagnostic",memory_diagnostic},
        {"profiling_diagnostic",state.profile.enabled},{"operator_profiling",profile_ops},
        {"cpu_build",WLLM_CPU_TARGET},{"resident_session",resident!=nullptr},
        {"llama_commit",WLLM_LLAMA_COMMIT},{"window_s",2},{"pid",GetCurrentProcessId()}}).dump() << '\n';
    std::cerr << "[Wllm] Q4_K_M CPU context=" << state.config.context << " thinking=" << (!task && state.config.thinking?"on":"off")
              << " mode=" << state.config.inference_mode
              << " cpu_build=" << WLLM_CPU_TARGET
              << " pid=" << GetCurrentProcessId() << " record=" << state.config.record << '\n';
    std::cerr << "[Wllm] model=" << state.config.model << " threads=" << state.config.threads
              << " threads_batch=" << state.config.effective_threads_batch()
              << " persistent_threadpool=" << state.config.persistent_threadpool
              << " prefill_last_only=" << state.config.prefill_last_only
              << " cpu_weight_repacking=" << state.config.cpu_weight_repacking
              << " batch=" << state.config.batch << " temperature=" << state.config.temperature
              << " top_k=" << state.config.top_k << " top_p=" << state.config.top_p
              << " presence_penalty=" << state.config.presence_penalty << '\n';
    if (state.config.cpu_weight_repacking)
        std::cerr << "[Wllm] Experimental CPU repacking: failed current baseline numerical checks; research use only.\n";
    // A decision is one JSON line, including on a real console; never place it in the alternate screen.
    state.terminal.start(quiet || task!=nullptr);
    std::thread monitor(observer,std::ref(state),std::ref(record),std::cref(id),quiet);
    std::string reason;
    try { reason=run(state,prompt,system,inspect,logits_file,resident,task); }
    catch(const std::exception & e) { reason=state.cancelled()?(state.timed_out?"timeout":"cancelled"):"error"; std::lock_guard<std::mutex> lock(state.mutex); state.metrics.error=e.what(); }
    if (task && (reason=="selected" || reason=="abstained") && state.cancelled())
        reason=state.timed_out?"timeout":"cancelled";
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.sample_memory(); state.metrics.ended=state.now(); state.metrics.reason=reason;
        state.metrics.stage=reason=="error"?"failed":reason=="cancelled"||reason=="timeout"?"cancelled":"finished";
        state.finished=true;
    }
    state.wake.notify_all(); if(monitor.joinable()) monitor.join();
    state.terminal.finish();
    const auto final=state.metrics.snapshot(state.now());
    if (task && inspect.empty()) {
        if (reason!="selected" && reason!="abstained")
            state.decision={{"schema_version",1},{"mode","candidate_score"},{"route","fast"},{"status",reason},
                {"selected_candidate_id",nullptr},{"top_candidate_id",nullptr},{"scores",nullptr},{"error",state.metrics.error}};
        state.decision["request_id"]=id;
        state.decision["metrics"]=final;
    }
    display(final,true);
    const auto profile_result=state.profile.dump();
    if (state.profile.enabled) {
        std::cerr << "[Wllm profile] Diagnostic timings (not a speedup benchmark):\n";
        for (const auto & row : profile_result.at("stages"))
            std::cerr << "  " << row.at("name").get<std::string>() << "=" << row.at("seconds")
                      << "s calls=" << row.at("calls") << '\n';
        if (profile_ops) std::cerr << "  Operator callbacks split/synchronize the graph; details in JSONL end.profile.operators\n";
    }
    if(!state.metrics.error.empty()) std::cerr << "[Wllm] " << state.metrics.error << '\n';
    record << json({{"event","end"},{"request_id",id},{"metrics",final},{"identity",state.identity},
        {"decision",state.decision},
        {"profile",profile_result},
        {"token_times_s",state.metrics.token_times},{"token_ids",state.metrics.token_ids},
        {"load_start_s",state.metrics.load_start},{"load_end_s",state.metrics.load_end},
        {"prefill_start_s",state.metrics.prefill_start},{"prefill_end_s",state.metrics.prefill_end}}).dump() << '\n';
    record.flush();
    if(!record) { std::cerr << "Metric record write failed\n"; return 1; }
    if (decision_out) *decision_out=state.decision;
    if (task && inspect.empty() && emit_decision) {
        std::cout << state.decision.dump() << '\n' << std::flush;
        if (!std::cout) { std::cerr << "Decision output write failed\n"; return 1; }
    }
    return reason=="error"?1:reason=="cancelled"?130:reason=="timeout"?124:0;
}

static int execute_system_one(const Config & config, const std::string & body, const std::string & cancel_file,
        bool quiet, bool profile, bool profile_ops, bool memory_diagnostic, ModelPtr & resident) {
    const auto started=Clock::now();
    const auto elapsed=[&] { return std::chrono::duration<double>(Clock::now()-started).count(); };
    const std::string id="systemone-"+std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(started.time_since_epoch()).count())+
        "-"+std::to_string(GetCurrentProcessId());
    json response={{"model",SystemOneRequest::actual_model},{"answers",json::object()},
        {"usage",{{"input_tokens",0},{"output_tokens",0}}},
        {"_wllm",{{"request_id",id},{"protocol","typesafe-system-one-text-v1"},
            {"model_path",config.model},
            {"execution","sequential isolated questions; retained local weights"},
            {"probabilities","candidate-normalized local logits; not calibrated correctness"},
            {"usage_scope","sum of completed prefill batches; aborted partial work is not measurable"}}}};
    int code=0;
    std::string active_question;
    bool validated=false;
    const auto check_budget=[&] {
        if (!cancel_file.empty() && fs::exists(fs::u8path(cancel_file))) interrupted=true;
        if (interrupted.load()) { code=130; throw std::runtime_error("System One request cancelled"); }
        if (config.timeout_s>0 && elapsed()>=config.timeout_s) { code=124; throw std::runtime_error("System One request timed out"); }
    };
    try {
        const auto request=SystemOneRequest::parse(json::parse(body));
        validated=true;
        response["_wllm"]["requested_model"]=request.requested_model;
        response["_wllm"]["questions"]=json::object();
        for (auto q:request.questions) {
            active_question=q.id;
            check_budget();
            if (q.task.candidates.size()==1) {
                // A one-option closed choice is a constant, not a model evaluation.
                response["answers"][q.id]=system_one_answer(q,json::array({{{"id",q.task.candidates[0].id},{"candidate_normalized_score",1.0}}}));
                response["_wllm"]["questions"][q.id]={{"constant_choice",true},{"model_evaluated",false}};
                continue;
            }
            auto effective=config;
            if (config.timeout_s>0) {
                effective.timeout_s=config.timeout_s-elapsed();
                if (effective.timeout_s<=0) { code=124; throw std::runtime_error("System One request timed out"); }
            }
            q.task.protocol_metadata={{"system_one_request_id",id},{"question_id",q.id},{"type",q.type}};
            json decision;
            code=execute_request(effective,request.prompt(),q.system_prompt(),"","",cancel_file,
                quiet,profile,profile_ops,memory_diagnostic,&resident,&q.task,false,&decision);
            if (decision.contains("metrics")) response["usage"]["input_tokens"]=
                response["usage"]["input_tokens"].get<int>()+decision["metrics"]["input_processed"].get<int>();
            if (code!=0) {
                auto message=decision.value("error",std::string());
                if (message.empty()) message=code==124?"Question timed out":code==130?"Question cancelled":"Question evaluation failed";
                throw std::runtime_error(message);
            }
            check_budget();
            response["answers"][q.id]=system_one_answer(q,decision.at("scores"));
            response["_wllm"]["questions"][q.id]={{"request_id",decision.at("request_id")},{"metrics",decision.at("metrics")},
                {"template_version",decision.at("template_version")}};
        }
        check_budget();
    } catch (const std::exception & e) {
        if (code==0) code=1;
        response["answers"]=json::object(); // Never present partial completion as a successful batch.
        response["error"]={{"type",code==130?"cancelled":code==124?"timeout":validated?"evaluation_error":"invalid_request"},
            {"message",e.what()},{"question_id",active_question.empty()?json(nullptr):json(active_question)}};
    }
    response["_wllm"]["elapsed_s"]=elapsed();
    response["_wllm"]["status"]=code==0?"completed":"failed";
    try {
        const auto path=fs::u8path(config.record);
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::ofstream record(path,std::ios::app|std::ios::binary);
        record << json({{"event","system_one_end"},{"request_id",id},{"response",response}}).dump() << '\n';
        record.flush();
        if (!record) throw std::runtime_error("Cannot write System One result record");
    } catch (const std::exception & e) {
        code=1; response["answers"]=json::object();
        response["_wllm"]["status"]="failed";
        response["error"]={{"type","record_error"},{"message",e.what()}};
    }
    std::cout << response.dump() << '\n' << std::flush;
    return std::cout?code:1;
}

static int main_utf8(int argc,char ** argv) {
    SetConsoleOutputCP(CP_UTF8); SetConsoleCP(CP_UTF8);
    SetConsoleCtrlHandler(console_handler,TRUE);
    std::setlocale(LC_NUMERIC,"C");
    try {
        check_cpu_target();
        json settings=json::object();
        for(int i=1;i<argc;++i) if(std::string(argv[i])=="--config") {
            if(++i==argc) throw std::runtime_error("--config needs a path");
            settings=json::parse(read_text(argv[i]));
        }
        std::string prompt,system,inspect,cancel_file,logits_file,resources_file,candidates_file,request_file;
        bool prompt_set=false, quiet=false, interactive=false, profile=false, profile_ops=false, memory_diagnostic=false;
        const std::map<std::string,std::string> ints={{"--context","context"},{"--threads","threads"},{"--threads-batch","threads_batch"},{"--batch","batch"},{"--gpu-layers","gpu_layers"},{"--max-tokens","max_tokens"},{"--seed","seed"}};
        const std::map<std::string,std::string> bools={{"--thinking","thinking"},{"--persistent-threadpool","persistent_threadpool"},{"--prefill-last-only","prefill_last_only"},{"--cpu-weight-repacking","cpu_weight_repacking"}};
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help" || arg=="-h") { usage(); return 0; }
            if(arg=="--quiet-metrics") { quiet=true; continue; }
            if(arg=="--interactive") { interactive=true; continue; }
            if(arg=="--profile") { profile=true; continue; }
            if(arg=="--profile-ops") { profile_ops=true; continue; }
            if(arg=="--memory-diagnostic") { memory_diagnostic=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing value for " + arg);
            const std::string v=argv[++i];
            if(arg=="--config") continue;
            if(arg=="--prompt" || arg=="--prompt-file") { if(prompt_set) throw std::runtime_error("Use only one prompt source"); prompt=arg=="--prompt"?v:read_text(v); prompt_set=true; }
            else if(arg=="--system") system=v;
            else if(arg=="--model") settings["model"]=v;
            else if(arg=="--record") settings["record"]=v;
            else if(arg=="--inspect") inspect=v;
            else if(arg=="--logits-file") logits_file=v;
            else if(arg=="--cancel-file") cancel_file=v;
            else if(arg=="--resources") resources_file=v;
            else if(arg=="--mode") settings["inference_mode"]=v;
            else if(arg=="--candidates") candidates_file=v;
            else if(arg=="--request-file") request_file=v;
            else if(bools.count(arg)) { if(v!="on" && v!="off") throw std::runtime_error(arg + " must be on or off"); settings[bools.at(arg)]=v=="on"; }
            else if(ints.count(arg)) { const json n=json::parse(v); if(!n.is_number_integer()) throw std::runtime_error("Expected integer for " + arg); settings[ints.at(arg)]=n; }
            else if(arg=="--temperature" || arg=="--timeout") { const json n=json::parse(v); if(!n.is_number()) throw std::runtime_error("Expected number"); settings[arg=="--timeout"?"timeout_s":"temperature"]=n; }
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        const auto config=Config::parse(settings);
        const bool system_one=config.inference_mode=="system_one";
        if (!system_one && !request_file.empty()) throw std::runtime_error("--request-file requires --mode system_one");
        if (system_one && (prompt_set || !system.empty() || !inspect.empty() || !logits_file.empty() || !candidates_file.empty() || !resources_file.empty()))
            throw std::runtime_error("system_one accepts state/questions JSON; do not combine with prompt, system, candidates, inspect, logits or resources");
        std::optional<CandidateTask> task;
        if (config.inference_mode=="candidate_score") {
            if (candidates_file.empty()) throw std::runtime_error("candidate_score requires --candidates FILE");
            task=CandidateTask::parse(json::parse(read_text(candidates_file)));
        } else if (!candidates_file.empty()) throw std::runtime_error("--candidates requires --mode candidate_score");
        if (config.gpu_layers && !WLLM_VULKAN) throw std::runtime_error("Vulkan was not compiled into this executable; use the separate validation build");
        if (!resources_file.empty()) {
            if (task || prompt_set || interactive || !inspect.empty() || !logits_file.empty() || profile || profile_ops || memory_diagnostic || !cancel_file.empty() || !system.empty())
                throw std::runtime_error("--resources is a standalone probe; do not combine with request/diagnostic options");
            State state; state.config = config;
            json report = {{"schema_version",1}, {"evidence", "resource probe only; no model execution"},
                {"unix_time_ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                {"llama_commit",WLLM_LLAMA_COMMIT}, {"cpu_build",WLLM_CPU_TARGET},
                {"system",system_resources()}, {"cpu",cpu_capabilities()}, {"dxgi",dxgi_adapters()}, {"config",config.dump()},
                {"backends", {{"CPU", "compiled; model execution checked separately"},
                    {"Vulkan", WLLM_VULKAN ? "compiled; experimental validation only" : "not compiled into this executable"}, {"SYCL", "not compiled into this executable"},
                    {"OpenVINO", "not verified in this executable"}}}};
            if (!config.model.empty()) { preflight(state, false); report["model"] = state.identity; }
            else report["model"] = nullptr;
            const auto path = fs::u8path(resources_file);
            if (path.has_parent_path()) fs::create_directories(path.parent_path());
            if (fs::exists(path)) throw std::runtime_error("Resource report exists; choose a new path");
            std::ofstream out(path, std::ios::binary); out << report.dump(2) << '\n'; out.flush();
            if (!out) throw std::runtime_error("Cannot write resource report");
            std::cout << "Resource report: " << resources_file << '\n';
            return 0;
        }
        if (interactive && (!inspect.empty() || !logits_file.empty()))
            throw std::runtime_error("Interactive mode cannot overwrite inspect/logits diagnostic files");
        BackendRuntime backend;
        ModelPtr resident(nullptr,llama_model_free);
        int code=0;
        if (system_one) {
            auto request=[&](const std::string & body) {
                const int result=execute_system_one(config,body,cancel_file,quiet,profile,profile_ops,memory_diagnostic,resident);
                if (result!=0) code=result;
                return result;
            };
            if (!request_file.empty()) request(read_text(request_file));
            else if (!interactive) request(std::string(std::istreambuf_iterator<char>(std::cin),{}));
            if (interactive && code!=130 && code!=124) {
                std::string line;
                while (!interrupted.load()) {
                    std::cerr << "SystemOne JSON> ";
                    if (!std::getline(std::cin,line) || line.empty() || line=="/exit") break;
                    const int result=request(line);
                    if (result==130 || result==124) break;
                }
            }
        } else {
        if (!prompt_set) { std::cerr << "Prompt> "; std::getline(std::cin,prompt); }
        while (true) {
            if (interactive && (prompt.empty() || prompt=="/exit")) break;
            code=execute_request(config,prompt,system,inspect,logits_file,cancel_file,quiet,profile,profile_ops,memory_diagnostic,interactive?&resident:nullptr,task?&*task:nullptr);
            if (!interactive || code!=0 || interrupted.load()) break;
            if (!task) std::cout << '\n' << std::flush;
            std::cerr << "[Wllm] Weights retained; each prompt starts a fresh context. /exit or empty line releases the model.\nPrompt> ";
            if (!std::getline(std::cin,prompt)) break;
        }
        }
        if (resident) {
            const auto release_start=Clock::now();
            resident.reset();
            const double release_s=std::chrono::duration<double>(Clock::now()-release_start).count();
            std::ofstream record(fs::u8path(config.record),std::ios::app|std::ios::binary);
            record << json({{"event","session_end"},{"pid",GetCurrentProcessId()},{"model_release_s",release_s}}).dump() << '\n';
            record.flush();
            if (!record) throw std::runtime_error("Cannot record session release");
            std::cerr << "[Wllm] Session ended; retained model released in " << release_s << "s\n";
        }
        return interrupted.load() ? 130 : code;
    } catch(const std::exception & e) { std::cerr << "Wllm: " << e.what() << '\n'; return 1; }
}

int wmain(int argc, wchar_t ** argv) {
    std::vector<std::string> args;
    for (int i=0;i<argc;++i) {
        const int length=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);
        std::string text(length,'\0');
        WideCharToMultiByte(CP_UTF8,0,argv[i],-1,text.data(),length,nullptr,nullptr);
        text.resize(length-1); args.push_back(std::move(text));
    }
    std::vector<char *> pointers;
    for(auto & arg:args) pointers.push_back(arg.data());
    return main_utf8(argc,pointers.data());
}
