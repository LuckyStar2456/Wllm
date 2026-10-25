#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <io.h>
#include <intrin.h>
#include "core.h"
#include "candidate_score.h"
#include "system_one.h"
#include "prefix_cache.h"
#include "resources.h"
#include "profile.h"
#include "terminal.h"
#include "console_ui.h"
#include "model_profile.h"
#include "service_protocol.h"
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

struct RequestIO {
    std::vector<common_chat_msg> messages;
    std::function<bool()> cancel, stopped;
    std::function<bool(const std::string &)> output;
};

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
    const RequestIO * io = nullptr;
    double now() const { return std::chrono::duration<double>(Clock::now() - start).count(); }
    bool cancelled() {
        if (config.timeout_s > 0 && now() >= config.timeout_s) timed_out = true;
        return interrupted.load() || timed_out.load() || (io && io->cancel && io->cancel());
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
        IO_COUNTERS process_io{};
        if (GetProcessIoCounters(GetCurrentProcess(), &process_io)) {
            metrics.io_available = true;
            metrics.io_read_bytes = process_io.ReadTransferCount; metrics.io_write_bytes = process_io.WriteTransferCount;
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
    s << " reused=" << j.at("model_reused") << " prefix_tokens=" << j.at("input_reused");
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
        if (records.is_open()) { records << json({{"event","sample"},{"request_id",id},{"metrics",snapshot}}).dump() << '\n'; records.flush(); }
        lock.lock();
        s.wake.wait_for(lock, std::chrono::milliseconds(s.config.refresh_ms), [&] { return s.finished; });
    }
}

static bool abort_callback(void * data) { return static_cast<State *>(data)->cancelled(); }
static bool load_progress(float, void * data) { return !abort_callback(data); }
static void llama_log(ggml_log_level level, const char * text, void * data) {
    // Keep backend diagnostics separate from generated text and live metrics.
    if (level >= GGML_LOG_LEVEL_WARN) { std::fputs(text, data ? static_cast<FILE *>(data) : stderr); std::fflush(data ? static_cast<FILE *>(data) : stderr); }
}

static void preflight(State & s, bool enforce = true) {
    const auto & c=s.config;
    const auto profile=inspect_model(c,load_progress,&s);
    const auto resources=system_resources();
    const uint64_t weights=profile.at("weight_bytes").get<uint64_t>();
    const uint64_t state=profile.at("context_plan_bytes").get<uint64_t>();
    const uint64_t compute=profile.at("compute_plan_bytes").get<uint64_t>();
    // Allocator/tokenizer/OS overhead is separate from backend graph buffers.
    const uint64_t overhead=64ULL<<20, reserve=uint64_t(c.reserve_mib*1048576);
    const uint64_t cache=uint64_t(c.prefix_cache_mib)<<20;
    const uint64_t estimated=budget_add(budget_add(weights,state),budget_add(compute,budget_add(overhead,cache)));
    const uint64_t extra=c.cpu_weight_repacking?weights:0;
    const uint64_t physical_required=budget_add(estimated,budget_add(extra,reserve));
    const uint64_t commit_required=budget_add(budget_add(state,compute),budget_add(overhead,budget_add(cache,budget_add(extra,reserve))));
    std::vector<std::string> reasons;
    if(physical_required>resources.at("physical_available_bytes").get<uint64_t>()) reasons.push_back("available_physical_below_resident_plan");
    if(commit_required>resources.at("commit_headroom_bytes").get<uint64_t>()) reasons.push_back("system_commit_headroom_below_plan");
    if(c.gpu_layers) reasons.push_back("GPU_requires_backend_specific_budget");
    if(c.cpu_weight_repacking) reasons.push_back("experimental_repacking");
    if(c.prefix_cache_mib) reasons.push_back("experimental_prefix_cache");
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.identity=profile;
        s.identity["cpu_build"]=WLLM_CPU_TARGET; s.identity["llama_commit"]=WLLM_LLAMA_COMMIT;
        s.identity["pid"]=GetCurrentProcessId();
        s.identity["working_budget_estimate_bytes"]=estimated;
        s.identity["resource_ledger"]={
            {"estimate_scope","backend CPU graph plan, one sequence; no context or precision fitting"},
            {"system_at_admission",resources},{"startup_physical_budget_bytes",physical_required},
            {"additional_commit_budget_bytes",commit_required},{"peaks_calibrated",false},
            {"performance_preflight_eligible",reasons.empty()},{"performance_rejection_reasons",reasons},
            {"components",{{"mapped_file_bytes",profile.at("file_bytes")},{"mapped_file_resident_bytes",nullptr},
                {"model_plan_bytes",weights},{"context_plan_bytes",state},{"compute_plan_bytes",compute},
                {"runtime_overhead_allowance_bytes",overhead},{"prefix_cache_budget_bytes",cache},
                {"extra_weight_copy_estimate_bytes",extra},{"system_reserve_bytes",reserve}}}};
    }
    if(!enforce) return;
    if(physical_required>resources.at("physical_total_bytes").get<uint64_t>() ||
        commit_required>resources.at("commit_headroom_bytes").get<uint64_t>() ||
        resources.at("physical_available_bytes").get<uint64_t>()<reserve)
        throw std::runtime_error("Insufficient memory budget for selected model/context; No automatic context/precision reduction");
    if(physical_required>resources.at("physical_available_bytes").get<uint64_t>())
        std::cerr << "[Wllm] Available RAM is below the resident graph plan; paging may occur. Settings were preserved.\n";
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

static void assign_system_one_labels(CandidateTask & task, const llama_vocab * vocab, const std::string & boundary) {
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
    for (const auto & label:pool) add(" "+label);
    for (llama_token token=0;next<task.candidates.size() && token<llama_vocab_n_tokens(vocab);++token) {
        char piece[32];
        const int n=llama_token_to_piece(vocab,token,piece,sizeof(piece),0,false);
        if (n<1 || n>16) continue;
        const std::string label(piece,size_t(n));
        if (std::all_of(label.begin(),label.end(),[](unsigned char ch) {
            return ch==' ' || (ch>='A' && ch<='Z') || (ch>='a' && ch<='z') || (ch>='0' && ch<='9');
        }) && std::any_of(label.begin(),label.end(),[](unsigned char ch){return ch!=' ';})) add(label);
    }
    if (next!=task.candidates.size()) throw std::runtime_error("Not enough validated single-token labels for this model");
}

static std::string prefix_identity(const Config & config, llama_model * model) {
    auto identity=config.dump();
    // Presentation, deadlines and reserve policy do not change model state.
    for (const auto * field:{"record","refresh_ms","timeout_s","reserve_mib","prefix_cache_mib"}) identity.erase(field);
    const auto path=fs::u8path(config.model);
    identity["model_instance"]=reinterpret_cast<uintptr_t>(model);
    identity["file_size"]=fs::file_size(path);
    identity["file_write_time"]=fs::last_write_time(path).time_since_epoch().count();
    identity["llama_commit"]=WLLM_LLAMA_COMMIT;
    identity["cpu_build"]=WLLM_CPU_TARGET;
    return identity.dump();
}

static void save_prefix(State & s, llama_context * ctx, const std::string & key,
        const std::vector<llama_token> & tokens, size_t count, PrefixCache & cache, PrefixCache & pending) {
    const auto started=Clock::now();
    llama_synchronize(ctx);
    const size_t size=llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_NONE);
    const uint64_t required=uint64_t(size)+count*sizeof(int32_t)+key.size();
    std::string status;
    if (!size) throw std::runtime_error("Cannot size the complete prefix sequence state");
    if (required > (uint64_t(s.config.prefix_cache_mib)<<20)) status="snapshot_exceeds_budget";
    else {
        // Release the old entry before allocating a replacement: one snapshot budget.
        cache.clear();
        const auto resources=system_resources();
        const uint64_t reserve=uint64_t(s.config.reserve_mib*1048576);
        if (required+reserve > resources.at("physical_available_bytes").get<uint64_t>() ||
            required+reserve > resources.at("commit_headroom_bytes").get<uint64_t>()) status="snapshot_resource_skipped";
        else {
            try {
                pending.key=key;
                pending.tokens.assign(tokens.begin(),tokens.begin()+count);
                pending.state.resize(size);
            } catch (const std::bad_alloc &) { pending.clear(); status="snapshot_allocation_skipped"; }
            if (status.empty()) {
                if (llama_state_seq_get_data_ext(ctx,pending.state.data(),size,0,LLAMA_STATE_SEQ_FLAGS_NONE)!=size) {
                    pending.clear(); throw std::runtime_error("Complete prefix snapshot write failed");
                }
                status="snapshot_staged";
            }
        }
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.metrics.prefix_save_s+=std::chrono::duration<double>(Clock::now()-started).count();
    s.metrics.prefix_cache_bytes=pending.state.empty()?cache.bytes():pending.bytes();
    if (s.metrics.input_reused==0) s.metrics.prefix_cache_status=status;
    s.identity["prefix_snapshot_status"]=status;
}

static std::string run(State & s, const std::string & prompt, const std::string & system, const std::string & inspect_path,
        const std::string & logits_path, ModelPtr * resident, const CandidateTask * task, PrefixCache * cache, PrefixCache & pending) {
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
        throw std::runtime_error("Model load failed; check the GGUF path, format and available memory");
    }
    const llama_vocab * vocab=llama_model_get_vocab(model.get());
    if(!llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()) || llama_model_is_diffusion(model.get()))
        throw std::runtime_error("Selected model does not provide the autoregressive decoder text execution capability");
    CandidateTask mapped_task;
    if(task) { mapped_task=*task; task=&mapped_task; }
    common_chat_templates_ptr templates;
    if(!c.raw) {
        const auto source=llama_model_chat_template(model.get(),nullptr);
        if(c.chat_template.empty() && (!source || !*source))
            throw std::runtime_error("GGUF has no chat template; use --raw for completion or --chat-template-file for an explicit Jinja template");
        templates=common_chat_templates_init(model.get(),c.chat_template);
    } else if(!system.empty()) throw std::runtime_error("Raw completion accepts the prompt exactly; omit --system or supply a chat template");
    const auto render=[&](const CandidateTask * candidate) {
        common_chat_params chat;
        if(c.raw) { chat.prompt=prompt; return chat; }
        common_chat_templates_inputs inputs;
        // The renderer removes explicit edge specials when the tokenizer adds
        // them, so templates that include BOS do not create a duplicate BOS.
        inputs.add_bos=llama_vocab_get_add_bos(vocab);
        inputs.add_eos=llama_vocab_get_add_eos(vocab);
        const bool thinking=!candidate && c.thinking;
        inputs.enable_thinking=thinking;
        inputs.chat_template_kwargs["enable_thinking"]=thinking?"true":"false";
        const std::string effective_system=candidate?candidate->system_prompt(system):system;
        const std::string effective_prompt=candidate?candidate->user_prompt(prompt):prompt;
        if(!candidate && s.io && !s.io->messages.empty()) inputs.messages=s.io->messages;
        else {
            if(!effective_system.empty()) { common_chat_msg msg; msg.role="system"; msg.content=effective_system; inputs.messages.push_back(msg); }
            common_chat_msg msg; msg.role="user"; msg.content=effective_prompt; inputs.messages.push_back(msg);
        }
        if(!candidate) return common_chat_templates_apply(templates.get(),inputs);
        // Ask the native template for a continuation in assistant *content*, not
        // reasoning. No architecture-specific assistant header or think tag.
        std::string marker="WLLM_CONTENT_BOUNDARY_49F71C";
        while(effective_prompt.find(marker)!=std::string::npos || effective_system.find(marker)!=std::string::npos) marker+="_";
        common_chat_msg answer; answer.role="assistant"; answer.content=marker; inputs.messages.push_back(answer);
        inputs.continue_final_message=COMMON_CHAT_CONTINUATION_CONTENT;
        inputs.add_generation_prompt=false;
        auto result=common_chat_templates_apply(templates.get(),inputs);
        const auto at=result.prompt.rfind(marker);
        if(at==std::string::npos || at+marker.size()!=result.prompt.size() || result.prompt.find(marker)!=at)
            throw std::runtime_error("System One requires a template with an unambiguous assistant-content continuation boundary");
        result.prompt.erase(at);
        return result;
    };
    if(task && task->automatic_labels) {
        const auto initial=render(task);
        assign_system_one_labels(mapped_task,vocab,initial.prompt);
    }
    const auto chat=render(task);
    const auto tokens=tokenize(vocab,chat.prompt);
    check_context(int(tokens.size()),c.context,!task);
    std::vector<int32_t> labels;
    if(task) labels=candidate_tokens(*task,vocab,chat.prompt,tokens);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.metrics.input_total=int(tokens.size());
        s.identity["template_source"]=c.raw?"raw_completion":c.chat_template.empty()?"GGUF tokenizer.chat_template":"explicit Jinja override";
        s.identity["thinking_requested"]=c.thinking;
        s.identity["generation_prompt"]=chat.generation_prompt;
        if(task) {
            s.identity["candidate_task"]=task->manifest(); s.identity["candidate_token_ids"]=labels;
            s.identity["decision_boundary"]="native assistant-content continuation";
            s.identity["generation_loop_entered"]=false;
        }
    }
    if (!inspect_path.empty()) {
        std::ofstream out(fs::u8path(inspect_path),std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write inspect file");
        json inspected={{"rendered_prompt",chat.prompt},{"token_ids",tokens},{"template",c.raw?std::string():common_chat_templates_source(templates.get())},{"config",c.dump()}};
        if (task) {
            inspected["candidate_task"]=task->manifest(); inspected["candidate_token_ids"]=labels;
            inspected["effective_system"]=task->system_prompt(system); inspected["effective_user"]=task->user_prompt(prompt);
        }
        out << inspected.dump(2);
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.load_end=s.now(); }
        return "inspected";
    }
    auto cp=execution_context(c);
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
    std::string cache_key;
    const size_t checkpoint=cache?PrefixCache::checkpoint(tokens.size(),c.batch):0;
    if (cache) {
        cache_key=prefix_identity(c,model.get());
        const auto reused=cache->match(cache_key,tokens,c.batch);
        { std::lock_guard<std::mutex> lock(s.mutex);
          s.metrics.prefix_cache_status=checkpoint?"miss":"input_shorter_than_batch";
          s.metrics.prefix_cache_bytes=cache->bytes(); }
        if (reused) {
            if (s.cancelled()) return s.timed_out?"timeout":"cancelled";
            const auto started=Clock::now();
            const auto restored=llama_state_seq_set_data_ext(ctx.get(),cache->state.data(),cache->state.size(),0,LLAMA_STATE_SEQ_FLAGS_NONE);
            if (restored!=cache->state.size() || llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)!=llama_pos(reused-1)) {
                cache->clear(); throw std::runtime_error("Complete prefix state restore failed; no implicit recomputation");
            }
            pos=int(reused);
            std::lock_guard<std::mutex> lock(s.mutex);
            s.metrics.input_reused=pos; s.metrics.prefix_cache_status="hit";
            s.metrics.prefix_restore_s=std::chrono::duration<double>(Clock::now()-started).count();
        }
    }
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
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.input_done=pos-s.metrics.input_reused;
          ++s.metrics.prefill_batches; if (output) ++s.metrics.prefill_logits_batches; }
        if (cache && size_t(pos)==checkpoint && !s.cancelled())
            save_prefix(s,ctx.get(),cache_key,tokens,checkpoint,*cache,pending);
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
        if(s.io && s.io->stopped && s.io->stopped()) return "stop";
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
    std::cout << "Wllm 0.5.0 - local CPU inference and System One decisions\n"
        "  wllm --config configs/local.json --prompt \"Hello\"\n"
        "  --tui (terminal interface) --ascii (compact banner) --version\n"
        "  --serve --service all|openai|systemone --host 127.0.0.1 --port 8080\n"
        "  --model-alias wllm-local --api-key-env ENV_NAME (optional Bearer auth)\n"
        "  --model PATH --context N (default 16384) --threads N --batch N\n"
        "  --raw (exact text completion) --chat-template-file UTF8_JINJA\n"
        "  --threads-batch N (0 = threads) --persistent-threadpool on|off\n"
        "  --prefill-last-only on|off (skip unused intermediate logits)\n"
        "  --cpu-weight-repacking on|off (experimental CPU weight layout)\n"
        "  --prefix-cache-mib N (experimental; 0=off; bounded CPU resident-session snapshots)\n"
        "  --prompt TEXT | --prompt-file UTF8_FILE (otherwise read one line)\n"
        "  --interactive (retain weights between independent prompts; /exit quits)\n"
        "  --system TEXT --max-tokens N (-1 = until EOS/context) --seed N\n"
        "  --temperature N --thinking on|off --timeout SECONDS\n"
        "  --record PATH --cancel-file PATH --inspect PATH --quiet-metrics\n"
        "  --logits-file PATH (diagnostic: final prefill logits, float32 little-endian)\n"
        "  --logits-directory NEW_DIR (interactive diagnostic: separate numbered files)\n"
        "  --mode generate|candidate_score --candidates UTF8_JSON (CPU; returns a JSON decision)\n"
        "  --mode system_one [--request-file UTF8_JSON] (state + questions; stdin/interactive JSON also supported)\n"
        "  --profile (phase timings) --profile-ops (intrusive operator diagnostic)\n"
        "  --resources PATH (write resource/CPU budget report; no inference or prompt read)\n"
        "  --memory-diagnostic (allocation/OS peaks; excluded from performance benchmarks)\n"
        "  --gpu-layers N (experimental Vulkan validation build only; default 0)\n"
        "Metrics: stderr; --record explicitly enables JSONL. Text/decisions: stdout.\n"
        "Requests are independent. No automatic configuration or history files.\n";
}

class BackendRuntime {
public:
    BackendRuntime() { llama_log_set(llama_log,nullptr); llama_backend_init(); }
    ~BackendRuntime() { llama_backend_free(); llama_log_set(nullptr,nullptr); }
};

static int execute_request(const Config & config, const std::string & prompt, const std::string & system,
        const std::string & inspect, const std::string & logits_file, const std::string & cancel_file,
        bool quiet, bool profile, bool profile_ops, bool memory_diagnostic, ModelPtr * resident, const CandidateTask * task,
        bool emit_decision=true, json * decision_out=nullptr, PrefixCache * cache=nullptr, json * presentation=nullptr,
        const RequestIO * io=nullptr) {
    State state; state.config=config; state.cancel_file=cancel_file;
    state.io=io;
    if(io && io->output) state.terminal.set_writer(io->output);
    state.metrics.prefix_cache_status=config.prefix_cache_mib?"not_started":"disabled";
    PrefixCache pending;
    state.memory_diagnostic=memory_diagnostic;
    state.profile.enabled=profile || profile_ops; state.profile.operators=profile_ops;
    if(prompt.empty() && (!io || io->messages.empty())) throw std::runtime_error("Prompt is empty");
    if(!inspect.empty() && !logits_file.empty()) throw std::runtime_error("--inspect cannot be combined with --logits-file");
    if(!cancel_file.empty() && fs::exists(fs::u8path(cancel_file))) throw std::runtime_error("Cancel file already exists; use a new path");
    state.start=Clock::now();
    std::ofstream record;
    if (!state.config.record.empty()) {
        const auto path=fs::u8path(state.config.record);
        if(path.has_parent_path()) fs::create_directories(path.parent_path());
        record.open(path,std::ios::app|std::ios::binary);
        if(!record) throw std::runtime_error("Cannot open metric record file");
    }
    const std::string id=std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())+"-"+std::to_string(GetCurrentProcessId());
    if (record.is_open()) record << json({{"event","start"},{"schema_version",1},{"request_id",id},{"config",state.config.dump()},
        {"candidate_task",task?task->manifest():json(nullptr)},
        {"logits_diagnostic",!logits_file.empty()},
        {"memory_diagnostic",memory_diagnostic},
        {"profiling_diagnostic",state.profile.enabled},{"operator_profiling",profile_ops},
        {"cpu_build",WLLM_CPU_TARGET},{"resident_session",resident!=nullptr},
        {"llama_commit",WLLM_LLAMA_COMMIT},{"window_s",2},{"pid",GetCurrentProcessId()}}).dump() << '\n';
    std::cerr << "[Wllm] CPU context=" << state.config.context << " thinking=" << (!task && state.config.thinking && !state.config.raw?"on":"off")
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
    if (state.config.prefix_cache_mib)
        std::cerr << "[Wllm] Experimental CPU prefix cache: explicit budget, separate numerical/session performance acceptance required.\n";
    // A decision is one JSON line, including on a real console; never place it in the alternate screen.
    state.terminal.start(quiet || task!=nullptr, presentation!=nullptr);
    std::thread monitor(observer,std::ref(state),std::ref(record),std::cref(id),quiet);
    std::string reason;
    try { reason=run(state,prompt,system,inspect,logits_file,resident,task,cache,pending); }
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
        state.decision["model"]=state.identity.value("name",fs::u8path(config.model).stem().u8string());
        state.decision["model_identity"]=json::object();
        for(const auto * key:{"name","architecture","quantization","parameter_count","expert_count","model_context","state_kind","file_bytes","shard_count"})
            if(state.identity.contains(key)) state.decision["model_identity"][key]=state.identity.at(key);
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
    if (record.is_open()) record << json({{"event","end"},{"request_id",id},{"metrics",final},{"identity",state.identity},
        {"decision",state.decision},
        {"profile",profile_result},
        {"token_times_s",state.metrics.token_times},{"token_ids",state.metrics.token_ids},
        {"load_start_s",state.metrics.load_start},{"load_end_s",state.metrics.load_end},
        {"prefill_start_s",state.metrics.prefill_start},{"prefill_end_s",state.metrics.prefill_end}}).dump() << '\n';
    if (record.is_open()) {
        record.flush();
        if(!record) { std::cerr << "Metric record write failed\n"; return 1; }
    }
    if (presentation) *presentation={{"mode",config.inference_mode},{"metrics",final},{"output",state.terminal.text()},{"decision",state.decision},{"model_identity",state.identity}};
    if (decision_out) *decision_out=state.decision;
    if (task && inspect.empty() && emit_decision) {
        std::cout << state.decision.dump() << '\n' << std::flush;
        if (!std::cout) { std::cerr << "Decision output write failed\n"; return 1; }
    }
    if (cache && !pending.state.empty() && (reason=="eos" || reason=="max_tokens" || reason=="context_limit" ||
            reason=="selected" || reason=="abstained")) cache->swap(pending);
    return reason=="error"?1:reason=="cancelled"?130:reason=="timeout"?124:0;
}

static int execute_system_one(const Config & config, const std::string & body, const std::string & cancel_file,
        bool quiet, bool profile, bool profile_ops, bool memory_diagnostic, ModelPtr & resident, PrefixCache * cache, json * response_out=nullptr,
        const RequestIO * io=nullptr, bool emit_response=true) {
    const auto started=Clock::now();
    const auto elapsed=[&] { return std::chrono::duration<double>(Clock::now()-started).count(); };
    const std::string id="systemone-"+std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(started.time_since_epoch()).count())+
        "-"+std::to_string(GetCurrentProcessId());
    json response={{"model",config.model.empty()?std::string("wllm-local"):fs::u8path(config.model).stem().u8string()},{"answers",json::object()},
        {"usage",{{"input_tokens",0},{"output_tokens",0}}},
        {"_wllm",{{"request_id",id},{"protocol","typesafe-system-one-text-v1"},
            {"model_path",config.model},
            {"execution","sequential isolated questions; retained local weights"},
            {"probabilities","candidate-normalized local logits; not calibrated correctness"},
            {"usage_scope","sum of completed newly computed prefill tokens; reused tokens in question metrics; aborted partial work is not measurable"}}}};
    int code=0;
    std::string active_question;
    bool validated=false;
    const auto check_budget=[&] {
        if (!cancel_file.empty() && fs::exists(fs::u8path(cancel_file))) interrupted=true;
        if (interrupted.load() || (io && io->cancel && io->cancel())) { code=130; throw std::runtime_error("System One request cancelled"); }
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
                quiet,profile,profile_ops,memory_diagnostic,&resident,&q.task,false,&decision,cache,nullptr,io);
            if (decision.contains("metrics")) response["usage"]["input_tokens"]=
                response["usage"]["input_tokens"].get<int>()+decision["metrics"]["input_processed"].get<int>();
            if (code!=0) {
                auto message=decision.value("error",std::string());
                if (message.empty()) message=code==124?"Question timed out":code==130?"Question cancelled":"Question evaluation failed";
                throw std::runtime_error(message);
            }
            check_budget();
            response["model"]=decision.at("model");
            response["_wllm"]["model_identity"]=decision.at("model_identity");
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
        if (!config.record.empty()) {
        const auto path=fs::u8path(config.record);
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::ofstream record(path,std::ios::app|std::ios::binary);
        record << json({{"event","system_one_end"},{"request_id",id},{"response",response}}).dump() << '\n';
        record.flush();
        if (!record) throw std::runtime_error("Cannot write System One result record");
        }
    } catch (const std::exception & e) {
        code=1; response["answers"]=json::object();
        response["_wllm"]["status"]="failed";
        response["error"]={{"type","record_error"},{"message",e.what()}};
    }
    if(response_out) *response_out=response;
    if(emit_response) std::cout << response.dump() << '\n' << std::flush;
    if (cache && (code!=0 || !std::cout)) cache->clear();
    return std::cout?code:1;
}

#include "console_app.h"
#include "http_service.h"

static int main_utf8(int argc,char ** argv) {
    std::ios::sync_with_stdio(false);
    ConsoleEnvironment console_environment;
    SetConsoleCtrlHandler(console_handler,TRUE);
    std::setlocale(LC_NUMERIC,"C");
    try {
        check_cpu_target();
        json settings=json::object();
        for(int i=1;i<argc;++i) if(std::string(argv[i])=="--config") {
            if(++i==argc) throw std::runtime_error("--config needs a path");
            settings=json::parse(read_text(argv[i]));
        }
        std::string prompt,system,inspect,cancel_file,logits_file,logits_directory,resources_file,candidates_file,request_file;
        bool prompt_set=false, quiet=false, interactive=false, profile=false, profile_ops=false, memory_diagnostic=false;
        bool serve=false, service_flags=false;
        ServiceOptions service;
        bool tui=argc==1 && console_handle(STD_INPUT_HANDLE) && console_handle(STD_OUTPUT_HANDLE), ascii=false;
        const std::map<std::string,std::string> ints={{"--context","context"},{"--threads","threads"},{"--threads-batch","threads_batch"},{"--batch","batch"},{"--gpu-layers","gpu_layers"},{"--prefix-cache-mib","prefix_cache_mib"},{"--max-tokens","max_tokens"},{"--seed","seed"}};
        const std::map<std::string,std::string> bools={{"--thinking","thinking"},{"--persistent-threadpool","persistent_threadpool"},{"--prefill-last-only","prefill_last_only"},{"--cpu-weight-repacking","cpu_weight_repacking"}};
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help" || arg=="-h") { usage(); return 0; }
            if(arg=="--version") { std::cout << "Wllm 0.5.0 / CPU " << WLLM_CPU_TARGET << " / llama.cpp " << WLLM_LLAMA_COMMIT << '\n'; return 0; }
            if(arg=="--tui") { tui=true; continue; }
            if(arg=="--serve") { serve=true; continue; }
            if(arg=="--ascii") { ascii=true; continue; }
            if(arg=="--raw") { settings["raw"]=true; continue; }
            if(arg=="--quiet-metrics") { quiet=true; continue; }
            if(arg=="--interactive") { interactive=true; continue; }
            if(arg=="--profile") { profile=true; continue; }
            if(arg=="--profile-ops") { profile_ops=true; continue; }
            if(arg=="--memory-diagnostic") { memory_diagnostic=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing value for " + arg);
            const std::string v=argv[++i];
            if(arg=="--config") continue;
            if(arg=="--service" || arg=="--host" || arg=="--port" || arg=="--model-alias" || arg=="--api-key-env") {
                service_flags=true;
                if(arg=="--service") service.type=v;
                else if(arg=="--host") service.host=v;
                else if(arg=="--model-alias") service.alias=v;
                else if(arg=="--api-key-env") service.key_env=v;
                else { const auto port=json::parse(v); if(!port.is_number_integer() || port<1 || port>65535) throw std::runtime_error("port must be 1..65535"); service.port=port.get<int>(); }
            }
            else if(arg=="--prompt" || arg=="--prompt-file") { if(prompt_set) throw std::runtime_error("Use only one prompt source"); prompt=arg=="--prompt"?v:read_text(v); prompt_set=true; }
            else if(arg=="--system") system=v;
            else if(arg=="--model") settings["model"]=v;
            else if(arg=="--chat-template-file") { const auto content=read_text(v); if(content.empty()) throw std::runtime_error("Chat template file is empty"); settings["chat_template"]=content; }
            else if(arg=="--record") settings["record"]=v;
            else if(arg=="--inspect") inspect=v;
            else if(arg=="--logits-file") logits_file=v;
            else if(arg=="--logits-directory") logits_directory=v;
            else if(arg=="--cancel-file") cancel_file=v;
            else if(arg=="--resources") resources_file=v;
            else if(arg=="--mode") settings["inference_mode"]=v;
            else if(arg=="--candidates") candidates_file=v;
            else if(arg=="--request-file") request_file=v;
            else if(bools.count(arg)) { if(v!="on" && v!="off") throw std::runtime_error(arg + " must be on or off"); settings[bools.at(arg)]=v=="on"; }
            else if(ints.count(arg)) { const json n=json::parse(v); if(!n.is_number_integer()) throw std::runtime_error("Expected integer for " + arg); settings[ints.at(arg)]=n; }
            else if(arg=="--temperature" || arg=="--timeout" || arg=="--top-p" || arg=="--min-p" || arg=="--presence-penalty" || arg=="--frequency-penalty" || arg=="--repeat-penalty") {
                const json n=json::parse(v); if(!n.is_number()) throw std::runtime_error("Expected number");
                std::string name=arg.substr(2); std::replace(name.begin(),name.end(),'-','_');
                settings[name=="timeout"?"timeout_s":name]=n;
            }
            else if(arg=="--top-k" || arg=="--repeat-last-n") {
                const json n=json::parse(v); if(!n.is_number_integer()) throw std::runtime_error("Expected integer");
                std::string name=arg.substr(2); std::replace(name.begin(),name.end(),'-','_'); settings[name]=n;
            }
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        const auto config=Config::parse(settings);
        if(service_flags && !serve) throw std::runtime_error("Service options require --serve");
        if(serve) {
            service.validate();
            if(tui || prompt_set || interactive || !system.empty() || !inspect.empty() || !logits_file.empty() || !logits_directory.empty() || !resources_file.empty() || !cancel_file.empty() || !candidates_file.empty() || !request_file.empty() || profile || profile_ops || memory_diagnostic || ascii)
                throw std::runtime_error("--serve uses HTTP requests; do not combine with TUI, CLI prompt/mode input or diagnostic options");
            BackendRuntime backend;
            return run_http_service(config,service);
        }
        if (tui) {
            if (!console_handle(STD_INPUT_HANDLE) || !console_handle(STD_OUTPUT_HANDLE) || !console_handle(STD_ERROR_HANDLE))
                throw std::runtime_error("--tui requires console input/output; use --interactive or --prompt for pipes");
            if(prompt_set || interactive || !inspect.empty() || !logits_file.empty() || !logits_directory.empty() || !resources_file.empty() || !cancel_file.empty() || profile || profile_ops || memory_diagnostic)
                throw std::runtime_error("--tui accepts model/config/mode/candidates/request settings; use the CLI for request or diagnostic options");
            BackendRuntime backend;
            return run_console_ui(settings,system,candidates_file,request_file,ascii);
        }
        const bool system_one=config.inference_mode=="system_one";
        if (!system_one && !request_file.empty()) throw std::runtime_error("--request-file requires --mode system_one");
        if (system_one && (prompt_set || !system.empty() || !inspect.empty() || !logits_file.empty() || !logits_directory.empty() || !candidates_file.empty() || !resources_file.empty()))
            throw std::runtime_error("system_one accepts state/questions JSON; do not combine with prompt, system, candidates, inspect, logits or resources");
        std::optional<CandidateTask> task;
        if (config.inference_mode=="candidate_score") {
            if (candidates_file.empty()) throw std::runtime_error("candidate_score requires --candidates FILE");
            task=CandidateTask::parse(json::parse(read_text(candidates_file)));
        } else if (!candidates_file.empty()) throw std::runtime_error("--candidates requires --mode candidate_score");
        if (config.gpu_layers && !WLLM_VULKAN) throw std::runtime_error("Vulkan was not compiled into this executable; use the separate validation build");
        if (!resources_file.empty()) {
            if (task || prompt_set || interactive || !inspect.empty() || !logits_file.empty() || !logits_directory.empty() || profile || profile_ops || memory_diagnostic || !cancel_file.empty() || !system.empty())
                throw std::runtime_error("--resources is a standalone probe; do not combine with request/diagnostic options");
            BackendRuntime backend;
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
        if (config.prefix_cache_mib && !interactive && !system_one)
            throw std::runtime_error("Prefix cache requires --interactive or system_one; no implicit resident session");
        if (!logits_directory.empty()) {
            if (!interactive || !inspect.empty() || !logits_file.empty())
                throw std::runtime_error("--logits-directory requires interactive mode without inspect/logits-file");
            if (!fs::create_directories(fs::u8path(logits_directory)))
                throw std::runtime_error("Logits directory must be new; refusing to overwrite diagnostics");
        }
        BackendRuntime backend;
        ModelPtr resident(nullptr,llama_model_free);
        PrefixCache prefix_cache;
        auto * cache=config.prefix_cache_mib?&prefix_cache:nullptr;
        int code=0;
        if (system_one) {
            auto request=[&](const std::string & body) {
                const int result=execute_system_one(config,body,cancel_file,quiet,profile,profile_ops,memory_diagnostic,resident,cache);
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
        size_t request_index=0;
        while (true) {
            if (interactive && (prompt.empty() || prompt=="/exit")) break;
            const auto diagnostic=logits_directory.empty()?logits_file:
                (fs::u8path(logits_directory)/(std::to_string(request_index++)+".f32")).u8string();
            code=execute_request(config,prompt,system,inspect,diagnostic,cancel_file,quiet,profile,profile_ops,memory_diagnostic,interactive?&resident:nullptr,task?&*task:nullptr,true,nullptr,cache);
            if (!interactive || code!=0 || interrupted.load()) break;
            if (!task) std::cout << '\n' << std::flush;
            std::cerr << "[Wllm] Weights retained; independent context, explicit prefix cache=" << bool(cache)
                      << ". /exit or empty line releases the model.\nPrompt> ";
            if (!std::getline(std::cin,prompt)) break;
        }
        }
        if (resident) {
            const auto cache_bytes=prefix_cache.bytes();
            prefix_cache.clear();
            const auto release_start=Clock::now();
            resident.reset();
            const double release_s=std::chrono::duration<double>(Clock::now()-release_start).count();
            if (!config.record.empty()) {
            std::ofstream record(fs::u8path(config.record),std::ios::app|std::ios::binary);
            record << json({{"event","session_end"},{"pid",GetCurrentProcessId()},{"model_release_s",release_s},
                {"prefix_cache_released_bytes",cache_bytes}}).dump() << '\n';
            record.flush();
            if (!record) throw std::runtime_error("Cannot record session release");
            }
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
