#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <io.h>
#include <intrin.h>
#include "core.h"
#include "terminal.h"
#include "llama.h"
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
    json identity = json::object();
    Clock::time_point start = Clock::now();
    std::mutex mutex;
    std::condition_variable wake;
    bool finished = false;
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
        }
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

static void preflight(State & s) {
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
    uint64_t heads = 0;
    if (heads_key < 0) throw std::runtime_error("Missing KV head metadata");
    if (gguf_get_kv_type(meta.get(), heads_key) == GGUF_TYPE_ARRAY && gguf_get_arr_type(meta.get(), heads_key) == GGUF_TYPE_UINT32) {
        const auto * a = static_cast<const uint32_t *>(gguf_get_arr_data(meta.get(), heads_key));
        for (size_t i=0; i<gguf_get_arr_n(meta.get(),heads_key); ++i) heads += a[i];
    } else {
        heads = meta_u32(meta.get(), "qwen35.attention.head_count_kv") *
            (meta_u32(meta.get(), "qwen35.block_count") / meta_u32(meta.get(), "qwen35.full_attention_interval"));
    }
    const uint64_t kv = uint64_t(c.context) * heads * 2 * (meta_u32(meta.get(),"qwen35.attention.key_length") + meta_u32(meta.get(),"qwen35.attention.value_length"));
    const uint64_t weights = fs::file_size(fs::u8path(c.model));
    // A conservative admission estimate, not a measured runtime allocation.
    const uint64_t estimated = weights + kv + (3072ULL << 20);
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) throw std::runtime_error("Cannot read system memory budget");
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.identity = {{"architecture","qwen35"},{"quantization","Q4_K_M"},{"weight_bytes",weights},
            {"kv_estimate_bytes",kv},{"working_budget_estimate_bytes",estimated},{"physical_total_bytes",m.ullTotalPhys},
            {"physical_available_at_admission",m.ullAvailPhys},{"model_context",train},
            {"llama_commit",WLLM_LLAMA_COMMIT},{"pid",GetCurrentProcessId()}};
        s.identity["cpu_build"] = WLLM_CPU_TARGET;
        const auto source_key = gguf_find_key(meta.get(), "wllm.source_sha256");
        if (source_key >= 0 && gguf_get_kv_type(meta.get(), source_key) == GGUF_TYPE_STRING)
            s.identity["original_weight_sha256"] = gguf_get_val_str(meta.get(), source_key);
    }
    const uint64_t reserve = uint64_t(c.reserve_mib * 1048576);
    if (estimated + reserve > m.ullTotalPhys || kv + (512ULL << 20) + reserve > m.ullAvailPageFile || m.ullAvailPhys < reserve) {
        std::ostringstream error;
        error << "Insufficient memory budget: estimated weights+KV+workspace=" << estimated / 1048576
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

static std::string run(State & s, const std::string & prompt, const std::string & system, const std::string & inspect_path, const std::string & logits_path, ModelPtr * resident) {
    ModelPtr transient(nullptr, llama_model_free);
    auto & model = resident ? *resident : transient;
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.model_reused = bool(model); }
    preflight(s);
    const auto & c=s.config;
    if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.stage="loading"; s.metrics.load_start=s.now(); }
    llama_model_params mp=llama_model_default_params();
    mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP; mp.progress_callback=load_progress; mp.progress_callback_user_data=&s;
    mp.use_extra_bufts=false;
    mp.vocab_only=!inspect_path.empty();
    if (!model) model.reset(llama_model_load_from_file(c.model.c_str(),mp));
    if (!model) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        throw std::runtime_error("Model load failed; see reports/backend.log");
    }
    const llama_vocab * vocab=llama_model_get_vocab(model.get());
    const char * template_src=llama_model_chat_template(model.get(),nullptr);
    if (!template_src || !*template_src) throw std::runtime_error("GGUF has no chat template; refusing to guess one");
    auto templates=common_chat_templates_init(model.get(),"");
    common_chat_templates_inputs inputs;
    inputs.enable_thinking=c.thinking;
    inputs.chat_template_kwargs["enable_thinking"]=c.thinking ? "true" : "false";
    if (!system.empty()) { common_chat_msg m; m.role="system"; m.content=system; inputs.messages.push_back(m); }
    common_chat_msg user; user.role="user"; user.content=prompt; inputs.messages.push_back(user);
    const auto chat=common_chat_templates_apply(templates.get(),inputs);
    const auto tokens=tokenize(vocab,chat.prompt);
    check_context(int(tokens.size()),c.context);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.metrics.input_total=int(tokens.size());
        s.identity["template_source"]="GGUF tokenizer.chat_template via upstream Jinja";
        s.identity["thinking_prompt_suffix"]=chat.generation_prompt;
    }
    if (!inspect_path.empty()) {
        std::ofstream out(fs::u8path(inspect_path),std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write inspect file");
        out << json({{"rendered_prompt",chat.prompt},{"token_ids",tokens},{"template",template_src},{"config",c.dump()}}).dump(2);
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.load_end=s.now(); }
        return "inspected";
    }
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=c.context; cp.n_batch=c.batch; cp.n_ubatch=c.batch; cp.n_seq_max=1;
    cp.n_threads=c.threads; cp.n_threads_batch=c.effective_threads_batch();
    cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=false; cp.op_offload=false; cp.no_perf=false;
    cp.abort_callback=abort_callback; cp.abort_callback_data=&s;
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
    if (!ctx) throw std::runtime_error("Context allocation failed; requested context was not reduced");
    if (llama_n_ctx(ctx.get()) != uint32_t(c.context)) throw std::runtime_error("Backend changed context size; refusing implicit change");
    CpuPool pool(ctx.get(), c);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.metrics.load_end=s.now(); s.metrics.stage="prefill"; s.metrics.prefill_start=s.now();
        s.sample_memory();
    }
    auto sp=llama_sampler_chain_default_params();
    std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(sp),llama_sampler_free);
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
    std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);
    int pos=0;
    while(pos<int(tokens.size())) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        int n=std::min(c.batch,int(tokens.size())-pos);
        // Intermediate batches update all attention/recurrent state but need no vocabulary projection.
        const bool output = !c.prefill_last_only || pos+n==int(tokens.size());
        set_batch(batch.get(),tokens.data()+pos,n,pos,output);
        const int rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
        if (rc!=0) { if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled"; throw std::runtime_error("Prefill failed: " + std::to_string(rc)); }
        pos+=n;
        { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.input_done=pos;
          ++s.metrics.prefill_batches; if (output) ++s.metrics.prefill_logits_batches; }
    }
    { std::lock_guard<std::mutex> lock(s.mutex); s.metrics.prefill_end=s.now(); s.metrics.stage="generating"; }
    if (!logits_path.empty()) {
        // Diagnostic only: full, unsampled vocabulary logits for this fixed input prefix.
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) throw std::runtime_error("No final prefill logits for diagnostic");
        std::ofstream out(fs::u8path(logits_path), std::ios::binary);
        out.write(reinterpret_cast<const char *>(logits), std::streamsize(llama_vocab_n_tokens(vocab)) * sizeof(float));
        if (!out) throw std::runtime_error("Cannot write logits diagnostic");
    }
    int generated=0;
    while(true) {
        if (s.cancelled()) return s.timed_out ? "timeout" : "cancelled";
        const llama_token token=llama_sampler_sample(sampler.get(),ctx.get(),-1);
        if (llama_vocab_is_eog(vocab,token)) return "eos";
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.metrics.token_times.push_back(s.now()); s.metrics.token_ids.push_back(token);
        }
        ++generated;
        char token_buffer[256]; int n=llama_token_to_piece(vocab,token,token_buffer,sizeof(token_buffer),0,true);
        std::string piece;
        if (n>=0) piece.assign(token_buffer,n);
        else { piece.resize(-n); n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true); if(n<0) throw std::runtime_error("Token rendering failed"); piece.resize(n); }
        s.terminal.write(piece);
        if (!std::cout) throw std::runtime_error("Output stream failed");
        if (c.max_tokens>0 && generated>=c.max_tokens) return "max_tokens";
        if (int(tokens.size())+generated>=c.context) return "context_limit";
        set_batch(batch.get(),&token,1,pos++);
        const int rc=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
        if (rc!=0) { if(s.cancelled()) return s.timed_out ? "timeout" : "cancelled"; throw std::runtime_error("Decode failed: " + std::to_string(rc)); }
    }
}

static void usage() {
    std::cout << "Wllm 0.2.0 - local Qwen3.5 Q4_K_M text inference\n"
        "  wllm --config configs/local.json --prompt \"Hello\"\n"
        "  --model PATH --context N (default 4096) --threads N --batch N\n"
        "  --threads-batch N (0 = threads) --persistent-threadpool on|off\n"
        "  --prefill-last-only on|off (skip unused intermediate logits)\n"
        "  --prompt TEXT | --prompt-file UTF8_FILE (otherwise read one line)\n"
        "  --interactive (retain weights between independent prompts; /exit quits)\n"
        "  --system TEXT --max-tokens N (-1 = until EOS/context) --seed N\n"
        "  --temperature N --thinking on|off --timeout SECONDS\n"
        "  --record PATH --cancel-file PATH --inspect PATH --quiet-metrics\n"
        "  --logits-file PATH (diagnostic: final prefill logits, float32 little-endian)\n"
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
        bool quiet, ModelPtr * resident) {
    State state; state.config=config; state.cancel_file=cancel_file;
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
        {"logits_diagnostic",!logits_file.empty()},
        {"cpu_build",WLLM_CPU_TARGET},{"resident_session",resident!=nullptr},
        {"llama_commit",WLLM_LLAMA_COMMIT},{"window_s",2},{"pid",GetCurrentProcessId()}}).dump() << '\n';
    std::cerr << "[Wllm] Q4_K_M CPU context=" << state.config.context << " thinking=" << (state.config.thinking?"on":"off")
              << " cpu_build=" << WLLM_CPU_TARGET
              << " pid=" << GetCurrentProcessId() << " record=" << state.config.record << '\n';
    std::cerr << "[Wllm] model=" << state.config.model << " threads=" << state.config.threads
              << " threads_batch=" << state.config.effective_threads_batch()
              << " persistent_threadpool=" << state.config.persistent_threadpool
              << " prefill_last_only=" << state.config.prefill_last_only
              << " batch=" << state.config.batch << " temperature=" << state.config.temperature
              << " top_k=" << state.config.top_k << " top_p=" << state.config.top_p
              << " presence_penalty=" << state.config.presence_penalty << '\n';
    state.terminal.start(quiet);
    std::thread monitor(observer,std::ref(state),std::ref(record),std::cref(id),quiet);
    std::string reason;
    try { reason=run(state,prompt,system,inspect,logits_file,resident); }
    catch(const std::exception & e) { reason=state.cancelled()?(state.timed_out?"timeout":"cancelled"):"error"; std::lock_guard<std::mutex> lock(state.mutex); state.metrics.error=e.what(); }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.sample_memory(); state.metrics.ended=state.now(); state.metrics.reason=reason;
        state.metrics.stage=reason=="error"?"failed":reason=="cancelled"||reason=="timeout"?"cancelled":"finished";
        state.finished=true;
    }
    state.wake.notify_all(); if(monitor.joinable()) monitor.join();
    state.terminal.finish();
    const auto final=state.metrics.snapshot(state.now());
    display(final,true);
    if(!state.metrics.error.empty()) std::cerr << "[Wllm] " << state.metrics.error << '\n';
    record << json({{"event","end"},{"request_id",id},{"metrics",final},{"identity",state.identity},
        {"token_times_s",state.metrics.token_times},{"token_ids",state.metrics.token_ids},
        {"load_start_s",state.metrics.load_start},{"load_end_s",state.metrics.load_end},
        {"prefill_start_s",state.metrics.prefill_start},{"prefill_end_s",state.metrics.prefill_end}}).dump() << '\n';
    record.flush();
    if(!record) { std::cerr << "Metric record write failed\n"; return 1; }
    return reason=="error"?1:reason=="cancelled"?130:reason=="timeout"?124:0;
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
        std::string prompt,system,inspect,cancel_file,logits_file;
        bool prompt_set=false, quiet=false, interactive=false;
        const std::map<std::string,std::string> ints={{"--context","context"},{"--threads","threads"},{"--threads-batch","threads_batch"},{"--batch","batch"},{"--max-tokens","max_tokens"},{"--seed","seed"}};
        const std::map<std::string,std::string> bools={{"--thinking","thinking"},{"--persistent-threadpool","persistent_threadpool"},{"--prefill-last-only","prefill_last_only"}};
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help" || arg=="-h") { usage(); return 0; }
            if(arg=="--quiet-metrics") { quiet=true; continue; }
            if(arg=="--interactive") { interactive=true; continue; }
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
            else if(bools.count(arg)) { if(v!="on" && v!="off") throw std::runtime_error(arg + " must be on or off"); settings[bools.at(arg)]=v=="on"; }
            else if(ints.count(arg)) { const json n=json::parse(v); if(!n.is_number_integer()) throw std::runtime_error("Expected integer for " + arg); settings[ints.at(arg)]=n; }
            else if(arg=="--temperature" || arg=="--timeout") { const json n=json::parse(v); if(!n.is_number()) throw std::runtime_error("Expected number"); settings[arg=="--timeout"?"timeout_s":"temperature"]=n; }
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        const auto config=Config::parse(settings);
        if (interactive && (!inspect.empty() || !logits_file.empty()))
            throw std::runtime_error("Interactive mode cannot overwrite inspect/logits diagnostic files");
        BackendRuntime backend;
        ModelPtr resident(nullptr,llama_model_free);
        if (!prompt_set) { std::cerr << "Prompt> "; std::getline(std::cin,prompt); }
        int code=0;
        while (true) {
            if (interactive && (prompt.empty() || prompt=="/exit")) break;
            code=execute_request(config,prompt,system,inspect,logits_file,cancel_file,quiet,interactive?&resident:nullptr);
            if (!interactive || code!=0 || interrupted.load()) break;
            std::cout << '\n' << std::flush;
            std::cerr << "[Wllm] Weights retained; each prompt starts a fresh context. /exit or empty line releases the model.\nPrompt> ";
            if (!std::getline(std::cin,prompt)) break;
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
