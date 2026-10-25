#pragma once
#include "core.h"
#include "llama.h"
#include "llama-ext.h"
#include "gguf.h"
#include "architecture.h"
#include <filesystem>
#include <memory>
#include <limits>

namespace wllm {
inline uint64_t budget_add(uint64_t a, uint64_t b) {
    if(b>std::numeric_limits<uint64_t>::max()-a) throw std::runtime_error("Model memory budget overflow");
    return a+b;
}
inline std::string model_meta_string(const llama_model * model, const char * key) {
    const int n=llama_model_meta_val_str(model,key,nullptr,0);
    if(n<0) return {};
    std::vector<char> text(size_t(n)+1);
    llama_model_meta_val_str(model,key,text.data(),text.size());
    return std::string(text.data(),size_t(n));
}
inline uint64_t gguf_unsigned(const gguf_context * meta, const char * name, uint64_t fallback=0) {
    const auto key=gguf_find_key(meta,name);
    if(key<0) return fallback;
    switch(gguf_get_kv_type(meta,key)) {
        case GGUF_TYPE_UINT8:return gguf_get_val_u8(meta,key);
        case GGUF_TYPE_UINT16:return gguf_get_val_u16(meta,key);
        case GGUF_TYPE_UINT32:return gguf_get_val_u32(meta,key);
        case GGUF_TYPE_UINT64:return gguf_get_val_u64(meta,key);
        default:throw std::runtime_error(std::string("Expected unsigned GGUF metadata: ")+name);
    }
}
inline std::vector<std::filesystem::path> model_files(const std::string & name) {
    namespace fs=std::filesystem;
    const auto file=fs::absolute(fs::u8path(name));
    if(!fs::is_regular_file(file)) throw std::runtime_error("GGUF file not found; set model or use --model");
    gguf_init_params params{}; params.no_alloc=true;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> meta(gguf_init_from_file(name.c_str(),params),gguf_free);
    if(!meta) throw std::runtime_error("Cannot read GGUF metadata");
    const auto count=gguf_unsigned(meta.get(),"split.count",1);
    const auto index=gguf_unsigned(meta.get(),"split.no",0);
    if(count<1 || count>65535 || index>=count) throw std::runtime_error("Invalid GGUF split metadata");
    if(count==1) return {file};
    if(index!=0) throw std::runtime_error("Select the first shard of a split GGUF model");
    const std::string absolute=file.u8string();
    std::vector<char> prefix(absolute.size()+32),part(absolute.size()+64);
    if(llama_split_prefix(prefix.data(),prefix.size(),absolute.c_str(),int(index),int(count))<=0)
        throw std::runtime_error("Split GGUF filename does not match its shard metadata");
    std::vector<fs::path> files;
    for(uint64_t i=0;i<count;++i) {
        if(llama_split_path(part.data(),part.size(),prefix.data(),int(i),int(count))<=0)
            throw std::runtime_error("Cannot resolve GGUF shard filename");
        const auto path=fs::u8path(part.data());
        if(!fs::is_regular_file(path)) throw std::runtime_error("Missing GGUF shard: "+path.u8string());
        files.push_back(path);
    }
    return files;
}
inline llama_context_params execution_context(const Config & c) {
    auto params=llama_context_default_params();
    params.n_ctx=c.context; params.n_batch=c.batch; params.n_ubatch=c.batch; params.n_seq_max=1;
    params.n_threads=c.threads; params.n_threads_batch=c.effective_threads_batch();
    params.type_k=GGML_TYPE_F16; params.type_v=GGML_TYPE_F16;
    params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv=c.gpu_layers>0; params.op_offload=false; params.no_perf=false;
    return params;
}

// The backend plans its own architecture's tensors, state and compute graph.
// No weight data is allocated or evaluated; this is not a process peak estimate.
inline json inspect_model(const Config & c, llama_progress_callback progress=nullptr, void * user=nullptr) {
    namespace fs=std::filesystem;
    const auto files=model_files(c.model);
    json signature=json::array();
    uint64_t file_bytes=0;
    for(const auto & path:files) {
        const auto bytes=fs::file_size(path); file_bytes=budget_add(file_bytes,bytes);
        signature.push_back({path.u8string(),bytes,fs::last_write_time(path).time_since_epoch().count()});
    }
    const std::string key=json({signature,c.context,c.batch,c.threads,c.effective_threads_batch(),c.cpu_weight_repacking,c.gpu_layers,c.chat_template}).dump();
    static std::string cached_key;
    static json cached_profile;
    if(key==cached_key) return cached_profile;
    auto mp=llama_model_default_params();
    mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_NONE; mp.no_alloc=true;
    mp.use_extra_bufts=c.cpu_weight_repacking; mp.progress_callback=progress; mp.progress_callback_user_data=user;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(c.model.c_str(),mp),llama_model_free);
    if(!model) throw std::runtime_error("Backend cannot load this GGUF; check architecture, tensor layout and file completeness");
    const auto architecture=model_meta_string(model.get(),"general.architecture");
    const auto train=llama_model_n_ctx_train(model.get());
    if(train>0 && c.context>train) throw std::runtime_error("Requested context exceeds model training context ("+std::to_string(train)+"); choose --context explicitly");
    if(!llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()) || llama_model_is_diffusion(model.get()))
        throw std::runtime_error("This execution path requires an autoregressive decoder-only text model; architecture="+architecture);
    const auto contract=native_architecture_contract(*model);
    const auto declared_window=model_meta_string(model.get(),(architecture+".attention.sliding_window").c_str());
    if(!declared_window.empty()) {
        const auto value=json::parse(declared_window);
        if(!value.is_number_unsigned()) throw std::runtime_error("Invalid sliding-window metadata");
        check_sliding_semantics(value.get<uint64_t>(),contract.at("sliding_window").get<uint64_t>());
    }
    const auto layers=llama_model_n_layer(model.get());
    if(c.gpu_layers>layers+1) throw std::runtime_error("GPU layer count exceeds the selected model");
    auto cp=execution_context(c);
    std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);
    if(!context) throw std::runtime_error("Backend could not plan the requested model/context configuration");
    if(llama_n_ctx(context.get())!=uint32_t(c.context)) throw std::runtime_error("Backend changed context size; choose a supported size explicitly");
    uint64_t weights=0,state=0,compute=0;
    json buffers=json::array();
    for(const auto & entry:llama_get_memory_breakdown(context.get())) {
        const auto & mb=entry.second;
        weights=budget_add(weights,mb.model); state=budget_add(state,mb.context); compute=budget_add(compute,mb.compute);
        buffers.push_back({{"buffer_type",ggml_backend_buft_name(entry.first)},{"model_bytes",mb.model},{"context_bytes",mb.context},{"compute_bytes",mb.compute}});
    }
    weights=std::max(weights,llama_model_size(model.get()));
    auto name=model_meta_string(model.get(),"general.name");
    if(name.empty()) name=files.front().stem().u8string();
    const bool hybrid=llama_model_is_hybrid(model.get()), recurrent=llama_model_is_recurrent(model.get());
    const auto tmpl=llama_model_chat_template(model.get(),nullptr);
    char description[512]{}; llama_model_desc(model.get(),description,sizeof(description));
    json profile={{"name",name},{"architecture",architecture},{"quantization",llama_ftype_name(llama_model_ftype(model.get()))},
        {"description",description},{"parameter_count",llama_model_n_params(model.get())},{"expert_count",llama_model_n_expert(model.get())},
        {"layer_count",layers},{"embedding_size",llama_model_n_embd(model.get())},{"model_context",train},
        {"weight_bytes",weights},{"file_bytes",file_bytes},{"shard_count",files.size()},{"context_plan_bytes",state},{"compute_plan_bytes",compute},
        {"state_kind",hybrid?"hybrid":recurrent?"recurrent":"attention"},{"recurrent_state",hybrid||recurrent?"f32":"not_applicable"},
        {"chat_template_available",!c.chat_template.empty() || (tmpl && *tmpl)},
        {"chat_template_source",!c.chat_template.empty()?"explicit_override":tmpl && *tmpl?"GGUF":"none"},
        {"plan_buffers",buffers},{"planning_method","backend no-allocation model and context graph"},
        {"architecture_contract",contract}};
    const auto source=model_meta_string(model.get(),"wllm.source_sha256");
    if(!source.empty()) profile["original_weight_sha256"]=source;
    cached_key=key; cached_profile=profile;
    return profile;
}
} // namespace wllm
