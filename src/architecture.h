#pragma once
#include "core.h"
#include "llama-model.h"
#include <set>

namespace wllm {
// This adapter is bound to dependencies.json. Native geometry and loaded tensor
// layouts describe capabilities; model names never select an execution path.
inline std::string attention_topology(uint32_t query,uint32_t kv) {
    if(!query || !kv) return "none";
    if(query==kv) return "mha";
    if(kv==1) return "mqa";
    return query>kv && query%kv==0?"gqa":"other";
}
inline json native_architecture_contract(const llama_model & model) {
    const auto & hp=model.hparams;
    json layers=json::array();
    std::set<std::string> attention;
    uint32_t recurrent=0,moe=0,shared=0,indexed=0,sliding=0;
    for(uint32_t i=0;i<hp.n_layer();++i) {
        const auto & layer=model.layers.at(i);
        const bool recr=hp.is_recr(i);
        const bool experts=layer.ffn_down_exps!=nullptr;
        const bool shared_path=layer.ffn_down_shexp!=nullptr;
        const bool indexer=layer.indexer_attn_k || layer.indexer_attn_q_b || layer.indexer_comp_wkv || layer.indexer_kpool_gate;
        const bool swa=hp.swa_type!=LLAMA_SWA_TYPE_NONE && hp.is_swa(i);
        const std::string kind=recr?"recurrent":hp.is_mla()?"mla":attention_topology(hp.n_head(i),hp.n_head_kv(i));
        attention.insert(kind);
        recurrent+=recr; moe+=experts; shared+=shared_path; indexed+=indexer; sliding+=swa;
        layers.push_back({{"index",i},{"attention",kind},{"query_heads",hp.n_head(i)},{"kv_heads",hp.n_head_kv(i)},
            {"sliding_window",swa},{"indexer",indexer},{"feed_forward",experts?"moe":"dense"},
            {"routed_experts",experts?hp.n_expert_used(i):0},{"shared_expert_path",shared_path}});
    }
    return {{"schema_version",1},{"source","pinned backend geometry and loaded tensor layouts"},
        {"scope","execution structure; not model quality or performance acceptance"},
        {"feed_forward",moe==0?"dense":moe==hp.n_layer()?"moe":"dense_moe"},
        {"state_kind",recurrent==0?"attention":recurrent==hp.n_layer()?"recurrent":"hybrid"},
        {"attention_kinds",std::vector<std::string>(attention.begin(),attention.end())},
        {"moe_layers",moe},{"shared_expert_layers",shared},{"recurrent_layers",recurrent},
        {"indexed_layers",indexed},{"sliding_layers",sliding},{"sliding_window",hp.n_swa},
        {"context_policy","explicit; no automatic shrinking"},{"precision_policy","artifact weights and configured native state"},
        {"routing_policy","execute every backend-selected expert; no resident-only pruning"},
        {"request_state_policy","fresh context and sampler; retained weights only"},
        {"task_policy","reasoning and code reuse architecture; template and task acceptance are separate"},
        {"layers",layers}};
}
inline void check_sliding_semantics(uint64_t declared,uint64_t effective) {
    if(declared>0 && effective!=declared)
        throw std::runtime_error("Backend changed the model's declared sliding-window semantics; this configuration is refused rather than silently changing attention");
}
} // namespace wllm
