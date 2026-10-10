#pragma once
#include "core.h"
#include "ggml.h"
#include <chrono>
#include <map>

namespace wllm {
// Diagnostic only. Node callbacks split/synchronize the graph and can inhibit
// fusion. These timings locate candidates; never use them as speedup evidence.
class Profile {
    using Clock = std::chrono::steady_clock;
    struct Entry { size_t calls = 0; double seconds = 0; };
    std::map<std::string, Entry> stages, nodes;
    Clock::time_point node_start;
    std::string node_key;
public:
    bool enabled = false, operators = false;
    std::string phase;
    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto & p = *static_cast<Profile *>(data);
        if (ask) {
            p.node_key = p.phase + "/" + ggml_op_name(tensor->op) + "/" +
                (tensor->src[0] ? ggml_type_name(tensor->src[0]->type) : "none");
            // Allocate outside the measured interval. Only aggregate by phase/op/type.
            p.nodes.try_emplace(p.node_key);
            p.node_start = Clock::now();
        } else {
            const double seconds = std::chrono::duration<double>(Clock::now() - p.node_start).count();
            auto & entry = p.nodes.at(p.node_key); ++entry.calls; entry.seconds += seconds;
        }
        return true;
    }
    json dump() const {
        if (!enabled) return nullptr;
        auto rows = [](const auto & entries) {
            json result = json::array();
            for (const auto & [key, e] : entries)
                result.push_back({{"name",key},{"calls",e.calls},{"seconds",e.seconds}});
            return result;
        };
        return {{"diagnostic",true},{"operator_callbacks",operators},
            {"timing_scope","wall time; process calls include backend synchronization; partial calls included on failure"},
            {"operator_warning","Callbacks split the graph and may inhibit fusion; not production kernel timings"},
            {"stages",rows(stages)},{"operators",rows(nodes)}};
    }
    class Span {
        Entry * entry = nullptr;
        Clock::time_point start;
    public:
        Span(Profile & p, const char * label) {
            if (p.enabled) {
                // Allocate before the operation, never while unwinding a failed call.
                entry = &p.stages[label]; start = Clock::now();
            }
        }
        ~Span() {
            if (entry) { ++entry->calls; entry->seconds += std::chrono::duration<double>(Clock::now()-start).count(); }
        }
        Span(const Span &) = delete;
        Span & operator=(const Span &) = delete;
    };
};
} // namespace wllm
