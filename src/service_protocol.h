#pragma once
#include "core.h"
#include <functional>
#include <set>
#include <utility>

namespace wllm {
struct ServiceOptions {
    std::string type="all", host="127.0.0.1", alias="wllm-local", key_env;
    int port=8080;
    void validate() const {
        if(type!="all" && type!="openai" && type!="systemone") throw std::runtime_error("service must be all, openai or systemone");
        if(port<1 || port>65535) throw std::runtime_error("port must be 1..65535");
        for(const auto & field:{host,alias})
            if(field.empty() || field.size()>255 || std::any_of(field.begin(),field.end(),[](unsigned char c){return c<=32 || c==127;}))
                throw std::runtime_error("Invalid service host or model alias");
        if(key_env.size()>128 || std::any_of(key_env.begin(),key_env.end(),[](unsigned char c){return !(c=='_' || (c>='0' && c<='9') || (c>='a' && c<='z') || (c>='A' && c<='Z'));}))
            throw std::runtime_error("Invalid API key environment variable name");
    }
};
inline void service_keys(const json & body, const std::set<std::string> & allowed) {
    if(!body.is_object()) throw std::runtime_error("Request body must be an object");
    for(auto it=body.begin();it!=body.end();++it)
        if(!allowed.count(it.key())) throw std::runtime_error("Unsupported request field: "+it.key());
}
inline json config_settings(const Config & config) {
    auto settings=config.dump();
    for(const auto * key:{"effective_thinking","sampling_applied","backend","cache_k","cache_v","recurrent_state","flash_attention","context_shift"}) settings.erase(key);
    return settings;
}
struct ServiceGeneration {
    Config config;
    std::vector<std::pair<std::string,std::string>> messages;
    std::string prompt;
    std::vector<std::string> stops;
    bool stream=false, include_usage=false, chat=true;
    static ServiceGeneration parse(const Config & base,const json & body,bool chat) {
        service_keys(body,{"model","messages","prompt","stream","stream_options","max_tokens","max_completion_tokens","temperature","top_p","presence_penalty","frequency_penalty","seed","stop","n","thinking","top_k","min_p","repeat_penalty","user","store","response_format","echo"});
        ServiceGeneration r; r.chat=chat;
        if(!body.contains("model") || !body.at("model").is_string() || body.at("model").get<std::string>().empty()) throw std::runtime_error("model is required");
        if(body.contains("n") && (!body["n"].is_number_integer() || body["n"]!=1)) throw std::runtime_error("Only n=1 is supported");
        if(body.contains("store") && body["store"]!=false) throw std::runtime_error("store must be false; requests are stateless");
        if(body.contains("user") && !body["user"].is_string()) throw std::runtime_error("user must be a string");
        if(body.contains("response_format") && body["response_format"]!=json({{"type","text"}})) throw std::runtime_error("response_format must be text");
        if(body.contains("echo") && (chat || body["echo"]!=false)) throw std::runtime_error("Only completion echo=false is supported");
        if(body.contains("stream")) r.stream=body["stream"].get<bool>();
        if(body.contains("stream_options") && !body["stream_options"].is_null()) {
            if(!r.stream) throw std::runtime_error("stream_options requires stream=true");
            service_keys(body["stream_options"],{"include_usage"});
            r.include_usage=body["stream_options"].value("include_usage",false);
        }
        auto settings=config_settings(base);
        settings["inference_mode"]="generate"; settings["raw"]=!chat;
        if(!chat) settings["chat_template"]="";
        for(const auto * key:{"temperature","top_p","presence_penalty","frequency_penalty","seed","thinking","top_k","min_p","repeat_penalty"})
            if(body.contains(key) && !body[key].is_null()) settings[key]=body[key];
        if(body.contains("max_tokens") && body.contains("max_completion_tokens")) throw std::runtime_error("Use either max_tokens or max_completion_tokens");
        for(const auto * key:{"max_tokens","max_completion_tokens"})
            if(body.contains(key) && !body[key].is_null()) {
                if(!body[key].is_number_integer() || body[key].get<int64_t>()<1) throw std::runtime_error("Output token limit must be a positive integer");
                settings["max_tokens"]=body[key];
            }
        r.config=Config::parse(settings);
        if(body.contains("stop") && !body["stop"].is_null()) {
            const auto items=body["stop"].is_string()?json::array({body["stop"]}):body["stop"];
            if(!items.is_array() || items.size()>4) throw std::runtime_error("stop must be a string or at most four strings");
            for(const auto & item:items) {
                const auto value=item.get<std::string>();
                if(value.empty() || value.size()>4096) throw std::runtime_error("stop must contain 1..4096 bytes");
                r.stops.push_back(value);
            }
        }
        if(chat) {
            if(body.contains("prompt")) throw std::runtime_error("Chat requests use messages");
            const auto & messages=body.at("messages");
            if(!messages.is_array() || messages.empty() || messages.size()>256) throw std::runtime_error("messages must contain 1..256 entries");
            for(const auto & message:messages) {
                service_keys(message,{"role","content"});
                const auto role=message.at("role").get<std::string>();
                if(role!="system" && role!="user" && role!="assistant") throw std::runtime_error("Text roles are system, user and assistant");
                const auto & content=message.at("content");
                std::string text;
                if(content.is_string()) text=content.get<std::string>();
                else if(content.is_array()) for(const auto & part:content) {
                    service_keys(part,{"type","text"});
                    if(part.at("type")!="text") throw std::runtime_error("Only text content parts are supported");
                    text+=part.at("text").get<std::string>();
                } else throw std::runtime_error("content must be text or text parts");
                r.messages.emplace_back(role,text);
            }
        } else {
            if(body.contains("messages") || body.contains("max_completion_tokens")) throw std::runtime_error("Completions use prompt and max_tokens");
            r.prompt=body.at("prompt").get<std::string>();
            if(r.prompt.empty()) throw std::runtime_error("prompt cannot be empty");
        }
        return r;
    }
};

// Preserve incomplete UTF-8 and possible stop prefixes across model tokens.
class ServiceText {
    std::vector<std::string> stops;
    std::function<bool(const std::string &)> send;
    std::string pending;
    bool ended=false;
    static size_t complete_prefix(const std::string & text,size_t limit) {
        if(!limit) return 0;
        size_t start=limit-1;
        while(start>0 && (static_cast<unsigned char>(text[start])&0xc0)==0x80) --start;
        const auto ch=static_cast<unsigned char>(text[start]);
        const size_t need=ch<0x80?1:ch>=0xc2 && ch<=0xdf?2:ch>=0xe0 && ch<=0xef?3:ch>=0xf0 && ch<=0xf4?4:1;
        return limit-start<need?start:limit;
    }
    bool emit(size_t count) {
        if(!count) return true;
        if(!send(pending.substr(0,count))) return false;
        pending.erase(0,count); return true;
    }
public:
    ServiceText(std::vector<std::string> values,std::function<bool(const std::string &)> output):stops(std::move(values)),send(std::move(output)) {}
    bool stopped() const { return ended; }
    bool write(const std::string & piece) {
        if(ended) return true;
        pending+=piece;
        size_t at=std::string::npos;
        for(const auto & stop:stops) at=std::min(at,pending.find(stop));
        if(at!=std::string::npos) { if(!emit(at)) return false; pending.clear(); ended=true; return true; }
        size_t keep=0;
        for(const auto & stop:stops) for(size_t n=1;n<stop.size() && n<=pending.size();++n)
            if(pending.compare(pending.size()-n,n,stop,0,n)==0) keep=std::max(keep,n);
        return emit(complete_prefix(pending,pending.size()-keep));
    }
    bool finish() { return ended || emit(pending.size()); }
};
} // namespace wllm
