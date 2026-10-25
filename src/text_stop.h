#pragma once
#include <functional>
#include <string>
#include <vector>
#include <algorithm>

namespace wllm {
// Preserve incomplete UTF-8 and possible stop prefixes across model tokens.
class TextStopFilter {
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
    TextStopFilter(std::vector<std::string> values,std::function<bool(const std::string &)> output):stops(std::move(values)),send(std::move(output)) {}
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
