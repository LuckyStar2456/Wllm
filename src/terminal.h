#pragma once
#include "core.h"
#include <windows.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace wllm {
// The interactive screen owns both streams while active. Redirected stdout stays raw.
class Terminal {
    std::mutex mutex;
    std::string generated;
    bool enabled = false;
    DWORD original_mode = 0;
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);

    static std::string number(const json & j, const char * key, double divisor = 1) {
        if (j.at(key).is_null()) return "--";
        std::ostringstream s; s << std::fixed << std::setprecision(2) << j.at(key).get<double>() / divisor; return s.str();
    }
public:
    bool active() const { return enabled; }
    void start(bool quiet) {
        DWORD error_mode;
        if (!quiet && GetConsoleMode(output, &original_mode) && GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &error_mode) &&
            SetConsoleMode(output, original_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            enabled = true;
            std::cout << "\x1b[?1049h\x1b[?25l" << std::flush;
        }
    }
    void write(const std::string & piece) {
        if (enabled) { std::lock_guard<std::mutex> lock(mutex); generated += piece; }
        else { std::cout.write(piece.data(), piece.size()); std::cout.flush(); }
    }
    void render(const json & j, int context) {
        std::lock_guard<std::mutex> lock(mutex);
        CONSOLE_SCREEN_BUFFER_INFO info{};
        int width=80, height=24;
        if (GetConsoleScreenBufferInfo(output,&info)) {
            width=std::max(30,int(info.srWindow.Right-info.srWindow.Left+1));
            height=std::max(10,int(info.srWindow.Bottom-info.srWindow.Top+1));
        }
        std::vector<std::string> lines(1);
        int column=0;
        // Wrap UTF-8 text without splitting a code point; hide terminal control sequences.
        for (size_t i=0;i<generated.size();) {
            unsigned char first=static_cast<unsigned char>(generated[i]);
            if(first=='\n') { lines.emplace_back(); column=0; ++i; continue; }
            if(first<32 || first==127) { ++i; continue; }
            size_t n=first<128?1:(first&0xE0)==0xC0?2:(first&0xF0)==0xE0?3:4;
            if(i+n>generated.size()) break;
            uint32_t cp=first & (n==1?0x7F:n==2?0x1F:n==3?0x0F:0x07);
            bool valid=true;
            for(size_t k=1;k<n;++k) { unsigned char b=static_cast<unsigned char>(generated[i+k]); if((b&0xC0)!=0x80) valid=false; cp=(cp<<6)|(b&0x3F); }
            if(!valid) { ++i; continue; }
            const int cells=(cp>=0x1100 && (cp<=0x115f || (cp>=0x2e80 && cp<=0xa4cf) || (cp>=0xac00 && cp<=0xd7af) || (cp>=0xf900 && cp<=0xfaff) || cp>=0x1f300))?2:1;
            if(column+cells>width-1) { lines.emplace_back(); column=0; }
            lines.back().append(generated,i,n); column+=cells; i+=n;
        }
        const int text_rows=height-8;
        std::ostringstream frame;
        const std::string title="Wllm 0.1 | CPU Q4_K_M | ctx=" + std::to_string(context) + " | Ctrl+C stop";
        frame << "\x1b[H\x1b[2K" << title.substr(0,width-1) << "\r\n";
        size_t begin=lines.size()>size_t(text_rows)?lines.size()-text_rows:0;
        for(int row=0;row<text_rows;++row) {
            frame << "\x1b[2K";
            if(begin+row<lines.size()) frame << lines[begin+row];
            frame << "\r\n";
        }
        std::vector<std::string> status={
            "--- " + j.at("stage").get<std::string>() + " | total " + number(j,"elapsed_s") + "s | load " + number(j,"load_s") + "s",
            "Input " + j.at("input_processed").dump() + "/" + j.at("input_tokens").dump() + " | " + number(j,"prefill_tps") + " tok/s | TTFT " + number(j,"ttft_s") + "s",
            "Output " + j.at("output_tokens").dump() + " | live " + number(j,"decode_window_tps") + " | avg " + number(j,"decode_average_tps") + " tok/s",
            "RAM " + number(j,"working_set_bytes",1048576) + " | peak " + number(j,"sampled_peak_working_set_bytes",1048576) + " MiB",
            "Private " + number(j,"private_bytes",1048576) + " | system free " + number(j,"system_available_bytes",1048576) + " MiB",
            "-- = pending / insufficient samples. Full output retained on exit."};
        for(auto & line:status) frame << "\x1b[2K" << line.substr(0,width-1) << "\r\n";
        std::cout << frame.str() << std::flush;
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if(enabled) {
            std::cout << "\x1b[?25h\x1b[?1049l" << generated << '\n' << std::flush;
            SetConsoleMode(output,original_mode);
            enabled=false;
        }
    }
    ~Terminal() { finish(); }
};
}
