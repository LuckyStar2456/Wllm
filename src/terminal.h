#pragma once
#include "core.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace wllm {
inline bool console_handle(DWORD stream) {
    DWORD mode = 0;
    return GetConsoleMode(GetStdHandle(stream), &mode) != FALSE;
}
inline std::wstring wide(const std::string & text) {
    if (text.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (!count) throw std::runtime_error("Invalid UTF-8 text");
    std::wstring result(count, 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), result.data(), count);
    return result;
}
inline std::string utf8(const std::wstring & text) {
    if (text.empty()) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0, nullptr, nullptr);
    std::string result(count, 0);
    WideCharToMultiByte(CP_UTF8, 0, text.data(), int(text.size()), result.data(), count, nullptr, nullptr);
    return result;
}
inline int console_width() {
    CONSOLE_SCREEN_BUFFER_INFO info{};
    return GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)
        ? std::max(1, int(info.srWindow.Right - info.srWindow.Left + 1)) : 80;
}
inline bool terminal_color() {
    wchar_t value[2]{};
    SetLastError(ERROR_SUCCESS);
    const DWORD length=GetEnvironmentVariableW(L"NO_COLOR",value,2);
    return length==0 && GetLastError()==ERROR_ENVVAR_NOT_FOUND && console_handle(STD_OUTPUT_HANDLE);
}
inline std::string visible_text(const std::string & text) {
    std::string clean;
    for (unsigned char c : text) {
        if (c >= 32 && c != 127) clean += char(c);
        else if (c == '\n' || c == '\r' || c == '\t') clean += char(c);
    }
    return clean;
}
// Model tokens may end in the middle of a UTF-8 character. Decode complete
// characters for the console, retaining the trailing bytes for the next token.
inline std::wstring console_characters(std::string & pending, bool final = false) {
    std::wstring out;
    size_t i=0;
    while(i<pending.size()) {
        const auto first=static_cast<unsigned char>(pending[i]);
        const size_t count=first<0x80?1:first>=0xc2 && first<=0xdf?2:first>=0xe0 && first<=0xef?3:first>=0xf0 && first<=0xf4?4:0;
        if(count && i+count>pending.size() && !final) break;
        uint32_t cp=count?first & (count==1?0x7f:count==2?0x1f:count==3?0x0f:0x07):0xfffd;
        bool valid=count && i+count<=pending.size();
        if(valid) for(size_t k=1;k<count;++k) {
            const auto next=static_cast<unsigned char>(pending[i+k]);
            if((next & 0xc0)!=0x80) valid=false;
            cp=(cp<<6)|(next & 0x3f);
        }
        if(!valid || (count==2 && cp<0x80) || (count==3 && cp<0x800) || (count==4 && cp<0x10000) ||
            cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff)) { cp=0xfffd; i+=1; }
        else i+=count;
        if((cp<32 && cp!='\n' && cp!='\r' && cp!='\t') || (cp>=0x7f && cp<=0x9f)) continue;
        if(cp<0x10000) out+=wchar_t(cp);
        else { cp-=0x10000; out+=wchar_t(0xd800+(cp>>10)); out+=wchar_t(0xdc00+(cp & 0x3ff)); }
    }
    pending.erase(0,i);
    return out;
}
// Scope host console changes, including early exits and exceptions.
class ConsoleEnvironment {
    UINT input_cp = GetConsoleCP(), output_cp = GetConsoleOutputCP();
    DWORD mode = 0;
    bool output_console = GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode) != FALSE;
public:
    ConsoleEnvironment() {
        if (input_cp) SetConsoleCP(CP_UTF8);
        if (output_cp) SetConsoleOutputCP(CP_UTF8);
        if (output_console) SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    ~ConsoleEnvironment() {
        if (output_console) SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), mode);
        if (input_cp) SetConsoleCP(input_cp);
        if (output_cp) SetConsoleOutputCP(output_cp);
    }
};

// Scrollback belongs to the user's terminal. No alternate screen or output replay.
class Terminal {
    std::mutex mutex;
    bool enabled = false, status_line = false, streaming = false;
    std::wstring original_title;
    std::string captured;
    std::string pending_utf8;
    bool capture = false;
    std::function<bool(const std::string &)> writer;
    static std::string number(const json & j, const char * key, double divisor = 1) {
        if (j.at(key).is_null()) return "--";
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << j.at(key).get<double>() / divisor;
        return out.str();
    }
    void clear_status() {
        if (!status_line) return;
        if (terminal_color()) std::cerr << "\r\x1b[2K";
        else std::cerr << '\n';
        status_line = false;
    }
public:
    void set_writer(std::function<bool(const std::string &)> output) { writer=std::move(output); }
    bool active() const { return enabled; }
    const std::string & text() const { return captured; }
    void start(bool quiet, bool retain_output = false) {
        capture=retain_output;
        enabled = !quiet && console_handle(STD_OUTPUT_HANDLE) && console_handle(STD_ERROR_HANDLE);
        if (enabled) {
            wchar_t title[32768]{};
            const auto n = GetConsoleTitleW(title, DWORD(std::size(title)));
            original_title.assign(title, n);
        }
    }
    void write(const std::string & piece) {
        std::lock_guard<std::mutex> lock(mutex);
        clear_status();
        if(capture) captured += piece;
        streaming = true;
        if(writer) {
            if(!writer(piece)) throw std::runtime_error("Output stream failed");
            return;
        }
        if(console_handle(STD_OUTPUT_HANDLE)) {
            pending_utf8+=piece;
            const auto chars=console_characters(pending_utf8);
            std::cout.flush();
            DWORD written=0;
            if(!chars.empty() && !WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE),chars.data(),DWORD(chars.size()),&written,nullptr))
                std::cout.setstate(std::ios::badbit);
        } else std::cout.write(piece.data(), std::streamsize(piece.size()));
        std::cout.flush();
    }
    void render(const json & j, int) {
        std::lock_guard<std::mutex> lock(mutex);
        const std::string status = "W-LLM | " + j.at("stage").get<std::string>() + " | input " +
            j.at("input_processed").dump() + "/" + j.at("input_tokens").dump() +
            " | output " + j.at("output_tokens").dump() + " | " + number(j,"decode_window_tps") +
            " tok/s | RAM " + number(j,"working_set_bytes",1048576) + " MiB";
        SetConsoleTitleW(wide(status).c_str());
        if (!streaming && terminal_color()) {
            std::cerr << "\r\x1b[2K" << status.substr(0, size_t(std::max(1, console_width()-1))) << std::flush;
            status_line = true;
        }
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if(!pending_utf8.empty()) {
            const auto chars=console_characters(pending_utf8,true);
            DWORD written=0;
            WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE),chars.data(),DWORD(chars.size()),&written,nullptr);
        }
        if(capture && !captured.empty()) {
            const int count=MultiByteToWideChar(CP_UTF8,0,captured.data(),int(captured.size()),nullptr,0);
            std::wstring normalized(count,0);
            MultiByteToWideChar(CP_UTF8,0,captured.data(),int(captured.size()),normalized.data(),count);
            captured=utf8(normalized);
        }
        if (!enabled) return;
        clear_status();
        if (streaming) std::cout << '\n' << std::flush;
        SetConsoleTitleW(original_title.c_str());
        enabled = false;
    }
    ~Terminal() { finish(); }
};
} // namespace wllm
