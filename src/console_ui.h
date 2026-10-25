#pragma once
#include "terminal.h"
#include "banner.h"
#include <atomic>
#include <cwctype>
#include <deque>

namespace wllm {
inline void show_banner(bool ascii = false) {
    const bool color = terminal_color();
    if (ascii || console_width() < 57) std::cout << "\n W-LLM\n";
    else {
        std::cout << '\n';
        for (const auto & row : banner_rows) {
            const auto chars = wide(row);
            for (wchar_t ch : chars) {
                if (color && ch == L'\x2588') std::cout << "\x1b[38;2;64;158;255m";
                else if (color && ch == L'\x2593') std::cout << "\x1b[38;2;37;90;145m";
                std::cout << utf8(std::wstring(1, ch));
            }
            if (color) std::cout << "\x1b[0m";
            std::cout << '\n';
        }
    }
    std::cout << "\n :: W-LLM ::  (v0.4.0)\n"
        " Local CPU inference / Reflex\n"
        " Independent requests. Settings and input history stay in this process.\n"
        " /help for commands. Enter sends; Shift+Enter adds a line; Ctrl+V pastes.\n\n" << std::flush;
}

inline size_t previous_character(const std::wstring & text, size_t pos) {
    if (!pos) return 0;
    --pos;
    if (pos && text[pos] >= 0xdc00 && text[pos] <= 0xdfff) --pos;
    WORD kind = 0;
    while (pos && GetStringTypeW(CT_CTYPE3, &text[pos], 1, &kind) && (kind & (C3_NONSPACING | C3_DIACRITIC))) --pos;
    return pos;
}
inline size_t next_character(const std::wstring & text, size_t pos) {
    if (pos >= text.size()) return text.size();
    if (text[pos] >= 0xd800 && text[pos] <= 0xdbff && pos+1 < text.size()) ++pos;
    ++pos;
    WORD kind = 0;
    while (pos < text.size() && GetStringTypeW(CT_CTYPE3, &text[pos], 1, &kind) && (kind & (C3_NONSPACING | C3_DIACRITIC))) ++pos;
    return pos;
}
inline int cell_width(const std::wstring & text, size_t pos) {
    const unsigned cp = text[pos];
    WORD kind = 0;
    if (GetStringTypeW(CT_CTYPE3, &text[pos], 1, &kind) && (kind & C3_NONSPACING)) return 0;
    return (cp >= 0x1100 && (cp <= 0x115f || (cp >= 0x2e80 && cp <= 0xa4cf) ||
        (cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0xf900 && cp <= 0xfaff) ||
        (cp >= 0xfe10 && cp <= 0xfe6f) || (cp >= 0xff01 && cp <= 0xff60) ||
        (cp >= 0xd800 && cp <= 0xdbff))) ? 2 : 1;
}

// A single editable viewport; the full request can contain multiple lines.
// Console scrollback remains available throughout input and streaming output.
class LineEditor {
    std::vector<std::wstring> history;
    std::deque<INPUT_RECORD> pending;
    static constexpr size_t limit = 1024 * 1024;
    static void insert(std::wstring & text, size_t & pos, const std::wstring & value) {
        std::wstring clean;
        for (wchar_t ch : value) {
            if (ch == L'\r') continue;
            if ((ch >= 32 && ch != 127 && !(ch >= 0x80 && ch < 0xa0)) || ch == L'\n' || ch == L'\t') clean += ch;
        }
        if (text.size() + clean.size() > limit) throw std::runtime_error("Input exceeds 1 Mi characters");
        text.insert(pos, clean); pos += clean.size();
    }
    static void paste(std::wstring & text, size_t & pos) {
        if (!OpenClipboard(nullptr)) return;
        const auto data = GetClipboardData(CF_UNICODETEXT);
        const wchar_t * value = data ? static_cast<const wchar_t *>(GlobalLock(data)) : nullptr;
        std::wstring copy;
        if (value) { copy = value; GlobalUnlock(data); }
        CloseClipboard();
        insert(text, pos, copy);
    }
    static void draw(const std::wstring & text, size_t pos, const std::wstring & prompt) {
        std::cout.flush();
        const auto output = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (!GetConsoleScreenBufferInfo(output, &info)) return;
        const int width = std::max(1, int(info.srWindow.Right-info.srWindow.Left+1));
        const int room = std::max(1, width - int(prompt.size()) - 2);
        size_t begin = pos;
        int before = 0;
        while (begin) {
            const size_t prior = previous_character(text, begin);
            const int cells = text[prior] == L'\n' || text[prior] == L'\t' ? 1 : cell_width(text, prior);
            if (before + cells > room * 2 / 3) break;
            before += cells; begin = prior;
        }
        std::wstring view = prompt;
        int used = 0;
        for (size_t at=begin; at < text.size();) {
            const int cells = text[at] == L'\n' || text[at] == L'\t' ? 1 : cell_width(text, at);
            if (used+cells > room) break;
            const auto next = next_character(text, at);
            view += text[at] == L'\n' ? L"|" : text[at] == L'\t' ? L" " : text.substr(at, next-at);
            used += cells; at=next;
        }
        const COORD start{info.srWindow.Left, info.dwCursorPosition.Y};
        DWORD written = 0;
        FillConsoleOutputCharacterW(output, L' ', DWORD(width), start, &written);
        SetConsoleCursorPosition(output, start);
        WriteConsoleW(output, view.data(), DWORD(view.size()), &written, nullptr);
        SetConsoleCursorPosition(output, {SHORT(start.X+std::min(width-1,int(prompt.size())+before)),start.Y});
    }
public:
    void clear() { history.clear(); }
    bool read(const std::string & label, std::string & result, std::atomic<bool> & interrupted) {
        if (!console_handle(STD_INPUT_HANDLE) || !console_handle(STD_OUTPUT_HANDLE)) return bool(std::getline(std::cin,result));
        const auto input = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode = 0; GetConsoleMode(input, &mode);
        struct Restore {
            HANDLE input; DWORD mode;
            bool bracket;
            ~Restore() { if (bracket) std::cout << "\x1b[?2004l" << std::flush; SetConsoleMode(input,mode); }
        } restore{input,mode,terminal_color()};
        SetConsoleMode(input,(mode | ENABLE_EXTENDED_FLAGS | ENABLE_WINDOW_INPUT | ENABLE_PROCESSED_INPUT) &
            ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_QUICK_EDIT_MODE | ENABLE_VIRTUAL_TERMINAL_INPUT));
        if (restore.bracket) std::cout << "\x1b[?2004h" << std::flush;
        std::wstring text, draft, escape;
        size_t pos=0, index=history.size();
        bool bracketed=false;
        const auto prompt = wide(label);
        draw(text,pos,prompt);
        for (;;) {
            if (interrupted.exchange(false)) {
                text.clear(); pos=0; escape.clear(); bracketed=false;
                draw(text,pos,prompt);
            }
            INPUT_RECORD events[128]{}; DWORD count=0;
            if(!pending.empty()) {
                while(!pending.empty() && count<128) { events[count++]=pending.front(); pending.pop_front(); }
            } else {
                if (WaitForSingleObject(input,100) != WAIT_OBJECT_0) continue;
                if (!ReadConsoleInputW(input,events,128,&count)) throw std::runtime_error("Cannot read console input");
            }
            for (DWORD n=0; n<count; ++n) {
                if (events[n].EventType != KEY_EVENT || !events[n].Event.KeyEvent.bKeyDown) continue;
                const auto & key=events[n].Event.KeyEvent;
                const bool ctrl=(key.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0;
                const bool shift=(key.dwControlKeyState & SHIFT_PRESSED) != 0;
                const wchar_t ch=key.uChar.UnicodeChar;
                if (ch==L'\x1b' || !escape.empty()) {
                    escape+=ch;
                    if (escape==L"\x1b[200~") { bracketed=true; escape.clear(); }
                    else if (escape==L"\x1b[201~") { bracketed=false; escape.clear(); }
                    else if (std::wstring(L"\x1b[200~").find(escape)!=0 && std::wstring(L"\x1b[201~").find(escape)!=0) escape.clear();
                    continue;
                }
                if (bracketed) { insert(text,pos,ch==L'\r'?L"\n":std::wstring(1,ch)); continue; }
                if ((ctrl && key.wVirtualKeyCode=='V') || (shift && key.wVirtualKeyCode==VK_INSERT)) { paste(text,pos); continue; }
                if (ctrl && key.wVirtualKeyCode=='D' && text.empty()) { std::cout << '\n'; return false; }
                if (ctrl && key.wVirtualKeyCode=='U') { text.clear(); pos=0; continue; }
                if (key.wVirtualKeyCode==VK_RETURN) {
                    if (shift) { insert(text,pos,L"\n"); continue; }
                    if (!text.empty() && (history.empty() || history.back()!=text)) {
                        history.push_back(text);
                        if(history.size()>32) history.erase(history.begin());
                    }
                    draw(text,text.size(),prompt);
                    std::cout << '\n' << std::flush;
                    for(DWORD rest=n+1;rest<count;++rest) pending.push_back(events[rest]);
                    result=utf8(text); return true;
                }
                if (key.wVirtualKeyCode==VK_LEFT) pos=previous_character(text,pos);
                else if (key.wVirtualKeyCode==VK_RIGHT) pos=next_character(text,pos);
                else if (key.wVirtualKeyCode==VK_HOME) pos=0;
                else if (key.wVirtualKeyCode==VK_END) pos=text.size();
                else if (key.wVirtualKeyCode==VK_BACK && pos) { const auto prior=previous_character(text,pos); text.erase(prior,pos-prior); pos=prior; }
                else if (key.wVirtualKeyCode==VK_DELETE && pos<text.size()) text.erase(pos,next_character(text,pos)-pos);
                else if (key.wVirtualKeyCode==VK_UP && index) { if(index==history.size()) draft=text; text=history[--index]; pos=text.size(); }
                else if (key.wVirtualKeyCode==VK_DOWN && index<history.size()) { ++index; text=index==history.size()?draft:history[index]; pos=text.size(); }
                else if (ch>=32 || ch==L'\t') for(WORD repeat=0;repeat<std::max(WORD(1),key.wRepeatCount);++repeat) insert(text,pos,std::wstring(1,ch));
            }
            draw(text,pos,prompt);
        }
    }
};
} // namespace wllm
