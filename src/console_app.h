// Included after engine request functions; this layer does not own inference state.
static void write_new_file(const std::string & name, const std::string & content) {
    const auto path=wide(name);
    const HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot create export; choose a new writable file path");
    DWORD written=0;
    const bool ok=WriteFile(file,content.data(),DWORD(content.size()),&written,nullptr)!=FALSE && written==content.size();
    CloseHandle(file);
    if(!ok) throw std::runtime_error("Export write failed");
}

static int run_console_ui(json settings, std::string system, std::string candidates_file,
        std::string request_file, bool ascii) {
    auto config=Config::parse(settings);
    const auto safe_config=[](const Config & c) {
        if(c.gpu_layers || c.cpu_weight_repacking || c.prefix_cache_mib)
            throw std::runtime_error("The terminal interface uses the CPU path with repacking and prefix cache off");
    };
    safe_config(config);
    ModelPtr resident(nullptr,llama_model_free);
    json last=nullptr;
    json selected_profile=nullptr;
    LineEditor editor;
    show_banner(ascii);
    const auto info=[&] {
        if(!selected_profile.is_null()) std::cout << " identity " << visible_text(selected_profile.value("name",std::string())) << " / " << visible_text(selected_profile.value("architecture",std::string())) << " / " << visible_text(selected_profile.value("quantization",std::string())) << " / " << visible_text(selected_profile.value("state_kind",std::string())) << '\n';
        if(selected_profile.is_object() && selected_profile.contains("architecture_contract")) {
            const auto & contract=selected_profile.at("architecture_contract");
            std::cout << " structure " << contract.at("feed_forward").get<std::string>() << " / " << contract.at("attention_kinds").dump() << " / " << contract.at("state_kind").get<std::string>() << '\n';
        }
        std::cout << " model    " << (config.model.empty()?"select with /model PATH":visible_text(config.model)) << '\n'
            << " backend  CPU / " << WLLM_CPU_TARGET << " / threads " << config.threads << ':' << config.effective_threads_batch() << '\n'
            << " context  " << config.context << " / KV F16 / state from model\n"
            << " mode     " << config.inference_mode << " / thinking " << (config.inference_mode=="generate" && config.thinking && !config.raw?"on":"off") << '\n'
            << " state    " << (resident?"weights loaded":"unloaded") << '\n'
            << " record   " << (config.record.empty()?"off":visible_text(config.record)) << "\n\n";
    };
    const auto help=[] {
        std::cout <<
            " /help                 Show commands and input keys\n"
            " /model PATH           Select a local GGUF text model\n"
            " /info                 Model, backend, settings and loaded state\n"
            " /load  /unload        Load or release model weights\n"
            " /mode MODE            generate | candidate_score | system_one\n"
            " /candidates FILE      Candidate JSON for candidate_score\n"
            " /request FILE         Select a System One JSON request file\n"
            " /run FILE             Submit a UTF-8 prompt or System One request\n"
            " /system TEXT          Set the generation system prompt; empty clears\n"
            " /config               Show effective configuration\n"
            " /set KEY JSON         Change one setting, for example /set threads 4\n"
            " /reset                Clear request display, system prompt and input history\n"
            " /metrics              Show the last request result and metrics\n"
            " /export FILE          Save the last result as JSON (new file only)\n"
            " /exit                 Release weights and exit; Ctrl+D also exits\n"
            " Enter sends; Shift+Enter inserts newline; Ctrl+V/Shift+Insert pastes.\n"
            " Left/Right/Home/End edit; Up/Down recall this process's inputs.\n"
            " Ctrl+U clears input. Ctrl+C cancels inference or clears input.\n"
            " Input/last result live in memory only. No conversation is appended.\n"
            " Streaming metrics appear in the window title; summaries follow output.\n\n";
    };
    info();
    std::string line;
    for (;;) {
        interrupted=false;
        if(!editor.read(config.inference_mode=="system_one"?"json> ":"wllm> ",line,interrupted)) break;
        if(line.empty()) continue;
        try {
            std::string body=line;
            if(line.front()=='/') {
                const size_t split=line.find_first_of(" \t\r\n");
                const std::string command=line.substr(0,split);
                std::string arg=split==std::string::npos?"":line.substr(split+1);
                const auto start=arg.find_first_not_of(" \t");
                arg=start==std::string::npos?"":arg.substr(start);
                if(arg.size()>=2 && arg.front()=='"' && arg.back()=='"' && command!="/set") arg=arg.substr(1,arg.size()-2);
                if(command=="/exit" || command=="/quit") break;
                if(command=="/help") { help(); continue; }
                if(command=="/info" || command=="/models") { info(); continue; }
                if(command=="/config") { std::cout << config.dump().dump(2) << '\n'; continue; }
                if(command=="/metrics") { std::cout << (last.is_null()?"No request result in this process.":last.dump(2)) << '\n'; continue; }
                if(command=="/system") { system=arg; std::cout << "System prompt updated.\n"; continue; }
                if(command=="/candidates") {
                    CandidateTask::parse(json::parse(read_text(arg)));
                    candidates_file=arg; std::cout << "Candidate file selected.\n"; continue;
                }
                if(command=="/request") {
                    SystemOneRequest::parse(json::parse(read_text(arg)));
                    request_file=arg; std::cout << "Request file selected; /run FILE submits it.\n"; continue;
                }
                if(command=="/model") {
                    if(arg.empty()) { info(); continue; }
                    auto next=settings; next["model"]=arg;
                    State probe; probe.config=Config::parse(next); preflight(probe,false);
                    resident.reset(); settings=next; config=probe.config; last=nullptr; selected_profile=probe.identity;
                    std::cout << "Model selected; weights load on request or /load.\n"; info(); continue;
                }
                if(command=="/mode") {
                    auto next=settings; next["inference_mode"]=arg;
                    const auto changed=Config::parse(next); safe_config(changed);
                    settings=next; config=changed; std::cout << "Mode: " << config.inference_mode << '\n'; continue;
                }
                if(command=="/set") {
                    const size_t separator=arg.find_first_of(" \t");
                    if(separator==std::string::npos) throw std::runtime_error("Use /set KEY JSON_VALUE");
                    const auto key=arg.substr(0,separator);
                    if(key=="model" || key=="inference_mode") throw std::runtime_error("Use /model or /mode");
                    auto next=settings; next[key]=json::parse(arg.substr(separator+1));
                    const auto changed=Config::parse(next); safe_config(changed);
                    settings=next; config=changed;
                    std::cout << "Setting applies to the next request.\n"; continue;
                }
                if(command=="/unload") { resident.reset(); std::cout << "Model weights released.\n"; continue; }
                if(command=="/load") {
                    if(resident) { std::cout << "Model weights already loaded.\n"; continue; }
                    State probe; probe.config=config; preflight(probe); selected_profile=probe.identity;
                    auto params=llama_model_default_params();
                    params.n_gpu_layers=0; params.load_mode=LLAMA_LOAD_MODE_MMAP; params.use_extra_bufts=false;
                    params.progress_callback=load_progress; params.progress_callback_user_data=&probe;
                    const auto start_load=Clock::now();
                    resident.reset(llama_model_load_from_file(config.model.c_str(),params));
                    if(!resident) throw std::runtime_error(interrupted?"Loading cancelled":"Model load failed");
                    std::cout << "Model weights loaded in " << std::chrono::duration<double>(Clock::now()-start_load).count() << "s.\n";
                    continue;
                }
                if(command=="/reset") { last=nullptr; system.clear(); editor.clear(); std::cout << "Input history, system prompt and last result cleared.\n"; continue; }
                if(command=="/export") {
                    if(last.is_null()) throw std::runtime_error("Run a request before exporting");
                    write_new_file(arg,last.dump(2)+"\n"); std::cout << "Result exported.\n"; continue;
                }
                if(command=="/run") {
                    const auto file=arg.empty()?request_file:arg;
                    if(file.empty()) throw std::runtime_error("Use /run FILE");
                    body=read_text(file);
                } else throw std::runtime_error("Unknown command; use /help");
            }
            last=nullptr;
            if(config.inference_mode=="system_one") {
                execute_system_one(config,body,"",false,false,false,false,resident,nullptr,&last);
            } else {
                std::optional<CandidateTask> task;
                if(config.inference_mode=="candidate_score") {
                    if(candidates_file.empty()) throw std::runtime_error("Select /candidates FILE first");
                    task=CandidateTask::parse(json::parse(read_text(candidates_file)));
                }
                execute_request(config,body,system,"","","",false,false,false,false,&resident,
                    task?&*task:nullptr,true,nullptr,nullptr,&last);
            }
            interrupted=false;
            if(last.is_object() && last.contains("model_identity")) selected_profile=last["model_identity"];
            else if(last.is_object() && last.contains("_wllm") && last["_wllm"].contains("model_identity")) selected_profile=last["_wllm"]["model_identity"];
            std::cout << "\n";
        } catch(const std::exception & e) {
            interrupted=false;
            std::cerr << "Wllm: " << visible_text(e.what()) << '\n';
        }
    }
    resident.reset();
    std::cout << "Model released. Goodbye.\n";
    return 0;
}
