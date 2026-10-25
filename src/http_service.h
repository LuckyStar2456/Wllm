// Included after engine functions: HTTP is an adapter over the same engine.
#include "httplib.h"

static std::string service_json(const json & value) {
    return value.dump(-1,' ',false,json::error_handler_t::replace);
}
static void service_reply(httplib::Response & response,const json & value,int status=200) {
    response.status=status;
    response.set_content(service_json(value),"application/json; charset=utf-8");
}
static json service_error(const std::string & message,const std::string & type="invalid_request_error",const std::string & code="invalid_request") {
    return {{"error",{{"message",message},{"type",type},{"param",nullptr},{"code",code}}}};
}
static int service_status(int result,const std::string & error) {
    if(result==0) return 200;
    if(result==124) return 504;
    if(result==130) return 499;
    if(error.find("Input leaves")!=std::string::npos || error.find("Candidate input")!=std::string::npos || error.find("Label must")!=std::string::npos) return 400;
    if(error.find("Insufficient memory")!=std::string::npos) return 503;
    return 500;
}
struct ServiceLease {
    std::atomic<bool> & busy;
    explicit ServiceLease(std::atomic<bool> & value):busy(value) {}
    ~ServiceLease() { busy=false; }
};

static int run_http_service(const Config & base,const ServiceOptions & options) {
    options.validate();
    if(base.inference_mode!="generate" || base.raw || base.gpu_layers || base.cpu_weight_repacking || base.prefix_cache_mib)
        throw std::runtime_error("Service uses the shared CPU engine; select API routes with --service, without --mode, --raw or experimental placement/cache options");
    std::string key;
    if(!options.key_env.empty()) {
        const auto name=wide(options.key_env);
        const DWORD length=GetEnvironmentVariableW(name.c_str(),nullptr,0);
        if(length<2 || length>4097) throw std::runtime_error("API key environment variable is missing, empty or too long");
        std::wstring value(length,L'\0');
        const DWORD count=GetEnvironmentVariableW(name.c_str(),value.data(),length);
        if(!count || count>=length) throw std::runtime_error("Cannot read API key environment variable");
        value.resize(count); key=utf8(value);
        if(key.find_first_of("\r\n")!=std::string::npos) throw std::runtime_error("Invalid API key value");
    }
    httplib::Server server;
    server.set_socket_options([](socket_t socket) {
        const BOOL exclusive=TRUE;
        if(setsockopt(socket,SOL_SOCKET,SO_EXCLUSIVEADDRUSE,reinterpret_cast<const char *>(&exclusive),sizeof(exclusive))!=0)
            throw std::runtime_error("Cannot reserve an exclusive service port");
    });
    server.new_task_queue=[] { return new httplib::ThreadPool(4,4,16); };
    server.set_payload_max_length(4ULL<<20);
    server.set_read_timeout(30); server.set_write_timeout(30); server.set_keep_alive_timeout(5);
    server.set_default_headers({{"Server","Wllm"},{"Cache-Control","no-store"}});
    server.set_pre_routing_handler([&](const httplib::Request & request,httplib::Response & response) {
        if(!key.empty()) {
            const auto supplied=request.get_header_value("Authorization"),expected="Bearer "+key;
            unsigned difference=unsigned(supplied.size()!=expected.size());
            for(size_t i=0;i<expected.size();++i) difference|=static_cast<unsigned char>(expected[i]) ^ (i<supplied.size()?static_cast<unsigned char>(supplied[i]):0u);
            if(difference) {
                response.set_header("WWW-Authenticate","Bearer");
                service_reply(response,service_error("Invalid or missing Bearer token","authentication_error","invalid_api_key"),401);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_error_handler([](const httplib::Request &,httplib::Response & response) {
        if(response.body.empty()) service_reply(response,service_error(response.status==404?"Endpoint is not enabled or does not exist":"HTTP request rejected"),response.status);
    });
    server.set_exception_handler([](const httplib::Request &,httplib::Response & response,std::exception_ptr error) {
        std::string message="Request processing failed";
        try { if(error) std::rethrow_exception(error); } catch(const std::exception & e) { message=e.what(); }
        service_reply(response,service_error(message,"server_error","internal_error"),500);
    });
    if(!server.bind_to_port(options.host,options.port)) throw std::runtime_error("Cannot bind service address/port; check the address and whether the port is occupied");
    State probe; probe.config=base; probe.config.timeout_s=0; preflight(probe);
    const auto identity=probe.identity;
    ModelPtr resident(nullptr,llama_model_free);
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP; mp.use_extra_bufts=false;
    mp.progress_callback=load_progress; mp.progress_callback_user_data=&probe;
    resident.reset(llama_model_load_from_file(base.model.c_str(),mp));
    if(!resident) throw std::runtime_error("Cannot load service model");
    std::atomic<bool> busy{false};
    const bool openai=options.type!="systemone",systemone=options.type!="openai";
    const auto health=[&](const httplib::Request &,httplib::Response & response) {
        service_reply(response,{{"status","ok"},{"service",options.type},{"model",options.alias},{"architecture",identity.at("architecture")},
            {"context",base.context},{"busy",busy.load()},{"parallel_inference",1},{"authentication",!key.empty()}});
    };
    server.Get("/health",health);
    server.Get("/v1/models",[&](const httplib::Request &,httplib::Response & response) {
        service_reply(response,{{"object","list"},{"data",json::array({{{"id",options.alias},{"object","model"},{"created",0},{"owned_by","wllm"}}})}});
    });
    const auto parse_body=[](const httplib::Request & request) {
        if(request.get_header_value("Content-Type").rfind("application/json",0)!=0) throw std::runtime_error("Content-Type must be application/json");
        return json::parse(request.body);
    };
    const auto acquire=[&](httplib::Response & response) -> std::shared_ptr<ServiceLease> {
        bool expected=false;
        if(!busy.compare_exchange_strong(expected,true)) {
            response.set_header("Retry-After","1");
            service_reply(response,service_error("Model is busy; retry after the active request completes","rate_limit_error","model_busy"),429);
            return {};
        }
        return std::make_shared<ServiceLease>(busy);
    };
    if(openai) for(const bool chat:{true,false}) {
        server.Post(chat?"/v1/chat/completions":"/v1/completions",[&,chat](const httplib::Request & request,httplib::Response & response) {
            ServiceGeneration input;
            try {
                const auto body=parse_body(request);
                input=ServiceGeneration::parse(base,body,chat);
                if(body.at("model")!=options.alias && body.at("model")!=identity.at("name")) {
                    service_reply(response,service_error("Unknown model; use an id from GET /v1/models","invalid_request_error","model_not_found"),404); return;
                }
                if(chat && !identity.at("chat_template_available").get<bool>()) throw std::runtime_error("Selected model needs a chat template; use /v1/completions or start with --chat-template-file");
            } catch(const std::exception & e) { service_reply(response,service_error(e.what()),400); return; }
            auto lease=acquire(response); if(!lease) return;
            const auto closed=request.is_connection_closed;
            const auto created=std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            static std::atomic<uint64_t> sequence{0};
            const auto id=std::string(chat?"chatcmpl-":"cmpl-")+std::to_string(GetCurrentProcessId())+"-"+std::to_string(++sequence);
            // Own the request data and lease through the content-provider lifetime.
            const auto execute=[&,input,closed,id,created,lease](httplib::DataSink * sink,json & result) {
                const auto chunk=[&](const json & choices,const json & usage=nullptr) {
                    json event={{"id",id},{"object",input.chat?"chat.completion.chunk":"text_completion"},{"created",created},{"model",options.alias},{"choices",choices}};
                    if(!usage.is_null()) event["usage"]=usage;
                    const auto data="data: "+service_json(event)+"\n\n";
                    return sink->write(data.data(),data.size());
                };
                if(sink && input.chat && !chunk(json::array({{{"index",0},{"delta",{{"role","assistant"}}},{"finish_reason",nullptr}}}))) return 499;
                std::string output;
                ServiceText text(input.stops,[&](const std::string & piece) {
                    if(sink) {
                        json choice={{"index",0},{"finish_reason",nullptr}};
                        if(input.chat) choice["delta"]={{"content",piece}};
                        else { choice["text"]=piece; choice["logprobs"]=nullptr; }
                        return chunk(json::array({choice}));
                    }
                    output+=piece; return true;
                });
                RequestIO io; io.cancel=closed; io.output=[&](const std::string & piece){return text.write(piece);}; io.stopped=[&]{return text.stopped();};
                for(const auto & message:input.messages) { common_chat_msg msg; msg.role=message.first; msg.content=message.second; io.messages.push_back(std::move(msg)); }
                json presentation;
                const int code=execute_request(input.config,input.prompt,"","","","",true,false,false,false,&resident,nullptr,false,nullptr,nullptr,&presentation,&io);
                const auto & metrics=presentation.at("metrics");
                const auto error=metrics.value("error",std::string());
                const int status=service_status(code,error);
                if(status!=200) { result=service_error(error.empty()?"Inference cancelled or timed out":error,"server_error",code==124?"timeout":"inference_error"); return status; }
                if(!text.finish()) return 499;
                const auto reason=metrics.at("end_reason").get<std::string>();
                const auto finish=reason=="max_tokens" || reason=="context_limit"?"length":"stop";
                const json usage={{"prompt_tokens",metrics.at("input_tokens")},{"completion_tokens",metrics.at("output_tokens")},
                    {"total_tokens",metrics.at("input_tokens").get<uint64_t>()+metrics.at("output_tokens").get<uint64_t>()}};
                if(sink) {
                    json choice={{"index",0},{"finish_reason",finish}};
                    if(input.chat) choice["delta"]=json::object(); else { choice["text"]=""; choice["logprobs"]=nullptr; }
                    if(!chunk(json::array({choice}))) return 499;
                    if(input.include_usage && !chunk(json::array(),usage)) return 499;
                    const std::string done="data: [DONE]\n\n";
                    return sink->write(done.data(),done.size())?200:499;
                }
                json choice={{"index",0},{"finish_reason",finish},{"logprobs",nullptr}};
                if(input.chat) choice["message"]={{"role","assistant"},{"content",output}}; else choice["text"]=output;
                result={{"id",id},{"object",input.chat?"chat.completion":"text_completion"},{"created",created},{"model",options.alias},{"choices",json::array({choice})},{"usage",usage}};
                return 200;
            };
            if(input.stream) {
                response.set_header("X-Accel-Buffering","no");
                response.set_chunked_content_provider("text/event-stream; charset=utf-8",[execute](size_t,httplib::DataSink & sink) {
                    json result;
                    try {
                        const int status=execute(&sink,result);
                        if(status!=200 && !result.is_null()) {
                            const auto event="data: "+service_json(result)+"\n\ndata: [DONE]\n\n";
                            if(!sink.write(event.data(),event.size())) return false;
                        }
                    } catch(const std::exception & e) {
                        const auto event="data: "+service_json(service_error(e.what(),"server_error","inference_error"))+"\n\ndata: [DONE]\n\n";
                        if(!sink.write(event.data(),event.size())) return false;
                    }
                    sink.done(); return true;
                });
            } else {
                json result;
                try { const int status=execute(nullptr,result); service_reply(response,result,status); }
                catch(const std::exception & e) { service_reply(response,service_error(e.what(),"server_error","inference_error"),500); }
            }
        });
    }
    if(systemone) {
        const auto decide=[&](const httplib::Request & request,httplib::Response & response) {
            json body;
            try { body=parse_body(request); SystemOneRequest::parse(body); }
            catch(const std::exception & e) { service_reply(response,service_error(e.what()),400); return; }
            const auto lease=acquire(response); if(!lease) return;
            RequestIO io; io.cancel=request.is_connection_closed; io.output=[](const std::string &){return true;};
            auto effective=base; effective.inference_mode="system_one";
            json result;
            const int code=execute_system_one(effective,body.dump(),"",true,false,false,false,resident,nullptr,&result,&io,false);
            result["_wllm"].erase("model_path");
            const auto error=result.contains("error")?result["error"].value("message",std::string()):std::string();
            service_reply(response,result,service_status(code,error));
        };
        server.Post("/v1/systemone",decide);
    }
    std::cerr << "[Wllm] HTTP " << options.type << " http://" << options.host << ':' << options.port
              << " model=" << options.alias << " context=" << base.context << " auth=" << (!key.empty()?"Bearer":"off") << '\n';
    std::atomic<bool> finished{false};
    std::thread shutdown([&] {
        while(!finished && !interrupted) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if(interrupted) server.stop();
    });
    const bool ok=server.listen_after_bind();
    finished=true; shutdown.join();
    return ok || interrupted?0:1;
}
