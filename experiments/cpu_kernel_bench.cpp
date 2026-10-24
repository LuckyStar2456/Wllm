#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <bcrypt.h>
#include <intrin.h>
#include "q4_pair.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
static constexpr size_t MiB = 1048576, reserve = 1024*MiB, cap = 256*MiB;
static size_t checks = 0;
static void require(bool ok, const char * message) { ++checks; if (!ok) throw std::runtime_error(message); }
static std::string environment(const char * name,const char * fallback) {
    char * raw=nullptr; size_t length=0;
    require(_dupenv_s(&raw,&length,name)==0,"Cannot read CPU environment");
    std::unique_ptr<char,decltype(&std::free)> value(raw,std::free);
    return raw?std::string(raw):std::string(fallback);
}
static void check_cpu() {
    int f[4]{}; __cpuid(f,1);
    constexpr uint32_t bits=(1u<<12)|(1u<<27)|(1u<<28)|(1u<<29); // FMA, OSXSAVE, AVX, F16C
    require((uint32_t(f[2])&bits)==bits,"CPU/OS FMA/AVX/F16C support required");
    require((_xgetbv(0)&6)==6,"OS AVX register state required");
    __cpuidex(f,7,0); require((uint32_t(f[1])&(1u<<5))!=0,"CPU AVX2 support required");
}
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static void admit(size_t allocation) {
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m);
    PERFORMANCE_INFORMATION p{}; p.cb=sizeof(p);
    require(GlobalMemoryStatusEx(&m) && GetPerformanceInfo(&p,sizeof(p)),"Cannot query resource headroom");
    require(allocation <= cap,"Operator allocation exceeds 256MiB cap");
    require(m.ullAvailPhys >= allocation+reserve,"Operator physical headroom below allocation plus 1GiB reserve");
    require(uint64_t(p.CommitLimit-p.CommitTotal)*p.PageSize >= allocation+reserve,"Operator commit headroom insufficient");
}
static json memory() {
    PROCESS_MEMORY_COUNTERS_EX p{}; p.cb=sizeof(p);
    require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&p),sizeof(p)),"Cannot query process memory");
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m); require(GlobalMemoryStatusEx(&m),"Cannot query physical memory");
    return {{"peak_working_set_bytes",p.PeakWorkingSetSize},{"peak_private_bytes",p.PeakPagefileUsage},
        {"physical_available_bytes",m.ullAvailPhys}};
}
static std::string sha256(const void * data,size_t size) {
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    require(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider");
    // BCryptCreateHash allocates its state when the object buffer is null on supported Windows.
    const auto created=BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0);
    if (created<0) { BCryptCloseAlgorithmProvider(alg,0); throw std::runtime_error("SHA256 initialization"); }
    std::array<unsigned char,32> digest{};
    const bool ok=BCryptHashData(hash,static_cast<PUCHAR>(const_cast<void *>(data)),ULONG(size),0)>=0 &&
        BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0)>=0;
    BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(alg,0); require(ok,"SHA256 digest");
    const char * digits="0123456789abcdef"; std::string result;
    for (auto b:digest) { result+=digits[b>>4]; result+=digits[b&15]; } return result;
}

// Match the pinned upstream block-partitioned Q8_K activation conversion.
static void convert(ggml_tensor * dst,int ith,int nth,void *) {
    const auto * src=dst->src[0]; const int64_t k=src->ne[0], blocks=k/256;
    const int64_t first=(ith*blocks)/nth, last=((ith+1)*blocks)/nth;
    const size_t block=ggml_type_size(GGML_TYPE_Q8_K);
    for (int64_t col=0;col<src->ne[1];++col)
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(
            reinterpret_cast<const float *>(static_cast<const char *>(src->data)+col*src->nb[1])+first*256,
            static_cast<char *>(dst->data)+col*dst->nb[1]+first*block,(last-first)*256);
}
static void multiply(ggml_tensor * dst,int ith,int nth,void * userdata) {
    const auto * a=dst->src[0]; const auto * b=dst->src[1];
    const bool paired=userdata!=nullptr;
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;
    const int64_t rows=a->ne[1], columns=b->ne[1];
    const int64_t first=rows*ith/nth,last=rows*(ith+1)/nth;
    // Keep the same small row tile in cache across activation pairs. Scanning
    // the entire weight matrix per pair lost locality in the first screening.
    for (int64_t tile=first;tile<last;tile+=16) for (int64_t col=0;col<columns;col+=2) {
        const auto * y=static_cast<const char *>(b->data)+col*b->nb[1];
        auto * out=reinterpret_cast<float *>(static_cast<char *>(dst->data)+col*dst->nb[1]);
        auto * out1=out+rows;
        for (int64_t row=tile;row<std::min(tile+16,last);++row) {
            const auto * x=static_cast<const char *>(a->data)+row*a->nb[1];
            if (paired && col+1<columns)
                wllm::experiment::q4_pair(int(a->ne[0]),out+row,out1+row,x,y,y+b->nb[1]);
            else {
                dot(int(a->ne[0]),out+row,0,x,0,y,0,1);
                if (col+1<columns) dot(int(a->ne[0]),out1+row,0,x,0,y+b->nb[1],0,1);
            }
        }
    }
}
struct Trial {
    std::string name;
    ggml_cgraph * graph=nullptr;
    ggml_tensor * output=nullptr;
    ggml_cplan plan{};
    int threads=4;
    std::vector<double> timings;
};

static json evaluate(const std::vector<uint8_t> & weights,int64_t k,int64_t rows,int columns,int threads,int repeats,int pattern,bool sweep=false) {
    admit(cap); // Fixed upper bound, checked before each case.
    constexpr size_t metadata=2*MiB;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({metadata,nullptr,true}),ggml_free);
    require(bool(ctx),"Operator metadata allocation");
    auto * a=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_K,k,rows);
    auto * b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,k,columns);
    auto * upstream=ggml_mul_mat(ctx.get(),a,b);
    ggml_tensor * input_args[]={b};
    auto * quant=ggml_custom_4d(ctx.get(),GGML_TYPE_Q8_K,k,columns,1,1,input_args,1,convert,GGML_N_TASKS_MAX,nullptr);
    ggml_tensor * args[]={a,quant};
    auto * scalar=ggml_custom_4d(ctx.get(),GGML_TYPE_F32,rows,columns,1,1,args,2,multiply,GGML_N_TASKS_MAX,nullptr);
    int paired_marker=1;
    auto * paired=ggml_custom_4d(ctx.get(),GGML_TYPE_F32,rows,columns,1,1,args,2,multiply,GGML_N_TASKS_MAX,&paired_marker);
    // The pinned upstream already has a tiled Q4_K path from 8 columns onward.
    // It has a different reduction order: replacing it with vec_dot is not an
    // unchanged-arithmetic optimization. Retain it, and retain single-column decode.
    const bool eligible=columns>=2 && columns<8;
    std::vector<Trial> trials(sweep?6:3);
    trials[0].name="upstream"; trials[0].output=upstream;
    trials[1].name="same_schedule_scalar"; trials[1].output=eligible?scalar:upstream;
    trials[2].name="pair_shared_unpack"; trials[2].output=eligible?paired:upstream;
    for (auto & trial:trials) trial.threads=threads;
    if (sweep) {
        for (int i=0;i<3;++i) {
            const int count=std::array<int,3>{1,2,8}[i];
            trials[size_t(i+3)].name="upstream_t"+std::to_string(count);
            trials[size_t(i+3)].output=upstream; trials[size_t(i+3)].threads=count;
        }
    }
    const int pool_capacity=sweep?8:threads;
    auto params=ggml_threadpool_params_default(pool_capacity);
    std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&params),ggml_threadpool_free);
    require(bool(pool),"Operator thread pool allocation");
    size_t work_size=0;
    for (auto & trial:trials) {
        trial.graph=ggml_new_graph_custom(ctx.get(),32,false);
        ggml_build_forward_expand(trial.graph,trial.output);
        trial.plan=ggml_graph_plan(trial.graph,trial.threads,pool.get());
        work_size=std::max(work_size,trial.plan.work_size);
    }
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),ggml_backend_cpu_buffer_type()),ggml_backend_buffer_free);
    require(bool(buffer),"Operator tensor allocation");
    const size_t output_bytes=ggml_nbytes(upstream);
    const size_t estimated=metadata+ggml_backend_buffer_get_size(buffer.get())+work_size+output_bytes+weights.size();
    require(estimated<=cap,"Combined operator allocation exceeds cap");
    require(weights.size()==ggml_nbytes(a),"Selected weight shape/size mismatch");
    std::memcpy(a->data,weights.data(),weights.size());
    // Deterministic synthetic activations; never described as captured model activations.
    auto * values=static_cast<float *>(b->data);
    uint32_t rng=0x12345678u+uint32_t(pattern);
    for (size_t i=0;i<size_t(ggml_nelements(b));++i) {
        rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
        values[i]=pattern==1?0.0f:(int(rng%65537)-32768)/8192.0f;
        if (pattern==2) values[i]*=(i%2?1e-3f:100.0f);
    }
    const auto activation_hash=sha256(b->data,ggml_nbytes(b));
    std::vector<uint8_t> work(work_size),reference(output_bytes);
    for (auto & trial:trials) trial.plan.work_data=work.data();
    auto execute=[&](Trial & trial) {
        require(ggml_graph_compute(trial.graph,&trial.plan)==GGML_STATUS_SUCCESS,"Operator compute failed");
    };
    execute(trials[0]); std::memcpy(reference.data(),upstream->data,output_bytes);
    for (size_t i=0;i<output_bytes/sizeof(float);++i)
        require(std::isfinite(static_cast<const float *>(upstream->data)[i]),"Non-finite reference output");
    // Zero tolerance: no relaxed numeric threshold is introduced for this candidate.
    auto compare=[&](Trial & trial) {
        if (std::memcmp(reference.data(),trial.output->data,output_bytes)!=0)
            throw std::runtime_error("Byte mismatch: "+trial.name+", k="+std::to_string(k)+", columns="+std::to_string(columns)+", threads="+std::to_string(threads));
        ++checks;
    };
    for (auto & trial:trials) for (int warmup=0;warmup<2;++warmup) { execute(trial); compare(trial); }
    json rounds=json::array();
    const int iterations=repeats>1 && columns<8?16:1;
    for (int round=0;round<repeats;++round) {
        admit(0);
        json row={{"round",round+1},{"order",json::array()},{"seconds",json::object()}};
        for (int position=0;position<int(trials.size());++position) {
            const int index=round%2?int(trials.size())-1-position:position;
            auto & trial=trials[size_t(index)];
            const auto started=Clock::now();
            for (int iteration=0;iteration<iterations;++iteration) execute(trial);
            const double elapsed=seconds(started)/iterations;
            // Equality and resource queries stay outside the timed interval.
            compare(trial); trial.timings.push_back(elapsed);
            row["order"].push_back(trial.name); row["seconds"][trial.name]=elapsed;
        }
        rounds.push_back(std::move(row));
    }
    json summaries=json::object();
    for (auto & trial:trials) {
        auto sorted=trial.timings; std::sort(sorted.begin(),sorted.end());
        const bool no_op_control=!eligible && (trial.name=="same_schedule_scalar" || trial.name=="pair_shared_unpack");
        summaries[trial.name]={{"threads",trial.threads},{"kind",no_op_control?"unchanged_upstream_control":"operator_profile"},
            {"samples",sorted.size()},{"median_s",sorted[sorted.size()/2]},
            {"min_s",sorted.front()},{"max_s",sorted.back()},{"all_outputs_byte_equal",true}};
    }
    return {{"k",k},{"rows",rows},{"activation_columns",columns},{"threads",threads},{"pattern",pattern},
        {"pair_dispatch",eligible?"custom_pair":"unchanged_upstream"},
        {"threadpool_capacity",pool_capacity},{"threadpool_poll",params.poll},
        {"activation_sha256",activation_hash},{"output_sha256",sha256(reference.data(),reference.size())},
        {"planned_payload_bytes",estimated},{"workspace_bytes",work_size},{"memory",memory()},
        {"warmups_per_profile",2},{"invocations_per_sample",iterations},
        {"equality_scope","preflight/warmup outputs and final output of each timed sample"},
        {"rounds",rounds},{"profiles",summaries}};
}

int main(int argc,char ** argv) {
    json report={{"schema_version",1},{"status","started"},{"llama_commit",WLLM_LLAMA_COMMIT},
        {"cpu_build",WLLM_CPU_TARGET},{"evidence","isolated operator experiment; no model inference or context allocation"},
        {"candidate_integrated_into_engine",false},{"activation_source","deterministic synthetic F32, pinned upstream Q8_K conversion"},
        {"arithmetic","original Q4_K weights; same Q8_K activation conversion, lane accumulation and F32 reduction order"},
        {"acceptance","all finite outputs byte equal; no tolerance relaxation"},
        {"timing_scope","warm operator graph execution including activation quantization; excludes weight IO, plan/pool/allocation and equality checks"},
        {"resource_policy",{{"allocation_cap_bytes",cap},{"system_reserve_bytes",reserve}}},
        {"cases",json::array()}};
    fs::path output;
    int code=0;
    try {
        bool self=false,list=false,sweep=false; std::string model,tensor;
        int columns=128,threads=4,repeats=9;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--self-test") self=true;
            else if (arg=="--list") list=true;
            else if (arg=="--thread-sweep") sweep=true;
            else {
                require(i+1<argc,"Missing argument value"); const std::string val=argv[++i];
                if (arg=="--model") model=val;
                else if (arg=="--tensor") tensor=val;
                else if (arg=="--output") output=fs::u8path(val);
                else if (arg=="--columns" || arg=="--threads" || arg=="--repeats") {
                    size_t consumed=0; const int integer=std::stoi(val,&consumed);
                    require(consumed==val.size(),"Expected an integer control");
                    if (arg=="--columns") columns=integer;
                    else if (arg=="--threads") threads=integer;
                    else repeats=integer;
                }
                else throw std::runtime_error("Unknown argument: "+arg);
            }
        }
        require(!output.empty() && !fs::exists(output),"Provide a new --output JSON path; never overwrite results");
        require(threads>=1 && threads<=8 && columns>=1 && columns<=129 && repeats>=1 && repeats<=99,"Out-of-range experiment controls");
        require(!sweep || threads==4,"Thread sweep uses the unchanged 4-thread baseline");
        require(environment("GGML_CPU_TILED_MM","1")=="1" && environment("GGML_CPU_TILED_MM_FORCE","0")=="0",
            "Non-default tiled-kernel environment is not an accepted baseline");
        require(std::string(WLLM_CPU_TARGET)=="avx2","Experiment requires the pinned AVX2 baseline build");
        check_cpu(); ggml_cpu_init();
        require(ggml_cpu_has_avx2() && ggml_cpu_has_fma() && ggml_cpu_has_f16c(),"AVX2/FMA/F16C required");
        admit(cap); report["memory_before"]=memory();
        if (self) {
            require(model.empty() && !list && tensor.empty(),"Self-test must not load model weights");
            for (int64_t k:{256,512,4096}) for (int pattern:{0,1,2}) {
                const int64_t rows=17;
                std::vector<float> values(size_t(k*rows));
                for (size_t i=0;i<values.size();++i) values[i]=float(int((i*37)%1021)-510)/128.0f;
                std::vector<uint8_t> weights(ggml_row_size(GGML_TYPE_Q4_K,k)*rows);
                ggml_get_type_traits(GGML_TYPE_Q4_K)->from_float_ref(values.data(),weights.data(),k*rows);
                for (int n:{1,2,3,7,8,16,129}) for (int t:{1,4})
                    report["cases"].push_back(evaluate(weights,k,rows,n,t,1,pattern));
            }
        } else {
            require(!model.empty(),"--model is required for actual tensor experiments");
            gguf_init_params params{}; params.no_alloc=true;
            std::unique_ptr<gguf_context,decltype(&gguf_free)> meta(gguf_init_from_file(model.c_str(),params),gguf_free);
            require(bool(meta),"Cannot read GGUF metadata");
            report["model_file_bytes"]=fs::file_size(fs::u8path(model));
            if (list) {
                report["q4_matrices"]=json::array();
                for (int64_t i=0;i<gguf_get_n_tensors(meta.get());++i) {
                    const auto * ne=gguf_get_tensor_ne(meta.get(),i);
                    if (gguf_get_tensor_type(meta.get(),i)==GGML_TYPE_Q4_K && ne[2]==1 && ne[3]==1)
                        report["q4_matrices"].push_back({{"name",gguf_get_tensor_name(meta.get(),i)},
                            {"shape",{ne[0],ne[1]}},{"bytes",gguf_get_tensor_size(meta.get(),i)}});
                }
            } else {
                const auto id=gguf_find_tensor(meta.get(),tensor.c_str()); require(id>=0,"Unknown GGUF tensor");
                const auto * ne=gguf_get_tensor_ne(meta.get(),id); const size_t size=gguf_get_tensor_size(meta.get(),id);
                require(gguf_get_tensor_type(meta.get(),id)==GGML_TYPE_Q4_K && ne[2]==1 && ne[3]==1 && ne[0]%256==0 && ne[0]<=65536,
                    "Experiment supports contiguous 2D Q4_K tensors only");
                require(size<=64*MiB,"Selected tensor exceeds 64MiB weight cap");
                // Bound all tensor/output buffers before allocation, including duplicated weights.
                const size_t estimate=2*size+size_t(ne[0])*columns*8+size_t(ne[1])*columns*16+8*MiB;
                require(estimate<=cap,"Selected case exceeds combined memory cap"); admit(estimate);
                const size_t offset=gguf_get_data_offset(meta.get())+gguf_get_tensor_offset(meta.get(),id);
                require(offset<=fs::file_size(fs::u8path(model)) && size<=fs::file_size(fs::u8path(model))-offset,"Truncated tensor data");
                std::vector<uint8_t> weights(size); std::ifstream file(fs::u8path(model),std::ios::binary);
                file.seekg(std::streamoff(offset)); file.read(reinterpret_cast<char *>(weights.data()),std::streamsize(size));
                require(bool(file),"Cannot read selected GGUF tensor");
                report["tensor"]={{"name",tensor},{"type","q4_K"},{"bytes",size},{"sha256",sha256(weights.data(),size)}};
                report["cases"].push_back(evaluate(weights,ne[0],ne[1],columns,threads,repeats,0,sweep));
            }
        }
        report["status"]="passed";
    } catch (const std::exception & e) { report["status"]="failed"; report["error"]=e.what(); code=1; }
    report["checks"]=checks;
    if (!output.empty() && !fs::exists(output)) {
        std::ofstream file(output,std::ios::binary); file<<report.dump(2)<<'\n';
        if (!file) { std::cerr<<"Cannot write experiment report\n"; return 1; }
    }
    std::cout<<"CPU operator experiment: "<<report["status"]<<", cases="<<report["cases"].size()<<'\n';
    if (code) std::cerr<<report.value("error",std::string())<<'\n';
    return code;
}
