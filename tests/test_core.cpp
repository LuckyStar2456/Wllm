#include "core.h"
#include <iostream>
#include <functional>
using namespace wllm;
static int checks=0;
static void require(bool ok,const char * name) { ++checks; if(!ok) throw std::runtime_error(name); }
static bool throws(const std::function<void()> & fn) { try { fn(); return false; } catch(...) { return true; } }
int main() {
    try {
        auto c=Config::parse(json::object());
        require(c.context==4096,"4K default");
        require(c.gpu_layers==0 && c.dump()["backend"]=="CPU","CPU remains default");
        require(Config::parse({{"gpu_layers",2}}).dump()["backend"]=="Vulkan","explicit Vulkan placement recorded");
        require(throws([]{Config::parse({{"gpu_layers",-1}});}),"no implicit GPU layer count");
        require(throws([]{Config::parse({{"gpu_layers",2.5}});}),"integer GPU layer count");
        require(throws([]{Config::parse({{"gpu_layers",2},{"cpu_weight_repacking",true}});}),"no mixed unaccepted layout candidate");
        require(c.dump()["recurrent_state"]=="f32","recurrent precision is explicit and separate from KV");
        require(!c.cpu_weight_repacking,"weight repack stays opt-in");
        require(Config::parse({{"cpu_weight_repacking",true}}).dump()["cpu_weight_repacking"]==true,"repack recorded");
        require(throws([]{Config::parse({{"cpu_weight_repacking","on"}});}),"reject string repack config");
        require(Config::parse({{"threads",2}}).effective_threads_batch()==2,"batch threads inherit decode threads");
        require(Config::parse({{"threads",2},{"threads_batch",6}}).dump()["threads_batch"]==6,"explicit prefill threads recorded");
        require(throws([]{Config::parse({{"threads_batch",-1}});}),"negative batch threads rejected");
        require(throws([]{Config::parse({{"persistent_threadpool","false"}});}),"boolean strings rejected");
        require(!Config::parse({{"persistent_threadpool",false},{"prefill_last_only",false}}).prefill_last_only,"explicit optimization rollback");
        require(Config::parse({{"context",262144}}).context==262144,"explicit large context");
        require(throws([]{Config::parse({{"context",0}});}),"no auto context");
        require(throws([]{Config::parse({{"context",4096},{"batch",8192}});}),"batch budget");
        require(throws([]{Config::parse({{"contex",2048}});}),"typo rejection");
        require(throws([]{Config::parse({{"temperature",-1}});}),"bad sampling");
        require(throws([]{Config::parse({{"context",4096.5}});}),"reject fractional context");
        require(throws([]{Config::parse({{"seed",-1}});}),"reject unsigned wraparound");
        require(throws([]{check_context(4096,4096);}),"no silent truncation");
        Metrics m;
        require(!m.average_tps() && !m.window_tps(10),"empty output");
        m.token_times={2};
        require(!m.average_tps(),"one token no average");
        require(*m.window_tps(3)==0,"no new tokens");
        m.token_times={2,2.5,3,3.5};
        require(std::abs(*m.average_tps()-2)<1e-9,"decode excludes first token");
        require(std::abs(*m.window_tps(3.5)-2)<1e-9,"partial window");
        require(*m.window_tps(6)==0,"stalled window expires");
        m.prefill_start=1; m.prefill_end=2; m.input_total=100; m.input_done=100;
        require(m.snapshot(5)["prefill_tps"]==100.0,"prefill independent of whole request");
        m.ended=4; m.reason="cancelled";
        require(m.snapshot(9)["elapsed_s"]==4.0,"stopped clock");
        require(m.snapshot(9)["output_tokens"]==4,"partial output retained");
        require(m.snapshot(9)["working_set_bytes"].is_null(),"unknown memory not zero");
        require(m.snapshot(9)["system_commit_total_bytes"].is_null(),"unknown system commit not zero");
        require(m.snapshot(9)["process_io_read_bytes"].is_null(),"unknown process IO not zero");
        Metrics next;
        require(next.snapshot(0)["output_tokens"]==0,"request reset");
        std::cout << checks << " checks passed\n";
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
