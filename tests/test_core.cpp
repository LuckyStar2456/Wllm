#include "core.h"
#include "candidate_score.h"
#include "system_one.h"
#include "prefix_cache.h"
#include <limits>
#include <iostream>
#include <functional>
using namespace wllm;
static int checks=0;
static void require(bool ok,const char * name) { ++checks; if(!ok) throw std::runtime_error(name); }
static bool throws(const std::function<void()> & fn) { try { fn(); return false; } catch(...) { return true; } }
int main() {
    try {
        auto c=Config::parse(json::object());
        require(c.context==16384,"16K default");
        require(c.prefix_cache_mib==0,"prefix cache disabled by default");
        require(Config::parse({{"prefix_cache_mib",128}}).prefix_cache_mib==128,"explicit bounded cache");
        require(throws([]{Config::parse({{"prefix_cache_mib",-1}});}),"negative cache rejected");
        require(throws([]{Config::parse({{"prefix_cache_mib",4097}});}),"cache budget bounded");
        require(throws([]{Config::parse({{"prefix_cache_mib",1.5}});}),"fractional cache rejected");
        require(throws([]{Config::parse({{"prefix_cache_mib",true}});}),"boolean cache rejected");
        require(throws([]{Config::parse({{"prefix_cache_mib",64},{"gpu_layers",1}});}),"GPU state cache not implicitly enabled");
        require(throws([]{Config::parse({{"prefix_cache_mib",64},{"cpu_weight_repacking",true}});}),"unaccepted layout cannot use cache");
        PrefixCache pc; pc.key="one"; pc.tokens={1,2,3,4}; pc.state={5,6};
        require(pc.match("one",{1,2,3,4,9},2)==4,"exact token prefix hit");
        require(pc.match("one",{1,2,8,4,9},2)==0,"changed prefix misses");
        require(pc.match("other",{1,2,3,4,9},2)==0,"changed identity misses");
        require(pc.match("one",{1,2,3,4},2)==0,"suffix must be recomputed");
        require(pc.match("one",{1},2)==0,"shorter prompt cannot restore longer state");
        require(pc.match("one",{1,2,3,4,9},3)==0,"batch boundaries preserved");
        require(PrefixCache::checkpoint(128,128)==0 && PrefixCache::checkpoint(129,128)==128 &&
            PrefixCache::checkpoint(256,128)==128 && PrefixCache::checkpoint(257,128)==256,"last full batch left for fresh logits");
        pc.clear(); require(pc.bytes()==0 && pc.match("one",{1,2,3,4,9},2)==0,"cache clear releases identity and state");
        require(Config::parse({{"context",4096}}).context==4096,"explicit context override preserved");
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
        require(throws([]{check_context(16384,16384);}),"no silent truncation at 16K");
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
        m.input_reused=128; m.input_total=228;
        require(m.snapshot(5)["input_processed"]==100 && m.snapshot(5)["input_reused"]==128 &&
            m.snapshot(5)["prefill_tps"]==100.0,"cached tokens are not counted as new throughput");
        m.ended=4; m.reason="cancelled";
        require(m.snapshot(9)["elapsed_s"]==4.0,"stopped clock");
        require(m.snapshot(9)["output_tokens"]==4,"partial output retained");
        require(m.snapshot(9)["working_set_bytes"].is_null(),"unknown memory not zero");
        require(m.snapshot(9)["system_commit_total_bytes"].is_null(),"unknown system commit not zero");
        require(m.snapshot(9)["process_io_read_bytes"].is_null(),"unknown process IO not zero");
        Metrics next;
        require(next.snapshot(0)["output_tokens"]==0,"request reset");
        require(c.inference_mode=="generate" && c.dump()["effective_thinking"]==true,"normal generation stays the default");
        const auto score_config=Config::parse({{"inference_mode","candidate_score"}});
        require(score_config.thinking && !score_config.dump()["effective_thinking"].get<bool>() &&
            !score_config.dump()["sampling_applied"].get<bool>(),"explicit score mode preserves requested settings and records effective behavior");
        require(throws([]{Config::parse({{"inference_mode","score"}});}),"unknown inference mode rejected");
        require(throws([]{Config::parse({{"inference_mode","candidate_score"},{"gpu_layers",2}});}),"GPU numerical rejection cannot be bypassed");
        require(throws([]{Config::parse({{"inference_mode","candidate_score"},{"cpu_weight_repacking",true}});}),"score mode rejects unaccepted repacking");
        check_context(16384,16384,false);
        require(throws([]{check_context(16385,16384,false);}),"score input respects exact context without truncation");
        const json rows=json::array({{{"id","yes"},{"label","A"},{"description","yes"}},
            {{"id","unknown"},{"label","B"},{"description","unknown"}}});
        auto spec=json({{"candidates",rows},{"unknown_id","unknown"}});
        auto task=CandidateTask::parse(spec);
        require(throws([&]{CandidateTask::parse({{"candidates",rows}});}),"closed set must be explicit");
        require(CandidateTask::parse({{"candidates",rows},{"exhaustive",true}}).exhaustive,"explicit exhaustive set");
        auto bad=spec; bad["candidates"][1]["id"]="yes";
        require(throws([&]{CandidateTask::parse(bad);}),"duplicate id rejected");
        bad=spec; bad["candidates"][1]["label"]="A";
        require(throws([&]{CandidateTask::parse(bad);}),"duplicate label rejected");
        bad=spec; bad["candidates"][1]["label"]=" B";
        require(throws([&]{CandidateTask::parse(bad);}),"label whitespace rejected");
        bad=spec; bad["unknown_id"]="missing";
        require(throws([&]{CandidateTask::parse(bad);}),"invalid unknown id rejected");
        bad=spec; bad["fallback"]="reasoning";
        require(throws([&]{CandidateTask::parse(bad);}),"unimplemented fallback cannot silently enable");
        const float logits[]={10000.0f,9999.0f,20000.0f};
        auto decision=select_candidate(task,{0,1},logits,3);
        require(decision["selected_candidate_id"]=="yes" && decision["logit_margin"]==1.0,"raw-logit selection independent of other vocabulary winners");
        require(std::abs(decision["scores"][0]["candidate_normalized_score"].get<double>()-1.0/(1+std::exp(-1.0)))<1e-12,"stable candidate-only normalization at extreme logits");
        const float tied[]={1.0f,1.0f};
        decision=select_candidate(task,{0,1},tied,2);
        require(decision["tie_for_first"] && decision["top_candidate_id"]=="yes","ties explicitly use stable input order");
        decision=select_candidate(task,{1,0},logits,3);
        require(decision["status"]=="abstained" && decision["selected_candidate_id"].is_null() && decision["top_candidate_id"]=="unknown","unknown winner is abstention");
        require(throws([&]{select_candidate(task,{0,0},logits,3);}),"duplicate token mapping rejected");
        require(throws([&]{select_candidate(task,{0,3},logits,3);}),"invalid vocabulary index rejected");
        const float invalid[]={1,2,std::numeric_limits<float>::quiet_NaN()};
        require(throws([&]{select_candidate(task,{0,1},invalid,3);}),"invalid unselected logits cannot be concealed");
        Metrics scored; scored.decision_start=2; scored.decision_ready=2.1;
        require(scored.snapshot(3)["output_tokens"]==0 && scored.snapshot(3)["decode_average_tps"].is_null() &&
            scored.snapshot(3)["decision_s"]==2.1,"decision metrics never fabricate generation throughput");
        require(Config::parse({{"inference_mode","system_one"}}).dump()["effective_thinking"]==false,"structured mode is explicitly non-thinking");
        require(throws([]{Config::parse({{"inference_mode","system_one"},{"gpu_layers",2}});}),"structured mode cannot bypass GPU rejection");
        const auto request_json=json::parse(R"({"model":"jev-latest","state":{"message":"hello"},"questions":{
            "decision_id":{"type":"choice","instructions":{"question":"route"},"criteria":{"one":null,"two":{"kind":"other"},"three":[]}},
            "boolean_id":{"type":"noul","instructions":"Is it true?","criteria":{"true":"yes"}},
            "score_id":{"type":"score","instructions":"Rate severity","criteria":["low","medium","high"]}}})");
        auto request=SystemOneRequest::parse(request_json);
        require(request.questions.size()==3 && request.requested_model=="jev-latest","mixed typed questions parsed without remote model substitution");
        require(request.questions[0].task.automatic_labels && request.questions[0].task.candidates[1].id=="two","choice ids preserved, labels internal");
        require(request.questions[0].system_prompt().find("decision_id")==std::string::npos &&
            request.prompt().find("boolean_id")==std::string::npos,"routing ids and sibling questions are not model input");
        auto alias=request_json; alias["ask"]=alias["questions"]; alias.erase("questions");
        require(throws([&]{SystemOneRequest::parse(alias);}),"ask is not an accepted request field");
        alias["questions"]=alias["ask"];
        require(throws([&]{SystemOneRequest::parse(alias);}),"ask is rejected even alongside questions");
        auto invalid_request=request_json; invalid_request["state"]=nullptr;
        require(throws([&]{SystemOneRequest::parse(invalid_request);}),"null state rejected");
        invalid_request=request_json; invalid_request["model"]="arbitrary-model";
        require(throws([&]{SystemOneRequest::parse(invalid_request);}),"unknown model does not silently change execution");
        invalid_request=request_json; invalid_request["assets"]={{"image",{{"type","image"}}}};
        require(throws([&]{SystemOneRequest::parse(invalid_request);}),"vision assets fail explicitly");
        invalid_request=request_json; invalid_request["questions"]["decision_id"]["type"]="box";
        require(throws([&]{SystemOneRequest::parse(invalid_request);}),"unsupported typed question rejected before batch execution");
        invalid_request=request_json; invalid_request["questions"]["score_id"]["criteria"]=json::array({"one"});
        require(throws([&]{SystemOneRequest::parse(invalid_request);}),"score requires at least two ordered levels");
        auto large_choice=request_json; auto & options=large_choice["questions"]["decision_id"]["criteria"];
        options=json::object();
        for (int i=0;i<255;++i) options[std::to_string(i)]="option";
        require(SystemOneRequest::parse(large_choice).questions[0].task.candidates.size()==255,"choice schema accepts 255 options");
        options["extra"]="option";
        require(throws([&]{SystemOneRequest::parse(large_choice);}),"choice schema rejects more than 255 options");
        const auto choice_rows=json::array({{{"id","one"},{"candidate_normalized_score",.6}},
            {{"id","two"},{"candidate_normalized_score",.3}},{{"id","three"},{"candidate_normalized_score",.1}}});
        const auto choice_answer=system_one_answer(request.questions[0],choice_rows);
        require(choice_answer["choice"]=="one" && std::abs(choice_answer["confidence"].get<double>()-.4)<1e-12,"published choice confidence formula");
        const auto boolean_answer=system_one_answer(request.questions[1],json::array({
            {{"id","false"},{"candidate_normalized_score",.25}},{{"id","true"},{"candidate_normalized_score",.75}}}));
        require(boolean_answer["noul"]==.75 && !boolean_answer.contains("confidence"),"noul returns true probability with no separate confidence");
        const auto score_answer=system_one_answer(request.questions[2],json::array({
            {{"id","0"},{"candidate_normalized_score",0.0}},{{"id","1"},{"candidate_normalized_score",.57}},
            {{"id","2"},{"candidate_normalized_score",.43}}}));
        require(std::abs(score_answer["score"].get<double>()-1.43)<1e-12 &&
            std::abs(score_answer["confidence"].get<double>()-.355)<1e-12,"score is expectation with published ordered-level confidence");
        require(score_answer["legend"]["2"]=="high","score legend uses original levels");
        auto invalid_rows=choice_rows; invalid_rows[0]["candidate_normalized_score"]=.8;
        require(throws([&]{system_one_answer(request.questions[0],invalid_rows);}),"invalid probability mass rejected");
        invalid_rows=choice_rows; invalid_rows[0]["id"]="wrong";
        require(throws([&]{system_one_answer(request.questions[0],invalid_rows);}),"probability ids cannot be misassigned");
        const auto constant=SystemOneRequest::parse(json::parse(R"({"state":"","questions":{"only":{"type":"choice","instructions":"choose","criteria":{"single":null}}}})"));
        require(!constant.prompt().empty() && system_one_answer(constant.questions[0],json::array({
            {{"id","single"},{"candidate_normalized_score",1.0}}}))["confidence"]==1.0,"one-option closed choice is deterministic");
        std::cout << checks << " checks passed\n";
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
