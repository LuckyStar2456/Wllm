# System One 文本请求协议

本协议是 Wllm **Reflex（反射决策）** 模式的结构化入口。Reflex 与普通生成共用 Wllm 推理引擎和仓库，当前参数沿用 `--mode system_one`，底层候选文件入口沿用 `--mode candidate_score`。

更新：2026-10-22。按 JEV / TypeSafe 的 `state + questions` 请求和具名 `answers` 响应适配本地 CPU 候选打分。只接受 `questions`，不接受 `ask`。这是 JSON 请求/响应适配，当前没有 HTTP 服务，也不会调用远程 JEV。实现和本机有界测试见[本轮报告](../reports/system-one-validation.md)。

2026-10-23 文档补充：[覆盖清单及优先级](hardware-model-coverage.md)区分主线全品类模型目标与 Reflex 适用子集，Reflex 不要求兼容纯生成等全部模型，也不扩大本协议的当前能力。GPU、目标内其他读出及多模态理解需要独立适配/验收；未来模态输入另作版本化设计，不把当前 `state` 的 JSON 接受能力当作图像/音频/视频理解。当前字段、题型、CLI 及 CPU 限制保持原样。

## 运行

```powershell
./scripts/run.ps1 -Mode system_one -RequestFile configs/system-one.example.json
./scripts/run.ps1 -Mode system_one -Interactive
```

文件入口支持 UTF-8 多行 JSON；交互入口每行一个完整 JSON，空行、EOF 或 `/exit` 退出。也可将一个完整 JSON 文档通过 stdin 传给 `wllm.exe --config configs/local.json --mode system_one`。stdout 每个请求一行 JSON，诊断与指标走 stderr。交互中无效 JSON 返回错误后可以重试；取消和超时结束会话，释放模型。交互中发生过错误时最终退出码保留该错误。

## 请求

```json
{
  "model": "wllm-local",
  "state": {"user_request": "请把客厅的灯打开"},
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "根据用户请求选择动作。不明确时选择 clarify。",
      "criteria": {
        "turn_on": "打开灯",
        "turn_off": "关闭灯",
        "clarify": "向用户澄清需求"
      }
    }
  }
}
```

| 字段 | 本地支持与边界 |
|---|---|
| `state` | 必填，字符串、对象或数组；空字符串/对象/数组也保留为数据。不自动截断 |
| `questions` | 必填，问题名称到问题定义的对象，本地限制 1–64 题。名称只用于关联响应和记录，不进入模型提示词；其他题目也不进入该题上下文 |
| `instructions` | 每题必填，字符串、对象或数组 |
| `model` | 可省略，默认 `wllm-local`；也接受 `wllm-qwen3.5-q4_k_m`、`jev-latest`、`jev-1.13.0` 作为协议路由别名。均执行配置中已有的本地 Qwen3.5 Q4_K_M，不下载、不切换模型、不调用 JEV；未知别名报错 |
| `assets` | 仅允许省略、null 或空对象；非空图像资产和 `box` 题型明确拒绝，当前只做文本 |

顶层或问题定义中的未知字段报错。`state`、`instructions` 和描述里的结构化内容按 JSON 编码保留。各题共享同一份 state 数据，但分别建立请求上下文、KV/循环状态；模型权重驻留复用，问题逐题串行执行，没有跨题前缀缓存或并行推理。

| 题型 | `criteria` | `answers[问题名]` |
|---|---|---|
| `choice` | 1–255 项的对象；选项名对应字符串、对象、数组或 null。null 表示用选项名本身描述含义 | `type`、选项名 `choice`、具名 `probabilities`、`confidence` |
| `noul` | 可省略/null，或只含 `true`/`false` 描述的对象；描述为字符串、对象或数组，缺少的一侧使用真假语义描述 | `type`、`noul`（true 的归一化分数，0–1），没有额外 confidence |
| `score` | 2–10 个等级的有序数组；每级为字符串、对象或数组 | `type`、`score`（等级下标的期望）、按 `"0"` 等下标索引的 `probabilities`、保留原描述的 `legend`、`confidence` |

choice 是封闭候选选择，需要不确定选项时由调用方在 criteria 中显式提供；不会套用旧入口的 `unknown_id` 或自动转成拒答状态。单个选项的 choice 是常量映射，直接返回该选项、概率和 confidence 均为 1，usage 为 0，`_wllm` 明示未执行模型。

## 响应与计量

以下数值仅为结构示例：

```json
{
  "model": "wllm-qwen3.5-q4_k_m",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "turn_on",
      "probabilities": {"turn_on": 0.8, "turn_off": 0.1, "clarify": 0.1},
      "confidence": 0.7
    }
  },
  "usage": {"input_tokens": 180, "output_tokens": 0},
  "_wllm": {"status": "completed", "requested_model": "wllm-local"}
}
```

实际 `_wllm` 还包括本地模型路径、父请求 ID、逐题指标、模板版本、串行执行说明和父请求耗时。`model` 如实标识本地模型系列/精度，实际模型文件以 `_wllm.model_path` 和 JSONL 中的权重来源摘要为准，不把传入的 JEV 别名冒充成执行模型。

程序自动为选项分配内部普通单 token 标签，在实际完整答案前缀校验分词和字节边界。使用完整主干前向与词表输出，读取原始 logits，候选内 softmax；不创建采样器、不生成答案 token。选项名称/描述进入该题提示词，内部标签和选项顺序会影响分布，因此不能把本地分数等同于 JEV 分数。

`usage.input_tokens` 是各题实际完成的 prefill 批次 token 数之和，包含模板、state、指令和选项；重复输入按实际计算重复计数。`output_tokens` 为 0，JSON 的字符数不算模型输出 token。解码速度为 null；逐题记录输入吞吐、决策时延和权重复用。父耗时从解析前计到答案聚合完成，包含各题执行和请求状态清理，不含进程启动、最终记录/终端写出及模型释放；最终释放单列 `session_end`。被中止批次内部已做但未返回的计算无法准确计入 usage，响应中注明此边界。

confidence 使用[官方公开公式](https://docs.typesafe.ai/confidence)，作用于本地归一化分布 p：choice 为 `(max(p) - 1/n) / (1 - 1/n)`；score 为 `max(0, 1 - E[|等级 - 众数等级|] / MAD_uniform)`，其中 `MAD_uniform = mean(|i - (n-1)/2|)`。score 数值为 `sum(i*p_i)`。平局按输入顺序确定，score 使用第一个最大项。单选概率及 confidence 均不是经过校准的正确率。

完整请求先校验全部题目，再执行；任一题失败、超时或取消时 `answers` 清空，返回顶层 `error` 和失败题名，已有消耗和完成题诊断仍保留在 usage/`_wllm`，不把部分结果当整批成功。退出码成功 0、参数/执行/记录错误 1、超时 124、取消 130。`--timeout` 对整个父请求统一计时，所有题及常量映射共享截止时间；没有自动重试或思考回退。记录包含响应内容，应作为本地输入输出资料保管。

JSON 请求内的错误会返回结构化响应；请求文件无法读取、无效 CLI 参数等进入协议处理前的错误仅走 stderr、退出码为 1。

## 证据与后续

2026-10-22 补充与 LLM2Jev、SemIf、snapjudge 等的[协议和算法对照](reflex-research.md#4-概率confidence-与校准的兼容层次)。结构兼容、概率定义、confidence、usage 和质量分别核对：当前 Wllm 的 score confidence 保留有序等级公式，不跟随其他项目的同名字段改算法；usage 保留当前完成 prefill 批次累计口径，未来缓存研究另列逻辑/计算/复用输入。本轮不新增协议字段、配置开关或策略。

已实现：上述文本协议、内部标签、三种答案映射、独立请求状态、统一超时与结构化错误。默认仍为 generate；旧 `candidate_score + 候选文件 + prompt` 入口继续保留，见[底层模式设计](candidate-score-mode.md)。Q4_K_M、16K、F16 KV/F32 循环状态不变。

研究候选：共享 state 前缀复用、批处理/并行题目、任务校准与完整思考回退、HTTP 服务、视觉支持。本地最多 255 选项是解析/映射能力边界，不表示最大规模任务的质量和延迟均已验收。

性能预期：可省去自回归答案和思考生成；长 state、多题和多选项仍增加 prefill。当前每题独立完整处理 state，批次时间大致累加各题成本。没有 JEV 的并行执行延迟保证，没有正式加速倍数或通用质量等价结论。

格式依据核对日期 2026-10-22：[System One API](https://docs.typesafe.ai/api/system-one)、[choice](https://docs.typesafe.ai/primitives/choice)、[noul](https://docs.typesafe.ai/primitives/noul)、[score](https://docs.typesafe.ai/primitives/score)。本地 `jev_test/internal/eval/client.go` 与 `types.go` 的调用结构也为 state/questions/answers；未读取其凭据或发送远程请求。
