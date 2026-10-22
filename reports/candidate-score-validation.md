# 候选打分快速决策：CPU 首版有界验收

日期：2026-10-22。按用户最新优先级实现 E3-07。结论为 **CPU 显式单 token 分支已实现并通过本轮有界检查**；不宣称通用任务质量等同完整思考，也没有稳定提速或倍数结论。完整数据摘要见[机器可读报告](candidate-score-validation.json)。

## 工件与范围

- 基于 `2ac9eb4` 开发，未改第三方源码、模型文件或本机默认配置。固定 llama.cpp 为 `f2918cabbffe8abf8e2a90c1c089085fa116c2cf`。
- 本机 Windows / i5-1135G7 / 16GB / Iris Xe，实际验证为 CPU AVX2；保持既有 Qwen3.5 Q4_K_M、16384 上下文、F16 注意力 KV 与 F32 循环状态。
- 最终验证 EXE SHA256：`5ab1d02b1dbaabb0ff8c36d50b86670ce91e3849aeaff0c9b3da6eaabe6fd90d`。新构建会使旧内存校准的工件绑定失效，不能继承旧构建的性能准入结论。
- 仅显式选择此模式才使用有效非思考模板。普通生成默认仍为思考开启和原采样设置；候选分支记录原设置及 `effective_thinking=false`、`sampling_applied=false`。

## 已实现行为

原生命令 `--mode candidate_score --candidates FILE` 和 PowerShell `-Mode candidate_score -CandidatesFile FILE` 已可用。候选文件提供至少两项唯一 ID/标签/描述，并以 `unknown_id` 指定不确定项，或显式声明 `exhaustive=true`。

首版核实实际非思考答案边界，验证“前缀+标签”的完整分词保持原前缀不变且只增加一个普通 token，检查标签解码字节和 token 唯一性；输入不自动截断。复用原 CPU 主干与完整词表投影，检查全词表有限值，再按原始 logits 选择。未创建采样器，不执行后续生成循环，结果在请求状态释放后以一行 JSON 输出，运行指标另记 stderr/JSONL。

unknown 获胜返回 `abstained`，`selected_candidate_id=null`；其 ID 保存在 `top_candidate_id`。候选内归一化分数不是校准正确率。平局明确标记，按输入顺序决胜。本轮不提供阈值校准、自动思考回退、多 token 标签/完整词语评分或跨任务自动分流。

GPU 放置和未验收的 CPU 重排在该模式下直接拒绝，原 GPU 数值失败状态保持；本轮没有验证其他 CPU 指令构建。启动前无效请求走 stderr/退出码 1；运行中错误、超时和取消均不返回有效选择，退出码分别为 1/124/130。

## 检查结果

| 检查 | 本轮结果与边界 |
|---|---|
| 构建及原生检查 | Windows Release AVX2 构建与 CTest 通过，原生共 58 项检查，覆盖配置/指标、数值稳定归一化、平局、不确定、非法候选/索引及非候选位置的 NaN |
| 新模式真实模型 | `tests/test_candidate_score.py` 7 项通过；采用当前 16K 配置，不换模型或降低精度 |
| 模板与映射 | 非思考 assistant 前缀匹配已记录 Qwen3.5 模板；A/B/C/D 在该前缀下分别为 token 32/33/34/35。非法多 token 和特殊标签在 prefill 前拒绝 |
| 原始输出对照 | 同一 174-token 非思考答案前缀下，与普通路径导出的全部 248,320 项 float32 logits 逐字节一致；不是只检查四个标签。普通参照实际生成 1 token，快速路径生成 0 token |
| 无生成与采样隔离 | 剖析中 prefill 有调用，采样、生成计算、逐 token 文本输出调用均为 0。改变生成 temperature/seed/原 thinking 选择后，快速候选得分不变；仅验证本轮固定输入 |
| 独立请求与有限任务 | 驻留模式依次开灯、关灯、重复开灯、无关配方问题，返回 turn_on/turn_off/turn_on/unknown；重复输入得分相同，各请求状态重建、最终权重释放。英文开灯在标签互换及选项逆序后仍选正确 ID，仅是小样本冒烟检查 |
| 超时与取消 | 早期超时退出 124；长输入完成至少一批 prefill 后通过自有取消文件合作退出 130，均无已生成 token、无有效选项；不是杀进程测试 |
| 普通生成回归 | 原 `tests/test_integration.py` 8 项通过，含默认思考完整回答、原模板、多批输入、上下文边界、错误、取消和超时；原模式仍输出文本 |
| 资源/汇总回归 | 基准汇总、资源与内存校准相关 19 项测试通过；没有放宽既定数值或资源阈值 |

原生、模型及回归用例的输入和阈值在运行前固定。首轮测试发现测试夹具把带空白标签的启动前拒绝当成运行中错误，修正为无空白但多 token 的标签后，最终 7 项模型检查完整复跑通过；没有为通过验收放宽接口或数值规则。

## 尚未证明的内容

上面的分类/路由仅覆盖灯光与未知类别、标签/顺序变更等少量自设问题，没有独立划分的广泛任务/校准集，没有证明与完整思考模式准确率相当。默认生成不会自动使用该分支。

所有本轮模型运行是有界功能诊断；其中完整 logits 对照启用了导出/阶段剖析，部分运行提示当前可用 RAM 低于保守工作预算，文件缓存和后台负载未受控。故不将这些计时纳入正式性能采用，不发布“无损加速”或 tok/s 倍数。首版仍处理整个输入并驻留主模型，短输出不能消除 prefill 成本；生成 token 为 0，解码速度必须为空/不适用，以决策延迟单独计量。

本地原始证据目录为 `reports/candidate-score-1792665455328303500/`，包含前缀检查、完整词表、各请求 JSONL 和输出；普通生成回归记录位于 `reports/acceptance/`。原始记录、模型和 EXE 不随仓库分发。后续先补代表性标注任务与校准，再研究显式回退和受控重复性能；GPU 与多 token 扩展单独验收。

## 运行与复现

```powershell
./scripts/build.ps1
py -3.11 -B tests/test_candidate_score.py
py -3.11 -B tests/test_integration.py
./scripts/run.ps1 -Mode candidate_score -CandidatesFile configs/light-switch.candidates.json -Prompt '请把客厅的灯打开'
```

接口、指标和剩余设计见[模式说明](../docs/candidate-score-mode.md)。
