# Reflex 参考研究与后续设计

核对日期：2026-10-22。Wllm 保持单一项目，Reflex 与普通生成共用推理引擎。本文依据作者仓库、固定提交的文档及部分源码，补充可借鉴机制和最小验收；没有安装或运行外部项目，没有修改实现、依赖锁、模型、本机配置或历史报告。参考编号 R46–R51 见[统一研究池](engine-references.md#13-reflex-参考项目2026-10-22)。

2026-10-23 范围补充：主线全品类模型目标与用户三档排序见[覆盖清单](hardware-model-coverage.md)；按随后澄清，Reflex 仅选择适用决策模型/任务，不要求兼容全部模型，纯生图/生视频/音频合成不列常规兼容。当前 CPU/Qwen/文本验证范围不变；单 token、yes/no 和选项树不能替代多模态理解的专用读出。硬件优先 x86/Intel GPU/NVIDIA GPU；主线 Dense/MoE/推理/代码最高，多模态/其他任务次之，其余后期完善，Reflex 在适用子集内沿用档位。扩散分类的研究例外及成本见[说明](hardware-model-coverage.md#41-生图图像判断与快速决策的边界)，不新增必做适配。此补充不代表重新核实此前全部外部项目或获得新运行结果。

## 1. 当前能力与研究边界

| 证据类型 | 本次确认的范围 |
|---|---|
| Wllm 已验证结果 | CPU 单 token 候选打分和 System One 文本协议已有有界验收，见[候选报告](../reports/candidate-score-validation.md)和[协议报告](../reports/system-one-validation.md)。证据仍绑定报告中的构建、模型、输入和 16K 配置；不扩展为广泛质量或正式性能结论 |
| Wllm 当前源码事实 | 同一题内的标签一起打分；多个问题逐题重建上下文，只复用权重；尚无共享前缀、候选并行、多 token 选项树、校准或自动思考回退 |
| 外部源码/文档事实 | 下述项目提供相似决策接口、不同读出方法或缓存/校准机制；固定源码和作者报告可复查，不代表 Wllm 已兼容或复现 |
| 研究候选 | 逐候选 yes/no、完整混合状态快照、布局与调度选择、任务校准、服务复用等均待实现/验收 |
| 性能预期 | 只给条件化成本模型和对照设计；不写可兑现的毫秒延迟、加速倍数或无损质量承诺 |

保持既有 Qwen3.5 Q4_K_M、16384 上下文、F16 注意力 KV/F32 循环状态。默认 generate 的思考、采样和停止语义保持；改变模板、打分分布或权重的实验分别评价。

## 2. 同一接口下的三种打分策略

以下名称只用于设计讨论，不是新增 CLI、配置字段或已实现的可切换策略。

| 策略 | 算法与已有依据 | 计算、语义及采用边界 |
|---|---|---|
| 直接标签 `direct_label` / 当前实现 | Wllm 将一题全部候选及内部普通单 token 标签放入提示词，读取同一答案位置 logits `z_i`，以 `softmax(z)_i` 得到候选分布 | 每题处理一个完整提示词，仍执行主干和完整词表投影。标签与顺序会影响结果；先作为后续策略的固定参照 |
| 逐候选二元判断 `binary_candidate` / 研究候选 | LLM2Jev 的 Choice/Score 为每个候选或等级渲染 yes/no 任务；从两个标签 logits 得到 `s_i=P(yes)`，默认以 `p_i=s_i/sum(s)` 汇总，总和为零时返回均匀分布。[提示词][l2j-prompt]、[二元读出][l2j-tf]、[组装][l2j-assemble]、[归一化][l2j-normalize] | N 个候选形成 N 个逻辑分支，可能批处理或共享前缀，但不等于请求只算一个序列。归一化后的 p 与直接标签 softmax 不同；多个候选都高/都低也不能据此证明正确率、候选覆盖或未知类别识别。先串行无缓存验证，再比较缓存/批量成本 |
| 多 token 选项树 `label_tree` / 研究候选 | snapjudge 将选项 token 化为前缀树，在分叉节点对允许 token 做局部 softmax，沿路径累乘。[引擎源码][snap-engine] | 可能需要多个分支前向；局部归一化的路径分数不等同完整词语的原模型序列概率。需另定结束/前缀选项、大小写、多条拼写路径和长度规则，不把它包装为当前单 token 分数的等价替换 |

LLM2Jev 的无 criteria Noul 直接使用一条 yes/no 任务；有 criteria 时会分别评分 true/false 描述后归一化，也不能假定与 Wllm 当前同题 false/true 标签分布一致。所谓 prefill-only 说明不做自回归答案生成，不免除候选分支、输入处理或模型驻留成本。[提示词][l2j-prompt]、[答案映射][l2j-assemble]

保留当前直接标签路径作为控制组，不因为另一项目覆盖更广就默认替换。策略、题型和候选数量分别评测；固定输入/标签/顺序与扰动实验分开记录。

## 3. 共享前缀、布局与调度

### 3.1 先确认实际 token 前缀

LLM2Jev 在 SGLang 中先评分一个真实候选，建立可复用前缀，再提交依赖该前缀的其他候选；候选不额外重复预热。MLX 路径有显式前缀处理。其文档明确 TransformersBackend 不使用该 staged 提交策略，不能把一个后端的缓存能力推广到全部后端。[共享前缀说明][l2j-cache]

Wllm 当前 `SystemOneQuestion::system_prompt()` 把每题 instructions 放在 system 文本中，state 和选项位于后续 user 内容。跨题 instructions 变化会在 state 之前破坏公共 token 前缀；因此“同一 state”不保证当前模板直接命中跨题缓存。已有模板也不是简单的完整 state-first 或 question-first 布局。[本地解析与模板](../src/system_one.h)、[候选提示词](../src/candidate_score.h)

后续可比较两类新模板：固定 system 后先放共享 state，再放每题问题/选项；或先放固定问题/选项，再放变化的 state。前者适合同一长材料多题，后者适合固定分类规则反复处理新输入。它们会改变提示词和模型分布，必须新建模板版本并独立检验质量。比较实际渲染 token 的公共前缀，不能按字符切分、语义相似或字段相同直接复用。

snapjudge 的布局选择和缓存快照提供参考，但其按字符长度选择的阈值来自 Apple MLX 工程；Wllm 应依据实际 token 长度、题数、状态复制和本机计量另定规则，不照搬阈值。[布局与缓存][snap-engine]

### 3.2 完整混合状态与隔离

SemIf 将共享 state 前向后的缓存展开为独立后缀行；snapjudge 明确为 Qwen 混合循环状态使用快照/恢复，不能依赖简单裁剪。两者只证明相关工程存在，不证明 llama.cpp 当前接口、Q4_K_M 或本机的等价路径已经打通。[SemIf shared][sem-shared]、[snapjudge 引擎][snap-engine]

更贴近本机的参考是 SemIf 的 [CPU/GGUF 后端][sem-cpu]：源码将 n_gpu_layers 固定为 0，核对参考 tokenizer 与 GGUF 的实际编码，使用 `llama_state_seq_get_size/get_data/set_data` 保存/恢复序列 0 的完整状态，再计算各题后缀。其 CPU shared 入口循环恢复各分支，不能套用 CUDA 并行后缀的时延。源码分列状态复制与后缀前向成本，可优先借鉴这一串行路径；不需要因此引入 Python 绑定或改 Wllm 的原生模板。

本地固定依赖的 [llama.h][llama-state] 确有对应序列状态接口。这仅是接口存在的源码证据；后续仍需核查保存覆盖范围、返回字节数、整序列清理、位置及 F16 KV/F32 循环状态恢复，并与重算完整 logits 对照。SemIf 的参考分词器校验也不证明其提示词与当前 Wllm 相同。

Wllm 方案必须同时覆盖 F16 注意力 KV、F32 卷积/循环状态、位置/序列信息及必要的后端状态。先在共享基座核查实际保存、复制和恢复接口，再做小型前缀恢复实验；不只保存 KV、不靠缩短窗口或近似重放替代循环历史。

设计要求：

- 缓存键绑定实际模型/容器与 tokenizer 身份、构建/后端及有效计算配置、完整渲染 token 前缀、模板/策略版本、上下文和状态格式；题目名称等未进入模型输入的路由元数据不单独改变语义键。
- 快照在完整前缀处理成功后的确定边界发布，作为不可变参照。每个后缀获得独立状态；题间执行顺序、修改或取消不污染其他分支，也不把失败后缀写回公共前缀。
- 模型、配置、布局变化或不支持的恢复条件触发失效/无缓存重算，不能静默采用近似状态。缓存预算包括注意力状态、循环状态、每个活动分支、复制临时缓冲、元数据和系统余量。
- 先完成串行快照/恢复，再研究分支批处理、写时复制或并行；总取消与父超时覆盖渲染、建缓存、恢复、分阶段调度、计算和清理，不为每个分支重置预算。
- 对照全量重算，检查完整词表 logits、选项 ID/原始分数、上下文边界、长短前缀、重复请求、配置失效和错误恢复。执行形状变化时沿用既定数值门槛；不能用少量选项相同掩盖其他 logits 或循环状态偏差。

共享基座统一维护此能力，generate 与 Reflex 的缓存命名空间及模板分别绑定；跨模式共享状态仍须单独证明等价。

### 3.3 选择有收益的调度方式

至少比较无缓存串行、串行前缀复用、一次提交全部分支、先建前缀再分阶段提交四种执行方式。另分离模型冷启动、模型驻留且相关前缀未命中、前缀已命中，不能把一次缓存命中称作模型首次启动加速。

在相同布局下，后缀分支的简化成本为：无复用约 `sum T_prefill(P+S_i)`；复用约 `T_prefill(P)+sum(T_restore_i+T_suffix_i)+T_cache_manage`，另加加载、渲染、排队、组装和释放。P 是实际共享 token 前缀，S_i 是分支后缀；该式不能直接当并行批次的墙钟预测，也不能据此比较不同模板的质量。

短前缀、少分支或已热缓存时，分阶段调度/复制可能比重算或一次提交更慢；16GB 共享内存设备不能假定分支并发越大越快。先提供显式实验策略，依据本机对照选择；负收益方案关闭，不给所有请求强制套用一种调度。

## 4. 概率、confidence 与校准的兼容层次

区分请求/响应结构兼容、概率定义、confidence 公式、usage 计量和任务质量。相同字段名称不保证以下语义相同：

| 项目/路径 | 已核对的差异 | Wllm 决策 |
|---|---|---|
| Wllm System One | 当前 choice 用最大概率相对均匀分布的集中度；score 使用有序等级到众数的离散度；noul 不带额外 confidence | 保留[现有协议](system-one-protocol.md)和[官方公式][typesafe-confidence]；本轮不改变接口或数值 |
| LLM2Jev | 固定快照的组装器对 Choice 和 Score 都调用同一集中度公式 | 记录为语义差异，不为仿照项目而静默改 Wllm 的 score confidence。[组装源码][l2j-assemble] |
| snapjudge | README 说明 confidence 使用归一化熵的补数 | 若将来比较策略，注明其公式，不能直接复用同一个 confidence 阈值。[接口说明][snap-readme] |
| LLM2Jev 缓存计量 | 文档中 usage.input_tokens 是完整候选输入的逻辑总量，缓存命中不减少它 | Wllm 当前 usage 是完成的 prefill 批次累计，不能直接跨项目比较该字段；未来保留协议版本，并分列逻辑 token、实际计算、复用 token、分支数和缓存命中。[计量说明][l2j-cache] |

概率和分布集中度均不自动等于正确率。借鉴 SemIf 的按工作负载温度缩放：对直接标签可拟合正温度 T，再计算 `softmax(z/T)`；这保持 argmax，不能提升选项准确率。数据应按来源/语义组隔离校准与测试，避免同一原题的改写版本泄漏；分列 NLL、Brier、ECE/可靠性图及分组置信区间，不只追求一个较小的 ECE。[校准方法][sem-calibration]

温度、门控阈值和数据版本绑定模型、量化、模板、策略、标签方案与任务域；分布/候选集变化需重验。二元候选归一化和选项树另定校准对象，不复用直接标签的温度。保留原始 logits/分数及校准输出，不覆盖原记录。先检验独立任务的准确率、覆盖率和自信错误，再考虑拒答或显式完整思考回退；回退成本和错误计入全部父请求，不只展示快速接受子集。

## 5. HTTP、原生决策模型与复用边界

llamacpp-jev 展示了在 stock llama-server 前封装 API、模板渲染、检查点、n_probs 与 GBNF 的路线。Wllm 当前直接读取采样前 logits，若研究该包装方案，须核对语法掩码/采样前后的概率、标签边界、输出 token 记账和额外进程成本，不能把受约束采样直接当成当前读出的逐位等价实现。[项目说明][lcj-readme]

本地固定 llama.cpp `f2918cabbffe8abf8e2a90c1c089085fa116c2cf` 已有 server 原生 `/v1/systemone`，并在缺少决策类型时拒绝模型；初始化读取决策元数据和名为 systemone 的模板。[固定上游路由][llama-route]、[模型准备][llama-decision]。Wllm 尚未构建/暴露该服务入口，普通 Qwen GGUF 不能因为端点存在就直接调用它。若研究不改变主权重的元数据/模板适配，或引入专用模型，分别验证其接口与状态；不预先宣称支持。

Kev 的训练适配/指针头及 jeff 的 GLiFormer 小模型属于不同模型路线，可参考隔离、校准、API 和 CPU/ONNX 工程；引入其权重属于 E4 的显式模型选项，不作为现有 Qwen3.5 Q4_K_M 的原样加速。[Kev 设计][kev-readme]、[jeff 配置][jeff-readme]

后续 HTTP 优先核查现有上游和薄适配层的复用，共用 Wllm 的模型生命周期、取消、资源与计量设施；不另建独立 Reflex 项目。是否新增服务或依赖，需在对应实施项准备完整方案和验收。

## 6. 外部结果与可推导的边界

下表均为作者公开结果，本轮没有复跑；它们不是 Wllm 指标或目标。

| 来源与条件 | 作者报告 | 对设计的意义 |
|---|---|---|
| LLM2Jev，Qwen3-1.7B BF16 / RTX 5090 / SGLang 0.5.20；5 次预热、每配置 30 次，模型/内核已热而相关 KV 未命中 | 长输入 staged 中位数 125.15ms，对照一次提交 611.77ms；短输入为 19.11ms 对照 11.49ms，反而更慢。计入渲染/分词/组装，不含模型加载及清缓存 | 证明必须覆盖负收益与缓存冷热；不把 GPU 上的倍数变成本机预期。[基准条件][l2j-benchmark] |
| LLM2Jev，231 道公开 JevBench 题；本地 Qwen3.5-4B，Jev 使用公开串行测量记录 | 准确率 76.2%，公开 Jev 1.13.0 为 85.1% | 仅为有限公开题对照，不证明与完整思考或 JEV 通用质量等价；不同端点延迟不当作纯引擎收益。[评测说明][l2j-jevbench] |
| SemIf，冻结 4B 模型 / RTX 3090；21 个二元判断、3 次中位数 | 直接读出 1.023s、0 生成 token；紧凑生成数组 5.332s、111 token，但两种读出仅 18/21 决策一致 | 同权重、少生成不能直接推出语义或质量相同。[作者记录][sem-readme] |
| SemIf，37 个 state × 21 个判断的 777 决策缓存实验 | 作者说明 BF16 快速复用路径相对重算改变 5–6 个 argmax | 不将缓存复用自动标为精确；Wllm 必须通过自己的数值、状态及任务检查。[作者限制][sem-readme] |
| SemIf，WANLI 256 题，按来源组隔离的 5 折交叉验证 | 作者报告温度校准后折外 ECE 从 0.208 降到 0.069；校准不改变 argmax | 可借鉴方法，不迁移温度或声称校准改善准确率；其他任务和混合温度并未都证明显著优劣。[详细方法][sem-calibration] |

snapjudge 的布局/延迟实验使用 Apple M2 Max 96GB 和 MLX 4-bit，部分自设样本很小；本机 i5/16GB/GGUF 无相同条件证据，不据其数字选默认布局或承诺延迟。[实验条件][snap-readme]

## 7. 最小验收矩阵与执行顺序

| 子项 / 状态 | 要解决的问题与参照 | 通过条件与边界 |
|---|---|---|
| E3-07A 当前基线 / 有界实现已交付，广泛任务待补 | 固定直接标签、实际模板、模型/构建与原始分数；普通思考选择及同模板非思考选择作不同参照 | 新增代表性标注任务及错误分析；公开任务与自设任务分列，不用小样本替代广泛质量 |
| E3-07B 策略比较 / 待实施 | 无缓存直接标签 vs 串行逐候选 yes/no；先锁定各自模板和映射，再比较候选数、歧义、候选外输入及标签/顺序/措辞扰动 | 同任务质量与全部资源/延迟分别报告；不同策略不要求 logits 相同，也不默认视为质量等价或升级默认 |
| E3-07C / E3-03 完整状态与布局 / 待实施 | 全量重算 vs 串行快照/恢复，先核查固定后端的注意力及循环状态接口 | 原模板可复用部分先验数值；新布局另验任务质量。前缀未命中、配置失效、长状态、混合题和分支隔离均通过后才评性能 |
| E3-07D / E3-02 调度与批处理 / 待实施 | 无缓存串行、串行复用、全提交和 staged；固定问题多 state、固定长 state 多题、短输入少候选各测 | 计入准备、复制、调度和峰值内存；冷/热、串行延迟/并发吞吐分列；失败/超时不计快速成功，负收益保持关闭 |
| E3-07E 校准与门控 / 待实施 | 原始分布、任务温度、拒答与显式思考回退；组隔离校准/测试 | 同时报告准确率、NLL/Brier/ECE、覆盖-风险关系、拒答/回退率及全部父请求耗时；门槛在实验前固定，不套通用 confidence 阈值 |
| E3-07F 多 token 选项树 / 后续候选 | 保留单 token 参照，另定义路径、终止、归一化与所需分支 | 验证完整路径评分及资源，不取首 token 冒充整项概率；未完成前继续明确拒绝不支持标签 |
| E3-07G API 与上游决策入口 / 后续候选 | 当前 JSON CLI、薄 HTTP 适配和上游原生决策模型分别核查 | 请求/响应、数值语义、usage、错误/超时/取消和模型身份分别验收；不混入未经许可的模型/精度切换 |

先补 E3-07A 的代表性任务与测量契约；B 的无缓存正确性对照、C 的后端接口核查可独立准备，之后再决定缓存和调度实现顺序。E3-07 不等待未通过的 Vulkan 重验，也不据此绕过 Vulkan 数值失败或 E3-04/MTP 的独立前置条件。

正式性能记录应含：模型/EXE 摘要、有效配置、策略/模板/布局版本、实际 token 与标签映射、逻辑/已计算/复用输入、候选与分支数量、缓存建成/命中/失效、准备/恢复/复制/调度耗时、父请求墙钟及序列化/最终释放、两类状态和分支峰值、内存准入、失败/取消/超时、样本及预热次数。未暴露或无法计量的字段明确为空/未知，不能用估计充当实测。

本轮仅完善文档；没有实现上述新字段、开关、策略或测试，没有新增推理、构建、校准或性能结果。

[l2j-prompt]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/src/llm2jev/inference/prompt.py
[l2j-tf]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/src/llm2jev/backend/transformers/backend.py
[l2j-assemble]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/src/llm2jev/inference/assembler.py
[l2j-normalize]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/src/llm2jev/inference/normalization.py
[l2j-cache]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/docs/en/shared-prefix-cache.md
[l2j-benchmark]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/docs/en/shared-prefix-benchmarks.md
[l2j-jevbench]: https://github.com/Yinsongxu/LLM2Jev/blob/93d4fcdcc81e0732026a65195eca3b45333af451/docs/en/jevbench.md
[sem-readme]: https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/README.md
[sem-shared]: https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/shared.py
[sem-cpu]: https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/llamacpp_backend.py
[sem-calibration]: https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/docs/CALIBRATION.md
[snap-readme]: https://github.com/Micha0827/snapjudge/blob/80012ca2abe8d000798e2dc7d717171058f1e6db/README.md
[snap-engine]: https://github.com/Micha0827/snapjudge/blob/80012ca2abe8d000798e2dc7d717171058f1e6db/snapjudge/engine.py
[lcj-readme]: https://github.com/NakliTechie/llamacpp-jev/blob/bb1c0a85e3a96a3699f2c2e6d67e607522689799/README.md
[kev-readme]: https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/README.md
[jeff-readme]: https://github.com/logan-markewich/jeff/blob/34b32f99a727c47b679adde33f4702a001e02979/README.md
[typesafe-confidence]: https://docs.typesafe.ai/confidence
[llama-route]: https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/tools/server/server-context.cpp#L5755
[llama-decision]: https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/tools/server/server-decision.cpp
[llama-state]: https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/include/llama.h#L885
