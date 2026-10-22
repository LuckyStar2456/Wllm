# 推理优化研究：统一候选比较

初次检索与核对日期：**2026-10-09**；**2026-10-10** 补充 BitNet 与 T-MAC；**2026-10-15** 复核部分官方资料并映射细化任务；**2026-10-17** 核实本地 MTP 张量、状态精度及固定依赖的 OpenVINO 说明，追加 Strata SSD 路径与 I/O 研究决策；**2026-10-21** 补入遗漏的 FreeToken，并通过 GitHub 搜索、上游文档及部分源码核对新增 R33–R44，见第 11 节；**2026-10-22** 补入 DeepSeek DeepInfer（R45），固定源码与适配边界见第 12 节。其他候选未因此全部重新核对。对象：Wllm 的 Windows / i5-1135G7 / 16GB / Iris Xe / Qwen3.5 Q4_K_M 专项优化，以及后续更大参数模型；当前默认上下文为 16K，历史实验保留原配置。

研究按可能解决的瓶颈组织。用户提到的名称仅是检索线索，所有可能有用的项目、论文和技术机制进入同一比较体系；不因被点名、热度高或发布时间新而获得优先地位。并列研究采用统一问题、证据要求和比较方法，实施顺序再由适用性和本机测量决定。

2026-10-09–15 为官方资料级筛选：读取仓库、技术文档、作者说明和论文摘要/版本记录，部分集成状态通过 API 核对。2026-10-17 另作本地源码、适配报告及 GGUF 元数据只读核查；没有完整源码审计、安装、权重下载或新增推理复现。以下“适用性”和“最小实验”是工程判断，未测收益一律未知。

证据分开标注：**已验证结果**链接 Wllm 历史运行报告并保留负载/样本边界；**研究候选**注明源码事实、文件事实或上游说明及未完成条件；**性能预期**仅为有假设的成本估算或实验前目标，不能写成预计可兑现的倍数。下表的上游支持和本地元数据核实都不代表 Wllm 已运行或已提速。

## 1. 统一研究问题

每个候选均需回答：

1. 改善的是加载、输入处理、解码、状态复用、内存容量还是并发吞吐？
2. 当前模型、量化、上下文、采样和混合状态能否保持？是否依赖转换、训练、剪枝或近似？
3. Windows、Iris Xe、共享内存和 16GB 容量下的前提是什么？可复用实现还是仅借鉴机制？
4. 文档、论文、代码、已合并集成、目标设备复现分别有什么证据？缺少什么？
5. 额外权重、状态、复制、同步、编译、I/O 和维护成本是多少？未知项明确保留。
6. 用什么最小实验验证，和哪个同条件参照比较？失败、负收益和质量回归怎样判定？

执行优化、推测算法、完整引擎、模型变体和服务缓存不是同层产品。按可用机制比较，避免把整个产品的端到端倍数当成单个算法的收益。参数规模、量化精度、内存与质量分开记录。

## 2. 持续补全的检索覆盖

| 研究方向 | 检索内容 | 目前的缺口 |
|---|---|---|
| 启动与加载 | 文件映射、加载复制、编译缓存、常驻生命周期 | 已有加载/驻留计量；进程初始化与 I/O 细分仍待补齐 |
| 输入处理与基础执行 | CPU SIMD、量化核、融合、布局、批量、CPU/核显后端 | 已有阶段及有干扰的算子诊断；带宽、缓存和同步瓶颈仍未确定 |
| 解码与推测 | 独立草稿、MTP、前缀查找、特征草稿、块/树并行验证 | 已核实原文件有 MTP、派生文件排除 MTP；恢复接口、采样和完整回滚未实现/验收 |
| 状态与长上下文 | 原地缓冲、前缀复用、KV 分块/分层、循环状态 | 当前注意力 KV F16、循环状态 F32 已从源码核实；前缀复用尚未实现，长上下文质量覆盖仍不足 |
| 大模型容量 | 张量/专家放置、热度、预取、RAM/SSD 分层 | 本机磁盘能力与更大模型候选未测 |
| 用户配置与调优 | 受硬约束的搜索、成本模型、帕累托候选、冷暖场景 | 不预设指标权重或最佳档位 |

检索覆盖 2026 年近期项目和 9 月新论文，同时保留仍有用的基础研究。不以“最新”排除成熟技术，也不声称穷尽全领域；没有创建自动监控。新增候选使用下表相同字段，不能只追加项目名称。

## 3. 执行、编译与加载候选

编号仅供追踪，不表示排名。“文档/仓库”表示读到上游说明，均不代表本机已验证。

| 编号 / 候选及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R01 [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp) / [bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) | 多后端、参数化 pp/tg；仓库/工具文档 | 推荐参照基础；固定现有 GGUF、状态和采样设置 | 同模型扫描线程/批量/放置；另测分词、采样和全程，核算常驻资源 |
| R02 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | CPU 计算、重排和融合；仓库说明 | 上游明确重点为 CPU/CUDA，不能沿用其 Vulkan 支持假设；额外量化另组 | 先比较同一 Q4_K_M 的 CPU 路径；核对 ISA、权重支持和与主线分叉的维护成本 |
| R03 [llama.cpp OpenVINO backend](https://github.com/ggml-org/llama.cpp/blob/f2918cabbffe8abf8e2a90c1c089085fa116c2cf/docs/backend/OPENVINO.md) / [延迟优化](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html) | 图执行、编译/模型缓存；固定依赖文档列出 Qwen3.5-9B Q4_K_M，验证平台不同于本机 | 本机未验证；不是 OpenVINO GenAI 直接 GGUF 加载入口；需核对运行时量化转换及状态化回滚限制 | 测冷编译/缓存启动、pp/tg、峰值内存与有效精度；普通前向、前缀恢复和推测组合分别验收 |
| R04 [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html) | past/present 缓冲复用；官方文档 | 可借鉴生命周期；转换格式不假定等价 | 剖析重复分配/复制；验证原地更新、取消和回滚，记录节省量与净收益 |
| R05 [MLC LLM/TVM](https://github.com/mlc-ai/mlc-llm/blob/main/docs/compilation/compile_models.rst) | 目标设备编译、模型库；文档含 Windows Vulkan | 固定形状专门化可借鉴；模型转换、精度与依赖需核对 | 同数学操作对照专门化前后，计入编译、载入、产物及额外缓冲 |
| R06 [NInfer](https://github.com/Neroued/ninfer) | 选定模型与 GPU 的专用运行时；仓库说明 | 当前 CUDA/RTX 5090 路径仅作机制参考，不是 Iris Xe 实现 | 找出适用的调度/融合机制，映射本机热点后再做局部原型 |
| R07 [Splash](https://github.com/incoai/splash) | 模型形状专用内核、加载布局、内存规划；仓库说明 | Metal/ANE 不可直接移植；默认状态精度不等于 Wllm 基线 | 研究布局/计划复用时同时计入初始化和质量检查，不套用 Apple 倍数 |
| R08 [goinfer](https://github.com/townsendmerino/goinfer) | 轻量本地运行、冷启动与交替对照；仓库基准 | 可借鉴生命周期和测量；未核实本机目标适配 | 对照常驻/非驻留启动与请求开销，比较收益及常驻内存，不先更换技术栈 |

R03 于 2026-10-17 对照本地固定 llama.cpp 提交 `f2918cabbffe8abf8e2a90c1c089085fa116c2cf` 的 `docs/backend/OPENVINO.md` 核实：其模型验证平台为 Lunar Lake / 32GB / Ubuntu，不是本机。状态化执行要求 Qwen3.5 使用未切分图、模型缓存且无循环状态回滚；状态保存/恢复、序列回退、上下文移动等不受支持。以上限制针对所述状态化路径，不能扩大成所有执行方式均不支持，也不能假定与前缀恢复或 MTP 自动兼容。

同一文档说明 CPU/GPU 默认将 Q5_K/Q6_K 转为 Q8_0_C，另有进一步降为 4-bit 和 `native` 配置。因此固定磁盘文件并不保证运行时权重表示不变；原生配置也不自动证明所有模型/设备可用。实施时逐张量核对转换、计算/累加及状态精度、内存和数值，无法满足保真约束时独立为配置实验。OpenVINO GenAI 的直接 GGUF 加载、离线转换和 llama.cpp 后端分别记录，某一入口的问题不能直接否定或证明另一入口。

### 3.1 补充：低比特 CPU 计算机制（2026-10-10）

以下编号延续原候选池，不表示优先级。证据为官方仓库与技术说明；尚未安装、集成或在本机复现，收益未知。

| 编号 / 候选及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R31 [Microsoft BitNet](https://github.com/microsoft/BitNet) / [CPU 优化说明](https://github.com/microsoft/BitNet/blob/main/src/README.md) | 原生三值模型的 I2_S 内核；权重解包跨多个输入位置复用、可调分块与并行度、分别优化 GEMM/GEMV；官方有 Windows 构建说明及 x86 AVX2 路径 | 可借鉴计算组织；I2_S 内核不能直接替代 Q4_K_M。原生 BitNet 模型属于 E4 独立模型选项，不是当前 Qwen3.5 的等价转换；嵌入层重新量化也不进入默认基线 | 先检查现有 Q4_K 热点的解包/访存是否重复，再对实际矩阵形状验证复用与分块；记录工作缓冲、准备开销、数值和完整请求表现。模型选项另测质量与资源 |
| R32 [Microsoft T-MAC](https://github.com/microsoft/T-MAC) | 查表执行低比特混合精度矩阵乘法，减少传统反量化计算；官方说明覆盖一般低比特模型、Intel CPU 与 Windows | “支持 4-bit/GGUF”不等于已适配当前 Q4_K_M；须核对混合张量类型、分组 scale/min、激活处理及 Qwen3.5 架构/状态支持。不能通过重新量化伪装成同权重加速 | 固定源码版本，先做实际张量的算子对照，再评估模型集成；计入查表构建、布局转换、额外内存、代码生成/编译与维护成本，分别测 prefill/decode |

BitNet 的“无损推理”针对其原生低比特模型及相应执行路径，不证明将现有 Qwen3.5 Q4_K_M 转为 1.58-bit 后能力不变。官方硬件和模型上的加速倍数不能作为本机预期收益。BitNet 官方将一般低比特模型的查表计算引向 T-MAC，因此两者按不同适配条件并列研究。

当前实验衔接：CPU 重排候选继续关闭并独立定位数值差异；资源与能力核实后，可开展不依赖该修复的后端对照、状态能力检查。解包复用、缓存分块或查表计算根据实际张量和瓶颈另选最小实验。分块尺寸和并行度在本机测定，不照搬上游服务器配置；长输入批处理与逐 token 生成分别验收。上游已具备的机制先核对，避免把已有能力重复包装为新优化。

每项实验先固定算子输出、完整 logits、采样/任务质量的检查条件；通过后再交替重复测量短/长输入、生成、首 token、完整请求与资源。数值不达标时定位差异并保持关闭，不为候选放宽既定门槛。换模型或改变精度则独立记录，不能与默认基线混算收益。

## 4. 解码与推测候选

全部对照无推测执行，统一测目标精度、采样语义、额外内存、草稿/验证/回滚/同步耗时和完整时延；接受率不能单独排名。

| 编号 / 候选及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R09 [普通独立草稿](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | 小模型提议、目标验证；上游文档 | 需 tokenizer/验证实现兼容；额外草稿可能占满内存 | 扫草稿大小与块宽，记录双模型加载及验证开销，保留原目标 |
| R10 [prompt lookup / n-gram](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | 从已有 token 匹配提出草稿；上游文档 | 无需训练草稿，不代表零开销；仍需目标验证与状态回滚 | 重复文本和无重复文本各测，记录索引内存、命中率及退化场景 |
| R11 [MTP](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | 利用模型预测层提出多 token；本地文件核实原文件有 15 个 MTP 张量、派生文件没有 | 尚未集成或运行；先核对固定上游接口和张量映射，恢复辅助能力不改主目标权重；后端须支持完整状态回滚 | 先验容器/接口/采样/回滚，再测不同草稿长度；额外头、嵌入/输出、缓冲与快照纳入预算，按接受输出和完整请求测净收益 |
| R12 [EAGLE-3](https://github.com/SafeAILab/EAGLE) / [论文](https://arxiv.org/abs/2503.01840) | 特征条件草稿与训练时测试；官方实现/论文 | 需要匹配检查点；不能将其他模型草稿视为可通用 | 与同条件其他草稿比总成本；无现成适配时记录训练成本，不擅自开展训练 |
| R13 [DFlash](https://github.com/z-lab/dflash) / [论文](https://arxiv.org/abs/2602.06036) | 块扩散并行草稿；官方列有 Qwen3.5-9B 草稿 | 仍需确认本地目标、隐藏层、格式、采样及后端 | 扫块宽并测额外模型/状态；混合状态恢复先过正确性 |
| R14 [DFlash 2](https://inco.ai/blog/dflash2/) | 候选路径选择与局部卷积；作者说明、集成 PR | 公开检查点不能与一代清单混用；本机目标匹配未核实 | 对照普通块草稿，计入新增选择/计算开销；现成权重不符则保留研究项 |
| R15 [DFlare](https://arxiv.org/abs/2606.02091v2) | 草稿各层融合更丰富目标层信息；2026-06-02 论文修订 | 本轮为论文级筛选；草稿训练/导出适配成本未知 | 先评估可用草稿和隐藏状态成本，再比较接受长度与总时间 |
| R16 [JetSpec](https://github.com/hao-ai-lab/JetSpec) | 并行树草稿与树验证；官方实现/实验 | 上游核心实验为 B200；本机树状态和掩码执行未验证 | 块和树在同预算下比较，计入无效分支及状态存储，不直接套用论文倍数 |

“无损”必须对应具体目标计算和采样算法。greedy 输出一致、采样分布等价、有限精度数值接近分别验收。各方法先单独测试，再测试兼容组合；不能相乘单项倍数。

**R11 本地文件核实（2026-10-17，无推理测试）：** `reports/model-adaptation.json` 记录原始文件和派生文件的身份、427 个保留张量校验及被排除名称；`scripts/prepare_model.py` 明确排除 `mtp.`。本次用固定依赖的 GGUFReader 只读元数据复核：原文件 883 个张量中有 15 个 `mtp.`，原始张量数据合计 **150,980,608 字节（约 144MiB）**；派生文件 427 个张量，MTP 为 0。字节数由各张量 `n_bytes` 求和，不含容器开销，更不等于运行时增量内存。文件哈希身份沿用原适配报告，本次未重算全文件哈希。

这些 MTP 张量属于辅助文本预测，不是视觉模块；旧报告字段 `skipped_nontext_tensors` 的命名不能改变其性质。后续准备独立容器/视图仍须核对上游命名、元数据、可能共享的嵌入/输出与完整 KV/循环状态快照，不能仅复制张量或增加开关就宣称可用。任务及回滚检查见[架构 6.1](architecture.md#61-当前文件的-mtp-前置条件2026-10-17)和 E3-04。

## 5. 放置、异构计算与容量候选

| 编号 / 候选及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R17 [ATSInfer](https://arxiv.org/abs/2607.10183v2) | 张量粒度放置、动态传输、异步协调；2026-07-14 论文修订 | 只核实论文；共享内存成本不同于独显 PCIe | 用本机剖析比较层级/张量级计划，计算减少同步是否抵消调度开销 |
| R18 [KTransformers / kt-kernel](https://github.com/kvcache-ai/ktransformers) | CPU 量化核、异构专家放置；[内核说明](https://kvcache-ai.github.io/ktransformers/en/kt-kernel/kt-kernel_intro.html) | AMX/AVX 路径按实际 ISA 选择；服务器/大 MoE 结果不可照搬 | 比较可用 CPU 核及专家放置，量化变化另组；核算线程/缓存争用 |
| R19 [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) / [PowerInfer-2](https://arxiv.org/abs/2406.06282) | 热/冷激活分工；第二代研究手机神经元簇计算与存储 | 依赖模型稀疏性及特定设备，不能当作任意 Dense 模型直接加速 | 先检查原模型是否具备必要性质；需换模型/重训则移入可选能力线 |
| R20 [Strata](https://github.com/Niko1221/Strata) / [设计](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md) / [细节](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) | 2026-10-15 记录 RAM/设备分工、热点缓存、预取及增量状态；2026-10-17 另核实 README 中部分高精度模型的 SSD 权重读取说明，见 5.2 | 不将旧 RAM 驻留描述推广为全部模式；也不将新增 SSD 路径推广为任意模型/容量可用。Iris Xe、16GB 和当前 9B Dense 均未验证 | 分别进入 E4-03/04/05、E3-03/04 与 E4-07；固定对应模式与版本，同权重/路由验证，不继承上游精度与速度结论 |
| R21 [Colibri](https://github.com/JustVugg/colibri) / [调优](https://github.com/JustVugg/colibri/blob/main/docs/tuning.md) | 专家流式加载、热度缓存、预取；仓库文档 | 有 Windows/Vulkan 路径，当前 9B 适配未核实；可选丢弃专家模式不适合默认基线 | 关闭近似路径，比较 CPU/GPU、缓存和预取，记录 I/O、资源、未命中及负收益 |

本机 CPU/核显共享带宽和功耗，不能把独显并发复制或“显存+RAM”容量相加的经验直接套用。MoE 总参数与激活参数分别报告。预测只能安排加载时机，未命中的必要专家仍需执行。

### 5.1 大模型机制到实施任务的映射（2026-10-15）

细化任务及验收位于[实施计划第 8 节](implementation-plan.md#8-剩余任务与参考实现明细)。以下是资料级设计映射，Wllm 尚未实现这些容量策略。

| 内容 | 参考项目的具体部分 | Wllm 任务与适配要求 |
|---|---|---|
| 热专家驻留与替换 | Strata 的自适应 GPU 专家缓存；Colibri 的路由热度、固定热点与 LRU | E4-02/03：先实现专家存储与正确路由，再比较静态/动态放置，记录冷/热命中与替换代价 |
| CPU/GPU 协同 | Strata 在 GPU 计算命中专家、CPU 计算未命中专家；R18 异构内核 | E4-04：相同数学语义下比较串行与重叠；核显共享带宽须本机实测 |
| 超出 RAM 的专家集合 | Colibri 的磁盘按需读取、批内去重与预取；Strata 后续核实的部分 SSD 权重读取路径（5.2） | E4-05：先正确按需读取，再测有界预取和负收益禁用；SSD n-gram 表与权重读取是不同证据，Windows I/O 另适配 |
| Dense 大模型流式执行 | llama.cpp mmap/放置与 R17 的张量粒度思路；借鉴 Colibri 的读取组织 | E4-06：Wllm 跨模型设计推导，非 Strata/Colibri 已验证通用方案；核算每 token 必需读取量，不假定 MoE 稀疏性 |
| 长上下文与连续请求 | Strata 的 KV 放置与增量对话；vLLM/SGLang 的前缀复用；R24 状态分层 | E3-03、E4-07：保持当前状态精度，适配完整 KV/循环状态，分别验证命中、未命中、取消及恢复 |
| 资源与用户需求匹配 | Strata 的硬件/模型配置流程、Colibri 的资源规划；Wllm 用户硬约束与偏好 | E4-01/08、E5-01–06：逐组合确定可行性并用实测校准，支持弱硬件；上游推荐不能直接作为本机最优结论 |

Strata 与 Colibri 的低比特状态、剪枝模型或近似专家省略等选项不进入 Wllm 默认保真路径。参考其放置与调度机制时保留当前模型/精度/上下文约束；独立模型及精度选项另行验收。

### 5.2 Strata 资料更新（2026-10-17）

保留 2026-10-15 的机制筛选日期与第 8 节历史快照，追加本次核实范围：[README 固定修订 `2c4005cee06605104e7f43d562761e257ff49a90`](https://github.com/Niko1221/Strata/blob/2c4005cee06605104e7f43d562761e257ff49a90/README.md#which-model-should-i-pick)（GitHub API 返回该文件最近修订时间为 2026-10-13 UTC）。其说明部分高精度模型在 RAM 不足以驻留时会在回答期间从 SSD 读取部分权重，并提示速度下降。因此“Strata 只有 RAM 驻留专家、SSD 仅存 n-gram”不能作为所有模式的结论；SSD n-gram 的描述本身仍不能证明权重流式执行。

这是上游文档证据，未作完整源码审计、安装或本机复现。该 README 的常规配置要求至少 32GB RAM 和独显，不能外推到本机 16GB / Iris Xe；未将其宣传速度、低比特/裁剪模型或缩短思考的配置纳入 Wllm 默认预期。E4-05 可增加此路径的源码与成本核查，具体模型、格式、平台和质量条件仍需锁定后逐项验证。

## 6. 状态、缓存与调优方法候选

| 编号 / 候选及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R22 [vLLM 缓存设计](https://docs.vllm.ai/en/latest/design/prefix_caching/) | KV 分块和前缀共享；官方文档 | 服务吞吐不等于单用户延迟，混合循环状态另适配 | 命中/未命中/失效/恢复分别测，记录额外驻留与状态正确性 |
| R23 [SGLang](https://github.com/sgl-project/sglang) / [原论文](https://arxiv.org/abs/2312.07104) | RadixAttention 等跨调用状态复用；仓库/论文 | 借鉴缓存组织，不能直接视为本机运行支持 | 与简单前缀缓存对照实际复用率、管理开销及取消隔离 |
| R24 [LMCache](https://github.com/LMCache/LMCache) | KV 复用、分层存储与控制接口；仓库/文档 | 不直接复用不兼容的混合状态；压缩精度独立核对 | 比较重算与读取的成本，测命中时首字、缓存容量及失效 |
| R25 [py-kvcache](https://github.com/atlarge-research/py-kvcache) / [2026-09 论文](https://arxiv.org/abs/2609.11744) | 外部 KV 预加载、成本门控和 DRAM 缓存；仓库/摘要 | io_uring 路径不能直接用于 Windows；可借鉴门控机制 | 先测 I/O 和重算交点，再决定预加载，计入无效读取及系统缓存 |
| R26 [SpecStream](https://arxiv.org/abs/2609.33184) | 流式 KV 验证与只卸载已提交历史；2026-09-27 摘要 | 论文级线索；质量接近不能等同严格输出一致 | 长上下文状态确为瓶颈后，验证完整注意力/回滚并测内存与总时间 |
| R27 [Spexis](https://github.com/mlsys-seo/spexis) / [论文](https://arxiv.org/abs/2609.34370) | 多 GPU 前瞻推测调度；2026-09-28 研究原型 | 当前多 GPU 轴超出本机范围，保留调度思想 | 只研究预测无效工作与资源争用的可迁移部分，不自动新增分布式实现 |
| R28 [边缘推测成本模型](https://arxiv.org/abs/2602.08060v2) | 异构划分与推测收益预测；Cortex-A/Mali 实验 | 机制可借鉴，平台成本须重测 | 用本机草稿/验证/同步测量校准，测试预测错误与禁用条件 |
| R29 [HyperQwen](https://github.com/syv-ai/HyperQwen) | Qwen 部署补丁、配置与重量化管线；仓库说明 | 补丁、量化、模型改变须拆开比较 | 各改动单独开关，记录原模型与变体差异及维护成本 |
| R30 [FlashAttention](https://arxiv.org/abs/2205.14135) | 分块和在线归约减少注意力 I/O；原始论文 | 是精确注意力思路，不等于 CUDA 实现能运行 Iris Xe，也不覆盖循环层 | 先查上游已有核；同状态精度测长输入的完整注意力层数值、内存与全程收益 |

## 7. 从并列筛选到实验

第 3–6 节保留原有 R01–R32，第 11 节补充 R33–R44，第 12 节补充 R45，共 45 项项目/机制参考，包含已归档的历史机制参照，并非 45 个可执行候选或都要安装的引擎，也未完成同深度源码审计。未知信息统一记录为未知，不用虚构分数给出排名。

每个候选进入下一阶段须满足：对应已测瓶颈、默认语义不变或明确独立为用户可选配置、硬件/模型路径可验证、成本可控制。若某候选需要当前没有的架构/硬件，保留其适用前提，不因项目名称直接淘汰或提前选中。下表保留原实验分组，新增参考的任务归属见第 11.4 节和第 12.3 节。

| 实验组 | 统一比较问题 | 候选覆盖 | 所需证据 |
|---|---|---|---|
| X1 基础执行 | 相同权重下哪个后端/执行配置合适 | R01–R08、R30–R32（保留权重的计算机制） | 冷/热启动、pp/tg、首字、全程、有效精度及资源；不可直接运行者先比机制 |
| X2 异构放置 | 同语义下如何减少复制、同步和计算等待 | R17–R21 | 张量/专家适用性、带宽、同步及净时延；Dense 和 MoE 分组 |
| X3 推测执行 | 哪种提议/验证方式带来实际净收益 | R09–R16、R28 | 草稿匹配、采样/回滚、额外内存及全流程成本；无草稿方法也并列比较 |
| X4 状态管理 | 复用、重算、驻留和读取怎样选择 | R04、R22–R26、R30 | 混合状态完整性、命中与未命中、I/O/重算成本和长上下文质量 |
| X5 可运行规模 | 相同用户约束下怎样扩大参数规模 | R17–R21、R24–R26、R31（原生模型独立组） | 资源、未命中、首字/尾延迟、模型质量与容量；不把能加载当成可用，换模型单列 |
| X6 调优与组合 | 单项收益能否在组合中保留 | R01–R32 中通过条件的候选 | 单变量实验后做组合消融，记录冲突、维护成本和多指标折中 |

首期 E0 仅做最小准备，E1 优先交付一个可用后端上的文本推理原型，同时提供可用的实时数值计量显示、结束汇总与可保存记录，不等待完整参照矩阵或所有研究候选筛选。原型后 E2 补齐评测并逐项做基础优化，E3 按瓶颈开展专项优化，E4 扩展模型/精度/参数规模，E5 扩展兼容性。候选按统一标准并列研究，实验按依赖和本机证据选择；每轮围绕有限优化项交付，不要求任一等级穷尽所有项目。

上游“已支持”、PR“已合并”、本机“已运行”、本机“已提速”分开标注。保持完整上下文和当前 Q4_K_M 默认基线；其他精度、剪枝/稀疏化或换模型归入用户可选配置，不能混入同等条件引擎加速。

## 8. 仓库快照与集成状态

以下 SHA 为 GitHub API 在检索时返回的默认分支 HEAD，用于追溯研究时点，**不是 Wllm 已锁定的依赖版本**。提交时间按 UTC 列示；网页文档可能继续变化，实现前重新检查相关文件与提交。

本表保留 2026-10-09 快照。2026-10-10 补充的 R31/R32 及 2026-10-15 的机制复核使用上述官方网页，未新增源码锁定；2026-10-17 的固定文件核实范围单列于 R03/R11/5.2，不改写历史 HEAD 表，也不增加运行依赖。进入代码实验前须补齐版本与许可证核对。下列历史 PR 状态不代表本次实时状态。

| 仓库 | HEAD SHA | 提交时间 UTC |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp/commit/a518119d30cade6260f7494120863428a3fe8ee5) | a518119d30cade6260f7494120863428a3fe8ee5 | 2026-10-09 14:20:25 |
| [DFlash](https://github.com/z-lab/dflash/commit/07ebd93db9f472af339b644bb70221ad8428328a) | 07ebd93db9f472af339b644bb70221ad8428328a | 2026-08-18 20:58:29 |
| [Strata](https://github.com/Niko1221/Strata/commit/fb58e0dbc8399662c0e47c76578c6e878b14f6cf) | fb58e0dbc8399662c0e47c76578c6e878b14f6cf | 2026-10-08 10:25:39 |
| [Colibri](https://github.com/JustVugg/colibri/commit/bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850) | bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850 | 2026-10-06 20:02:24 |
| [Splash](https://github.com/incoai/splash/commit/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7) | 0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7 | 2026-10-06 11:27:05 |
| [NInfer](https://github.com/Neroued/ninfer/commit/81c8ce093b2c1646a87566a8e59d807fcf0ec95c) | 81c8ce093b2c1646a87566a8e59d807fcf0ec95c | 2026-10-07 10:36:40 |
| [HyperQwen](https://github.com/syv-ai/HyperQwen/commit/3acb93f7acdcdbc3cfe67e2849e27537c0524b0a) | 3acb93f7acdcdbc3cfe67e2849e27537c0524b0a | 2026-10-08 09:17:59 |
| [JetSpec](https://github.com/hao-ai-lab/JetSpec/commit/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f) | 2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f | 2026-06-27 10:07:40 |

关键集成核对：llama.cpp #27342 为 closed 且 merged=true；Strata #1426 为 open 且 merged=false。前者是上游集成证据，后者只是未合并实验。两者都未在本机验证。

## 9. 后续研究记录格式

每个候选保存：来源与版本 → 解决的瓶颈 → 模型/硬件假设 → 是否改变语义/精度 → 最小验证实验 → 质量/采样结果 → 全流程速度与资源 → 采用或淘汰原因。复用代码前核对对应许可证与第三方来源。

完成一次新技术筛选后再决定是否投入实现，不按项目热度批量安装依赖，不以未验证宣传填充性能目标。

## 10. 普通 SSD、查表与地址计算的研究决策（2026-10-17）

以下是本次讨论形成的设计取舍与成本推导，不是已实现机制或性能结果；本轮不改实现、不新增模型或训练任务。

| 方向 | 当前取舍与理由 | 重新进入实验的条件 |
|---|---|---|
| CPU 缓存/寄存器中的局部查表 | 保留 R32 候选。将小块低比特运算组织为查表，仍由处理器构表、寻址和归约；不能据此推导 SSD 查表有相同收益 | 现有 Q4_K_M 类型及 scale/min 可正确适配；含构表/布局/访存的成本低于参照，完整数值与质量通过 |
| 全模型或整层穷举后以磁盘查表 | 暂不进入近期主线。即使输入有限精度，若一个块有 d 个独立输入、每个 b 位，直接枚举也有 `2^(b*d)` 个输入组合；整层状态、位置和上下文还增加索引维度。固定权重不消除动态激活/状态空间 | 证明具体子问题的表容量、构建/更新成本和随机依赖读取可控，并保持目标语义；不以近似匹配或压缩损失替代质量证明 |
| 地址变化代替数值运算 | 寻址能表达索引/置换/路由等操作，作为布局与调度机制研究；普通 SSD 地址变化本身不会执行目标矩阵乘加、归约或非线性。纯查表模拟仍受表规模、索引生成和串行读取约束 | 给出明确算子映射与等价性，含地址生成、读取和归约的全成本对照；不能将 CPU 执行的工作从统计中移除 |
| SSD 权重/状态分层与 I/O 重叠 | 纳入 E4 容量线，已有数据按需加载与有界预取可降低驻留需求；是否加速取决于缓存、必要读取及计算重叠 | 按[架构 7.1](architecture.md#71-ssd-容量与速度的成本边界设计估算未实测)记录物理字节、请求、带宽、依赖等待、冷暖及完整请求；容量改善和速度改善分别验收 |

最小收益条件是：在相同目标语义与工作负载下，构表/准备的摊销、索引计算、数据读取、归约及必要目标校验的总成本低于参照，并满足内存、质量和稳定性约束。纯 I/O 可表达有限计算不等于存在有利的存储量和时延；普通 SSD 的并行请求也不等同于增加算术单元。模型特制查表层、重新训练或改变路由的机制只能列为独立模型研究，不默认转换当前模型。

近期计划只承诺形成可复现的比较与采用/拒绝结论，不承诺某个参数规模、tok/s 或加速倍数。现有 Q4_K_M 是保真参照，不声称等同未经量化的原始模型；超过实验噪声的改善与满足用户“明显加速”的体验目标分别记录。

## 11. GitHub 补充研究（2026-10-21）

本次补入此前遗漏的 **FlashML-org/FreeToken**，并围绕 CPU 量化计算、Intel 后端、混合循环状态、MoE 分层和弱硬件扩展搜索 GitHub。读取项目维护者的 README、技术文档及部分源码，用 GitHub API 记录默认分支 SHA、归档状态与许可证识别结果。仅为资料筛选和设计映射：未安装引擎、下载模型、运行上游代码或新增 Wllm 推理测试；上游宣称的速度不作为本机性能预期。原 R01–R32 未全面重新审计。

### 11.1 新增项目与最小验证

| 编号 / 项目及来源 | 可借鉴机制与证据 | 本机适配 / 保真约束 | 最小验证与代价 |
|---|---|---|---|
| R33 [FreeToken](https://github.com/FlashML-org/FreeToken) | 带宽自适应 CPU/GPU 专家分工、双缓冲 prefill、专家缓存、KV/GDN 检查点和弹性显存预算；README、CLI 文档及部分源码，详见 11.2 | 安装文档主路径为 Linux/NVIDIA CUDA，另有 Windows 桌面及 WSL/WDDM 路径说明；不构成 Iris Xe 支持证明。GGUF 注册表含 `qwen35`/`qwen35moe`，但未验证本地文件、状态精度或完整回滚；CPU 专家执行不等于整个引擎可 CPU-only 运行 | 先核对 Dense 路径与所需状态接口；把状态缓存和带宽校准机制分别映射，不直接照搬 PCIe 分流公式；计入权重副本、锁页、缓存/快照和编译峰值 |
| R34 [QuestWend](https://github.com/kishida/questwend) | 基于 ggml 的 Qwen 运行时、RAM/SSD 专家层级、热缓存、CPU miss compute、启动调优和 MTP；README 与 resident-decode 技术文档 | 有 Vulkan 构建说明，但其专家卸载仅记录 CUDA/Metal 实践，不能认定 Intel 路径已验收。`resident-decode` 会掩蔽非驻留专家并替换路由，属于近似模式；GitHub API 未识别项目许可证，不能直接引入代码 | 保真实验关闭 resident-only routing 和 prefill pruning，完整计算所有实际路由专家；先核实许可证、量化表示、状态和 MTP 回滚，再比较未命中与 I/O |
| R35 [FastLLM](https://github.com/ztxz16/fastllm/blob/master/README_EN.md) | C++ CPU/NUMA 内核、专家独立放置、磁盘辅助、Qwen3.5 MTP 与外置草稿；上游 README | Windows/NVIDIA 和 CPU 源码构建分别有入口；NUMA/多 GPU 机制不自动适合本机。支持部分 GGUF 不等于当前全部 Q4_K_M 张量保持原表示；草稿默认量化须单独核实 | 先查实际张量与 CPU 算子；MTP 对照保留主目标并核对草稿转换、采样和状态恢复；不能因存在外置 MTP 接口就认定本地容器可直接运行 |
| R36 [llamafile](https://github.com/mozilla-ai/llamafile) | 跨平台单文件分发与 CPU SGEMM/量化 mixmul 调度；README 及 `llamafile/sgemm.cpp` | 本轮源码含 AVX2 等 ISA 分派；仍须检查现有固定 ggml 是否已有相同机制，避免重复实现。单文件打包本身不证明推理加速 | 用实际 prefill GEMM/decode GEMV 形状比较，计入布局、线程与准备开销；按具体借用文件核对许可证，不把整个引擎作为强制替换 |
| R37 [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) | C++/Python 推理管线、编译执行、前缀缓存和推测解码；官方 README | 与 R03 的 llama.cpp OpenVINO 后端分开；示例使用模型导出/IR，不证明现有 GGUF Q4_K_M 可无变更复用。硬件、模型、状态操作分别验证 | 先辨明入口、导出和有效精度；涉及重新量化则作为独立模型/精度实验；普通前向、缓存恢复与推测回滚分验，记录编译与内存峰值 |
| R38 [Lemonade](https://github.com/lemonade-sdk/lemonade) | 多引擎/后端管理、设备能力展示和模型配置组织；README 的后端矩阵与 `lemonade backends` | 它是运行与服务管理参考，不是独立量化核。不同后端拥有各自硬件要求；NPU/其他 GPU 路径不代表本机可用或自动最优 | 借鉴后端探测、显式配置和失败解释；验证物理设备去重、未知能力及不可行组合，不隐式下载或切换模型 |
| R39 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Gated DeltaNet 等线性/循环算子的 chunk 与 recurrent 实现；README、安装矩阵 | 当前资料含 CUDA/ROCm/XPU 等入口，但不证明 Iris Xe/Windows 可执行对应核；CPU 安装项的 Triton 标为 import-only，不能据此声称 CPU 加速。训练用结构变体不进入默认模型 | 对照 Qwen3.5 已有循环层的精确方程、门控、卷积和 F32 状态；分别验证逐 token、批量、边界与恢复；不把 FlashAttention 的完整注意力优化套到循环层 |
| R40 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | 注意力、分组 GEMM/MoE、采样、推测验证和预编译/JIT 内核组织；README | 当前所查实现围绕 NVIDIA/CUDA，作为算子与接口机制参考；不是 Iris Xe 可直接使用的后端，低精度选项不进入当前保真基线 | 先检查 Wllm 是否存在对应热点，再研究分块、融合与准备摊销；需要新硬件时另立实验，不为研究安装 CUDA 栈 |
| R41 [ssdmoe](https://github.com/RasoulNik/ssdmoe) | safetensors 专家字节索引、按路由 `pread`、批量读取与提示状态持久化；README | 明确为 Apple Silicon/MLX，不能当作 Windows 运行支持；其 4-bit 权重不等于当前 GGUF Q4_K_M。当前 Dense 9B 不适用专家卸载 | 容量线先用索引/读取夹具核对实际路由专家、字节和缓存；Windows 原生 I/O 单独实现，比较按需、预取、重算与完整请求 |
| R42 [FlexLLMGen（原 FlexGen）](https://github.com/FMInference/FlexLLMGen) | GPU/CPU/磁盘分层、I/O 调度、批量摊销与成本规划；README | API 核实仓库已归档，仅作历史机制参照；主要面向高吞吐大批量，README 明确小批量可能很慢。压缩和旧模型支持不能直接迁移 | 区分单用户首字/解码与批量吞吐，先测 Dense 必读数据与存储成本；保持当前精度，压缩另组，不作为默认运行依赖 |
| R43 [MNN / MNN-LLM](https://github.com/alibaba/MNN) | 端侧低内存执行、转换/融合、移动及 PC 部署；README 与 `transformers/README.md` | 上游记录 Qwen3.5 支持和 Windows 构建，但 MNN 模型导出与量化不是现有 Q4_K_M 的自动等价路径；本机完整模型未验收 | 近期先借鉴低内存计划；完整引擎对照需核对逐张量转换与状态语义，改变量化时单列质量和性能；弱硬件/移动端留待真实设备验证 |
| R44 [KleidiAI](https://github.com/ARM-software/kleidiai) | Arm 量化矩阵微内核、packing、tile 与 ISA 变体组织；README 和文件树 | GitHub 为 Arm 官方 GitLab 的只读镜像；面向 Arm，不是 i5 x86 可执行内核。API 未自动识别许可证，已见源码 SPDX 与 `LICENSES` 中 Apache-2.0/BSD-3-Clause，仍按文件核实 | 作为跨硬件 E5 参考；取得 Arm 设备后按实际类型与 ISA 检查 packing/数值/准备成本；不把 Arm 特定收益列入本机预期 |

### 11.2 FreeToken 的具体参考边界

FreeToken 应补入参考池：其资源校准、混合状态复用和动态预算与 Wllm 长期目标相关。以下按机制分别研究，不能从上游整机宣传推出本机收益：

1. **实测后决定 CPU 计算或搬运。** 固定源码 `python/freetoken/moe/benchbw.py` 比较实际 CPU 专家核与 PCIe gather，CLI 将校准结果用于 hybrid 分工。Wllm 可参考 E5-04 的测量与禁用条件；本机 CPU/Iris Xe 共享 DRAM，必须重新建模，不能沿用独显 PCIe 成本。
2. **prefill 与 decode 使用不同策略。** `python/freetoken/moe/offload_cache.py` 含专家格式布局、LRU、双缓冲 prefill 及 hybrid 执行组织。参考 E3-02/E4-03/04，分别计入全层准备、有效命中、缓冲与同步；不能假设小批量 decode 和长输入同样获益。
3. **完整混合状态检查点。** `python/freetoken/kvcache/hybrid_radix_cache.py` 同时组织完整注意力 KV 与 GDN 状态快照，前缀匹配回退到仍有有效循环状态的边界。可参考 E3-03/E4-07；语义锚点并不授权近似匹配其他文本，Wllm 仍需 token/配置匹配及当前 F16 KV/F32 循环状态的恢复一致性。
4. **专家缓存与状态缓存共用预算。** README 描述运行时弹性显存管理，可供 E4-01/E5-05 研究。须先核实具体分配/迁移实现与安全生效边界；本轮只抽查缓存文件，未完成弹性管理源码审计。共享内存设备不能把缓存预算视为额外物理容量。
5. **格式与平台按入口核对。** 固定 `python/freetoken/models/gguf/config.py` 注册 `qwen35` 和 `qwen35moe`；这比仅有 MoE README 名称更具体，但仍不证明本地派生容器可用。`docs/install.md` 主路径要求 Linux x86_64/NVIDIA/CUDA 13，README 另列 Windows 桌面入口，模型/CLI 文档还有 WSL/WDDM 锁页内存处理；不能概括为完全不支持 Windows，也不能据此声称支持 Iris Xe。其 `cpu` 策略是专家分工，不自动证明 CPU-only 完整模型服务。

可追溯文件均使用下节 FreeToken SHA： [安装说明][ft-install]、[CLI][ft-cli]、[模型说明][ft-models]、[带宽校准源码][ft-bw]、[专家缓存源码][ft-cache]、[混合前缀缓存源码][ft-hybrid]、[GGUF 注册表][ft-gguf]。未在本机运行 FreeToken，也未改动 Wllm 默认模型或执行策略。

### 11.3 本次 GitHub 快照与许可证线索

SHA 是本次读取默认分支时的快照，不新增 Wllm 依赖锁定。API 的 `null`/`NOASSERTION` 不代表无许可证或可任意复制；复制前仍需核对具体文件与第三方来源。未归档也不等于已维护到可用或已完成兼容性验收。

| 项目 | 默认分支快照 SHA | 归档 / 来源状态 | 许可证线索 |
|---|---|---|---|
| FreeToken | `cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465` | 未归档 | API/README：Apache-2.0 |
| QuestWend | `86fc9da74a0d9d0ed4e70f57f0fada685a366ad7` | 未归档 | API：null，项目许可待核实 |
| FastLLM | `d04af65dca2770ea365fb4e7f353a08d4f9b5117` | 未归档 | API：Apache-2.0 |
| llamafile | `a3ccf098ed9c7c77687f5430b0143156d3bcc45a` | 未归档 | API：NOASSERTION；顶层 LICENSE 为 Apache-2.0，子组件另查 |
| OpenVINO GenAI | `9345dfd47d1b3db7c2ce439d9f1617e0ab1bf938` | 未归档 | API：Apache-2.0 |
| Lemonade | `05d32f0167a4136bd80aa38408d85d2d33dd6c81` | 未归档 | API：Apache-2.0 |
| flash-linear-attention | `4a8dd3f5fdfb84f066170c31fad86fe788532a83` | 未归档 | API：MIT |
| FlashInfer | `fd4de5acce18fd1c747c8425555f082dd4452cc3` | 未归档 | API：Apache-2.0 |
| ssdmoe | `f12bb92ec2826ce8dc7fce0e6ef7d14946be934a` | 未归档 | API：MIT |
| FlexLLMGen | `004ffef82b46e8dc8685c55d0cdda650bdaf1269` | 已归档；FlexGen 旧地址重定向到此 | API：Apache-2.0 |
| MNN | `85c9ba38cbf48ec9969a93a1375e20b353e4df90` | 未归档 | API：Apache-2.0 |
| KleidiAI | `6233410e039011aa7422d0cd7c36d0f8c59af239` | GitHub 只读镜像，主开发在 GitLab | API：null；源码 SPDX 与 LICENSES 提供 Apache-2.0/BSD-3-Clause 线索 |
| IPEX-LLM（另列历史线索） | `de6bce27133ab250f13fd5d549c197519ce16d30` | 已归档，见 11.5 | API：Apache-2.0 |

除 FreeToken 的固定文件链接外，其余行可按 `https://github.com/<owner>/<repo>/blob/<SHA>/<文件路径>` 复查。本次其余文件核对范围：QuestWend `README.md`、`docs/resident_decode.md`、`CMakeLists.txt`；FastLLM `README_EN.md`；llamafile `README.md`、`LICENSE`、`llamafile/sgemm.cpp`；OpenVINO GenAI/Lemonade/FlashInfer/FlexLLMGen/ssdmoe 的 `README.md`；flash-linear-attention `README.md`、`INSTALL.md`；MNN `README.md`、`transformers/README.md`、`transformers/llm/config.json`；KleidiAI `README.md`、`LICENSES/Apache-2.0.txt`、`LICENSES/BSD-3-Clause.txt`。文件树与注册表只作为定位和局部事实依据，不构成完整路径审计。

### 11.4 新增参考到任务的映射

| Wllm 任务 | 新增参考 | 下一步成立条件 |
|---|---|---|
| E2-02/03、E3-01 CPU 数值与量化矩阵核 | R35 FastLLM、R36 llamafile | 相同实际张量/精度下先查既有能力，再单项对照布局、分块和线程成本 |
| E2-04 后端、E5-01/03 能力矩阵 | R34 QuestWend、R37 OpenVINO GenAI、R38 Lemonade | 区分引擎入口、模型转换、设备和状态操作；不从后端枚举推定完整支持 |
| E3-03/04/05 状态、推测和循环热点 | R33 FreeToken、R34 QuestWend、R35 FastLLM、R39 FLA、R40 FlashInfer | 完整状态保存/恢复、取消和回滚先通过；MTP 仍需本地辅助头容器，不直接使用其他模型草稿 |
| E4-02/03/04/05 MoE 容量和分工 | R33 FreeToken、R34 QuestWend、R35 FastLLM、R41 ssdmoe | 先有 MoE 与可信路由参照；保持所有被选专家，排除常驻限定路由与剪枝；读取、计算和同步全计 |
| E4-01/06/07、E5-04/05 资源与动态预算 | R33 FreeToken、R38 Lemonade、R41 ssdmoe、R42 FlexLLMGen | Dense/MoE、单请求/吞吐、容量/速度分开；先测本机带宽与 I/O，再决定预算、重算、缓存或卸载 |
| E5-06 弱硬件和其他架构 | R43 MNN、R44 KleidiAI | 模型转换/精度单列，取得对应真实设备再验收；不扩大本轮 Windows/i5 的支持声明 |

以上只扩展参考与可选最小实验，不改变 8.5 的近期执行关口，不将所有新增项目变成待安装依赖，也不要求全部研究完才能继续有限优化。

### 11.5 历史项目与暂未纳入条目的线索

- [IPEX-LLM](https://github.com/intel/ipex-llm)：GitHub 与 API 均确认已归档（页面标注 2026-01-28），只保留旧 Intel CPU/XPU 路径参考，不推荐作为新增默认依赖；不把归档状态扩大成 OpenVINO 或所有 Intel 推理栈的状态。
- [SSD-LLaMA 论文](https://arxiv.org/abs/2609.18110)：本轮读到 SSD 原生 I/O、三层专家存储与完整路由执行的论文线索，但未找到并核实对应的作者 GitHub 实现，因此暂不编入 R33–R44；更不能把名称相近的 ssdmoe 当作该论文实现。
- GPU/Apple 实验、Intel 新一代 Core Ultra 基准和已有项目的配置封装不因搜索命中就逐个扩充参考编号。可作为后续检索线索；需要明确新增机制、可追溯版本和与当前设备/模型的区别，再进入统一表格。本次筛选不声称穷尽 GitHub。

[ft-install]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/docs/install.md
[ft-cli]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/docs/cli.md
[ft-models]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/docs/models.md
[ft-bw]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/python/freetoken/moe/benchbw.py
[ft-cache]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/python/freetoken/moe/offload_cache.py
[ft-hybrid]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/python/freetoken/kvcache/hybrid_radix_cache.py
[ft-gguf]: https://github.com/FlashML-org/FreeToken/blob/cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465/python/freetoken/models/gguf/config.py

## 12. DeepInfer 补充研究（2026-10-22）

**R45 [DeepSeek DeepInfer][di-readme]**，与 DeepInfra 云推理平台区分。研究快照为 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（提交时间 2026-10-21 04:55:43 UTC），通过官方 GitHub API、README、部署问答、基准说明及部分测试/执行源码核对。根目录 [LICENSE][di-license] 为 Apache-2.0；实际复用仍须逐文件核对第三方来源。这是参考快照，不新增 Wllm 依赖锁，也不表示已经移植、安装或在本机运行 DeepInfer。

### 12.1 项目范围与本机适用性

该快照 README 列出 DeepSeek-V4.1-Flash、NVIDIA SM100 GPU 与 Huawei Ascend 950 NPU；环境要求 Linux、Python 3.12、GCC 13 及对应 CUDA 13 / CANN 9.2。没有列出当前 Windows / Iris Xe / Qwen3.5 GGUF 路径，不能作为 Wllm 可直接替换的本机后端。公开版本提供离线推理；完整 KV 缓存系统、VE/Prefill/Decode 分离部署和 DP 副本间负载均衡列为未包含组件，不能把生产系统说明当作全部已公开实现。[项目范围][di-readme]

### 12.2 可借鉴机制与最小实验

| 机制 / 已核对依据 | 对 Wllm 的参考价值与最小实验 | 前提与边界 |
|---|---|---|
| 固定 token 轨迹与批次不变性：[test_batch_invariant.py][di-batch] 先批量生成，再用 `golden_token_ids` 单独重放，比较 token、logprobs 和 top-logprobs | 近期优先借鉴诊断方法：固定相同 token 输入，逐步比较 CPU/Vulkan logits；针对已发现的输出投影差异捕获相同输入、临时激活与归约结果，定位首处分歧 | 上游批次不变性不等于跨 CPU/GPU 逐位一致。固定种子、相同短文本或强制相同 token 都不能代替数值通过；仍用既定阈值，完整轨迹回放尚未在 Wllm 实现 |
| 无干扰基准：[examples/README.md][di-bench] 分离 prefill/decode、日志和 profiler，并指出 profiler 退出后的残余影响 | E2 基准增加受控测量段，与正常记录模式分开；完整用户请求仍计入实际日志/展示开销，保持加载、TTFT、pp/tg 和总耗时分别报告 | 其 decode 吞吐公式按每步全部草稿接受计算，不等于实际接受输出吞吐。Wllm 仍按实际接受 token 和整次净耗时验收；不套用上游倍数、并发度或热缓存口径 |
| 采样分布验证：[test_sampler.py][di-sampler-test] 用随机化 PIT 与 KS 检验检查采样分布 | 采样器/MTP 修改时，除 logits 和固定轨迹外增加分布测试；覆盖温度、top-p、接受/拒绝与失败样本 | 属于统计证据，不能替代逐算子数值、任务质量和完整回滚检查；测试规模与耗时按本机预算另定，不直接照搬上游夜间测试规模 |
| 置信度驱动推测验证：[schedule_mtp_varlen.py][di-varlen] 计算累计置信度，按总预算选择验证长度；[部署问答][di-qa] 结合硬件吞吐曲线说明策略 | 后续 E3-04 在固定主目标下，依据本机草稿、验证、同步与状态成本选择长度，并测试负收益禁用条件 | 当前 CUDA/TileLang 内核不能直接用于 Iris Xe；需要匹配的草稿/置信度来源及完整 F16 KV/F32 循环状态回滚，不能假定现有 Qwen3.5 MTP 文件自动具备 DSpark 能力 |
| CPU/GPU 异步调度：[llm_engine.py][di-engine] 与[部署问答][di-qa] 组织调度、执行、采样和后处理重叠 | E3-02 在剖析证明 host 调度/同步造成空闲后，再做单项流水实验，计入事件、复制、线程和缓冲成本 | 数据中心并发吞吐收益不自动适用于单用户共享内存设备；只在前向与取消/释放正确后比较，不能隐藏 CPU 工作或必要同步 |

### 12.3 采用边界与既有任务映射

[部署问答][di-qa] 明确说明 SWA 有界重放重建的是近似状态，且其低精度 KV/专家格式不同于当前基线。这些机制不直接进入 Wllm 的 Q4_K_M、16K、F16 KV/F32 循环状态保真实验；完整状态恢复仍须按 Qwen3.5 的实际方程与历史依赖验证。生产 KV 存储/负载均衡未完整公开，也不能作为现成的本地 RAM/SSD 缓存实现。

| 既有任务 | R45 的映射 | 验收次序 |
|---|---|---|
| E2-01 / E2-06 基线与回归；E2-04 GPU 正确性 | 固定轨迹、概率/数值比较、低干扰计量与口径核实 | 近期先补诊断并定位输出投影差异；数值/质量与资源通过后再比较性能，不放宽阈值或退回 4K |
| E3-02 执行计划与同步 | 异步调度、事件依赖、采样/后处理重叠 | 有热点证据才进入单项实验，保留同步参照及取消/错误释放检查 |
| E3-04 / E3-06 推测与组合 | 动态验证预算、采样分布测试、本机成本驱动禁用 | 先验证匹配草稿、完整状态与采样，再测实际接受输出的净收益；不将其列为全部优化的前置 |
| E3-03 / E4-07 状态复用与分层 | 缓存命中、重算及状态依赖的分析方法 | 只研究可保持当前状态语义的部分；近似 SWA 重放不自动采用，缓存命中不计为新计算吞吐 |

R45 补充方法与研究线索，不提升当前 GPU 候选状态。当前 16K CPU 容量验证与 GPU 数值失败以[本机报告](../reports/context16k-validation.md)为准；本次文档补充不构成 DeepInfer 本机复现、数值修复或性能验收。

[di-readme]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md
[di-license]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/LICENSE
[di-batch]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/tests/models/test_batch_invariant.py
[di-bench]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/README.md
[di-sampler-test]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/tests/models/test_sampler.py
[di-varlen]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py
[di-engine]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py
[di-qa]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md
