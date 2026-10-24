# CPU 基础优化与复现实验

2026-10-24 新增 `prefix_cache_mib` / `--prefix-cache-mib N`（PowerShell `-PrefixCacheMiB N`）的有界 CPU 单条前缀快照实验，默认 0；只用于显式驻留会话，GPU/重排组合拒绝。真实恢复/数值/隔离和性能未验收，用户选择本轮保留关闭候选；现有单请求内存校准不能授权此会话，不能据本轮运行计时报告加速。默认路径有界回归、资源阻断和后续验收命令见[报告](../reports/prefix-cache-candidate.md)。

当前固定已有 Qwen3.5 Q4_K_M 文本权重、注意力 KV F16、循环状态 F32、思考和采样设置；2026-10-21 按用户最新要求将默认上下文从 4096 扩展为 16384。执行配置不修改模型或自动缩短上下文。下文历史 4K 样本保持原记录，不当作 16K 校准或性能证据。

2026-10-21 新增资源准入与负载准备，见[报告](../reports/resource-validation.md)。基准在预热/每次样本前调用原生资源探测，拒绝原因写入 schema 2 的 `resource_preflights`；拒绝时没有生成样本，不能填 0 耗时或宣称失败请求更快。旧可执行文件没有 `--resources`，需先构建当前版本。

同日后续增加 `--memory-diagnostic`：通过固定依赖的 `llama-ext.h` WIP 接口查询模型/上下文/计算缓冲区，在上下文就绪、prefill 完成和释放前记录；该接口受精确上游提交约束，不修改第三方源码。额外记录 Windows `PeakWorkingSetSize` 与 `PeakPagefileUsage`，后者作为进程提交峰值保存于 `process_peak_private_bytes`。这些是进程生命周期累计峰值，不能在驻留会话中冒充单请求增量，也不等于物理 SSD 读量。内存诊断与算子/logits 诊断一样，不进入性能汇总。

`scripts/calibrate_memory.py` 串行执行聊天/代码/长输入三组有界内存诊断，生成绑定模型 SHA256、可执行文件 SHA256、内存配置、硬件与负载文件的校准。基准只有显式传入 `--memory-calibration PATH` 才采用它；任何身份/配置改变、失败样本、长输入覆盖不足、非法峰值或不足的安全余量都会拒绝。正式请求前仍重新检查系统资源。校准预算为完整映射文件 + 实测进程提交峰值 + `max(256MiB, 峰值的25%)` + 原系统预留，不再次叠加已包含在进程峰值中的 KV/循环/计算缓冲，也不把进程工作集峰值当整份权重已驻留的证明。它仅适用于对应有界配置与负载，不是通用内存上限。

运行记录新增显式 `recurrent_state=f32`、系统提交量/额度、进程累计缺页和累计 I/O 字节。进程缺页包含软缺页，I/O 包含缓存与其他句柄传输，二者都不能当作物理 SSD 读量或硬缺页测量；驻留会话内这些进程累计数不会按请求归零。系统、进程和 GPU 是不同记账视图，不能相加。

`benchmarks/representative-workloads.json` 按用户确认覆盖聊天、代码、长文，分别提供人工质量检查点；长文仍为构造材料，不能代表用户真实文档。`scripts/inspect_workloads.py` 只核对词表/模板与长度，质量状态为 `not_evaluated`；基准中没有精确答案检查的任务标记 `manual_review_required`，执行完成不等于质量通过。固定 32 token 的长输入样本与自然 EOS 的完整任务分组记录。

2026-10-17 文档修订：状态精度由 `src/main.cpp` 的 KV 配置和固定 llama.cpp 提交 `f2918cabbffe8abf8e2a90c1c089085fa116c2cf` 的 `src/llama-model.cpp` 混合状态构造核实。历史报告中的“F16 状态”是笼统表述，应按上述两类精度解读；历史样本与数值保持原样。本次未运行新推理或新增性能结果。

## 本机指令集构建

```powershell
# 通用起点，保持首版 AVX2 构建方式
./scripts/build.ps1 -CpuTarget avx2
# 仅研究候选：本轮 AVX512 未通过数值门槛，不作为推荐运行版本
./scripts/build.ps1 -CpuTarget avx512
```

两个选项共用 `build/` 输出位置；切换会重新编译相关后端并替换程序。需要保留对照时先复制旧程序。本机已通过 CPU 与操作系统能力核查，不代表其他机器可运行这个 AVX512 版本。程序在加载后端前检查必要能力，不满足时明确报错。默认构建脚本仍选 AVX2，运行记录的 `cpu_build` 标注实际构建。

AVX512 复用固定上游实现中的量化算子，改变向量指令宽度与计算顺序，不改权重、KV 精度或上下文。跨指令构建另用完整 logits 数值容差（绝对 `1e-3`、相对 `1e-4`）及原始 softmax KL 上限 `1e-6` 检查；不要求跨指令逐 bit 一致，也不将少量前缀检查等同于完整任务质量证明。

本次 AVX512 诊断观察到最大绝对 logits 差异约 1.224、KL 约 0.000139，超出预设门槛。虽然该样本的 32 个生成 token 相同，仍未采用，当前交付程序已恢复为 AVX2；需要进一步解释差异并完成质量评测后才能重新考虑。

## 复用模型加载

```powershell
./scripts/run.ps1 -Interactive
```

连续输入独立问题时，模型权重保留在进程中。每次仍完整处理输入、重新建立 KV/循环状态和采样器，不复用前缀、不携带之前的问题，不改变上下文预算。空行、`/exit` 或 EOF 释放模型；错误、取消、超时也结束会话。详细计量与退出约定见[使用说明](quickstart.md)。

`tests/test_session.py` 对三轮“单独启动”和“驻留模型”交替分组验证，检查生成 token 相同、请求计数重置、上下文不串扰、异常与取消后的释放。后续请求的加载和首 token 改善单独报告，不能宣称首次冷启动同样获得加速；最终模型释放耗时也必须计入整段会话成本。

## 执行控制

| 配置 / 原生命令参数 | 含义 |
|---|---|
| `threads` / `--threads` | 单 token 生成线程数，1–256 |
| `threads_batch` / `--threads-batch` | 多 token 输入计算线程数；0 表示沿用 `threads` |
| `batch` / `--batch` | 输入批次大小，不得超过所选上下文 |
| `persistent_threadpool` / `--persistent-threadpool on\|off` | 在一次请求内复用 CPU 工作线程；结束、异常和取消时释放 |
| `prefill_last_only` / `--prefill-last-only on\|off` | 中间输入批次更新完整 KV/循环状态，只在最后一批请求输出 logits |

复用线程池采用同一个容量为两阶段最大线程数的池，后端按当前阶段选择参与线程数。它不是模型常驻或跨请求缓存。跳过中间 logits 不跳过输入 token，也不裁剪注意力或循环状态。

当前默认仍为 4 线程、输入线程沿用、批次 128；线程池复用和仅最后一批 logits 均保持关闭。这两项已通过基础正确性检查，但本轮筛选未显示稳定性能收益，保留为显式实验开关。

### 按阶段和算子定位耗时

```powershell
./scripts/run.ps1 -Prompt '解释计算机如何执行程序' -MaxTokens 32 -Profile
# 更细的诊断会改变执行方式；只用来找候选热点
./scripts/run.ps1 -Prompt '解释计算机如何执行程序' -MaxTokens 16 -ProfileOps
```

默认不启用剖析。`-Profile` / 原生 `--profile` 记录输入计算、单步生成计算、采样、文本转换与输出的调用次数和墙钟耗时；计算范围包含 `llama_process` 和显式后端同步。结束时在 stderr 显示，并写入 JSONL 的 `end.profile.stages`。它不包含加载、批次构造、所有调度间隙和卸载，因此各项之和不等于整次请求耗时；生成计算次数也不等于输出 token 数（第一个输出来自输入计算，末尾通常无需再计算）。

`-ProfileOps` / `--profile-ops` 额外按输入/生成阶段、算子类型、第一输入的数据类型汇总到 `end.profile.operators`。它让调度器逐节点执行并同步，可能阻止融合，测量包含调度/同步开销。算子占比只在本次诊断内解释，不能作为正常运行的精确耗时占比，更不能作为优化加速证据。真正的性能对照须关闭两种剖析及 logits 导出。

`start.profiling_diagnostic` 与 `start.operator_profiling` 明确标记诊断模式；没有启用时 `end.profile` 为 null。驻留请求各自重新计量，失败保留已完成的统计；中途终止的计算调用计入阶段耗时，但未完成的节点不计入算子汇总。采样次数可能包含最终 EOS，文本输出次数不包含 EOS。

### CPU 权重重排候选

新增 `cpu_weight_repacking` / `--cpu-weight-repacking on|off` / `-CpuWeightRepacking on|off`，默认 false。它允许上游为适用张量选择 CPU 重排缓冲，改变内存布局和算子路径，不改磁盘 GGUF 或重新量化权重；不能据此假设浮点结果相同或内存用量降低。

本机 AVX2 首个固定前缀的最大 logits 差异约 1.563、原始 softmax KL 约 0.0005563，超出实验前沿用的 `atol=1e-3`、`rtol=1e-4`、`KL<=1e-6` 门槛。即使该样本的 32 个生成 token 相同，仍停止采用和性能评选。它只保留为显式研究选项，不属于通过验收的优化配置；数值门槛失败也不等同于已经证明任务质量降低。

复查可用 `tests/test_cpu_builds.py --baseline <保留的AVX2程序> --candidate-repacking`；该命令当前预期会在数值门槛报错。模型加载复用、上下文预算和默认采样均不因此改变。

PowerShell 入口支持同样的控制，例如：

```powershell
./scripts/run.ps1 -Threads 4 -ThreadsBatch 4 -Batch 128 -Prompt '你好'
# 显式回到 0.1 的执行方式，保留当前模型、上下文和采样配置：
./scripts/run.ps1 -Threads 4 -ThreadsBatch 4 -Batch 128 -PersistentThreadpool off -PrefillLastOnly off -Prompt '你好'
```

有效配置写入每次 JSONL 的 `start.config`，其中 `threads_batch` 为解析后的实际线程数。`prefill_batches` 和 `prefill_logits_batches` 分别记录完成的输入批次与请求输出概率的批次。它们是请求计数，不是底层算子剖析结果。

## 对照工具

Python 3.11 标准库即可运行，无需额外 Python 依赖。先构建程序并准备 `configs/local.json`：

```powershell
# 单项开关和组合消融，每组 3 次，顺序逐轮反转
py -3.11 scripts/benchmark.py --cases short medium --repeats 3
# 分阶段线程及批次候选
py -3.11 scripts/benchmark.py --profiles benchmarks/thread-profiles.json --cases short medium --repeats 3
# 较长输入、完整回答单独运行，避免与固定输出步数混比
py -3.11 scripts/benchmark.py --cases long completion --repeats 3
```

`benchmarks/workloads.json` 定义合成短、中、长输入及简单算术完整回答。实际 token 数以运行记录为准；长输入为重复事实，仅用于计算与边界测试，不能代表长文理解质量。固定步数样本必须达到约定输出 token 数，提前 EOS 不算成功的固定步数样本。完整回答必须自然 EOS 且匹配预设答案。

配置清单可为每个候选指定 `exe`，用于比较旧二进制。未指定时使用当前构建。候选只能覆盖线程、批次及上述开关，不能在对照中改模型、上下文、量化或采样。种子和输出上限由同一个工作负载指定。

报告位于新建的 `reports/cpu-benchmark-*/summary.json`，包括二进制与模型 SHA256、输入摘要、有效配置、每次记录、失败数、成功样本的中位数和最小/最大值。原始输出和 token 记录留在本地，不随仓库分发。无样本的指标不填 0，失败/超时不混入成功性能统计。

新增 `benchmarks/extended-workloads.json`：长输入固定 16 token、短输入固定 128 token 输出，以及长输入中的单事实提取（自然 EOS 且核对答案）。复现本轮筛选：

```powershell
py -3.11 scripts/benchmark.py --profiles benchmarks/long-input-profiles.json --workloads benchmarks/extended-workloads.json --cases long long-output long-retrieval --repeats 1
```

这是单轮筛选，不能直接提升为默认配置或用单样本统计稳定收益。重复文本和单个事实只能检查计算边界及有限检索能力，不代表真实文档理解质量。基准工具拒绝将标记为剖析或 logits 导出的诊断记录纳入成功性能统计。

每个候选先单独预热，再串行交替运行；各请求重新启动进程，不复用前缀。操作系统文件缓存和后台程序未受控，因此不能将其称为冷盘测试，或根据小样本报告可靠 P99。

## 正确性与采用条件

```powershell
py -3.11 tests/test_cpu_optimizations.py
py -3.11 tests/test_integration.py
py -3.11 tests/test_benchmark.py
py -3.11 tests/test_session.py
# 旧 AVX2 0.2 二进制需先自行保存；该文件不随仓库分发
py -3.11 tests/test_cpu_builds.py --baseline reports/e2-baseline/wllm-avx2-v0.2.exe
```

执行改动用相同输入前缀的完整词表 logits 做数值对照（预设绝对/相对容差均为 `1e-5`），并比较固定种子下的生成 token、正文和多批次状态连续性。`--logits-file PATH` 可导出最后一批输入的完整 float32 小端 logits；仅用于诊断，带导出的时间不纳入性能比较。它包含输入相关信息，应按本地推理资料保管。

性能筛选预设 5% 的改善/回归界线，分别检查加载、首 token、输入处理、生成与总耗时，并查看内存和重复样本波动。超过噪声且可重复的收益才构成采用依据。线程/批次切换可能改变浮点计算顺序；同种子文本不同不自动等于质量退化，但没有充分数值与任务检查的配置不得提升为默认。

上述 5% 是既有 CPU 实验的筛选界线，不是产品“明显加速”的承诺。后续实验先固定代表负载、数值/质量条件、重复次数、噪声及允许回归，再确定体验目标；未测候选不填写预计 tok/s 或套用外部倍数。近期资源核实、后端对照和状态能力检查按[实施计划 8.5](implementation-plan.md#85-执行顺序交付与追踪)推进；失败的重排候选保持关闭，其修复不阻塞其他独立实验。

本轮属于 E2 的 CPU 第一批实验。尚不包含 GPU 后端、跨引擎完整性能比较、带历史的会话缓存或推测解码，也不承诺某组参数适用于所有输入或电脑。

`benchmarks/build-profiles.json` 保留首版与 AVX512 实验构建的比较定义，因数值门槛失败，未开展后续重复性能验收。它明确引用本地保存的 `reports/e2-baseline/wllm-v0.1.exe` 和 `wllm-avx512-experimental.exe`，不会误用已恢复的 AVX2 程序。新克隆不包含这些文件；旧版应从 `deeceda` 在独立检出中构建，且使用旧版支持的配置字段。不要用新程序冒充旧版基线。
