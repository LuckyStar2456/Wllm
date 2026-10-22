# System One 文本协议适配验收

日期：2026-10-22。基于 `a950e70` 新增 JEV / TypeSafe 的 state/questions/answers 文本适配，不接受 ask。当前结论限定为本地 CPU 的协议、计算映射和少量功能检查；不证明与远程 JEV 的质量、概率或延迟相同。

## 工件

保持本机 Windows / i5-1135G7 / 16GB、既有 Qwen3.5-9B Q4_K_M 文本容器、16384 上下文、F16 注意力 KV / F32 循环状态和默认生成设置。第三方代码、模型和本机配置未改。llama.cpp 固定 `f2918cabbffe8abf8e2a90c1c089085fa116c2cf`；Release AVX2 EXE SHA256：`978d7bdfd40f1281d29458116f798e5600171a376964887f97c70e21b40dfe18`。新工件不继承旧构建的内存校准准入。

## 已验证

构建和 CTest 通过，原生共 79 项检查，覆盖三种题型的解析、结构化描述、choice/score 的公开 confidence 公式、noul 的 true 分数、无效概率映射、ask 拒绝及 choice 255/256 项解析边界。最大选项数尚未开展整模型质量/性能验收。

`tests/test_system_one.py` 的 6 项真实入口检查完整通过：

1. 一个 state 的 choice/noul/score 混合批次返回具名答案，内部单 token 打分、无生成 token、公式和结构化 legend 正确；请求使用 jev-latest 别名时响应仍标识本地 Qwen。
2. 重命名问题并加入无关常量题，原题结果及分布一致；题名和其他题目不进入其模型上下文。
3. 不支持的 box、缺失 state、ask 和未知模型别名明确报错；整批先校验，不先执行有效题。
4. 多行 stdin JSON、空数组 state 与单选项 choice 可用；常量选择 usage 为零，不假装执行模型。
5. 连续输入无效 JSON、开灯状态、关灯状态，可从无效请求恢复并分别返回 on/off；权重复用、上下文重建，最终释放另有记录。
6. 多题共享父超时预算；超时 124 和取消 130 均清空 answers，常量题已完成也不构成部分成功。

旧 `tests/test_candidate_score.py` 的 7 项模型检查在本轮 EXE 上完整复跑通过，包括 248,320 项完整 logits 与同前缀普通非思考参照逐字节一致、无生成/采样、标签重映射、驻留独立请求与 unknown、超时和 prefill 中合作取消。原始记录为 `reports/candidate-score-1792667899713738500/`。基准汇总、内存校准和资源相关的 19 项测试也通过，未放宽既定阈值。

原 `tests/test_integration.py` 的 8 项默认生成回归在本轮 EXE 上通过，覆盖原思考模板、完整思考回答、多批输入、上下文边界、内存拒绝、超时及生成中取消；对应本轮记录位于 `reports/acceptance/`。

PowerShell 的实际用户入口 `-Mode system_one -RequestFile configs/system-one.example.json` 通过，示例返回 `action.choice=turn_on`，处理输入 195 token、生成 0 token。响应与诊断分别保存在本地 `reports/system-one-wrapper.stdout.json` / `.stderr.txt`，该例也写入机器可读报告。

本机原始记录为 `reports/system-one-1792667730916626600/`，含请求、stdout/stderr、JSONL 和 results。混合样本为 488 个实际处理输入 token、0 输出 token；这是模板/题目/候选在内的逐题累计数，不是去重后的 state 长度。

旧候选入口、默认生成及包装脚本的当前构建复核结果见[机器可读报告](system-one-validation.json)。历史[候选首版报告](candidate-score-validation.md)绑定旧 EXE，原样保留，不以其数据冒充新构建验收。

## 失败记录与限制

首轮连续请求测试遇到系统可用 RAM 约 1007MiB、低于既定 1024MiB 余量时的准入拒绝；没有放宽阈值、降低精度或缩短上下文。该失败原始记录保留于 `reports/system-one-1792667489184657400/`。首轮超时夹具传入 `.01` 不符合现有 JSON 数字语法，改为 `0.01` 后重跑；最终 6 项完整复跑通过。首轮还包含后来按用户要求删除的 ask 别名，因此不能用首轮通过项作为最终接口验收。

本轮测试是有界功能诊断，背景负载和文件缓存未受控，运行中可用内存曾接近余量，部分运行提示可能换页。不会将这些计时作为正式性能采用依据。多题当前逐题完整处理同一 state，只有权重驻留，没有共享前缀缓存/并行；长 state、选项多和题目多会增加输入处理时间。

choice/noul/score 都由本地 Qwen 的候选归一化 logits 映射。confidence 按公开公式计算，仍不是已校准正确率。小样本灯光任务不证明广泛准确率、复杂推理能力或与完整思考/JEV 等价。默认 generate 仍保留思考和采样；结构化分支只在显式选择时启用。

研究候选为跨题缓存/批处理、任务校准和思考回退、HTTP 及视觉；本轮没有新增这些能力或获得加速倍数。使用方式和外部格式来源见[协议说明](../docs/system-one-protocol.md)。原始记录、模型、配置及 EXE 不随仓库分发。

## 复现

```powershell
./scripts/build.ps1 -Jobs 4
py -3.11 -B tests/test_system_one.py
py -3.11 -B tests/test_candidate_score.py
py -3.11 -B tests/test_integration.py
./scripts/run.ps1 -Mode system_one -RequestFile configs/system-one.example.json
./scripts/run.ps1 -Mode system_one -Interactive
```

模型检查串行运行，避免多个推理进程争用本机资源。标准输出为 JSON，阶段指标另走 stderr；取消和错误不触发自动重试或完整思考回退。
