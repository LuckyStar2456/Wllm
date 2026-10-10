# Wllm 0.2 使用说明

本机 Windows / CPU 文本路径。默认上下文 **4096 token**，用户可以显式调整；程序不会因内存不足或输入过长自动缩短上下文、截断输入或改变精度。默认思考开启，采用现有模型记录的 temperature=1、top_k=20、top_p=0.95、presence_penalty=1.5。

## 准备与构建

在 Wllm 根目录使用 PowerShell。需要 Visual Studio 2022 的 C++ x64 工具与 CMake，以及 Git。运行时是原生 C++ 程序，不依赖 Ollama 服务。

```powershell
./scripts/build.ps1
```

构建脚本检查 `dependencies.json` 锁定的 llama.cpp 提交。下载仅涉及源码依赖，不下载新模型；上游源码位于 `vendor/llama.cpp`，当前 CPU AVX2 构建供本机使用，不代表其他硬件已验证。

已有 Ollama GGUF 使用了不同的字段格式和内嵌视觉张量，需要先准备文本容器：

```powershell
py -3.11 scripts/prepare_model.py
```

适配工具使用 Python 3.11、NumPy 及锁定源码内的 gguf 模块。当前环境已有 NumPy；其他环境请先在独立 Python 环境安装 gguf 模块所需依赖。工具仅接受已记录的源文件 SHA256，原文件只读。它调整 KV 头数与位置编码字段格式、重命名 SSM 偏置，排除首期不用的视觉等非文本张量；不做重量化或修改文本张量数据。每个保留张量按类型、形状、SHA256 校验，报告位于 `reports/model-adaptation.json`。派生 GGUF 占用额外磁盘，输出到 `models/`，不能把文件变小当成推理加速或参数档位变化。

首次成功准备后生成 `configs/local.json`；已有配置不会被覆盖。也可使用符合上游格式的 Qwen3.5 Q4_K_M 文件显式配置路径，但当前验收仅针对本机模型。

验收机器已完成准备，可直接运行；新克隆的仓库需要执行上述构建与模型准备步骤。适配工具会拒绝覆盖已存在的模型输出，已准备的机器无需重复执行。

## 运行与上下文调整

```powershell
./scripts/run.ps1 -Prompt '用中文简短说明什么是推理引擎。'
./scripts/run.ps1 -Context 8192 -Prompt '计算1+1。'
./scripts/run.ps1 -PromptFile ./prompt.txt
./scripts/run.ps1 -Interactive
```

连续提问模式保留已加载权重，每行发起一个独立请求，输入 `/exit`、空行或 EOF 退出并释放模型；Ctrl+C、取消、超时或请求失败也会退出并释放。该模式不累积聊天历史：每次重新建立上下文、KV/循环状态和采样器，不通过丢弃历史冒充对话缓存。首个请求仍需加载模型，后续请求可省下重复加载。等待输入期间会继续占用模型内存。

记录中的 `model_reused` 区分首次加载与复用，`load_s` 仍计入本次模板与上下文准备耗时。每个请求单独计时，不把用户输入等待算作推理耗时；最终 `session_end.model_release_s` 单列会话结束时的模型释放耗时。单次模式保持请求完成即释放模型。

分阶段线程、批次、执行开关、构建候选和对照方法见 [CPU 基础优化](cpu-optimization.md)。

默认 4K 是当前用户选择，替代此前讨论过的 2048/262144 默认值。上下文预算包含输入与生成内容；如输入已占满预算会报错，生成触及预算以 `context_limit` 结束并明确显示，不滑动或丢弃历史。可选范围为 512–262144，是否可运行受实际资源限制。

直接使用可执行文件可获得全部入口：

```powershell
./build/bin/Release/wllm.exe --help
./build/bin/Release/wllm.exe --config configs/local.json --context 4096 --prompt '你好' --max-tokens 128
```

`--max-tokens` 是用户显式输出上限；默认 -1 表示生成到模型停止标记或上下文边界。`--timeout` 显式设置整次请求时限，默认不设限。Ctrl+C/Break 中止；测试及自动化还可使用 `--cancel-file 路径`，创建该文件触发同一中止路径。`--thinking off`、`--temperature`、`--seed` 等是可选覆盖，会在有效配置记录中明确展示，不用于默认加速。

## 数值显示与记录

交互终端使用独立的回答区和指标区，约每 500 ms 刷新；长回答显示末尾，退出后保留完整生成内容。重定向时，标准输出仅输出模型生成内容，标准错误显示计量行。指标包括阶段、总耗时、加载耗时、输入批次进度和吞吐、首 token 延迟、生成 token 数、最近两秒速度、平均持续生成速度、工作集、私有提交量和系统可用内存。`--` 表示事件尚未发生或样本不足，不是 0。吞吐计入思考 token，不按字符或终端刷新次数计数。

平均持续生成速度按 `(N-1)/(最后一个token时刻-首token时刻)` 计算，EOS 控制 token 不算可输出 token。首次输出不足两 token 时没有平均速度。窗口速度在停顿后会下降，不保留过期读数。工作集峰值是采样峰值，不能当成精确瞬时峰值。

```powershell
./build/bin/Release/wllm.exe --config configs/local.json --prompt '你好' > answer.txt 2> metrics.txt
```

正常、取消、超时及运行错误均保存结束摘要；详细数据默认追加到 `reports/runs.jsonl`。`start` 记录有效配置，`sample` 记录资源与阶段快照，`end` 记录汇总、token 时间与 ID，可复算速度。模型正文默认不额外写入计量记录，但 token ID 可还原内容，应按本地输入输出资料保管。`--record` 可以指定独立记录文件。`--quiet-metrics` 仅关闭周期显示，仍采样、记录并显示最终摘要。

后端诊断写入 `reports/backend.log`。`--inspect 文件.json` 只加载词表和模板、保存格式化提示词及 token ID，不执行推理，不能作为生成验收。

需要定位慢在哪里时，可使用 `./scripts/run.ps1 -Prompt '你好' -MaxTokens 16 -Profile`，结束时查看输入计算、生成计算、采样和输出耗时。`-ProfileOps` 会改变图执行方式，只用于算子诊断；这两种模式不用于性能验收，详细边界见 [CPU 优化说明](cpu-optimization.md)。

## 当前范围

默认每次进程处理一个文本请求；`-Interactive` 可在同一进程处理多个独立请求。当前不含 GUI、HTTP 服务、带历史的多轮对话、视觉或推测解码。模型加载使用内存映射，默认关闭额外 CPU 权重重排缓冲；该研究开关未通过本机数值门槛。空闲 RAM 较少时仍可能发生换页。资源检查是保守估计，不是运行内存的精确预报。错误时不会结束其他应用或修改 Ollama 配置。

本机验证结果和限制见 `reports/prototype-validation.md`；没有跨引擎加速结论。
