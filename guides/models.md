# 模型接入

Wllm 通过 llama.cpp / ggml 加载 GGUF 自回归文本模型。架构、词表、权重精度、专家数和状态布局取自模型与后端；接入不按品牌或 Qwen 版本设置白名单。普通生成与 Reflex 使用同一模型加载和资源管理路径。

## 指令与基础模型

指令模型使用 GGUF 中的原生 `tokenizer.chat_template`，例如：

```powershell
wllm --tui --model "D:\Models\qwen2.5-0.5b-instruct-q8_0.gguf" --context 4096
wllm --model "D:\Models\SmolLM2-135M-Instruct-Q4_K_M.gguf" --context 4096 --prompt "Hello"
```

基础模型可直接补全文本。`--raw` 不添加 system/user/assistant 对话包装，分词仍遵循模型的特殊 token 策略：

```powershell
wllm --model "D:\Models\base.gguf" --context 4096 --raw --prompt "The capital of France is"
```

模型没有模板或应用需要特定格式时，显式提供 UTF-8 Jinja 模板文件：

```powershell
wllm --model "D:\Models\model.gguf" --context 4096 --chat-template-file chat.jinja --prompt "Hello"
```

配置中的对应字段为 `raw` 和 `chat_template`（后者是 Jinja 源码字符串）。`--raw` 与模板覆盖互斥，且只用于普通生成。`--thinking on|off` 传递给模型模板；效果取决于模板及模型的训练方式。

## 文件与执行配置

- 按 GGUF 文件中的权重精度执行；不自动重新量化，不自动缩小上下文。F16、Q8_0、Q4_K_M 等由后端加载其实际张量类型。
- 分片 GGUF 选择 `*-00001-of-000NN.gguf`，其余分片放在同目录；程序检查完整性后交给原生加载器。
- 上下文默认 16384。较短上下文模型需显式设置 `--context`；超过模型训练上下文时给出错误。该入口不隐式启用长上下文外推。
- 稠密、MoE、注意力、循环及混合模型的具体架构由固定版本后端解析；原生加载、计算图建立和内存准入均需成功。
- 当前执行入口为自回归 decoder-only 文本计算。多模态模型的投影器、媒体预处理，以及编码器或扩散模型使用其他执行接口。

TUI `/info` 与结果中的 `model_identity` 展示实际模型架构、精度、参数量、模板与状态类型。`--resources report.json` 可导出原生无分配计算图的权重、状态和计算缓冲区预算；预算与运行时实测峰值含义不同。

## 实测文件

测试机器：Windows 11 x64，Intel Core i5-1135G7，16GB RAM，CPU AVX2。

| 模型 | 精度 | 上下文配置 | 完成检查 |
|---|---|---:|---|
| Qwen3.5-9B 文本 GGUF | Q4_K_M | 16384 | 生成、候选打分、System One、独立请求、终端取消恢复与导出 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | Q8_0 | 4096 | 生成、System One 三类题型、权重复用与请求隔离 |
| [SmolLM2-135M-Instruct](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF)，Llama 架构 | Q4_K_M / F16 | 4096 | 生成、System One 三类题型、权重复用与请求隔离 |

另以复制的 SmolLM2 文件移除模板元数据，验证缺失提示、原始文本补全与显式模板覆盖。模型测试检查执行和接口契约，任务准确率需使用业务数据单独评估。

`scripts/prepare_model.py` 是特定旧 Ollama Qwen3.5 容器的字节保留转换工具；标准 GGUF 直接加载，不经过该工具。
