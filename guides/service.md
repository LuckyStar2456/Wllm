# HTTP 服务

## 启动

同一进程同时开放文本生成和 System One，指定监听端口：

```powershell
wllm --serve --service all --host 127.0.0.1 --port 8080 --model "D:\Models\model.gguf" --context 4096
```

| 服务类型 | 开放的推理接口 |
|---|---|
| `openai` | `/v1/chat/completions`、`/v1/completions` |
| `systemone` | `/v1/systemone` |
| `all`（默认） | 上述全部接口，共用一份模型权重 |

`--model-alias NAME` 设置客户端使用的模型名，默认 `wllm-local`。`--host` 默认 `127.0.0.1`，`--port` 默认 `8080`；端口范围 1–65535，已占用时启动失败。`--host 0.0.0.0` 显式监听全部 IPv4 网卡，客户端使用机器的实际 IP 访问。模型在启动时加载，准备完成后开始处理请求；终端 `Ctrl+C` 停止服务并释放模型。

`GET /health` 返回就绪状态、服务类型、模型别名、上下文和忙碌状态。`GET /v1/models` 返回当前模型列表。服务类型控制推理端点，这两个查询端点始终开放。

## 认证

可从指定环境变量读取本地 Bearer token：

```powershell
$env:WLLM_API_KEY = 'replace-with-your-local-token'
wllm --serve --service all --port 8080 --api-key-env WLLM_API_KEY --model "D:\Models\model.gguf" --context 4096
```

启用后所有端点要求 `Authorization: Bearer <token>`。该 token 用于本地服务认证，不是 OpenAI 或 Jev 的云端凭据；服务不会调用远端模型。传输为 HTTP，认证配置不写入文件。

## 文本生成

通用客户端设置 Base URL 为 `http://127.0.0.1:8080/v1`，模型名为 `wllm-local` 或显式设置的别名。

客户端若要求填写 API key，未启用服务认证时可使用本地占位值；启用 `--api-key-env` 时填写对应的本地 token。

`POST /v1/chat/completions`，`Content-Type: application/json`：

```json
{
  "model": "wllm-local",
  "messages": [
    {"role": "system", "content": "请简洁回答。"},
    {"role": "user", "content": "什么是质数？"}
  ],
  "max_tokens": 128,
  "stream": false
}
```

响应包含 `id`、`object: "chat.completion"`、`created`、`model`、`choices[].message`、`finish_reason` 和 `usage`。输入和输出 token 数按实际模型分词与执行计量。

兼容的文本请求字段：

- `messages`：1–256 条 `system`、`user`、`assistant` 消息；内容为字符串或 `type: "text"` 的文本块数组。完整采用调用方提供的历史，每次请求重新建立上下文。
- `max_tokens` 或 `max_completion_tokens`：显式正整数输出上限，两者互斥。省略时使用启动配置。
- `temperature`、`top_p`、`presence_penalty`、`frequency_penalty`、`seed`：覆盖当前请求的采样参数；`n` 为 1。
- `stop`：一个字符串或最多四个字符串；支持跨 token 匹配，返回正文不包含停止串。
- `stream: true`：返回 SSE，逐步输出 `chat.completion.chunk`、`choices[].delta`，结束发送 `data: [DONE]`。`stream_options: {"include_usage": true}` 在结尾额外返回计量块。
- Wllm 扩展：`thinking`、`top_k`、`min_p`、`repeat_penalty`，沿用本地参数含义。

`user` 可作为调用方标识传入；`store` 为 false，`response_format` 为 `{"type":"text"}`。未支持的功能或字段返回明确的 400 JSON 错误，接口兼容范围以上述文本字段为准。接口形状参照 [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 和 [SSE 流式事件](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。

`POST /v1/completions` 接收字符串 `prompt`，使用模型原始文本补全路径，响应为 `choices[].text`；支持相同的采样、停止串、SSE 参数及 `max_tokens`。

## System One 决策

`POST /v1/systemone` 使用 `state + questions`，业务选项放在 `criteria`：

```json
{
  "model": "wllm-local",
  "state": {"request": "请开灯"},
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "选择应执行的动作。",
      "criteria": {"on": "开灯", "off": "关灯", "clarify": "澄清需求"}
    }
  }
}
```

响应返回 `answers.action.choice`、各选项 `probabilities` 和 `confidence`，并包含实际模型身份与 `usage`。内部标签由引擎自动映射；候选决策不生成答案 token，`output_tokens` 为 0。多题按顺序处理，权重共用、上下文独立；任一题失败时不返回部分成功答案。

支持 `choice`、`noul` 和 `score`，详见 [System One 决策说明](systemone.md)。路径和请求结构参照 [TypeSafe System One](https://docs.typesafe.ai/api/system-one)，已有客户端可修改 Base URL 对接本地服务。请求中的 `model` 为记录的调用方别名，实际模型由启动参数决定。

## 生命周期与错误

同一进程同时处理一个推理请求。健康查询保持可用；忙碌时其他推理请求返回 **429** 和 `Retry-After: 1`。断开连接取消该请求；启动参数 `--timeout SECONDS` 限制每个请求的总推理时间。模型权重在请求之间常驻，每次请求的上下文、采样器和输入历史独立。

默认不创建配置、请求记录或会话文件。`--record FILE` 可显式启用记录。请求体上限为 4 MiB，上下文和模型精度不会自动缩减；模板、上下文及线程数由启动配置确定，客户端不能通过请求切换本地模型文件。

| HTTP 状态 | 含义 |
|---|---|
| 200 | 正常 JSON 或 SSE 响应 |
| 400 | 请求格式、字段、候选标签或上下文输入错误 |
| 401 | Bearer 认证失败 |
| 404 | 模型名错误、端点不存在或未开放 |
| 413 | 请求体超过上限 |
| 429 | 推理槽忙碌 |
| 500 / 503 | 执行失败 / 内存准入不足 |
| 504 | 推理超时 |

普通 JSON 错误包含 `error`；System One 执行错误保留其 `answers`、`usage` 和诊断结构。SSE 发送响应头后发生的错误通过流内 `error` 事件对象返回，再结束流。
