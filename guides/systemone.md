# System One 决策模式

System One 使用模型输入末端的单 token 标签 logits 比较候选，直接返回结构化决策。普通生成与 System One 共用模型加载和 CPU 引擎。候选归一化分数用于当前题目内的相对比较。

## 请求与响应

请求使用 `state`、`questions`，响应返回具名 `answers`：

```json
{
  "state": {"user_request": "请把客厅的灯打开"},
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "根据状态选择动作。",
      "criteria": {"turn_on": "打开灯", "turn_off": "关闭灯"}
    }
  }
}
```

```powershell
.\wllm.exe --model "D:\Models\model.gguf" --mode system_one --request-file configs/system-one.example.json
```

终端中使用 `/mode system_one` 后粘贴 JSON，或 `/run FILE`。`choice` 做互斥选择，`noul` 对每个条件分别判定，`score` 对有序级别计算期望分数。标签自动映射；多题按顺序独立计算并复用权重。父请求统一超时和取消，错误返回错误对象。

HTTP 服务使用 `--service systemone`，端点为 `POST /v1/systemone`，请求体与 CLI 完全一致。详见[HTTP 服务说明](service.md)。

执行使用本地模型；System One 格式也可供已有应用集成。决策响应记录输入计算量、输出 token 数和耗时；候选打分不生成回答文本。

## 底层 CLI 兼容入口

```powershell
.\wllm.exe --model "D:\Models\model.gguf" --mode candidate_score --candidates configs/light-switch.candidates.json --prompt "请把客厅的灯打开"
```

候选文件包含 `candidates` 列表，每项为 `id`、`label`、`description`；`label` 使用模型词表中的单 token 标签。`unknown_id` 指定拒答候选。

终端中使用 `/mode candidate_score`、`/candidates FILE`，再输入待判断文本。

## 跨模型接入

System One 使用所选 GGUF 的原生词表和聊天模板，定位 assistant 正文续写边界，比较该位置的候选 logits。System One 自动挑选并验证可用标签；手工 `candidate_score` 标签必须在完整实际输入末端保持前缀不变，且恰好追加一个普通 token。不同词表可能需要不同标签，包括带前导空格的标签。

请求中的可选 `model` 是调用方别名，记录在 `_wllm.requested_model`；实际执行模型由 `--model`、显式配置或 TUI `/model` 选择，响应给出实际模型身份。该字段不会下载或切换模型。

模型需要支持明确的 assistant 正文续写模板；原始文本补全模式用于普通生成。候选分数是当前候选集合内的相对偏好，模型、模板与候选变化都可能影响数值。选择合适的指令模型，并以实际任务评估决策质量。
