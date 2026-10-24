# Reflex 决策模式

Reflex 使用模型输入末端的单 token 标签 logits 比较候选，直接返回结构化决策。普通生成与 Reflex 共用模型加载和 CPU 引擎。候选归一化分数用于当前题目内的相对比较。

## 候选打分

```powershell
.\wllm.exe --model "D:\Models\model.gguf" --mode candidate_score --candidates configs/light-switch.candidates.json --prompt "请把客厅的灯打开"
```

候选文件包含 `candidates` 列表，每项为 `id`、`label`、`description`；`label` 使用模型词表中的单 token 标签。`unknown_id` 指定拒答候选。

终端中使用 `/mode candidate_score`、`/candidates FILE`，再输入待判断文本。

## System One

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

执行使用本地模型；System One 格式也可供已有应用集成。决策响应记录输入计算量、输出 token 数和耗时；候选打分不生成回答文本。
