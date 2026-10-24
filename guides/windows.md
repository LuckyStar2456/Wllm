# Windows 使用说明

## 启动与模型

使用 Scoop 可直接安装并创建 `wllm` 命令：

```powershell
scoop bucket add wllm https://github.com/LuckyStar2456/Wllm
scoop install wllm/wllm
wllm --tui
```

`scoop update wllm` 升级，`scoop uninstall wllm` 卸载；`scoop prefix wllm` 显示安装目录，通用示例位于该目录的 `configs` 下。

将 Windows x64 压缩包解压到任意目录，在终端执行：

```powershell
.\wllm.exe --tui
.\wllm.exe --tui --model "D:\Models\qwen3.5-9b-q4_k_m-text.gguf"
```

进入界面后也可使用 `/model PATH` 选择 Qwen3.5 Q4_K_M 文本 GGUF。路径含空格时，`/model` 后可直接输入完整路径或使用双引号。按需 `/load`，或直接提交请求自动加载；`/unload` 释放权重。

普通生成、Reflex 共用本地引擎。连续请求复用模型权重，每个请求建立独立的推理上下文；输入历史不拼接到模型输入。思考开关、精度、采样及上下文由明确配置决定。默认上下文 16384、思考开启，KV F16、循环状态 F32。

## 输入和命令

| 操作 | 按键或命令 |
|---|---|
| 提交 | Enter |
| 换行 | Shift+Enter，输入视口用 `|` 表示换行 |
| 粘贴 | Ctrl+V、Shift+Insert；支持带换行的文本 |
| 编辑 | 左右方向键、Home、End、Backspace、Delete |
| 当前进程输入历史 | 上下方向键，最多 32 条 |
| 清空输入 | Ctrl+U；Ctrl+C 也可清空当前输入 |
| 取消推理 | Ctrl+C，完成清理后可继续请求 |
| 模型信息 | `/info`、`/model` |
| 模型加载/释放 | `/load`、`/unload` |
| 切换模式 | `/mode generate`、`/mode candidate_score`、`/mode system_one` |
| 选择候选文件 | `/candidates configs/light-switch.candidates.json` |
| 选择请求文件 | `/request configs/system-one.example.json` |
| 提交文件内容 | `/run PATH`；System One 可用 `/run` 提交已选请求文件 |
| 系统提示词 | `/system TEXT`；空参数清除 |
| 参数 | `/config`、`/set threads 4`、`/set thinking false` |
| 清理当前进程的输入历史、系统提示词和结果 | `/reset` |
| 上次结果及指标 | `/metrics` |
| 显式导出上次结果 | `/export result.json`；创建新文件 |
| 帮助/退出 | `/help`、`/exit`；空输入时 Ctrl+D 退出 |

运行期间展示加载和输入处理状态；流式生成的实时 token/速率/内存也显示在终端窗口标题中。结束后输出完整指标摘要。正文使用终端原生滚动，不切换全屏缓冲区；窗口变窄时输入区水平滚动。启动宽度小于 57 列时使用紧凑标题，`--ascii` 可显式选择紧凑标题。

设置 `NO_COLOR` 环境变量可关闭颜色。程序退出会恢复终端模式、代码页和窗口标题。

## CLI

```powershell
.\wllm.exe --model "D:\Models\model.gguf" --prompt "用中文解释质数。"
.\wllm.exe --model "D:\Models\model.gguf" --prompt-file prompt.txt --max-tokens 128
.\wllm.exe --model "D:\Models\model.gguf" --interactive
.\wllm.exe --config my-config.json --mode system_one --request-file configs/system-one.example.json
.\wllm.exe --config my-config.json --mode candidate_score --candidates configs/light-switch.candidates.json --prompt "打开灯"
.\wllm.exe --model "D:\Models\model.gguf" --prompt "你好" --record run.jsonl
```

默认不自动读取配置、不创建日志、不保存输入历史。`--config FILE` 显式加载配置，命令行参数覆盖配置。可复制 `configs/default.json` 并填写自己的模型路径。界面中的参数修改只在本进程生效。

生成文本及决策 JSON 写入 stdout；状态、指标与错误写入 stderr。重定向时不输出横幅或终端控制序列。退出码：0 成功，1 请求/配置错误，124 超时，130 取消。`--help` 查看完整参数。

模型文件由使用者提供。符合当前格式的文本 GGUF 可直接选择；Ollama Qwen3.5 容器转换工具见 `scripts/prepare_model.py --help`，工具保留主文本张量原始字节，输出独立文件，原文件只读。

## 源码构建

需要 Windows x64、AVX2 CPU、Git、Visual Studio 2022 的 C++ 桌面开发工具及 CMake 3.24+。

```powershell
git clone https://github.com/LuckyStar2456/Wllm.git
cd Wllm
.\scripts\build.ps1 -BuildDirectory build/public
.\build\public\bin\Release\wllm.exe --tui
```

依赖由 `dependencies.json` 固定到具体提交。构建脚本下载源码依赖，构建并运行原生测试；模型单独准备。

## 排查

- 模型路径或格式错误：核对路径、Qwen3.5 文本 GGUF 与 Q4_K_M 格式，使用 `/model` 重新选择。
- 内存预算不足：释放其他应用占用，或显式选择适合资源的上下文；程序保持用户指定的精度及上下文。
- 无颜色或块字：使用 `--ascii` 或 `NO_COLOR`；采用支持 Unicode 的等宽终端字体。
- 从脚本调用：使用 `--prompt`、`--prompt-file` 或 `--interactive`，将 stdout 与 stderr 分开捕获。
