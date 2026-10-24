# Wllm

![W-LLM](assets/banner.png)

**Windows 本地文本推理与 Reflex 快速决策工具。** C++ 实现，基于 llama.cpp/ggml；提供纯终端界面和可供脚本调用的 CLI，使用 CPU 即可运行。

[下载 Windows x64 版](https://github.com/LuckyStar2456/Wllm/releases/latest) · [使用说明](guides/windows.md) · [Reflex 协议](guides/reflex.md) · [贡献指南](CONTRIBUTING.md)

## 功能

- **纯终端界面**：蓝色实心大字、深蓝阴影，沿用终端背景；原生滚动输出、窄窗口紧凑标题、无颜色模式。
- **输入编辑**：中文、光标移动、多行输入与粘贴、当前进程输入历史。
- **本地生成**：流式文本输出、system 提示词、思考开关、上下文及采样设置。
- **Reflex**：单 token 候选打分与拒答；System One 的 `state/questions/answers`，支持 choice、noul、score。
- **模型管理**：选择本地 GGUF、加载与释放、连续请求复用权重。
- **运行计量**：加载、输入处理、首 token/决策耗时、生成速率、token 数及内存摘要。
- **交互控制**：参数查看与修改、Ctrl+C 取消、超时、错误后继续操作、显式结果导出。
- **脚本集成**：文本/JSON 输出、文件输入、stdout/stderr 分离、明确退出码。

## 快速开始

解压 Windows x64 发布包，在 PowerShell 或 CMD 中执行：

```powershell
.\wllm.exe --tui --model "D:\Models\qwen3.5-9b-q4_k_m-text.gguf"
```

也可直接启动 `wllm.exe`，进入界面后使用 `/model PATH` 选择模型。输入文本并回车开始推理，`/help` 查看命令。

```text
/model D:\Models\qwen3.5-9b-q4_k_m-text.gguf
/load
/set threads 4
用中文解释质数。
/mode system_one
/run configs/system-one.example.json
/unload
/exit
```

CLI 单次调用：

```powershell
.\wllm.exe --model "D:\Models\model.gguf" --prompt "你好"
.\wllm.exe --model "D:\Models\model.gguf" --mode system_one --request-file configs/system-one.example.json
```

模型文件由使用者提供，当前模型入口为 **Qwen3.5 Q4_K_M 文本 GGUF**。CPU 发行包面向 Windows x64、AVX2 指令集。默认上下文 16384、思考开启；模型精度、上下文及推理参数由使用者明确选择。

## 无状态交付

- 解压后运行，模型路径通过参数或界面指定。
- 默认不写日志、配置或输入历史，不自动读取本机配置。
- 请求之间共享已加载权重，每次请求独立建立上下文；历史输入不自动发给模型。
- `/export FILE`、`--record FILE` 显式保存结果或指标，`--config FILE` 显式读取配置。
- 发布包只包含程序、通用示例、使用文档、品牌资源与依赖许可证。

## 测试配置

本版在 **Windows 11 x64、Intel Core i5-1135G7、16GB RAM、CPU AVX2** 配置上测试。

| 检查 | 配置与内容 |
|---|---|
| 原生测试 | 配置校验、计量、候选打分、System One 请求与回答映射 |
| CLI | 空目录启动、无默认状态文件、显式配置/记录、Unicode 路径、JSON 输出、重定向 |
| 终端 | Windows ConPTY，PowerShell/CMD，38/48/80/100 列；横幅、编辑、粘贴、历史、缩放和退出恢复 |
| 模型 | Qwen3.5-9B Q4_K_M 文本 GGUF，约 5.15 GiB；16K 配置、文本生成、Reflex、权重复用与独立请求 |

## 从源码构建

安装 Git、Visual Studio 2022 C++ 桌面开发工具及 CMake 3.24+：

```powershell
git clone https://github.com/LuckyStar2456/Wllm.git
cd Wllm
.\scripts\build.ps1 -BuildDirectory build/public
.\build\public\bin\Release\wllm.exe --tui
```

依赖版本锁定于 `dependencies.json`。测试与打包方法见 [CONTRIBUTING.md](CONTRIBUTING.md)，具体命令和输入操作见[使用说明](guides/windows.md)。

## 开源与致谢

Wllm 使用 [MIT](LICENSE) 许可证。推理引擎复用 [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp)，JSON 使用 nlohmann/json；依赖说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型遵循各自许可证。

终端命令与用户文档组织参考 [OpenCode](https://opencode.ai/docs/tui/)、[Ollama](https://github.com/ollama/ollama)，启动横幅呈现参考 Spring Boot。
