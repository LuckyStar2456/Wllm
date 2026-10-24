# Wllm

<p align="center">
  <img src="assets/banner.png" width="640" alt="W-LLM 蓝色立体字符横幅">
</p>

<div align="center">

**在终端里运行本地模型，生成文本，直接做出决策。**

C++ · llama.cpp / ggml · CPU AVX2 · Windows x64

[![Release](https://img.shields.io/github/v/release/LuckyStar2456/Wllm?color=409eff)](https://github.com/LuckyStar2456/Wllm/releases/latest)
[![Windows](https://github.com/LuckyStar2456/Wllm/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/LuckyStar2456/Wllm/actions/workflows/windows.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-409eff)](LICENSE)
[![Scoop](https://img.shields.io/badge/Scoop-custom_bucket-255a91)](bucket/wllm.json)

[快速开始](#快速开始) · [功能](#功能) · [Reflex](#reflex直接返回决策) · [使用文档](#文档) · [下载](https://github.com/LuckyStar2456/Wllm/releases/latest)

</div>

## 快速开始

通过 [Scoop](https://scoop.sh/) 安装。首次添加 Wllm 安装源：

```powershell
scoop bucket add wllm https://github.com/LuckyStar2456/Wllm
scoop install wllm/wllm
```

指定自己的 **Qwen3.5 Q4_K_M 文本 GGUF**，打开终端界面：

```powershell
wllm --tui --model "D:\Models\qwen3.5-9b-q4_k_m-text.gguf"
```

输入文字并回车即可开始；`/help` 查看命令，`Ctrl+C` 取消当前请求。

| 获取方式 | 入口 |
|---|---|
| Scoop 安装与更新 | 添加一次源，后续 `scoop update wllm`；卸载用 `scoop uninstall wllm` |
| Windows 压缩包 | [下载发布包](https://github.com/LuckyStar2456/Wllm/releases/latest)，解压后运行 `wllm.exe` |
| 从源码构建 | [构建说明](#从源码构建)，依赖由 `dependencies.json` 锁定 |

<details>
<summary>直接通过远端 Scoop 清单安装</summary>

```powershell
scoop install https://raw.githubusercontent.com/LuckyStar2456/Wllm/main/bucket/wllm.json
```

这是 Wllm 仓库提供的自建安装源。添加安装源的方式便于后续使用 Scoop 更新。

</details>

## 功能

Wllm 将本地文本生成与 **Reflex 候选决策**放在同一个推理引擎上，提供交互终端和可供脚本调用的 CLI。

| 能力 | 使用体验 |
|---|---|
| 纯终端交互 | 蓝色实心字符横幅、原生滚动、中文编辑、多行粘贴、进程内输入历史 |
| 本地文本生成 | 流式输出、system 提示词、思考开关、上下文与采样设置 |
| Reflex 快速决策 | 比较单 token 候选 logits，输出选择、分数或拒答；支持 System One 请求格式 |
| 模型与参数 | 本地 GGUF 选择、显式加载/释放、权重复用、参数查看与修改 |
| 过程与结果 | 加载/输入处理状态、首 token 或决策耗时、生成速率、token 数与内存摘要 |
| 执行控制 | 取消、超时、错误后继续操作、显式导出结果 |
| CLI 集成 | 文本或 JSON 输出、文件输入、stdout/stderr 分离、明确退出码 |

CPU 发行包面向 **Windows x64 / AVX2**，使用者自行提供模型文件。默认上下文 **16384**、思考开启，注意力 KV 为 F16、循环状态为 F32；参数通过显式设置调整。

## 两种使用方式

**交互终端**适合直接提问、切换模式、查看状态：

```text
/load
/system 请用简洁的中文回答。
请解释什么是质数。
/metrics
/export result.json
/unload
```

**CLI** 适合脚本调用，生成文本写入 stdout，指标与错误写入 stderr：

```powershell
wllm --model "D:\Models\model.gguf" --prompt "用中文解释质数。"
wllm --model "D:\Models\model.gguf" --prompt-file prompt.txt --max-tokens 128
```

## Reflex：直接返回决策

Reflex 在处理输入后比较候选，直接返回结构化结果。它与普通生成共用本地模型加载、CPU 执行和资源管理。

| 模式 | 输入 | 输出 |
|---|---|---|
| `candidate_score` | 待判断文本 + 候选 JSON | 候选分数、选择或拒答 |
| `system_one` | `state` + `questions` | 具名 `answers`；支持 choice、noul、score |

```powershell
wllm --model "D:\Models\model.gguf" --mode system_one --request-file request.json
```

请求示例：

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

查看 [Reflex 使用说明](guides/reflex.md)和[通用请求示例](configs/system-one.example.json)。Scoop 用户可用 `scoop prefix wllm` 找到安装目录中的 `configs` 示例。

## 默认无状态

- **配置与记录按需启用**：默认不写日志、配置或输入历史，不自动读取本机配置；`--config FILE` 显式加载配置。
- **每次请求上下文独立**：连续请求复用已加载权重，历史输入不自动加入下一次模型输入。
- **保存由使用者决定**：`/export FILE` 导出上次结果，`--record FILE` 记录指标；界面参数修改只在当前进程生效。
- **交付内容明确**：发布包包含程序、通用示例、使用文档、品牌资源、许可证和校验清单；模型文件另行准备。

## 测试配置

本版在 **Windows 11 x64 · Intel Core i5-1135G7 · 16GB RAM · CPU AVX2** 配置上测试，模型为 **Qwen3.5-9B Q4_K_M 文本 GGUF，约 5.15 GiB，上下文配置 16384**。

| 范围 | 完成检查 |
|---|---|
| 原生组件 | 100 项配置、计量、候选打分、System One 映射与 Unicode 输出检查 |
| CLI | 空目录启动、默认无状态文件、显式配置/记录、Unicode 路径、JSON 与重定向 |
| 终端 | Windows ConPTY、PowerShell/CMD；38/48/80/100 列，横幅、编辑、粘贴、历史、缩放与退出恢复 |
| 真实模型 | 文本生成、Reflex/System One、权重复用、独立请求、取消恢复与结果导出 |
| 分发安装 | Scoop 下载、SHA256 校验、解压、命令入口，以及发布包解压启动 |

## 从源码构建

准备 Git、Visual Studio 2022 C++ 桌面开发工具和 CMake 3.24+：

```powershell
git clone https://github.com/LuckyStar2456/Wllm.git
cd Wllm
.\scripts\build.ps1 -BuildDirectory build/public
.\build\public\bin\Release\wllm.exe --tui
```

构建脚本获取固定版本的源码依赖，构建并运行原生检查。模型单独准备。

## 文档

| 面向使用者 | 面向开发者 |
|---|---|
| [Windows 使用说明](guides/windows.md) | [构建、测试与贡献](CONTRIBUTING.md) |
| [Reflex 与 System One](guides/reflex.md) | [依赖版本](dependencies.json) · [第三方声明](THIRD_PARTY_NOTICES.md) |
| [默认配置](configs/default.json) · [候选示例](configs/light-switch.candidates.json) | [更新记录](CHANGELOG.md) · [问题反馈](https://github.com/LuckyStar2456/Wllm/issues) |

## 开源与致谢

Wllm 使用 [MIT](LICENSE) 许可证，模型遵循各自许可证。

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)：本地推理引擎与张量计算基础。
- [nlohmann/json](https://github.com/nlohmann/json)：JSON 解析与序列化。
- [OpenCode](https://opencode.ai/docs/tui/) / [Ollama](https://github.com/ollama/ollama)：终端命令与用户文档组织参考；启动横幅呈现参考 Spring Boot。
