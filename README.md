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

[快速开始](#快速开始) · [功能](#功能) · [System One](#system-one直接返回决策) · [使用文档](#文档) · [下载](https://github.com/LuckyStar2456/Wllm/releases/latest)

</div>

## 快速开始

通过 [Scoop](https://scoop.sh/) 安装。首次添加 Wllm 安装源：

```powershell
scoop bucket add wllm https://github.com/LuckyStar2456/Wllm
scoop install wllm/wllm
wllm
```

Scoop 会创建 `wllm` 命令，在 PowerShell 或 CMD 的任意目录输入即可打开终端界面，无需进入安装文件夹。若刚安装 Scoop 后旧终端尚未刷新 PATH，重新打开终端即可。

首次进入后可用 `/model PATH` 选择模型，再输入文字；也可以启动时指定自己的 **GGUF 文本模型**。按模型能力显式设置上下文：

```powershell
wllm --tui --model "D:\Models\model.gguf" --context 4096
```

输入文字并回车即可开始；`/help` 查看命令，`Ctrl+C` 取消当前请求。

| 获取方式 | 入口 |
|---|---|
| Scoop 安装与更新 | 添加一次源，后续 `scoop update wllm`；卸载用 `scoop uninstall wllm` |
| Windows 压缩包 | [下载发布包](https://github.com/LuckyStar2456/Wllm/releases/latest)，解压目录内运行 `.\wllm.exe`；便携版不自动修改 PATH |
| 从源码构建 | [构建说明](#从源码构建)，依赖由 `dependencies.json` 锁定 |

<details>
<summary>直接通过远端 Scoop 清单安装</summary>

```powershell
scoop install https://raw.githubusercontent.com/LuckyStar2456/Wllm/main/bucket/wllm.json
```

这是 Wllm 仓库提供的自建安装源。添加安装源的方式便于后续使用 Scoop 更新。

</details>

## 功能

Wllm 将本地文本生成与 **System One 候选决策**放在同一个推理引擎上，提供交互终端和可供脚本调用的 CLI。

| 能力 | 使用体验 |
|---|---|
| 纯终端交互 | 蓝色实心字符横幅、原生滚动、中文编辑、多行粘贴、进程内输入历史 |
| 本地文本生成 | 流式输出、system 提示词、思考开关、上下文与采样设置 |
| System One 快速决策 | 接收业务选项，返回选择、候选分数和置信度；支持 choice、noul、score |
| 通用模型入口 | 由 GGUF 元数据识别架构、词表、精度、分片和原生聊天模板；本地选择、加载/释放与权重复用 |
| 架构契约 | 从逐层几何与张量布局描述 Dense、MoE、注意力及混合状态；推理/代码任务共用底层架构 |
| 文本与模板 | 指令模型使用自带模板；基础模型使用 `--raw` 补全；支持显式 Jinja 模板覆盖 |
| 过程与结果 | 加载/输入处理状态、首 token 或决策耗时、生成速率、token 数与内存摘要 |
| 执行控制 | 取消、超时、错误后继续操作、显式导出结果 |
| HTTP 服务 | 按 `openai`、`systemone`、`all` 开放接口，指定地址/端口，支持 SSE 与可选 Bearer 认证 |
| CLI 集成 | 文本或 JSON 输出、文件输入、stdout/stderr 分离、明确退出码 |

CPU 发行包面向 **Windows x64 / AVX2**，使用者自行提供模型文件。默认上下文 **16384**、思考开启，注意力 KV 为 F16，其他状态由模型架构决定；参数通过显式设置调整。模型按文件自带精度执行，使用原生词表和模板；思考开关是否生效由所选模型模板决定。详见[模型接入说明](guides/models.md)。

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

## HTTP 服务

```powershell
wllm --serve --service all --port 8080 --model "D:\Models\model.gguf" --context 4096
```

通用客户端使用 `http://127.0.0.1:8080/v1`，模型名 `wllm-local`；文本生成使用 `/v1/chat/completions`，决策请求使用 `/v1/systemone`。可用 `--service openai` 或 `--service systemone` 单独开放对应接口。详见[服务与请求协议](guides/service.md)。

## System One：直接返回决策

System One 在处理输入后比较候选，直接返回结构化结果。它与普通生成共用本地模型加载、CPU 执行和资源管理。

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

查看 [System One 使用说明](guides/systemone.md)和[通用请求示例](configs/system-one.example.json)。Scoop 用户可用 `scoop prefix wllm` 找到安装目录中的 `configs` 示例。

## 默认无状态

- **配置与记录按需启用**：默认不写日志、配置或输入历史，不自动读取本机配置；`--config FILE` 显式加载配置。
- **每次请求上下文独立**：连续请求复用已加载权重，历史输入不自动加入下一次模型输入。
- **保存由使用者决定**：`/export FILE` 导出上次结果，`--record FILE` 记录指标；界面参数修改只在当前进程生效。
- **交付内容明确**：发布包包含程序、通用示例、使用文档、品牌资源、许可证和校验清单；模型文件另行准备。

## 测试配置

本版在 **Windows 11 x64 · Intel Core i5-1135G7 · 16GB RAM · CPU AVX2** 配置上测试。

| 实测模型 | 文件大小 | 精度 | 上下文配置 |
|---|---:|---|---:|
| Qwen3.5-9B 文本 GGUF | 5.15 GiB | Q4_K_M | 16384 |
| Qwen2.5-0.5B-Instruct | 644.41 MiB | Q8_0 | 4096 |
| SmolLM2-135M-Instruct（Llama 架构） | 100.57 / 258.34 MiB | Q4_K_M / F16 | 4096 |

| 范围 | 完成检查 |
|---|---|
| 原生组件 | 124 项配置、计量、候选打分、System One 映射、架构策略与 Unicode 输出检查 |
| 架构结构 | 随机权重微型 GGUF 实际执行 MHA/MQA/GQA、MoE/共享专家、MLA 和滑动注意力；核对 logits、请求隔离、专家算子和 HTTP，语言模型实测单列 |
| CLI | 空目录启动、默认无状态文件、显式配置/记录、Unicode 路径、JSON 与重定向 |
| 终端 | Windows ConPTY、PowerShell/CMD；38/48/80/100 列，横幅、编辑、粘贴、历史、缩放与退出恢复 |
| 真实模型 | 跨架构与精度的文本生成、System One 三类题型、权重复用、独立请求；Qwen3.5 的取消恢复与结果导出 |
| 模板与补全 | 模型原生模板、模板缺失提示、显式 Jinja 覆盖与原始文本补全 |
| HTTP 服务 | SmolLM2 Q4_K_M、Qwen2.5 Q8_0：JSON/SSE、三类决策题型、认证、端口冲突、请求隔离、断线/超时恢复与 Ctrl+C 停止；OpenAI Python 2.41.0 客户端对接 |
| 分发安装 | Scoop 下载、SHA256 校验、解压、命令入口，以及发布包解压启动 |

v0.5.0 的 Qwen3.5-9B 同配置 CLI/HTTP 固定种子对照中，11 个输入 token、8 个输出 token 与原版本一致。

本轮还完成该 9B / 16K 基线的模板、输入 token、采样和状态配置对照；架构结构检查与真实模型运行记录分别列示。

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
| [Windows 使用说明](guides/windows.md) · [HTTP 服务](guides/service.md) | [构建、测试与贡献](CONTRIBUTING.md) |
| [模型接入说明](guides/models.md) · [System One 决策](guides/systemone.md) | [依赖版本](dependencies.json) · [第三方声明](THIRD_PARTY_NOTICES.md) |
| [默认配置](configs/default.json) · [候选示例](configs/light-switch.candidates.json) | [更新记录](CHANGELOG.md) · [问题反馈](https://github.com/LuckyStar2456/Wllm/issues) |

## 开源与致谢

Wllm 使用 [MIT](LICENSE) 许可证，模型遵循各自许可证。

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)：本地推理引擎与张量计算基础。
- [nlohmann/json](https://github.com/nlohmann/json)：JSON 解析与序列化。
- [OpenCode](https://opencode.ai/docs/tui/) / [Ollama](https://github.com/ollama/ollama)：终端命令与用户文档组织参考；启动横幅呈现参考 Spring Boot。
