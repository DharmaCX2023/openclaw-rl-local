# Harness Skills 可用性审计

## 结论

当前审计范围是 OpenClaw 源码中的 53 个 bundled skills，加上 harness
额外注入的 1 个 managed skill `serper`，合计 54 个候选。

| 分类 | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| 当前可运行 | 4 | 7.4% | 当前 harness 已具备所需基础工具或运行入口 |
| 条件可运行 | 41 | 75.9% | 六工具架构兼容，但当前缺 CLI、凭据、账号、模型或设备 |
| 当前不可运行 | 9 | 16.7% | 依赖六工具之外的专用工具/API，或者实现本身缺失 |
| 合计 | 54 | 100% | 53 bundled + 1 managed |

因此，按严格的“现在就能执行”口径：

- 能用：4 个。
- 不能用：50 个。
- 50 个不能用项中，41 个可以通过补环境变成可用，9 个不能直接用于当前六工具 harness。

这里的“当前可运行”表示现有 skill 可以通过 `read`、`write`、`edit`、
`exec`、`process`、`web_fetch` 完成其声明的核心工作，并且从仓库的镜像构建
定义可以确认基础入口存在。它不代表该 skill 对 ClawGym 任务一定有价值。

## 当前可运行：4

| Skill | 主要入口 | Harness 实用性 |
|---|---|---|
| `healthcheck` | `exec` 调用 OpenClaw CLI 和系统诊断命令 | 可运行，但检查的是临时训练容器，通常与任务无关 |
| `node-connect` | `exec` 调用 OpenClaw CLI | 可运行，但 rollout 通常没有真实 companion node |
| `skill-creator` | `read`、`write`、`edit`、Python 脚本 | 可运行；对创建、修改和审计 task-owned skills 有实际价值 |
| `weather` | `exec` + `curl`，或 `web_fetch` | 可运行；依赖外部网络，但不需要 API key |

如果进一步按“对当前 ClawGym 文件任务是否通常有用”筛选，优先级较高的只有：

1. `skill-creator`
2. `weather`，仅在任务确实要求实时天气时

`healthcheck` 和 `node-connect` 虽然可以执行，但不建议仅为了保持旧 prompt 而固定注入。

## 条件可运行：41

这些 skill 的现有工作流可以映射到六个工具，主要通过 `exec` 调用 CLI 或
HTTP 客户端。但当前 harness 没有证明已经提供它们所需的全部依赖，因此不能算
“现在能用”。

### 缺专用 CLI、模型或本地运行时

- `1password`
- `blogwatcher`
- `clawhub`
- `coding-agent`
- `gemini`
- `gifgrep`
- `github`
- `mcporter`
- `model-usage`
- `nano-pdf`
- `obsidian`
- `openai-whisper`
- `oracle`
- `session-logs`
- `sherpa-onnx-tts`
- `songsee`
- `summarize`
- `tmux`
- `video-frames`

### 缺 API key、登录态或外部账号配置

- `gog`
- `goplaces`
- `himalaya`
- `notion`
- `openai-whisper-api`
- `sag`
- `trello`
- `xurl`

### 依赖特定设备或局域网服务

- `blucli`
- `camsnap`
- `eightctl`
- `openhue`
- `ordercli`
- `sonoscli`
- `spotify-player`
- `wacli`

### 依赖 macOS 或 macOS 应用

- `apple-notes`
- `apple-reminders`
- `bear-notes`
- `imsg`
- `peekaboo`
- `things-mac`

给这些名字加入 `skills.allowBundled` 并不会自动令它们可用。OpenClaw 可能因为
`metadata.openclaw.requires` 不满足而不展示它们；没有完整依赖声明的 skill 则
可能被展示，但会在第一次执行时失败。

## 当前不可运行：9

| Skill | 原因 |
|---|---|
| `bluebubbles` | 现有指令依赖未开放的 `message` 工具 |
| `canvas` | 现有指令依赖未开放的 `canvas` 工具和连接节点 |
| `clawflow` | 指令调用 `createFlow()` 等 API，但当前源码和六工具中没有可调用入口 |
| `clawflow-inbox-triage` | 是基于 ClawFlow API 的示例，不是可独立执行的 skill |
| `discord` | 现有指令依赖未开放的 `message(channel=discord)` |
| `gh-issues` | 核心流程依赖 subagent/session 类工具，不在六工具中 |
| `slack` | 现有指令依赖未开放的 Slack 专用工具 |
| `voice-call` | 现有指令依赖未启用的 voice-call 插件工具 |
| `serper` | `scripts/search.py` 缺失，且 harness 没有向容器注入 `SERPER_API_KEY` |

`serper` 与前八个不同：它在设计上可以只使用 `exec`、Python 和网络，因此可以
修复成六工具兼容。但在脚本和凭据注入真正补齐并通过端到端测试之前，它仍属于
当前不可运行。

## 不计入 54 个候选的文件

仓库里一共可以扫描到 401 份 `SKILL.md`，但不能把它们都算作 harness skills：

- 210 份位于历史 rollout 输出中；
- 72 份位于 `node_modules` 中；
- 24 份位于 `dist` 或 `dist-runtime` 构建产物中；
- 22 份是 ClawGym task-owned 输入，分布在 14 个任务里；
- 7 份位于 OpenClaw 的仓库维护者 `.agents/skills` 中；
- 其余还包括上述目录的重复版本。

Task-owned skills 位于 `/root/.openclaw/workspace/input/...`，不是 OpenClaw 自动发现的
`/root/.openclaw/workspace/skills` 或 `.agents/skills`。它们是题目要求读取、创建、
修复或审计的材料，不属于固定 runtime skill catalog。

## Harness 建议

在不改镜像、不增加凭据的前提下，建议固定白名单先缩为：

```json
["skill-creator", "weather"]
```

如果确实需要 OpenClaw 部署诊断，再加入：

```json
["healthcheck", "node-connect"]
```

`serper` 应在以下条件全部满足后再加入：

1. 提供非空且可执行的 `scripts/search.py`；
2. harness 复制完整 skill 目录，而不是只复制 `SKILL.md`；
3. 显式、安全地向本地 Docker、远端 Docker 和 chroot 三条路径注入 API key；
4. 增加真实执行搜索并校验 JSON 结果的端到端 smoke test；
5. 明确它是原实现还是 documented compatible replacement。

只检查 skill 名是否出现在 system prompt 中，不足以证明 skill 可用。
