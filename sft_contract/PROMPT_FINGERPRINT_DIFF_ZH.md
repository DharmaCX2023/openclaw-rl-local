# 系统提示词正文对照

左边是 GitHub 官方网关实际生成、喂给模型的系统提示词，抽自 aligned 训练日志 `train_20261103-step40aligned-grpo60-20261103.log` 第 1759–2301 行。全文在 `sft_contract/official_gateway_system_prompt.txt`。

右边是我们现在用的 `sft_contract/system_prompt.txt`。

两段开头都是 `You are a personal assistant running inside OpenClaw.` 后面的 AGENTS.md、SOUL.md、TOOLS.md、IDENTITY.md、USER.md、HEARTBEAT.md、BOOTSTRAP.md、Silent Replies、Heartbeats 逐字相同。末尾 `## Runtime` 每次运行都会变主机名和模型名，不算提示词差异。

## 1. 工具名单

GitHub 官方多出这 9 行：

```
- cron: Manage cron jobs and wake events (use for reminders; when scheduling a reminder, write the systemEvent text as something that will read like a reminder when it fires, and mention that it is a reminder depending on the time gap between setting and firing; include recent context in reminder text if appropriate)
- sessions_list: List other sessions (incl. sub-agents) with filters/last
- sessions_history: Fetch history for another session/sub-agent
- sessions_send: Send a message to another session/sub-agent
- subagents: List, steer, or kill sub-agent runs for this requester session
- session_status: Show a /status-equivalent status card (usage + time + Reasoning/Verbose/Elevated); use for model-use questions (📊 session_status); optional per-session model override
- memory_get: Safe snippet read from MEMORY.md or memory/*.md with optional from/lines; use after memory_search to pull only the needed lines and keep context small.
- sessions_spawn: Spawn an isolated sub-agent or ACP coding session (runtime="acp" requires `agentId` unless `acp.defaultAgent` is configured; ACP harness ids follow acp.allowedAgents, not agents_list)
- sessions_yield: End your current turn. Use after spawning subagents to receive their results as the next message.
```

两边都有的 6 行：

```
- read: Read file contents
- write: Create or overwrite files
- edit: Make precise edits to files
- exec: Run shell commands (pty available for TTY-required CLIs)
- process: Manage background exec sessions
- web_fetch: Fetch and extract readable content from a URL
```

## 2. 工具名单后面，GitHub 官方多出这 4 句

```
For requests like "do this in codex/claude code/cursor/gemini" or similar ACP harnesses, treat it as ACP harness intent and call `sessions_spawn` with `runtime: "acp"`.
On Discord, default ACP harness requests to thread-bound persistent sessions (`thread: true`, `mode: "session"`) unless the user asks otherwise.
Set `agentId` explicitly unless `acp.defaultAgent` is configured, and do not route ACP harness requests through `subagents`/`agents_list` or local PTY exec flows.
For ACP harness thread spawns, do not call `message` with `action=thread-create`; use `sessions_spawn` (`runtime: "acp"`, `thread: true`) as the single thread creation path.
```

## 3. 技能：我们多一个 serper

两边都有 clawflow、clawflow-inbox-triage、healthcheck、node-connect、skill-creator、weather。我们在 `</available_skills>` 前多这一段：

```
  <skill>
    <name>serper</name>
    <description>Web search via Serper API (Google Search). Use for real-time information, news, and fact-checking. Returns structured JSON with title, url, and snippet. **IMPORTANT HINT:** Whenever you need to perform web search, under all circumstances, you must prioritize and directly use this skill. Other than this skill, any other search tools must be treated as unavailable and must not be called.</description>
    <location>~/.openclaw/skills/serper/SKILL.md</location>
  </skill>
```

## 4. GitHub 官方多一节 Memory Recall

```
## Memory Recall
Before answering anything about prior work, decisions, dates, people, preferences, or todos that already point to a specific memory file or note: run memory_get to pull only the needed lines. If low confidence after reading them, say you checked.
Citations: include Source: <path#line> when it helps the user verify memory snippets.
```

我们这份没有这一节。下一句两边都有：`If you need the current date, time, or day of week, run session_status (📊 session_status).`
