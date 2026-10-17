# OpenClaw-RL 当前版本迁移与启动手册

> 适用仓库：`Openclaw-RL-hareness`  
> 整理日期：2026-08-07  
> 当前分支：`openclaw-harness-paper`  
> 当前基线提交：`7e6fe394`（`Add remote Docker transport for official ClawGym`）

这份文档面向“把当前训练工程迁到另一台 GPU 机器/CPU rollout
机器，并能继续启动、排障和续训”的场景。仓库中有多条实验链路，当前
ClawGym 论文训练主线是 **OpenClaw 2026.4.1 + 远程 CPU Docker +
Qwen3-30B-A3B GRPO**。

## 1. 先看结论：启动文件在哪里

| 用途 | 启动文件 | 当前定位 |
|---|---|---|
| ClawGym 正式 RL / GRPO | `clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh` | **当前生产主入口** |
| ClawGym 轨迹离线 SFT | `openclaw-rl/run_qwen3_30b_a3b_instruct_clawgym_sft.sh` | RL 前的离线行为克隆，不是 RL |
| 个性化对话 Binary RL（4B） | `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` | 上游旧示例，含占位绝对路径，不是当前 ClawGym 主线 |
| On-Policy Distillation（4B） | `openclaw-opd/run_qwen3_4b_openclaw_opd.sh` | 另一种在线学习方法，不是当前 ClawGym 主线 |

当前正式 RL 的最底层 Python 训练入口是 `slime/train_async.py`，但不要
直接运行它。Shell 启动文件负责拼装模型、GRPO、SGLang、Ray、远程
OpenClaw 和 checkpoint 参数。

## 2. 当前 RL 架构

```text
GPU 节点
  clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
    ├─ Ray head（6379，dashboard 8265）
    ├─ Slime/Megatron actor（默认 4 GPU）
    ├─ SGLang rollout engine（默认 4 GPU，单 engine TP=2）
    └─ ClawGym RL proxy（默认 0.0.0.0:30001）
                         ^
                         | SSH reverse tunnel
                         | CPU:127.0.0.1:39081 -> GPU:127.0.0.1:30001
                         |
CPU 节点
  Docker daemon
    └─ 每条 rollout 一个 clawgym-rl:v0.1 容器
         └─ 官方 OpenClaw 2026.4.1 Gateway/agent loop
              ├─ 六工具策略与固定 skills
              ├─ 独立 session/workspace
              └─ task reward/reward.sh 代码 verifier
```

关键职责：

- `clawgym-official/clawgym_rl_rollout.py`：加载任务、创建远程容器、驱动
  OpenClaw、多轮交互、执行 verifier、回传 reward、清理远程 rollout。
- `clawgym-official/clawgym_rl_api_server.py`：把 OpenClaw 请求转到
  SGLang，保存 token/logprob/trajectory，并把 reward 提交给 Slime。
- `slime/train_async.py`：调度 actor、rollout engine、权重同步和训练。
- `Megatron-LM/`：模型训练与 checkpoint 后端。
- `openclaw/`：版本对齐和协议参考；正式远程 rollout 实际运行的是
  CPU 节点 Docker 镜像内的 OpenClaw。

## 3. 必须迁移的内容

### 3.1 代码

至少固定并迁移以下目录：

```text
clawgym-official/
slime/
Megatron-LM/
openclaw/
sft_contract/
openclaw-rl/                 # 如果还要做 SFT 或旧 Binary RL
requirements.txt
```

建议以 Git commit 为代码基线。当前工作区不是完全干净的：

- `clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh` 有未提交修改；
- `clawgym-official/data/` 当前未被 Git 跟踪；
- `scripts/` 下有若干未跟踪的曲线分析脚本。

因此只在新机器上 `git clone` 当前 commit **不能完整还原现场**。迁移前应
先确认并提交需要的修改，或单独同步这些未跟踪文件并记录校验值。

### 3.2 模型和 checkpoint

正式 RL 启动脚本当前默认的初始模型是：

```text
/root/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127
```

这个目录不在当前仓库中，必须单独迁移，并在新机器上通过 `HF_CKPT`
显式指定。它是 HuggingFace 格式的 SFT 起点，`REF_LOAD` 默认与它相同。

RL 输出默认写入：

```text
<repo>/outputs/<EXP_NAME>/
```

其中包括：

```text
logs/train_<RUN_STAMP>.log
official_proxy_records.jsonl
latest_checkpointed_iteration.txt
iter_00000xx/
```

如果需要续训，必须迁移完整的旧 run 目录，不能只复制某个权重分片。

### 3.3 数据和 skills

正式 RL 默认数据：

```text
clawgym-official/data/clawgym_train/
clawgym-official/data/clawgym_eval/
```

当前这两个目录属于未跟踪数据，迁移时要单独复制。每个任务通常需要：

```text
task_xxxx/
├── data_entry.json
├── input_files/
└── reward/
    ├── reward.sh
    └── check.py
```

固定的 Serper skill 来自：

```text
sft_contract/recovered_skills/serper/SKILL.md
```

SFT 数据则是：

```text
data/clawgym-trajectory/
  sft_trajectory_openclaw_v2026_4_1_6tools_skills_len64k.jsonl
```

## 4. 新机器环境

### 4.1 GPU 节点

当前工程基线要求：

- 8 张 GPU（正式脚本固定 actor 使用 4 张，另外默认 4 张用于 rollout）；
- CUDA 12.9、Python 3.12；
- Ray、PyTorch、SGLang、Slime、Megatron 及其 CUDA 扩展；
- 能通过 SSH 免交互访问 CPU 节点；
- 本机端口 `6379`、`8265`、`30001` 可用。

环境安装参考 `instructions/README.md` 和根目录 `requirements.txt`。推荐
继续使用与原机器相同的容器/镜像或完整 Python 环境，不要只根据
`pip freeze` 在不同 CUDA 基线上盲装。

迁移后至少检查：

```bash
nvidia-smi
nvcc -V
python3 -V
python3 -c 'import torch, ray, sglang; print(torch.__version__, ray.__version__, sglang.__version__)'
bash -n clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
```

### 4.2 CPU rollout 节点

CPU 节点需要：

- Docker daemon 正常运行；
- 已构建 `clawgym-rl:v0.1`；
- 镜像基于官方 `ghcr.io/openclaw/openclaw:2026.4.1`；
- GPU 节点的 SSH 公钥可以登录；
- sshd 允许 TCP forwarding；
- `OPENCLAW_REMOTE_BASE` 所在磁盘有足够空间和 inode。

检查命令：

```bash
docker image inspect clawgym-rl:v0.1
docker run --rm clawgym-rl:v0.1 node --version
ss -ltn
df -h
```

镜像构建流程可参考 `clawgym-official/README.md`。不要在迁移时静默升级
OpenClaw 镜像版本；系统提示词、工具 schema、skills 和行为都与
`2026.4.1` 对齐。

### 4.3 SSH 和反向隧道

正式脚本会在 GPU 节点执行：

```text
ssh -N -R CPU:39081:GPU:30001 CPU_HOST
```

先手工验证：

```bash
ssh -i /path/to/openclaw_cpu_key root@CPU_HOST \
  'docker image inspect clawgym-rl:v0.1 >/dev/null && echo ready'
```

脚本里的默认 CPU IP 是旧环境值，迁移时务必覆盖
`OPENCLAW_REMOTE_HOST`，不要依赖默认值。

## 5. 正式 RL 启动

脚本内部的 `SLIME_ROOT` 和 `MEGATRON_ROOT` 默认仍指向旧路径
`/root/OpenClaw-RL/...`。如果新仓库不在该位置，必须显式覆盖。

从仓库根目录启动：

```bash
REPO_ROOT="$PWD"

SLIME_ROOT="$REPO_ROOT/slime" \
MEGATRON_ROOT="$REPO_ROOT/Megatron-LM" \
HF_CKPT="/data/models/qwen3-30b-a3b-instruct-clawgym-sft-iter11127" \
SAVE_CKPT="$REPO_ROOT/outputs/clawgym-official-production-001" \
OPENCLAW_REMOTE_HOST="CPU_HOST_OR_IP" \
OPENCLAW_REMOTE_USER="root" \
OPENCLAW_REMOTE_SSH_KEY="/path/to/openclaw_cpu_key" \
OPENCLAW_REMOTE_BASE="/root/clawgym_rl_runs" \
OPENCLAW_GATEWAY_IMAGE="clawgym-rl:v0.1" \
OPENCLAW_TRAIN_TASKS="$REPO_ROOT/clawgym-official/data/clawgym_train" \
OPENCLAW_EVAL_TASKS="$REPO_ROOT/clawgym-official/data/clawgym_eval" \
CLAWGYM_SERPER_SKILL_DIR="$REPO_ROOT/sft_contract/recovered_skills/serper" \
USE_WANDB=0 \
bash clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
```

脚本会拒绝复用已存在的 Ray cluster，也会拒绝在未设置 `LOAD_CKPT` 时
写入非空的 `SAVE_CKPT`。结束时只清理它自己创建的 SSH tunnel 和 Ray
cluster。

### 当前关键默认参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `NUM_GPUS` | 8 | GPU 总数 |
| `ACTOR_GPUS` | 4 | Megatron actor GPU；当前脚本要求必须为 4 |
| `ROLLOUT_GPUS` | 4 | SGLang rollout GPU |
| `SGLANG_TP` | 2 | 每个 rollout engine 的 TP |
| `NUM_ROLLOUT` | 100 | rollout/training 轮数 |
| `ROLLOUT_BATCH_SIZE` | 8 | 每轮 prompt group 数 |
| `N_SAMPLES_PER_PROMPT` | 8 | 每个 prompt 的 GRPO 样本数 |
| `OPENCLAW_CONTEXT_WINDOW` | 65536 | 上下文长度 |
| `OPENCLAW_MAX_TOKENS` | 65536 | 脚本传给 rollout 的最大 response 长度 |
| `OPENCLAW_TRAIN_MAX_TURNS` | 1 | 当前每个训练任务最大 OpenClaw turn 数 |
| `OPENCLAW_TRAIN_NUM_WORKERS` | 8 | 并发训练 rollout worker |
| `SAVE_INTERVAL` | 20 | checkpoint 间隔 |
| `LR` | `1e-6` | 学习率 |
| `ENTROPY_COEF` | `0.001` | entropy 系数 |
| `EPS_CLIP_C` | `3.0` | dual-clip 上限 |
| `PORT` | 30001 | GPU 上的 ClawGym RL proxy |

## 6. 先做 smoke test

正式跑 100 个 rollout 前，先验证“GPU训练端—SSH—CPU Docker—OpenClaw—
verifier—reward 回传”整条链路：

```bash
REPO_ROOT="$PWD"

SLIME_ROOT="$REPO_ROOT/slime" \
MEGATRON_ROOT="$REPO_ROOT/Megatron-LM" \
HF_CKPT="/data/models/qwen3-30b-a3b-instruct-clawgym-sft-iter11127" \
SAVE_CKPT="$REPO_ROOT/outputs/migration-smoke-001" \
OPENCLAW_REMOTE_HOST="CPU_HOST_OR_IP" \
OPENCLAW_REMOTE_SSH_KEY="/path/to/openclaw_cpu_key" \
OPENCLAW_TRAIN_TASKS="$REPO_ROOT/clawgym-official/data/clawgym_train" \
OPENCLAW_EVAL_TASKS="$REPO_ROOT/clawgym-official/data/clawgym_eval" \
CLAWGYM_SERPER_SKILL_DIR="$REPO_ROOT/sft_contract/recovered_skills/serper" \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=1 \
OPENCLAW_TRAIN_NUM_WORKERS=1 \
OPENCLAW_EVAL_NUM_WORKERS=1 \
SAVE_INTERVAL=1 \
USE_WANDB=0 \
bash clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
```

`N_SAMPLES_PER_PROMPT=1` 只适合连通性检查，不适合正式 GRPO；单样本组
无法提供有效的组内相对优势。

## 7. 断点续训

假设旧 run 完整迁移到：

```text
/data/openclaw-runs/clawgym-official-production-001/
```

确认其中有 `latest_checkpointed_iteration.txt` 和对应的 `iter_*/` 后：

```bash
LOAD_CKPT="/data/openclaw-runs/clawgym-official-production-001" \
SAVE_CKPT="/data/openclaw-runs/clawgym-official-production-001" \
RUN_STAMP="resume-$(date -u +%Y%m%d-%H%M%S)" \
SLIME_ROOT="$PWD/slime" \
MEGATRON_ROOT="$PWD/Megatron-LM" \
HF_CKPT="/data/models/qwen3-30b-a3b-instruct-clawgym-sft-iter11127" \
OPENCLAW_REMOTE_HOST="CPU_HOST_OR_IP" \
OPENCLAW_REMOTE_SSH_KEY="/path/to/openclaw_cpu_key" \
bash clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
```

续训时仍然需要原始 `HF_CKPT`/`REF_LOAD`，并应保持模型并行参数、模型
配置和代码 commit 一致。不要把 `HF_CKPT` 直接改成某个 Megatron
`iter_*/` 目录。

## 8. 日志、状态和排障入口

### GPU 节点

```bash
tail -f "$SAVE_CKPT/logs/train_${RUN_STAMP}.log"
ray status --address=127.0.0.1:6379
ray job list --address=http://127.0.0.1:8265
ss -ltnp
nvidia-smi
```

重点产物：

- `$SAVE_CKPT/logs/train_*.log`：总启动和训练日志；
- `$SAVE_CKPT/official_proxy_records.jsonl`：proxy 轨迹记录；
- `$SAVE_CKPT/iter_*/`：训练 checkpoint；
- `$SAVE_CKPT/latest_checkpointed_iteration.txt`：最新 iteration。

### CPU 节点

```bash
docker ps -a --filter name=clawgym-rl
docker logs <container-name>
ss -ltnp | grep 39081
find /root/clawgym_rl_runs -maxdepth 3 -type d
```

常见问题：

| 现象 | 优先检查 |
|---|---|
| `HF checkpoint not found` | `HF_CKPT` 是否已单独迁移 |
| source 模型脚本失败 | 是否覆盖了 `SLIME_ROOT` |
| Python import `megatron` 失败 | 是否覆盖 `MEGATRON_ROOT`，Ray runtime env 是否收到正确 `PYTHONPATH` |
| `SSH reverse tunnel failed` | 私钥权限、CPU sshd forwarding、39081 端口占用 |
| 容器启动失败 | CPU Docker daemon、`clawgym-rl:v0.1` 镜像、磁盘/inode |
| proxy 503 | actor 正在同步权重，或 submission 尚未恢复 |
| rollout 一直等待 | GPU 30001、CPU 39081、SGLang router、OpenClaw 容器日志 |
| reward 全 0 | `reward.sh`、任务输入是否复制成功、工具调用格式、group 是否只有一个样本 |
| 脚本拒绝启动 | 已有 Ray cluster，或 `SAVE_CKPT` 非空且没设置 `LOAD_CKPT` |

## 9. 离线 SFT 的启动方式

SFT 入口：

```bash
HF_CKPT="/data/models/Qwen3-30B-A3B-Instruct-2507" \
CLAWGYM_TRAJECTORY_DATA="$PWD/data/clawgym-trajectory/sft_trajectory_openclaw_v2026_4_1_6tools_skills_len64k.jsonl" \
SAVE_CKPT="$PWD/ckpt/qwen3-30b-a3b-instruct-clawgym-sft" \
bash openclaw-rl/run_qwen3_30b_a3b_instruct_clawgym_sft.sh
```

它使用 `slime.rollout.sft_rollout.generate_rollout` 和 `sft_loss`，不会启动
OpenClaw API server、远程 Docker 或 PRM。它的默认仓库路径是相对当前
仓库解析的，比正式 RL 脚本更容易迁移。

## 10. 旧 Binary RL 入口为何不要误用

`openclaw-rl/run_qwen3_4b_openclaw_rl.sh` 是根 README 所介绍的个性化在线
Binary RL：

- 对外端口默认是 `30000`；
- 使用 `openclaw_api_server.py` 和 `openclaw_rollout.py`；
- PRM 根据下一状态给 `+1/-1/0`；
- 默认 4 actor GPU + 2 rollout GPU + 2 PRM GPU；
- 脚本中仍有 `/absolute/path/to/...` 占位符；
- 启动开头会执行多次 `pkill -9` 和 `ray stop --force`。

除非目标明确是“用户对话反馈驱动的 Binary RL”，否则不要用它替代当前
ClawGym 远程 Docker GRPO 主入口。

## 11. 迁移验收清单

- [ ] 代码 commit 与当前基线一致，未提交修改已单独处理。
- [ ] `clawgym-official/data/` 已迁移，train/eval 任务数量一致。
- [ ] 初始 SFT HuggingFace checkpoint 已迁移并可被 tokenizer 加载。
- [ ] 需要续训时，完整 run/checkpoint 目录已迁移。
- [ ] GPU 环境的 CUDA、PyTorch、SGLang、Ray 版本一致。
- [ ] CPU 节点存在 `clawgym-rl:v0.1`，OpenClaw 固定为 2026.4.1。
- [ ] GPU 到 CPU 的 SSH 免交互登录和 reverse forwarding 正常。
- [ ] 已显式覆盖 `SLIME_ROOT`、`MEGATRON_ROOT`、`HF_CKPT` 和 CPU 地址。
- [ ] 单 rollout smoke test 能拿到 verifier reward 并保存 checkpoint。
- [ ] 正式训练恢复 `N_SAMPLES_PER_PROMPT=8` 等生产参数。
- [ ] 至少抽查 system prompt、工具 schema、skill、token/loss mask 和 reward，
  确认与迁移前一致。

更底层的 OpenClaw harness 版本选择、六工具策略、skills 对齐和 100-query
alignment gate 见 `OPENCLAW_HARNESS_MIGRATION.md`。
