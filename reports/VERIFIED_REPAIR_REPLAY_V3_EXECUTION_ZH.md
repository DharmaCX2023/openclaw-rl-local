# Verified Action Repair v3：执行协议与进度

## 冻结目标

在既有 8B SFT 与静态 anchors 上，重做 Teacher 自主选择 agent step、动作重放、完整 Student 续跑的最小实验。旧快照／3-step 协议保留为历史基线，不据其低产率判定 DPO 或 margin 无效。

固定数据来自 v2 的 selection.json：16 题 × 4 条源轨迹，原 12 train／4 held-out 划分不变；5 题工程 smoke 独立。先修复 seeded sampler 后重新生成，不覆盖旧样本。

可靠偏好数据门槛：至少 16 对、覆盖至少 8 个 train 任务。held-out 任务不参与门槛、训练或调参。门槛不足则报告 NO-GO，不通过追加挑题制造通过。门槛通过后，以同一训练对与更新预算比较 NLL、DPO、DPO + all-blocked margin，再独立复评 held-out。

## 关键约束

- Teacher 同时输出修复的 agent step、call、原因和完整替代动作；可以不修复。诊断与动作 span 分离，不提供总 reward、隐藏 verifier 代码或逐项失败答案。
- 短轨迹全量；长轨迹使用机械目录与只读 read_trace 分页。不得把省略片段伪称为完整记录，不加入额外总结／judge 模型。
- 每个源 agent step 保存真实请求、输出 token、工具反馈和轻量文件哈希。不逐 step 保存文件内容快照；仅保存初始／终态必要证据。
- 新建专属容器，真实执行记录的前缀动作；前缀不调用 Student 推理。修复点执行 Teacher 替代动作，再让 Student 运行至正常结束／原任务预算。
- 主 agent-step 索引与 compact 辅助请求分开。保留正常 pruning、compaction 和原生工具协议。
- 同一对使用相同逐 step seed 方案；前缀消耗计入 agent-step 预算，重放基础设施耗时单列。取消旧 3-step／300 秒／32768-token 额外分支限制。
- 原配置内部 agent 时间上限为 1000 秒，外层传输等待为 1200 秒。修复续跑仅获得 1000 减去原前缀已用时间；重放容器的 2400 秒物理外壳只容纳前缀重放，Proxy 和控制器独立执行剩余策略时间限制，不等于给 Student 2400 秒。
- 对轻微环境变化记录差异，不按字符串逐字相等统一判失败；关键文件或依赖变化不得假装成同一可比较状态。
- 初筛使用源完整结果。改善候选再做原／修复同预算配对复验；正常失败与基础设施失败分开统计。
- Teacher 动作及强制重放前缀都是辅助数据，不制造 rollout logprob，不伪装成 on-policy GRPO。
- SGLang 修正使用每次实验独立的 import overlay，校验共享源文件指纹，所有 spawn worker 继承；不得编辑共享 venv。

## 当前进度

- 基线主仓：0904238c；旧修复实验提交：3157b071。
- 已实现隔离 sampler import overlay、机械轨迹目录／分页读取、Teacher 联合定位、真实动作重放、终态归档及同预算完整续跑运行器。
- 共 56 项修复相关 CPU 测试通过（旧 34 项 + 新 22 项），包括真实 8B tokenizer 的原始前缀保留、强制动作无伪造 logprob、压缩请求与主 step 分离，以及剩余时间扣除。
- 隔离子进程中实际 SGLang sampler 的 CPU 和 CUDA 编译路径端点测试均通过：已知 hash 端点的被屏蔽类别不再被采中；共享安装文件 SHA256 未变化。真实 SGLang scheduler PID 2580172 已记录补丁加载证据，见新输出目录的 sampler_overlay_imports.jsonl 与 sampler_overlay_cuda_check.json。
- 测试发现原生 qwen25 工具格式要求标签内换行；已修正测试输入、明确 Teacher 格式，不静默修补模型生成。
- 新输出：outputs/verified-repair-v3-20260907-replay-a1。固定 16 题及 12/4 划分已从旧 selection.json 复制，未更换任务。
- 工程 smoke 计划：GPU 0，一个 TP1 引擎，workers=2、Gateway 启动并发=2；GPU 1–7 烤机。端口 31780/31781/39781 启动前检查。远端只读预检显示 available 约 200 GiB、/data 可用 177 GiB；58 个已有容器不做清理。
- 尚无本轮修复效果或训练结论。smoke 闭环、三条真实 Teacher 案例和采样 span 审计通过审查后，才解除固定 64 条采集的工程门槛。

## 2026-09-07 首轮 smoke 运行记录（已结束，工程门槛未通过）

- 代码提交：4c38307f。启动时间：11:01:29 UTC。tmux 会话：verified_repair_v3_smoke_20260907；父进程链已核对为 tmux server → supervisor → runner → SGLang。
- 源轨迹已完成 5/5：task_0126（4 step，reward 0）、task_0133（9 step，reward 1）、task_0140（4 step，reward 0.95）、task_0158（36 step，reward 0）、task_0162（4 step，reward 0.866667）。这些是本次源轨迹，不是修复收益或正式 eval。
- task_0126 的第 4 step 生成 8192 tokens，重复诊断 max_ngram_repeat=8139，工具调用未闭合；终态没有两个目标输出文件，31 项 criterion 均未通过。这是已保存的策略失败，不是 verifier/归档故障。
- task_0158 实际执行约 1000 秒，以 original_task_wall_budget 正常结束；最大已完成 step 的 prompt 为 48689 tokens。未人为截成 3-step 窗口；该条未触发可观测 compaction，不能据此声称真实压缩重放已通过。
- 注意：OpenClaw 的 .openclaw/workspace-state.json 含 bootstrapSeededAt。跨容器差异必须结合原始哈希与文件内容审计；不能把运行时初始化时间变化直接判为 Teacher 错误，也不能未经核实抹掉所有时间字段。
- 本轮远端容器只在 /data/clawgym_rl_runs/verified-repair-v3-20260907-replay-a1 下创建与清理。其余七卡持续烤机，当前 optimizer_updates=0。

首轮共 1426.6 秒，11:25:33 UTC 结束并恢复八卡烤机。4 条非满分轨迹均尝试 Teacher，均被协议校验拒绝，没有执行修复分支：

| 任务 | 原始 Teacher 表现 | 判断 |
| --- | --- | --- |
| 0126 | 直接续写失败 write，重复到单轮上限 | 未完成 reviewer 角色切换 |
| 0140 | 回复“任务已完成”，没有结构化决定 | 不能把自然语言汇报自动猜成 no_repair |
| 0158 | 继续原任务的 exec，没有修复 step | 不能猜测其修复点 |
| 0162 | 给出 step、call、diagnosis 和原生工具动作，但缺外层闭合标签 | 输出契约应兼容原生动作自然结束；其手算数值仍须执行验证 |

这轮未通过端到端工程门槛，不能据此判定修复算法无效，更不能构造偏好数据。

## 第二轮冷启动复验

- 使用新目录 outputs/verified-repair-v3-20260907-replay-a2；通过 --source-output 只读复用 a1 的 5 条完整源轨迹，记录逐文件 SHA256，不覆盖原始结果。
- 在记录末尾明确切换 reviewer 角色；仍由同一 8B Teacher 联合选择 step/call/替代动作。允许最多两次纯格式反馈，记录每次原文，不给 verifier 反馈、不替 Teacher 选位置。
- 替代动作是响应的最后一个完整原生 action suffix；外层闭合标签可省略。未闭合工具块、超长 Teacher 决定或尾部额外诊断仍拒绝。源轨迹中未解析成功的 attempted tool-call block 也可以被明确定位修复。
- 对唯一变化为 .openclaw/workspace-state.json 初始化时间的情况，必须用终态文件哈希证明它与各自修复前内容相同，且版本、字段存在性、权限、所有其他文件及工具观察均一致；否则仍待审查。该检查不证明外部进程/网络状态完全相同。
- 当前共 62 项相关 CPU 回归测试通过，包括完整 Teacher 格式重试流程、隐藏反馈隔离、原生动作结束边界及元数据例外限制。
- 复验端口另设 31880/31881/39881；仍为一卡推理、其余七卡烤机。正式 16×4 尚未启动，optimizer_updates 仍为 0。

第二轮于 11:56:06 UTC 结束，仍未通过工程门槛：0/5 接受对，未进入修复分支。task_0162 返回 no_repair；task_0126 与 task_0140 经有界格式反馈后仍无有效决定；task_0158 的审查上下文耗尽。原始响应证明 8B 模型倾向继续任务或自然语言汇报，说明自由文本 XML 提交协议与该 SFT 模型的原生工具训练契约不匹配。

同时发现静态 anchor 覆盖限制：task_0140 的唯一隐藏失败为 sre_keywords_ok，但其 anchor 将通用关键词条件绑定到 backend_engineer.md，无法向 Teacher 提供精确的 SRE 文件要求；task_0162 的 anchor 描述了排除窗口和 sample standard deviation，却不足以保证模型心算值正确。按协议，不把隐藏逐项 verifier 失败注入 Teacher，因此这些情况保留为方法覆盖率损失。

## 第三轮工程接口复验

- Teacher 审查改用模型原生、受 JSON schema 约束的三个 reviewer tools：read_trace、submit_repair、decline_repair。原始 read/write/edit/exec/process/web_fetch schema 仅作为引用记录，不可在审查阶段调用。
- submit_repair 参数仍包含 agent_step、call_index、diagnosis 和完整 replacement_action。仅 replacement_action 重新 tokenize 后成为辅助训练目标；submit 包装、诊断和 reviewer tokens 均不进入动作 span。
- 这项修改只修复结构化交互接口，不提供修复位置、答案、reward 或 criterion pass/fail，不改变任务选择和 Teacher 模型。语义错误仍由真实重放与 verifier 拒绝。
- 长轨迹若完整 reviewer 输入超过 32768 tokens，或任一原始动作超过 12000 字符，直接使用机械目录和 read_trace 分页，避免完整历史与重复读取同时占用 64K。
- 63 项修复相关 CPU 回归测试通过。下一步在新目录进行真实 SGLang 约束解码 smoke；若仍不能产生可执行候选，不再继续 prompt 迭代或扩大到固定 64。

第三轮 a3 在首个任务即发现并终止了一个确定的桥接缺陷：SGLang 对 tool_choice=required 生成的约束格式是 JSON 调用数组（真实输出如 [{"name":"read_trace","parameters":...}]），而 qwen 原生 parser 只识别 <tool_call> 标签。三个合法受约束 read_trace 调用因此被误判。该运行由专属 tmux 清理钩子以 exit 130 停止，未进入修复重放，随后恢复八卡烤机。

修复方式严格依据当次 sampling_params.json_schema：只接受非空调用数组、已声明 reviewer tool、精确对象字段及通过 schema 的参数，并生成确定性 reviewer call ID。该 JSON 是审查控制面，不作为动作 target；submit_repair 内的 replacement_action 仍按原生 qwen 工具语法独立解析和 tokenize。64 项相关回归测试通过。下一轮使用新输出目录，a3 保留为接口失败证据。

## 后续接口收敛与最终工程 smoke

a4--a10 逐项定位并修复了约束解码与 Teacher 状态机问题：单次 reviewer
只允许一个调用；完整 JSON 后的约束空白可在长度结束时接受；重复分页读取不再
重复注入原文；连续重复后强制最终决定；定位阶段与原生动作生成拆成两个 pass；
Teacher call index 明确为 0-based；内部审计字段不再污染严格提交 schema；长轨迹
选中 step 后必须先曝光完整原文，再由同一 Teacher 在下一次 decision-only 请求中
确认或放弃。替代动作由受约束 JSON 机械转换为原生 Qwen tool-call 文本，只有该
动作 span 可成为训练目标。

最终代码提交包括：

- `c88fa26a`：两阶段、原生环境工具动作生成；
- `e4a6d02c`：明确 0-based Teacher call index；
- `1649f3a9`：补齐选中 step 的精确证据曝光；
- `0db6359f`：拒绝语义 no-op，并允许 action pass 主动放弃；
- `dce866f0`：曝光原文后必须再次确认，不能直接生成动作。

当前 72 项 Verified Repair 回归测试全部通过。

### A11：修复前对照

A11 使用 5 条固定 source，真实执行了三条修复分支：

| 任务 | 现象 | 最终判定 |
| --- | --- | --- |
| 0126 | 替代动作只是把原来的 7 个 read 重新输出一遍 | reward 0→0，无提升 |
| 0140 | Teacher 对完全相同的路径/大小写产生自相矛盾诊断 | reward 0.95→0.95，无提升 |
| 0158 | 初筛 0→0.470588，但两个固定配对中 original 与 repair 分别同为 0.470588、0.529412，并伴随 criterion 互换 | repeat_not_confirmed |
| 0162 | 三次确定性动作生成均在 8192 tokens 处形成未闭合 JSON | teacher_invalid_proposal |

A11 因而为 0/5 接受。它证明真实重放和配对复验能阻止“续跑随机性”伪装成修复
收益，同时暴露了强制动作生成造成的 no-op 与伪修复问题。

### A12：最终工程 smoke

A12 复用相同 5 条 source，在全新目录和端口上验证上述修复：

| 任务 | 新行为 | 最终判定 |
| --- | --- | --- |
| 0126 | 规范化工具名和参数后识别为原动作语义 no-op，未启动分支 | teacher_no_proposal |
| 0133 | source 已满分 | source_already_satisfies_all_criteria |
| 0140 | action pass 调用 abandon_repair_action，未执行伪修复 | teacher_no_repair |
| 0158 | 初筛 0→0.588235，但固定配对复验不成立 | repeat_not_confirmed |
| 0162 | 合法生成并真实执行 8 个 exec，完整续跑 12 step | reward 0.866667→0.866667，无提升 |

A12 总耗时 717.0 秒，5/5 pair 均终态结束，optimizer update 为 0，最终 0/5
接受。该结果通过的是工程门槛：端口、SGLang、Teacher、远端专属容器、前缀重放、
完整续跑、终态归档、verifier、状态审计、no-op 拒绝和主动放弃均工作。它没有
通过、也不替代数据质量门槛。

## 固定 16×4 数据实验（A13，已结束）

- 输出：`outputs/verified-repair-v3-20260907-fixed16x4-a13`；
- 启动：2026-09-07 13:49:48 UTC；
- 规模：16 tasks × 4 samples = 64 条 source，原 12 train／4 held-out 划分；
- 并发：4；GPU 0 为确定性 8B SGLang，GPU 1--7 持续烤机；
- 隔离端口：32980／32981／40981；
- 本阶段 optimizer update 为 0；
- 数据 gate：至少 16 个 accepted train pair，且覆盖至少 8 个 train task。

只有 A13 通过数据 gate，才比较同一数据和预算下的 NLL、DPO 与 DPO +
all-blocked multi-position margin。未通过则按预注册条件停止，不启动训练。

A13 于 2026-09-07 15:36:33 UTC 正常结束，总墙钟 6391.8 秒。64/64 条
source 与 64/64 个候选均落盘，64 条 source 和 9 条 repair branch 均为
`completed`；optimizer update 始终为 0。最终只有 9 个合法 proposal：5 个
完整续跑后 reward/criterion 不变，4 个发生 criterion 回退，初筛改善为 0，
因而没有候选进入两次配对复验。最终 accepted train pair/task 为 0/0，明确
未达到 16 pair/8 task 门槛，结论为 NO-GO，不启动 NLL、DPO 或 Margin 训练。

其余候选包括 16 个规范化后的语义 no-op、5 个明确 no-repair 和 34 个无效
proposal。无效 proposal 中 28 个是 `call_index: -`，至少 20 个同时用文字表达
“已完成/无需修复”，应同时记为控制格式不稳，不能全部解释为轨迹理解失败。
但即使宽松合并为 decline，9 个真实可执行修复仍为 0 改善，方法结论不变。

长轨迹目录模式共 13 条，合法 proposal 4 条，平均 Teacher 耗时 164.4 秒；
完整历史模式 51 条，合法 proposal 5 条，平均 18.8 秒。目录分页没有阻断修复
提案，但平均审查成本约为 8.7 倍。18 份原生 sampler audit 覆盖 32993 个动作
token，其中 270 个（0.818%）位于实际采样 support 外；Margin 触发集合存在，
但因可靠偏好对为 0，不具备训练前提。

终态 `data-summary.json` SHA256 为
`47f7ca6542dffb297f42701e66acca61967d68d47e171da15396573c968019ca`；
`data-execution.json` SHA256 为
`53431cd735c6a2d8fd62605e72d521b757d3c246ed9f4aa4cff21dd3acf07074`。
远端按本次专属命名前缀核对容器残留为 0，本地 runner/SGLang 已退出，八卡
烤机恢复。简洁终稿见 `reports/VERIFIED_ACTION_REPAIR_V3_FINAL_ZH.md`。

## 启动入口

在 tmux 中运行（由 tmux server 持久托管）：

~~~bash
bash scripts/run_verified_repair_replay_stage.sh \
  /root/openclaw-rl-local/outputs/verified-repair-v3-20260907-replay-a1 smoke 2
~~~

启动器只停止已核对命令的本机烤机会话；结束后检查 GPU 进程，避免重叠，并恢复八卡烤机。它不包含全机 ray stop、全局 Docker 清理或权重删除。

完成时更新此文档，附真实 prompt、三条人工可核查案例、采样器生效证据、复验结果、成本和明确结论。所有运行脚本必须只管理本次注册的远端容器；结束后恢复八卡烤机。
