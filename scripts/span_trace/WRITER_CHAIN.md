# 实际写入链追溯

入口：`analyze_writer_chain.py AUDIT_ROOT`。输出 `SUMMARY_WRITER_CHAIN.json` 和每条轨迹的 `span_analysis_writer_chain.json`。旧 `analyze.py`/`span_analysis.json` 为读取脚本近似版，不应当作本方法。

1. exec：成功写入 syscall 定位目标文件、PID、toolcall；直接 write/edit：工具目标与前后快照确认，单独标识为 `direct_tool_snapshot`，不声称有写入 PID。
2. 用 execve/execveat 确定写入进程的入口脚本，用 fork/vfork/clone 继承执行祖先；cwd 随 chdir/fchdir 更新。
3. 入口脚本连接到执行前已经记录的写入操作；该操作如由另一个执行文件生成，再递归回溯。绑定当时生产者，禁止引用未来写入。
4. artifact 起点取所有实际写入及上述祖先写入的最早 turn；终点只取该 artifact 本身最后写入 turn。
5. 普通 read/open/cat 不建依赖；同一个 toolcall 的兄弟进程不互相继承脚本。

这是进程/文件级生成来源，不是逐语句数据流分析。动态 import/source、stdin 脚本、管道输入来源及未支持的解释器参数不强行猜测；相对路径 rename、未完整解析的 syscall 与 cwd 缺失仍需看审计计数。文件以路径聚合，删除重建和脚本局部重写尚未按 inode/内容版本拆分。任务 artifact 名单仍来自 prompt 的显式路径候选，并非完整 verifier 文件清单。

测试：`python scripts/span_trace/test_writer_chain.py`；另使用真实 strace smoke archive 验证 generate.py → a.json 和并行无关 b.json。
