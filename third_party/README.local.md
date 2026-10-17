# 第三方源码与本地补丁

本目录的两个上游 Git 仓库是本机工作副本，不作为主仓库的嵌套仓库或 gitlink 提交。主仓库保存版本、补丁和恢复说明。

| 项目 | 上游 | 固定版本 | 本地修改 |
| --- | --- | --- | --- |
| OpenRLHF | https://github.com/OpenRLHF/OpenRLHF.git | `3c3be6234e0cb353e76bb8019947db9dfe99fca7` | [SFT 补丁](../patches/openrlhf-clawgym-sft.patch) |
| ClawGym-Agents | https://github.com/ClawGym/ClawGym-Agents.git | `0e8526d8f5e0999896c74cee4d6b998fb8a0429e` | 无；2026-09-07 检查工作树清洁 |

## OpenRLHF 修改范围

- SFT 渲染保留样本 tools schema，并支持预处理数据。
- epoch 尾部与梯度累积／全局 token 归一化窗口保持一致。
- 在不改变 optimizer-step 样本成员的前提下，按长度调度。
- 保存归一化与长度调度回归测试。

这些是既有实验修改的归档，本次没有改变训练逻辑。

本机 OpenRLHF 提交：`a7fa0f5a590b1f84d7d58fb5d908dbe41a33a1ea`。该提交未推送到上游；其他机器应使用固定上游版本加补丁恢复。

补丁 SHA256：`8798b5ff8e9fb77aeb1d97ea355e85a079c14a40b0c0aad10f489f84ca38d291`。

## 在新机器恢复

以下命令从主仓库根目录运行，**仅用于尚不存在的第三方目录**；不要覆盖已有工作副本。

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
git -C third_party/OpenRLHF switch --detach 3c3be6234e0cb353e76bb8019947db9dfe99fca7
git -C third_party/OpenRLHF apply --check ../../patches/openrlhf-clawgym-sft.patch
git -C third_party/OpenRLHF apply ../../patches/openrlhf-clawgym-sft.patch

git clone https://github.com/ClawGym/ClawGym-Agents.git third_party/ClawGym-Agents-upstream
git -C third_party/ClawGym-Agents-upstream switch --detach 0e8526d8f5e0999896c74cee4d6b998fb8a0429e
```

本次验证：OpenRLHF 源码编译检查及全部 26 项单元测试通过；没有启动训练或改变运行环境。今后修改第三方代码时，必须同步更新此补丁和版本说明，不能只依赖被忽略的工作副本。
