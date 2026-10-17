# OpenClaw Harness Migration Handoff

## Objective

Replace the custom Python OpenClaw-compatible harness with the real OpenClaw
runtime, while preserving the GRPO stability fixes and using isolated remote
CPU Docker environments for every rollout.

## Canonical repository and snapshot

- Working repository: `/root/Openclaw-RL-hareness`
- Branch: `openclaw-harness-paper`
- OpenClaw-RL commit: `47e79683e2894ffa722e1629d11362d4fda2b790`
- Canonical OpenClaw version: `2026.4.1`
- Upstream tag/commit: `v2026.4.1` / `da64a978e5814567f7797cc34fbe29b61b7eae7a`
- The working tree has been upgraded to that official release. It contains the
  real Gateway, agent loop, system-prompt builder, tool-policy pipeline, skills
  subsystem, sandbox implementation, and Dockerfiles.
- RL-specific request headers are implemented in:
  `openclaw/src/agents/pi-embedded-runner/extra-params.ts`
  - `X-Session-Id`
  - `X-Turn-Type`

## Resolved version decision

The canonical runtime is OpenClaw `2026.4.1`, not the previously bundled
`2026.3.2-beta.1` snapshot. The decision is based on independent evidence:

1. All 24,497 raw SFT trajectories contain one identical system prompt, SHA-256
   `2866a28510c621af541c59a78644858d57da5509d1110a37f5e73c8a95ce60f1`.
2. Official OpenClaw `v2026.4.1` generates that prompt byte-for-byte when given
   the recorded runtime context and skills catalog.
3. The official ClawGym RL README selects the
   `ghcr.io/openclaw/openclaw:2026.4.1` image.
4. The bundled skill files actually read by the trajectories match the same tag
   byte-for-byte.

The raw 19-tool schemas come from ClawGym's hard-coded SFT preprocessing table;
they are not a competing runtime-version signal. See `sft_contract/` for the
machine-readable version and skill manifest.

## Target tool surface

Use a real OpenClaw tool allowlist containing exactly:

1. `read`
2. `write`
3. `edit`
4. `exec`
5. `process`
6. `web_fetch`

The allowlist must be applied through OpenClaw configuration/tool policy. Do
not hard-code a replacement prompt or hand-written tool schemas.

## Skills are separate from tools

Skills are not additional model tools. OpenClaw lists eligible skills in the
system prompt, and the model reads `SKILL.md` using `read` or runs skill scripts
using `exec`.

Version-aligned skill findings:

- The prompt lists six official bundled skills: `clawflow`,
  `clawflow-inbox-triage`, `healthcheck`, `node-connect`, `skill-creator`, and
  `weather`. All six are present in OpenClaw `2026.4.1`.
- The dominant trajectory observations for the five bundled skills that are
  actually read match the `2026.4.1` files byte-for-byte. `node-connect` is
  listed but is not opened by a raw trajectory.
- `serper` is an extra managed skill, version `0.1.0`, outside the official
  OpenClaw repository. No exact public GitHub source was found.
- Of 2,842 trajectories that contain a Serper `SKILL.md` observation, 1,590 use the canonical clean file,
  854 differ only by two trailing spaces, 396 use an older substantive revision,
  and 2 contain truncated read observations.
- The canonical recovered `serper/SKILL.md` and all hashes are pinned under
  `sft_contract/`. Its exact `scripts/search.py` was not exposed by the dataset,
  so exact RL reproduction of this managed skill remains a documented gate.

The aligned 64K filter has now been regenerated from the pinned runtime prompt
and schemas. It keeps 22,631 rows and writes 1,866 auditable rejects. The old
20,496-row `.precontract` file remains obsolete and is not training-ready.

Choose one skills policy:

- Skills enabled: pin an exact skill manifest and copy the identical skill
  files into every SFT and RL environment. Prompt ordering, descriptions,
  locations, scripts, dependencies, environment variables, and API
  availability must match.
- Skills disabled: remove trajectories that depend on external/runtime skills.
  Use path-aware filtering. Do not remove task trajectories merely because the
  task itself creates or audits files such as `input/SKILL.md`.

## SFT migration

1. Start pinned OpenClaw `2026.4.1` with the six-tool allowlist and selected
   fixed skills policy. Runtime version pinning and the six-tool export are
   complete.
2. Use OpenClaw's actual `buildAgentSystemPrompt()` output as the canonical
   system prompt.
3. Export the actual post-policy tool schemas and their exact ordering.
4. Filter out every trajectory that calls a tool unavailable in the target
   runtime or uses unsupported arguments.
5. Apply the selected runtime-skill filtering policy.
6. Replace/normalize SFT system messages only with the system prompt captured
   from this exact runtime/configuration.
7. Attach the exact exported tool schemas.
8. Retain empty-assistant cleanup and context-length filtering where needed.
9. Verify chat-template token IDs and assistant-only loss masks.

Do not reuse the hand-written prompt and schemas in:

- `scripts/normalize_clawgym_sft_for_openclaw.py`
- `clawgym-rl/sft_harness.py`

Those files describe the previous compatibility implementation, not the real
OpenClaw runtime.

## RL runtime migration

Replace these compatibility components with the real OpenClaw Gateway and
agent loop:

- `clawgym-rl/openclaw_harness.py`
- `clawgym-rl/simple_agent.py`
- the custom `AgentRunner` loop where it duplicates OpenClaw turn handling

The thin ClawGym adapter should only:

1. Allocate a rollout.
2. Seed task input files.
3. Assign a unique OpenClaw session.
4. route model inference to the GPU Slime/SGLang endpoint;
5. run the task verifier and return reward;
6. persist trajectory/token/logprob/audit artifacts;
7. close and clean up the rollout.

OpenClaw itself must own:

- system-prompt construction;
- tool schemas and policy filtering;
- tool-call parsing and execution;
- observation formatting;
- assistant/tool history;
- iteration and completion behavior;
- session JSONL and compaction behavior;
- skills injection;
- sandbox routing.

## Remote CPU Docker architecture

Recommended topology:

```text
GPU node: Slime / SGLang / GRPO
                    |
                    | model requests + token logprobs
                    v
CPU worker: real OpenClaw Gateway + Docker daemon
                    |
                    v
one OpenClaw session + one Docker container per rollout
                    |
                    v
ClawGym verifier and reward
```

Required OpenClaw sandbox semantics:

- `agents.defaults.sandbox.mode = "all"`
- `agents.defaults.sandbox.scope = "session"`
- no elevated/host exec;
- one unique session key per rollout attempt;
- one persistent container/workspace for all turns in that rollout;
- identical initial task snapshot for each GRPO sample of the same query;
- independent process namespace and filesystem for concurrent rollouts.

Rollouts may execute concurrently. "One container per rollout" does not mean
serial execution.

Copy task inputs once at rollout reset, not once per turn. Preserve the same
workspace throughout the multi-turn trajectory. At completion, run the
verifier against that workspace, collect artifacts, and destroy the container.

If ClawGym tasks require different Docker images, the CPU allocator must select
the task-specific image before the OpenClaw session starts. Do not fall back to
host execution or directory-only isolation.

## GRPO and training fixes to preserve

The following changes are independent of the harness migration and should be
kept:

- GRPO reward normalization per trajectory within each prompt group;
- group mean centering and standard-deviation normalization;
- constant-reward groups produce zero advantages;
- dual clipping with `eps_clip_c = 3.0`;
- PPO clipping values currently used by the run configuration;
- entropy coefficient currently set to `0.001`;
- full-history chat-template loss masking;
- no silent token-level middle truncation;
- raw generated tool-call text preserved for token-history alignment;
- transport retries and failed-rollout discard behavior;
- trajectory persistence and checkpoint/evaluation utilities;
- router/SGLang stability fixes.

## Required alignment gate before training

Use at least 100 fixed queries. For SFT reconstruction and RL rollout, compare:

- exact system prompt;
- exact tool schemas and ordering;
- skills block and skill files;
- user prompt;
- every assistant history entry;
- every tool call;
- every observation byte/string;
- chat-template token IDs at every turn;
- generated-token logprobs;
- assistant loss mask;
- completion/iteration-limit behavior;
- workspace outputs and verifier reward.

Do not restart full GRPO until these comparisons pass or every intentional
difference is documented.

## Summary of the intended migration

1. Pin one real OpenClaw version.
2. Configure the exact six-tool allowlist.
3. Pin or disable skills independently.
4. Export the real OpenClaw system prompt and tool schemas.
5. Re-filter and normalize SFT data for that exact configuration.
6. Replace the Python compatibility harness with the real OpenClaw loop.
7. Run every rollout in a unique remote CPU Docker session.
8. Preserve the existing GRPO/dual-clip/entropy/stability fixes.
9. Pass the 100-query alignment gate before training.
