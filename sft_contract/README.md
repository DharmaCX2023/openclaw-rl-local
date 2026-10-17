# ClawGym SFT Version Alignment

## Result

The raw 24,497-trajectory SFT corpus identifies OpenClaw `2026.4.1` as the canonical runtime code family. The working tree has been upgraded to that release and the harness-specific request metadata has been ported. No training was started.

## Evidence

- Every raw trajectory has the same system prompt (SHA-256 `2866a28510c621af541c59a78644858d57da5509d1110a37f5e73c8a95ce60f1`).
- Official OpenClaw tag `v2026.4.1`, commit `da64a978e5814567f7797cc34fbe29b61b7eae7a`, generates that prompt byte-for-byte when supplied the catalog and runtime context recorded by the dataset.
- The official ClawGym RL README selects `ghcr.io/openclaw/openclaw:2026.4.1`.
- The six bundled skills listed by the SFT prompt are present in that tag. Five are actually read in the trajectories and their dominant contents match the tag byte-for-byte; `node-connect` is listed but not read.
- The 19-tool schema in the raw SFT file exactly matches the hard-coded `tools` list in ClawGym's SFT preprocessing script. It is preprocessing metadata, not evidence for a different OpenClaw runtime.

## Serper exception

`serper` is a managed skill outside the OpenClaw repository. Its exact Git source was not published in the official repositories and exact-phrase GitHub searches found no match. The SFT observations expose the `SKILL.md`, so the dominant clean `0.1.0` file is recovered under `sft_contract/recovered_skills/serper/`. They do not expose `scripts/search.py`, so this recovered directory is an SFT contract artifact, not yet a runnable claim of exact RL reproduction.

Observed serper trajectories:

- 1,590: canonical 1,446-byte file.
- 854: same logical content with two trailing spaces. These are normalized to the canonical file.
- 396: older 1,207-byte revision without the mandatory-search hint.
- 2: truncated read observations.

## Verification completed

- OpenClaw reports `2026.4.1`.
- The migrated extra-params test file passes all 116 tests.
- A full OpenClaw build completes successfully.
- `X-Session-Id` and `X-Turn-Type` propagation is retained and covered by a regression test.

## SFT filter completed

The real six-tool prompt and schemas were exported from the pinned runtime. The 65,536-token filter kept 22,631 of 24,497 trajectories and wrote 1,866 auditable reject records. See `SKILL_COMPATIBILITY_REPORT.md` and the generated JSON report beside the dataset. No training was started.
