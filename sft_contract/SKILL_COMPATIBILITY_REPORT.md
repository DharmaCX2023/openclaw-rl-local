# SFT skill compatibility report

Target contract: OpenClaw `2026.4.1`
(`da64a978e5814567f7797cc34fbe29b61b7eae7a`), six tools, 65,536-token
maximum. No training was started.

## Fixed runtime skill catalog

The generated system prompt always advertises these seven skills:

- `clawflow`
- `clawflow-inbox-triage`
- `healthcheck`
- `node-connect`
- `skill-creator`
- `weather`
- `serper`

The six tools and the seven skills are different concepts. Skills are
instructions; they implement their operations through the allowed tools.

## Runtime skill observations that are rejected

- Old Serper revision: 396 trajectories. Examples: source lines 14, 57, 79,
  179, 224. The complete list is under
  `reject_source_lines.runtime_skill_revision_mismatch:serper_old` in the JSON
  report.
- Limited Serper `SKILL.md` reads at source lines 16691 and 17205 are retained because both calls explicitly request `limit=30`.
- A different bundled-skill revision: source line 21425 contains incompatible
  `github`, `gh-issues`, and `weather` observations.

This does **not** mean that Serper or weather is removed. Canonical Serper is
retained, and 854 whitespace-only Serper observations were normalized to the
canonical content. The fixed prompt retains weather and Serper.

## Skills that remain valid

- Task-owned skills below `input/skills/...` and equivalent task workspace
  paths remain part of their trajectories.
- Exact v2026.4.1 bundled skill observations remain.
- Reads of absent paths that returned a reproducible `ENOENT` error remain;
  they are failed lookup attempts, not incompatible skill contents.

## Output

- Kept: 22,631 trajectories.
- Rejected: 1,866 trajectories.
- Removed empty assistant messages: 653.
- Over 65,536 tokens: 54 trajectories.
- Maximum retained length: 64,999 tokens.

The machine-readable report and every rejected source trajectory are stored
beside the generated dataset.
