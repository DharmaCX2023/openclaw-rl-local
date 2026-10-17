# ClawGym verifier structure audit

## All 2,000 train tasks

- Tasks: **2,000**
- Static criterion dictionaries: **1,838 (91.9%)**
- Criterion vector emitted in verifier JSON: **739 (37.0%)**
- Criteria per structured verifier: mean **15.63**, median **15.0**, P90 **25.0**, max **56**
- Verifiers invoking subprocesses: **146**
- Verifiers with explicit timeout: **107**
- Verifiers printing JSON: **1,996**

## Formal-RL 800 tasks

- Tasks: **800**
- Static criterion dictionaries: **736 (92.0%)**
- Criterion vector emitted in verifier JSON: **308 (38.5%)**
- Criteria per structured verifier: mean **15.75**, median **15.0**, P90 **24.0**, max **39**
- Verifiers invoking subprocesses: **57**
- Verifiers with explicit timeout: **42**
- Verifiers printing JSON: **798**

## Implication

A substantial subset already computes criterion-level booleans and emits them in the same JSON object as the scalar reward. Capturing this vector adds essentially no verifier execution cost. It can support criterion coverage, all-failed-group learning, and artifact-localized credit; unsupported verifiers must fall back to scalar GRPO/OPSD.
