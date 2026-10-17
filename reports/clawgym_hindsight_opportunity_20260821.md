# ClawGym failure-hindsight opportunity audit

## Coverage

- Parsed trajectories: **6,395**
- Rollouts: **100** (0–99)
- Unique tasks: **800**
- Complete 8-trajectory task groups: **796**
- Incomplete groups: **4**

## Group-level reusable signal

- Mixed-reward groups: **750/796 (94.2%)**
- All-zero groups: **3**
- All-perfect groups: **14**
- Equal partial-reward groups: **29**
- Trajectories strictly below a sibling best: **4,118 (64.7%)**
- Low-reward (<=0.5) trajectories with a better sibling: **1,191**
- Zero-reward trajectories with a positive-reward sibling: **364**
- Median reward gap within mixed groups: **0.3000**; P90 **0.8677**

## Reward and cost buckets

| Reward bucket | Count | Share | Mean steps | P90 steps | Max-step share |
|---|---:|---:|---:|---:|---:|
| zero | 389 | 6.1% | 34.39 | 96.0 | 23.1% |
| low_(0,0.5] | 992 | 15.5% | 29.47 | 96.0 | 14.0% |
| mid_(0.5,1) | 4,479 | 70.0% | 15.66 | 37.0 | 2.4% |
| perfect | 535 | 8.4% | 11.25 | 21.6 | 0.9% |

## Terminations

- `natural_completion`: 5,615
- `max_agent_steps`: 337
- `length_limited_incomplete_tool_call`: 326
- `length_limited_valid_response`: 117

## Interpretation

Mixed groups expose an immediate same-task ranking signal, while zero/constant groups are precisely where scalar GRPO has no within-group advantage signal. The cost bucket table shows whether failure-conditioned repair can replace full resampling with short suffix rollouts. This audit does not yet claim causal recoverability; that requires replaying selected workspace states.
