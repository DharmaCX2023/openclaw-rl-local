"""Artifact-group GRPO signal construction, independent of model execution.

A scores are supplied by a checked verifier/path manifest, never inferred from
file existence. A group compares the SAME artifact across logical trajectories.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict


def artifact_scores(components, manifest):
    """Attribute verifier outputs to artifacts using an explicit reviewed map.

    manifest = {artifact_id: {checks: {output_key: positive_weight}}}.
    Multi-artifact checks must have explicit apportioned weights in the manifest.
    Missing/invalid scores are errors, not zero-reward artifacts.
    """
    result = {}
    for artifact, spec in manifest.items():
        checks = spec['checks']
        if not checks:
            raise ValueError(f'{artifact}: no verifier checks mapped')
        weighted = mass = 0.0
        for key, weight in checks.items():
            value = components[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{artifact}/{key}: invalid verifier score')
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f'{artifact}/{key}: invalid weight')
            weighted += float(value) * weight
            mass += weight
        result[artifact] = weighted / mass
    return result


def _normalize(values, eps=1e-6):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('non-finite or missing group scores')
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return [(v - mean) / (std + eps) for v in values]


def build_group_signals(trajectories):
    """One row per logical trajectory, not one row per routing/context segment.

    Input: group_id, trajectory_id, outcome, artifact_scores, artifact_spans,
    turns. artifact_spans maps stable artifact IDs to inclusive turn bounds.
    """
    groups = defaultdict(list)
    seen = set()
    for row in trajectories:
        identity = (row['group_id'], row['trajectory_id'])
        if identity in seen:
            raise ValueError('duplicate logical trajectory in group statistics')
        seen.add(identity)
        if row['turns'] < 1:
            raise ValueError('empty trajectory')
        groups[row['group_id']].append(row)
    result = {}
    for group in groups.values():
        artifacts = set(group[0]['artifact_scores'])
        if not artifacts or any(set(r['artifact_scores']) != artifacts for r in group):
            raise ValueError('artifact score IDs must match across the entire group')
        global_adv = _normalize([r['outcome'] for r in group])
        local = {a: _normalize([r['artifact_scores'][a] for r in group]) for a in sorted(artifacts)}
        for index, row in enumerate(group):
            turns = row['turns']
            values = [0.0] * turns
            active = [[] for _ in range(turns)]
            spans = {}
            fallback = []
            for a, normalized in local.items():
                span = row['artifact_spans'].get(a)
                if span is None:
                    span = (1, turns)
                    fallback.append(a)
                start, end = span
                if not (isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= turns):
                    raise ValueError(f'invalid artifact span: {a}: {span}')
                spans[a] = [start, end]
                for turn in range(start-1, end):
                    values[turn] += normalized[index]
                    active[turn].append(a)
            segments = []
            for turn, covering in enumerate(active, 1):
                if segments and segments[-1]['artifacts'] == covering:
                    segments[-1]['end'] = turn
                else:
                    segments.append(dict(start=turn, end=turn, artifacts=covering, local=values[turn-1]))
            result[(row['group_id'], row['trajectory_id'])] = dict(
                outcome_advantage=global_adv[index],
                artifact_advantages={a: v[index] for a, v in local.items()},
                effective_spans=spans, full_trajectory_fallback=fallback,
                turn_signals=values, segments=segments,
            )
    return result


def project_tokens(signal, chunks, strength=1.0):
    """Project all chunks of ONE trajectory together (to avoid chunk bias).

    chunks: [{turn_ids: 1-based IDs (0 allowed only under a zero loss mask),
              loss_mask: [...]}, ...].
    Return per-chunk token advantages plus endpoint diagnostics. Implements
    the draft's endpoint-constrained baseline + centered local allocation.
    Constant local signals are recorded but cancel under the endpoint constraint.
    """
    local_chunks = []
    total = mass = 0.0
    for chunk in chunks:
        ids, masks = chunk['turn_ids'], chunk['loss_mask']
        if len(ids) != len(masks):
            raise ValueError('token turn IDs and loss mask length differ')
        local = []
        for turn, mask in zip(ids, masks):
            if mask not in (0, 1):
                raise ValueError('expected binary policy loss mask')
            if mask:
                if not 1 <= turn <= len(signal['turn_signals']):
                    raise ValueError(f'trainable token has invalid turn ID {turn}; archived turns={len(signal["turn_signals"])}')
                value = signal['turn_signals'][turn-1]
                total += value
                mass += 1
            else:
                value = 0.0
            local.append(value)
        local_chunks.append(local)
    if mass == 0:
        raise ValueError('logical trajectory has no trainable tokens')
    center = total / mass
    g = signal['outcome_advantage']
    projected = [[g + strength * (v-center) if mask else 0.0
                  for v, mask in zip(values, chunk['loss_mask'])]
                 for values, chunk in zip(local_chunks, chunks)]
    endpoint = sum(sum(v) for v in projected) / mass
    return projected, dict(trainable_tokens=int(mass), local_mean=center,
                           cumulative_endpoint=endpoint, endpoint_error=endpoint-g)


def project_samples(samples, records, strength=1.0):
    """Join explicit scored records to samples, preserving logical trajectory IDs.

    records keyed by (group_index, sample.index). sample.opsd_step_ids must
    already be 1-based trajectory turn IDs. This is the routing replay path.
    """
    grouped = defaultdict(list)
    for position, sample in enumerate(samples):
        identity = (sample.group_index, sample.index)
        if identity not in records:
            raise ValueError(f'no verified artifact record for trajectory {identity}')
        grouped[identity].append((position, sample))
    signals = build_group_signals([records[k] for k in grouped])
    output = [None] * len(samples)
    audit = []
    for key, chunks in grouped.items():
        data = []
        for _, sample in chunks:
            if sample.opsd_step_ids is None or sample.loss_mask is None:
                raise ValueError('artifact GRPO needs explicit token turn IDs and loss masks')
            if len(sample.opsd_step_ids) != sample.response_length:
                raise ValueError('token turn IDs do not align with response')
            data.append(dict(turn_ids=sample.opsd_step_ids, loss_mask=sample.loss_mask))
        try:
            values, diagnostics = project_tokens(signals[key], data, strength)
        except ValueError as exc:
            raise ValueError(f'trajectory={key}, session={(chunks[0][1].metadata or {}).get("session_id")}: {exc}') from exc
        for (position, _), v in zip(chunks, values):
            output[position] = v
        audit.append(dict(group_id=key[0], trajectory_id=key[1], **signals[key], **diagnostics))
    return output, audit
