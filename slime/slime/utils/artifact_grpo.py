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
    A successful verifier may omit checks that did not pass. Treat those checks
    as false/zero, matching verifier implementations that initialize sparse
    result dictionaries. Present non-numeric values remain errors.
    """
    result = {}
    for artifact, spec in manifest.items():
        checks = spec['checks']
        if not checks:
            raise ValueError(f'{artifact}: no verifier checks mapped')
        weighted = mass = 0.0
        for key, weight in checks.items():
            value = components.get(key, 0.0)
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


def project_samples(samples, records, strength=1.0, skip_broken=False):
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
    output = [None] * len(samples)
    audit = []

    def drop(key, chunks, message):
        print(f'[ArtifactGRPO] dropped unusable trajectory: {message}', flush=True)
        for position, sample in chunks:
            output[position] = [0.0] * sample.response_length
            if sample.loss_mask is not None:
                sample.loss_mask[:] = [0] * len(sample.loss_mask)
        audit.append(dict(group_id=key[0], trajectory_id=key[1], dropped=message))

    # Validate each logical trajectory before group normalization. Otherwise a
    # token-alignment failure discovered later would still distort its siblings'
    # means and standard deviations.
    prepared = {}
    for key, chunks in grouped.items():
        data = []
        try:
            # A one-row signal build validates score types, turns, and spans
            # without coupling this trajectory to any sibling.
            build_group_signals([records[key]])
            turns = records[key]['turns']
            trainable = 0
            for _, sample in chunks:
                if sample.opsd_step_ids is None or sample.loss_mask is None:
                    raise ValueError('artifact GRPO needs explicit token turn IDs and loss masks')
                if len(sample.opsd_step_ids) != sample.response_length:
                    raise ValueError('token turn IDs do not align with response')
                if len(sample.loss_mask) != sample.response_length:
                    raise ValueError('loss mask does not align with response')
                for turn, mask in zip(sample.opsd_step_ids, sample.loss_mask):
                    if mask not in (0, 1):
                        raise ValueError('expected binary policy loss mask')
                    if mask:
                        trainable += 1
                        if not 1 <= turn <= turns:
                            raise ValueError(
                                f'trainable token has invalid turn ID {turn}; archived turns={turns}'
                            )
                data.append(dict(turn_ids=sample.opsd_step_ids, loss_mask=sample.loss_mask))
            if trainable == 0:
                raise ValueError('logical trajectory has no trainable tokens')
        except (KeyError, TypeError, ValueError) as exc:
            message = (
                f'trajectory={key}, session={(chunks[0][1].metadata or {}).get("session_id")}: {exc}'
            )
            if not skip_broken:
                raise ValueError(message) from exc
            drop(key, chunks, message)
            continue
        prepared[key] = (chunks, data)

    # Score IDs must match only within a prompt group. If a malformed record is
    # the sole outlier, discard it instead of crashing all 64 trajectories.
    by_group = defaultdict(list)
    for key in prepared:
        by_group[key[0]].append(key)
    for group_id, keys in by_group.items():
        by_artifacts = defaultdict(list)
        for key in keys:
            by_artifacts[tuple(sorted(records[key]['artifact_scores']))].append(key)
        keep_artifacts, keep_keys = max(
            by_artifacts.items(), key=lambda item: (len(item[1]), item[0])
        )
        for artifact_ids, mismatched_keys in by_artifacts.items():
            if artifact_ids == keep_artifacts:
                continue
            for key in mismatched_keys:
                chunks, _ = prepared.pop(key)
                message = (
                    f'trajectory={key}, session={(chunks[0][1].metadata or {}).get("session_id")}: '
                    f'artifact score IDs {artifact_ids} do not match group IDs {keep_artifacts}'
                )
                if not skip_broken:
                    raise ValueError(message)
                drop(key, chunks, message)

        valid_keys = [key for key in keep_keys if key in prepared]
        if not valid_keys:
            continue
        try:
            signals = build_group_signals([records[key] for key in valid_keys])
        except (KeyError, TypeError, ValueError) as exc:
            message = f'group={group_id}: unusable artifact group signal: {exc}'
            if not skip_broken:
                raise ValueError(message) from exc
            for key in valid_keys:
                chunks, _ = prepared.pop(key)
                drop(key, chunks, message)
            continue
        for key in valid_keys:
            chunks, data = prepared[key]
            try:
                values, diagnostics = project_tokens(signals[key], data, strength)
            except ValueError as exc:
                message = (
                    f'trajectory={key}, session={(chunks[0][1].metadata or {}).get("session_id")}: {exc}'
                )
                if not skip_broken:
                    raise ValueError(message) from exc
                drop(key, chunks, message)
                continue
            for (position, _), value in zip(chunks, values):
                output[position] = value
            audit.append(dict(group_id=key[0], trajectory_id=key[1], **signals[key], **diagnostics))

    if skip_broken and not any('dropped' not in row for row in audit):
        raise RuntimeError('No usable artifact-GRPO trajectories remain after validation')
    return output, audit
