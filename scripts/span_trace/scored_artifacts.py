"""Select file artifacts using an explicit, checker-versioned role manifest.
A verifier file access is evidence for path resolution, not evidence of target role.
Unknown paths never silently become scored artifacts. Role manifests are task-specific.
"""
import hashlib


def resolve_targets(policy, checker_bytes, rollout_date, existing_paths):
    if hashlib.sha256(checker_bytes).hexdigest() != policy['checker_sha256']:
        raise ValueError('Verifier changed: review target roles before counting artifacts')
    targets = {p.replace('{rollout_date}', rollout_date): dict(meta)
               for p, meta in policy['targets'].items()}
    for group in policy.get('alternative_path_groups', []):
        group = [p.replace('{rollout_date}', rollout_date) for p in group]
        selected = next((p for p in group if p in existing_paths), group[0])
        for p in group:
            if p != selected:
                targets.pop(p, None)
        targets[selected]['accepted_aliases'] = group
    return targets


def classify_unwritten(path, initial_paths, terminal_paths, verifier_generated_paths):
    if path in verifier_generated_paths:
        return 'generated_during_original_verification'
    if path in initial_paths and path in terminal_paths:
        return 'initial_target_without_actor_write'
    if path in terminal_paths:
        return 'existing_target_write_unresolved'
    return 'target_absent_without_actor_write'
