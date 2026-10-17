import json
import os
from pathlib import Path

import pytest

from artifact_grpo_bridge import (
    IncompleteArtifactBatch,
    _build_task_manifest,
    _extend_policy,
)


def _session(root: Path, task_dir: Path, session_id: str) -> Path:
    import hashlib

    path = root / hashlib.sha256(session_id.encode()).hexdigest()[:24]
    path.mkdir()
    (path / "runtime.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_id": "task_test",
                "task_dir": str(task_dir),
            }
        )
    )
    (path / "verifier.json").write_text(
        json.dumps(
            {
                "returncode": 0,
                "components": {"a_exists": 1.0, "b_exists": 0.0},
            }
        )
    )
    (path / "span_analysis_writer_chain.json").write_text(
        json.dumps(
            {
                "requested_paths": ["output/a.txt", "output/b.txt"],
                "files": [
                    {"path": "output/a.txt"},
                    {"path": "output/b.txt"},
                ],
            }
        )
    )
    return path


def test_dynamic_manifest_is_frozen_before_scoring(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    (task_dir / "reward" / "check.py").write_text(
        """
from pathlib import Path
def grade(_, workspace):
    scores = {}
    scores["a_exists"] = float((Path(workspace) / "output/a.txt").exists())
    scores["b_exists"] = float((Path(workspace) / "output/b.txt").exists())
    return scores
"""
    )
    root = tmp_path / "sessions"
    root.mkdir()
    session_id = "train-task-test-0"
    session = _session(root, task_dir, session_id)

    manifest = _build_task_manifest("task_test", [session])
    assert set(manifest["artifacts"]) == {"output/a.txt", "output/b.txt"}
    assert manifest["unmapped_checks"] == []

    path = tmp_path / "manifest.json"
    path.write_text("{}")
    identities = {(0, 0): session_id}
    with pytest.raises(IncompleteArtifactBatch, match="missing tasks"):
        _extend_policy(root, identities, {}, path)

    monkeypatch.setenv("OPENCLAW_ARTIFACT_GRPO_DYNAMIC_MANIFEST", "1")
    policy = {}
    _extend_policy(root, identities, policy, path)
    assert json.loads(path.read_text()) == policy
    assert "task_test" in policy


def test_dynamic_manifest_skips_sessions_without_evidence(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    (task_dir / "reward" / "check.py").write_text(
        """
from pathlib import Path
def grade(_, workspace):
    scores = {}
    scores["a_exists"] = float((Path(workspace) / "output/a.txt").exists())
    scores["b_exists"] = float((Path(workspace) / "output/b.txt").exists())
    return scores
"""
    )
    root = tmp_path / "sessions"
    root.mkdir()
    good = _session(root, task_dir, "good")
    bad = root / "deadbeefdeadbeefdeadbeef"
    bad.mkdir()
    (bad / "runtime.json").write_text(
        json.dumps(
            {
                "session_id": "bad",
                "task_id": "task_test",
                "task_dir": str(task_dir),
            }
        )
    )
    manifest = _build_task_manifest("task_test", [bad, good])
    assert "output/a.txt" in manifest["artifacts"]


def test_dynamic_manifest_resolves_paths_loaded_from_task_config(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    (task_dir / "input_files" / "input").mkdir(parents=True)
    (task_dir / "input_files" / "input" / "config.json").write_text(
        json.dumps({"output_path": "output/a.txt"})
    )
    (task_dir / "reward" / "check.py").write_text(
        """
import json
from pathlib import Path
def grade(_, workspace):
    scores = {"a_exists": 0.0}
    cfg = json.loads((Path(workspace) / "input/config.json").read_text())
    artifact = Path(workspace) / cfg.get("output_path")
    if artifact.exists():
        scores["a_exists"] = 1.0
    return scores
"""
    )
    root = tmp_path / "sessions"
    root.mkdir()
    session = _session(root, task_dir, "config-path")

    manifest = _build_task_manifest("task_test", [session])

    assert manifest["artifacts"] == {
        "output/a.txt": {
            "checks": {"a_exists": 1.0},
            "attribution": "runtime frozen verifier dataflow",
        }
    }
    assert manifest["unmapped_checks"] == ["b_exists"]
