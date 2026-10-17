from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
SCRIPT = REPO_ROOT / "scripts" / "workspace_provenance.py"
SPEC = importlib.util.spec_from_file_location("workspace_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)

EXTRACTOR_SPEC = importlib.util.spec_from_file_location(
    "extract_mosaic_artifact_credit",
    REPO_ROOT / "scripts" / "extract_mosaic_artifact_credit.py",
)
assert EXTRACTOR_SPEC is not None and EXTRACTOR_SPEC.loader is not None
extractor = importlib.util.module_from_spec(EXTRACTOR_SPEC)
sys.modules[EXTRACTOR_SPEC.name] = extractor
EXTRACTOR_SPEC.loader.exec_module(extractor)
JOIN_SPEC = importlib.util.spec_from_file_location(
    "join_mosaic_provenance_credit",
    REPO_ROOT / "scripts" / "join_mosaic_provenance_credit.py",
)
assert JOIN_SPEC is not None and JOIN_SPEC.loader is not None
join = importlib.util.module_from_spec(JOIN_SPEC)
sys.modules[JOIN_SPEC.name] = join
JOIN_SPEC.loader.exec_module(join)


def _observer(tmp_path: Path):
    def no_remote(*args, **kwargs):
        raise AssertionError("remote transport should not be used")

    return provenance.WorkspaceProvenanceObserver(
        tmp_path / "ledger.jsonl", no_remote
    )


def _runtime(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SimpleNamespace(
        session_id="session_demo",
        workspace_dir=workspace,
        remote_root=None,
    )


def _records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_manifest_excludes_protected_and_runtime_paths(tmp_path):
    runtime = _runtime(tmp_path)
    (runtime.workspace_dir / "output").mkdir()
    (runtime.workspace_dir / "output" / "answer.txt").write_text("ok")
    (runtime.workspace_dir / "input").mkdir()
    (runtime.workspace_dir / "input" / "secret.txt").write_text("protected")
    (runtime.workspace_dir / "reward").mkdir()
    (runtime.workspace_dir / "reward" / "check.py").write_text("protected")
    (runtime.workspace_dir / "AGENTS.md").write_text("runtime")

    manifest = provenance._local_manifest(runtime.workspace_dir, 1024)

    assert set(manifest) == {"output/answer.txt"}


def test_single_tool_result_maps_exact_file_delta(tmp_path):
    runtime = _runtime(tmp_path)
    observer = _observer(tmp_path)
    observer.register(runtime)
    output = runtime.workspace_dir / "output"
    output.mkdir()
    (output / "answer.txt").write_text("answer")
    server = SimpleNamespace(
        _conversation_logs={
            runtime.session_id: [
                {"tool_calls": [{"id": "call_1", "name": "write"}]}
            ]
        }
    )

    observer.observe(
        server,
        runtime.session_id,
        [{"tool_call_id": "call_1", "content": "Successfully wrote file"}],
    )

    event = _records(observer.record_file)[-1]
    assert event["provenance_eligible"] is True
    assert event["attribution_unit"] == "single_action"
    assert event["tool_call_ids"] == ["call_1"]
    assert event["mutations"][0]["path"] == "output/answer.txt"
    assert event["mutations"][0]["change"] == "added"
    assert event["mutations"][0]["after"]["sha256"]


def test_multi_tool_turn_is_kept_only_as_action_group(tmp_path):
    runtime = _runtime(tmp_path)
    observer = _observer(tmp_path)
    observer.register(runtime)
    (runtime.workspace_dir / "a.txt").write_text("A")
    server = SimpleNamespace(
        _conversation_logs={
            runtime.session_id: [
                {
                    "tool_calls": [
                        {"id": "call_a", "name": "write"},
                        {"id": "call_b", "name": "read"},
                    ]
                }
            ]
        }
    )

    observer.observe(
        server,
        runtime.session_id,
        [
            {"tool_call_id": "call_a", "content": "wrote"},
            {"tool_call_id": "call_b", "content": "read"},
        ],
    )

    event = _records(observer.record_file)[-1]
    assert event["provenance_eligible"] is True
    assert event["attribution_unit"] == "action_group"


def test_background_result_taints_current_and_future_events(tmp_path):
    runtime = _runtime(tmp_path)
    observer = _observer(tmp_path)
    observer.register(runtime)
    server = SimpleNamespace(
        _conversation_logs={
            runtime.session_id: [
                {"tool_calls": [{"id": "call_exec", "name": "exec"}]}
            ]
        }
    )
    (runtime.workspace_dir / "partial.txt").write_text("partial")
    observer.observe(
        server,
        runtime.session_id,
        [
            {
                "tool_call_id": "call_exec",
                "content": "Command still running (session blue-oak, pid 3).",
            }
        ],
    )
    first = _records(observer.record_file)[-1]
    assert first["provenance_eligible"] is False
    assert "background_process_started" in first["rejection_reasons"]

    server._conversation_logs[runtime.session_id].append(
        {"tool_calls": [{"id": "call_write", "name": "write"}]}
    )
    (runtime.workspace_dir / "later.txt").write_text("later")
    observer.observe(
        server,
        runtime.session_id,
        [{"tool_call_id": "call_write", "content": "wrote"}],
    )
    second = _records(observer.record_file)[-1]
    assert second["provenance_eligible"] is False
    assert "inherited_background_taint" in second["rejection_reasons"]


def test_join_selects_one_matching_write_inside_multi_tool_turn():
    wanted = "long announcement"
    digest = __import__("hashlib").sha256(wanted.encode()).hexdigest()
    actions = [
        {
            "name": "write",
            "path": "output/announcements/long.md",
            "arguments": {"content": wanted},
            "call_index": 0,
        },
        {
            "name": "write",
            "path": "output/announcements/short.md",
            "arguments": {"content": "short"},
            "call_index": 1,
        },
        {"name": "read", "path": "input/a", "arguments": {}, "call_index": 2},
    ]

    action, reasons = join._choose_action(
        actions, "output/announcements/long.md", digest
    )

    assert reasons == []
    assert action["call_index"] == 0


def test_join_uses_single_exec_for_script_generated_artifact():
    action, reasons = join._choose_action(
        [{"name": "exec", "path": None, "arguments": {}, "call_index": 0}],
        "output/generated.json",
        "unused",
    )
    assert reasons == []
    assert action["name"] == "exec"


def test_join_rejects_artifact_if_same_path_mutates_later():
    events = [
        {"event_index": 1, "mutations": [{"path": "output/a.txt"}]},
        {"event_index": 2, "mutations": [{"path": "output/a.txt"}]},
    ]
    assert join._later_path_mutation(events, 1, "output/a.txt") is True
    assert join._later_path_mutation(events, 2, "output/a.txt") is False
