import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tarfile
import urllib.error
from types import SimpleNamespace


import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_rollout as rollout


SEARCH_SCRIPT = REPO_ROOT / "sft_contract/recovered_skills/serper/scripts/search.py"


def _load_search_module():
    spec = importlib.util.spec_from_file_location(
        "serper_search_for_test", SEARCH_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gateway_port_range_is_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_BASE", "30000")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_STRIDE", "2")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET", "1000")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_MAX_SLOTS", "100")

    assert rollout._gateway_port_for_task(3, is_train=False) == 30006
    assert rollout._gateway_port_for_task(3, is_train=True) == 31006

    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_BASE", "65000")
    with pytest.raises(ValueError, match="exceeds 65535"):
        rollout._gateway_port_for_task(99, is_train=True)


def test_gateway_healthcheck_uses_dynamic_port():
    command = rollout._gateway_healthcheck_command(31006)

    assert "http://127.0.0.1:31006/healthz" in command
    assert "18789" not in command


def test_train_batch_is_fixed_by_seed_and_rollout(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_TRAIN_SHUFFLE_SEED", "17")
    args = SimpleNamespace(rollout_batch_size=3, n_samples_per_prompt=2)
    tasks = [
        rollout.TaskSpec(tmp_path, f"task_{index:04d}", "query", None, {})
        for index in range(10)
    ]

    first = rollout._train_batch(args, tasks, rollout_id=4)
    repeated = rollout._train_batch(args, tasks, rollout_id=4)
    next_rollout = rollout._train_batch(args, tasks, rollout_id=5)

    assert [(group, task.task_id) for group, task in first] == [
        (group, task.task_id) for group, task in repeated
    ]
    assert [(group, task.task_id) for group, task in first] != [
        (group, task.task_id) for group, task in next_rollout
    ]
    assert [group for group, _ in first] == [0, 0, 1, 1, 2, 2]


def test_group_interleaved_submission_preserves_stable_batch_indices(tmp_path):
    tasks = [
        rollout.TaskSpec(tmp_path, f"task_{index:04d}", "query", None, {})
        for index in range(3)
    ]
    batch = [
        (group_index, tasks[group_index])
        for group_index in range(3)
        for _ in range(2)
    ]

    assert rollout._train_submission_indices(batch, "sequential") == [0, 1, 2, 3, 4, 5]
    assert rollout._train_submission_indices(batch, "group_interleaved") == [0, 2, 4, 1, 3, 5]
    assert sorted(rollout._train_submission_indices(batch, "group_interleaved")) == list(
        range(len(batch))
    )


def test_train_submission_rejects_unknown_scheduling(tmp_path):
    task = rollout.TaskSpec(tmp_path, "task_0000", "query", None, {})
    with pytest.raises(ValueError, match="unsupported OPENCLAW_TRAIN_SCHEDULING"):
        rollout._train_submission_indices([(0, task)], "unknown")


def test_rollout_registers_task_and_seed_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_TRAIN_SHUFFLE_SEED", "17")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(rollout.requests, "post", fake_post)
    task = rollout.TaskSpec(tmp_path, "task_0042", "query", None, {})
    rollout._register_session_metadata(
        30001,
        "task-12-3",
        task,
        rollout_id=12,
        task_index=3,
        group_index=7,
    )

    assert captured["url"] == "http://127.0.0.1:30001/register_session"
    assert captured["timeout"] == 10
    assert captured["json"]["session_id"] == "task-12-3"
    assert captured["json"]["metadata"] == {
        "task_id": "task_0042",
        "phase": "train",
        "rollout_id": 12,
        "task_index": 3,
        "group_index": 7,
        "task_selection_seed": 17,
        "fixed_rollout_seed": None,
    }


def test_eval_sampling_identity_is_checkpoint_independent():
    first = rollout._eval_sampling_identity("clawgym_eval", "task_0042", 0)
    repeated = rollout._eval_sampling_identity("clawgym_eval", "task_0042", 0)
    second_sample = rollout._eval_sampling_identity("clawgym_eval", "task_0042", 1)

    assert first == repeated
    assert first == "eval:v1:clawgym_eval:task_0042:sample:0"
    assert second_sample != first


def test_eval_registration_sends_stable_sampling_identity(monkeypatch, tmp_path):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(rollout.requests, "post", fake_post)
    task = rollout.TaskSpec(tmp_path, "task_0042", "query", None, {})
    identity = rollout._eval_sampling_identity(
        "clawgym_eval", task.task_id, 0
    )
    rollout._register_session_metadata(
        30001,
        "eval-task-99-42",
        task,
        rollout_id=99,
        task_index=42,
        group_index=None,
        sampling_identity=identity,
    )

    assert captured["json"]["metadata"]["sampling_identity"] == identity
    assert captured["json"]["metadata"]["phase"] == "eval"


def test_load_web_credentials_from_dotenv_without_printing_values(
    monkeypatch, tmp_path, capsys
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SERPER_URL=https://search.invalid\n"
        "SERPER_API_KEY=serper-test-secret\n"
        "JINA_URL=https://reader.invalid\n"
        "JINA_KEY=jina-test-secret\n",
        encoding="utf-8",
    )
    for key in rollout._WEB_CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENCLAW_ENV_FILE", str(env_file))

    credentials = rollout._load_web_credentials()

    assert set(credentials) == set(rollout._WEB_CREDENTIAL_KEYS)
    assert credentials["SERPER_API_KEY"] == "serper-test-secret"
    assert "JINA_URL" not in credentials
    assert "JINA_KEY" not in credentials
    assert capsys.readouterr() == ("", "")


def test_legacy_serper_key_is_translated_to_canonical_runtime_name(
    monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SERPER_URL=https://search.invalid\n"
        "SERPER_KEY=legacy-test-secret\n",
        encoding="utf-8",
    )
    for key in (*rollout._WEB_CREDENTIAL_KEYS, "SERPER_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENCLAW_ENV_FILE", str(env_file))

    credentials = rollout._load_web_credentials()

    assert credentials == {
        "SERPER_URL": "https://search.invalid",
        "SERPER_API_KEY": "legacy-test-secret",
    }
    assert "SERPER_KEY" not in credentials


@pytest.mark.parametrize("missing_key", rollout._WEB_CREDENTIAL_KEYS)
def test_missing_web_credentials_report_names_without_values(
    monkeypatch, tmp_path, missing_key
):
    values = {
        "SERPER_URL": "https://search.invalid",
        "SERPER_API_KEY": "secret-must-not-appear",
    }
    env_file = tmp_path / ".env"
    env_file.write_text(
        "".join(
            f"{key}={value}\n" for key, value in values.items() if key != missing_key
        ),
        encoding="utf-8",
    )
    for key in rollout._WEB_CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENCLAW_ENV_FILE", str(env_file))

    with pytest.raises(RuntimeError) as exc_info:
        rollout._load_web_credentials()

    message = str(exc_info.value)
    assert missing_key in message
    assert "secret-must-not-appear" not in message
    assert "search.invalid" not in message


def test_secret_env_file_is_atomic_0600_and_serper_only():
    credentials = {
        "SERPER_URL": "https://search.invalid",
        "SERPER_API_KEY": "secret-must-not-appear",
    }
    path = rollout._write_secret_env_file("credential-test", credentials)
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.read_text(encoding="utf-8") == (
            "SERPER_URL=https://search.invalid\n" "SERPER_API_KEY=secret-must-not-appear\n"
        )
        assert "JINA" not in path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)



def test_optional_gateway_runtime_env_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("OPENCLAW_BUNDLED_PLUGINS_DIR", raising=False)
    assert rollout._load_optional_gateway_runtime_env() == {}


def test_optional_gateway_runtime_env_accepts_built_plugins(monkeypatch):
    monkeypatch.setenv("OPENCLAW_BUNDLED_PLUGINS_DIR", "/app/dist/extensions")
    assert rollout._load_optional_gateway_runtime_env() == {
        "OPENCLAW_BUNDLED_PLUGINS_DIR": "/app/dist/extensions"
    }


@pytest.mark.parametrize(
    "value", ("app/dist/extensions", "/app/dist/extensions\nINJECTED=yes")
)
def test_optional_gateway_runtime_env_rejects_unsafe_values(monkeypatch, value):
    monkeypatch.setenv("OPENCLAW_BUNDLED_PLUGINS_DIR", value)
    with pytest.raises(RuntimeError):
        rollout._load_optional_gateway_runtime_env()


def test_secret_env_file_includes_validated_optional_runtime_env():
    credentials = {
        "SERPER_URL": "https://search.invalid",
        "SERPER_API_KEY": "secret-must-not-appear",
    }
    path = rollout._write_secret_env_file(
        "runtime-env-test",
        credentials,
        {"OPENCLAW_BUNDLED_PLUGINS_DIR": "/app/dist/extensions"},
    )
    try:
        assert path.read_text(encoding="utf-8").endswith(
            "OPENCLAW_BUNDLED_PLUGINS_DIR=/app/dist/extensions\n"
        )
    finally:
        path.unlink(missing_ok=True)


def test_remote_archive_contains_only_mode_0600_serper_env(tmp_path):
    config_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    config_dir.mkdir()
    workspace_dir.mkdir()
    (config_dir / "openclaw.json").write_text("{}\n", encoding="utf-8")
    runtime = rollout.ContainerRuntime(
        container_name="archive-test",
        gateway_url="http://127.0.0.1:16000",
        token="token",
        config_dir=config_dir,
        workspace_dir=workspace_dir,
        session_id="session",
        model_id="main__session",
    )
    credentials = {
        "SERPER_URL": "https://search.invalid",
        "SERPER_API_KEY": "secret-must-not-appear",
    }
    secret_path = rollout._write_secret_env_file("archive-test", credentials)
    try:
        archive_bytes = rollout._remote_state_archive(runtime, secret_path)
    finally:
        secret_path.unlink(missing_ok=True)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        member = archive.getmember("runtime.env")
        assert stat.S_IMODE(member.mode) == 0o600
        file_handle = archive.extractfile(member)
        assert file_handle is not None
        contents = file_handle.read().decode("utf-8")
    assert contents == (
        "SERPER_URL=https://search.invalid\n" "SERPER_API_KEY=secret-must-not-appear\n"
    )
    assert "JINA" not in contents


@pytest.mark.parametrize(
    "invalid_url",
    [
        "https://invalid host.example/search",
        "http://plaintext.example/search",
        "https://user:password@invalid.example/search",
    ],
)
def test_serper_invalid_endpoint_is_redacted_without_request(
    monkeypatch, capsys, invalid_url
):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", invalid_url)
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(sys, "argv", ["search.py", "-q", "endpoint validation"])

    def unexpected_urlopen(*args, **kwargs):
        raise AssertionError("network request must not be attempted")

    monkeypatch.setattr(search.urllib.request, "urlopen", unexpected_urlopen)
    assert search.main() == 2

    captured = capsys.readouterr()
    assert "invalid endpoint URL" in captured.err
    assert "test-secret" not in captured.err
    assert invalid_url not in captured.err


def test_serper_success_returns_structured_json(monkeypatch, capsys):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(
        sys,
        "argv",
        ["search.py", "-q", "OpenClaw RL", "--limit", "1"],
    )

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://search.invalid/search"
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 30
        return io.BytesIO(
            json.dumps(
                {
                    "organic": [
                        {
                            "title": "Result",
                            "link": "https://example.invalid/result",
                            "snippet": "Snippet",
                        }
                    ]
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    assert search.main() == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["count"] == 1
    assert output["results"][0]["source"] == "web"
    assert "test-secret" not in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("status_code", [401, 403])
def test_serper_http_error_is_redacted(monkeypatch, capsys, status_code):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid/search")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(sys, "argv", ["search.py", "-q", "failure"])

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            status_code,
            "authorization failure",
            {},
            io.BytesIO(b"server echoed test-secret"),
        )

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    assert search.main() == 1

    captured = capsys.readouterr()
    assert f"HTTP {status_code}" in captured.err
    assert "test-secret" not in captured.err
    assert "search.invalid" not in captured.err


def test_serper_invalid_json_is_redacted(monkeypatch, capsys):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid/search")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(sys, "argv", ["search.py", "-q", "invalid json"])

    def fake_urlopen(request, timeout):
        return io.BytesIO(b"not-json test-secret")

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    assert search.main() == 1

    captured = capsys.readouterr()
    assert "invalid JSON response" in captured.err
    assert "test-secret" not in captured.err
    assert "search.invalid" not in captured.err


def test_serper_rate_limit_retries_at_most_three_times(monkeypatch, capsys):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid/search")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(sys, "argv", ["search.py", "-q", "rate limited"])
    calls = []
    delays = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url, 429, "limited", {}, io.BytesIO(b"secret")
            )
        return io.BytesIO(b'{"organic":[]}')

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(search.time, "sleep", delays.append)
    assert search.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["count"] == 0
    assert len(calls) == search.MAX_ATTEMPTS == 3
    assert delays == [0.25, 0.5]


def test_serper_timeout_stops_after_retry_limit(monkeypatch, capsys):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid/search")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    monkeypatch.setattr(sys, "argv", ["search.py", "-q", "timeout"])
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise TimeoutError("endpoint and secret must stay redacted")

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(search.time, "sleep", lambda _: None)
    assert search.main() == 1

    captured = capsys.readouterr()
    assert len(calls) == search.MAX_ATTEMPTS == 3
    assert "TimeoutError" in captured.err
    assert "test-secret" not in captured.err
    assert "search.invalid" not in captured.err


def test_serper_site_chinese_and_repeated_calls_share_stable_contract(
    monkeypatch, capsys
):
    search = _load_search_module()
    monkeypatch.setenv("SERPER_URL", "https://search.invalid")
    monkeypatch.setenv("SERPER_API_KEY", "test-secret")
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return io.BytesIO(b'{"organic":[]}')

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    queries = ["site:example.com OpenClaw", "强化学习 工具调用"]
    for query in queries:
        monkeypatch.setattr(
            sys,
            "argv",
            ["search.py", "-q", query, "--gl", "CN", "--hl", "zh-cn"],
        )
        assert search.main() == 0
        output = json.loads(capsys.readouterr().out)
        assert output == {"query": query, "count": 0, "results": []}

    assert [payload["q"] for payload in payloads] == queries
    assert all(payload["gl"] == "CN" for payload in payloads)
    assert all(payload["hl"] == "zh-cn" for payload in payloads)


def test_gateway_config_copies_complete_serper_skill(monkeypatch, tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    (source / "scripts/search.py").write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setenv("CLAWGYM_SERPER_SKILL_DIR", str(source))

    config_dir = tmp_path / "state"
    config_dir.mkdir()
    runtime = rollout.ContainerRuntime(
        container_name="test",
        gateway_url="http://127.0.0.1:16000",
        token="token",
        config_dir=config_dir,
        workspace_dir=tmp_path / "workspace",
        session_id="session",
        model_id="main__session",
    )
    rollout._build_gateway_config(runtime, rl_port=30001)

    copied = config_dir / "skills/serper"
    assert (copied / "SKILL.md").is_file()
    assert (copied / "scripts/search.py").is_file()

    config = json.loads((config_dir / "openclaw.json").read_text(encoding="utf-8"))
    models = config["models"]["providers"]["rl-server"]["models"]
    assert [model["id"] for model in models] == ["main__session", "compact__session"]
    assert config["agents"]["defaults"]["compaction"]["model"] == (
        "rl-server/compact__session"
    )


def test_gateway_config_rejects_serper_skill_without_search_script(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    monkeypatch.setenv("CLAWGYM_SERPER_SKILL_DIR", str(source))

    config_dir = tmp_path / "state"
    config_dir.mkdir()
    runtime = rollout.ContainerRuntime(
        container_name="test",
        gateway_url="http://127.0.0.1:16000",
        token="token",
        config_dir=config_dir,
        workspace_dir=tmp_path / "workspace",
        session_id="session",
        model_id="main__session",
    )

    with pytest.raises(RuntimeError, match="scripts/search.py"):
        rollout._build_gateway_config(runtime, rl_port=30001)


def test_remote_chat_readiness_waits_for_routes_to_settle(monkeypatch, tmp_path):
    runtime = rollout.ContainerRuntime(
        container_name="test",
        gateway_url="http://127.0.0.1:16000",
        token="token",
        config_dir=tmp_path / "state",
        workspace_dir=tmp_path / "workspace",
        session_id="session",
        model_id="main__session",
        remote_root="/root/clawgym-runs/test",
    )
    monkeypatch.setattr(
        rollout,
        "_remote_run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=b""),
    )
    sleeps = []
    monkeypatch.setattr(rollout.time, "sleep", sleeps.append)
    monkeypatch.setattr(rollout, "_CHAT_READY_SETTLE_SECONDS", 2.0)

    rollout._wait_for_chat_ready(runtime)

    assert sleeps == [2.0]


def test_eval_workspace_archive_excludes_verifier_assets(monkeypatch, tmp_path):
    workspace = tmp_path / "runtime/workspace"
    (workspace / "input").mkdir(parents=True)
    (workspace / "output").mkdir()
    (workspace / "reward").mkdir()
    (workspace / "input/source.txt").write_text("source", encoding="utf-8")
    (workspace / "output/answer.txt").write_text("answer", encoding="utf-8")
    (workspace / "reward/check.py").write_text("hidden", encoding="utf-8")
    (workspace / ".openclaw_reward_payload.json").write_text(
        "secret verifier payload", encoding="utf-8"
    )
    runtime = rollout.ContainerRuntime(
        container_name="eval-container",
        gateway_url="http://127.0.0.1:16000",
        token="token",
        config_dir=tmp_path / "runtime",
        workspace_dir=workspace,
        session_id="eval-task-19-0",
        model_id="side__eval-task-19-0",
        use_chroot=True,
    )
    task = rollout.TaskSpec(tmp_path / "task", "task_0001", "query", None, {})
    args = SimpleNamespace(save=str(tmp_path / "save"))
    monkeypatch.setenv("OPENCLAW_EVAL_SAVE_WORKSPACES", "1")

    artifact = rollout._archive_terminal_workspace(
        args, runtime, task, rollout_id=19, train_step=20
    )

    assert artifact is not None
    assert artifact["workspace_frozen_before_verifier"] is True
    assert artifact["reward_assets_excluded"] is True
    archive_path = Path(artifact["archive"])
    assert archive_path.is_file()
    assert artifact["archive_sha256"] == rollout._sha256_file(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = {member.name.removeprefix("./") for member in archive.getmembers()}
    assert "input/source.txt" in names
    assert "output/answer.txt" in names
    assert not any(name == "reward" or name.startswith("reward/") for name in names)
    assert ".openclaw_reward_payload.json" not in names


def test_remote_eval_workspace_freeze_retries_transient_failure(monkeypatch):
    monkeypatch.setenv("OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS", "3")
    monkeypatch.setenv("OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS", "0")
    results = iter(
        [
            SimpleNamespace(returncode=1, stderr=b"temporary docker failure"),
            SimpleNamespace(returncode=0, stderr=b""),
        ]
    )
    calls = []

    def fake_remote_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(results)

    monkeypatch.setattr(rollout, "_remote_run", fake_remote_run)

    rollout._pause_remote_eval_container("eval-container")

    assert len(calls) == 2
    assert all(call[0] == "docker pause eval-container" for call in calls)
    assert all(call[1] == {"timeout": 30, "check": False} for call in calls)


def test_remote_eval_workspace_freeze_reports_exhausted_retries(monkeypatch):
    monkeypatch.setenv("OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS", "2")
    monkeypatch.setenv("OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(
        rollout,
        "_remote_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr=b"container state changed"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="could not freeze remote eval container after 2 attempts",
    ):
        rollout._pause_remote_eval_container("eval-container")


def test_eval_artifact_bundle_maps_semantic_step_to_framework_checkpoint(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    (task_dir / "reward/reward.sh").write_text("echo 1", encoding="utf-8")
    task = rollout.TaskSpec(task_dir, "task_0001", "query", None, {"x": 1})
    source = rollout.TaskSource("eval", [task])
    save = tmp_path / "save"
    checkpoint = save / "iter_0000019"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("metadata", encoding="utf-8")
    (checkpoint / "common.pt").write_bytes(b"common")
    args = SimpleNamespace(save=str(save), hf_checkpoint=str(tmp_path / "hf"))
    rows = [
        {
            "dataset_name": "eval",
            "flat_index": 0,
            "task_id": "task_0001",
            "status": "completed",
            "reward": 1.0,
        }
    ]
    summaries = {"eval": {"average_reward": 1.0}}

    manifest_path = rollout._write_eval_artifact_bundle(
        args, 19, 20, [source], rows, summaries
    )

    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["semantic_train_step"] == 20
    assert manifest["checkpoint"]["framework_iteration"] == 19
    assert manifest["checkpoint"]["exists"] is True
    assert manifest["checkpoint"][".metadata_sha256"] == rollout._sha256_file(
        checkpoint / ".metadata"
    )
    result_rows = [
        json.loads(line)
        for line in (manifest_path.parent / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert result_rows == rows
