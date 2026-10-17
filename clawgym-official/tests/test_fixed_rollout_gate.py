import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts/run_fixed_seed_8x8_rollout_gate.py"
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_rollout as rollout


def test_train_metrics_expose_context_and_multi_tool_health():
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1)
    old_metrics = rollout._TRAIN_METRICS
    try:
        rollout._TRAIN_METRICS = {
            "submitted": 2,
            "steps_total": 6,
            "compactions_total": 3,
            "compaction_model_calls_total": 5,
            "tool_result_pruning_events_total": 4,
            "tool_results_pruned_total": 9,
            "tool_result_truncation_events_total": 2,
            "tool_results_truncated_total": 2,
            "unclassified_history_rewrites_total": 1,
            "context_overflows_total": 4,
            "assistant_turns_total": 10,
            "multi_tool_turns_total": 2,
            "raw_tool_calls_total": 12,
            "parsed_tool_calls_total": 11,
            "returned_tool_calls_total": 10,
            "max_identical_tool_call_turns": 3,
            "turns_hit_response_limit": 2,
            "trajectories_hit_response_limit": 1,
            "incomplete_tool_calls_at_limit": 1,
            "response_limit_tokens": 8192,
            "repeated_ngram_ratios": [0.0, 0.5],
            "max_ngram_repeat": 128,
            "repetition_degenerated_trajectories": 1,
            "termination_length_limited_incomplete_tool_call": 1,
        }

        metrics = rollout._build_train_metrics(args, [])

        assert metrics["openclaw/avg_compactions_per_trajectory"] == 1.5
        assert metrics["openclaw/avg_session_compactions_per_trajectory"] == 1.5
        assert metrics["openclaw/compaction_model_calls_total"] == 5
        assert metrics["openclaw/tool_result_pruning_events_total"] == 4
        assert metrics["openclaw/avg_tool_result_pruning_events_per_trajectory"] == 2
        assert metrics["openclaw/tool_results_pruned_total"] == 9
        assert metrics["openclaw/tool_result_truncation_events_total"] == 2
        assert metrics["openclaw/tool_results_truncated_total"] == 2
        assert metrics["openclaw/unclassified_history_rewrites_total"] == 1
        assert metrics["openclaw/avg_context_overflows_per_trajectory"] == 2.0
        assert metrics["openclaw/multi_tool_turn_frac"] == 0.2
        assert metrics["openclaw/tool_call_parse_retention"] == pytest.approx(11 / 12)
        assert metrics["openclaw/tool_call_return_retention"] == pytest.approx(10 / 11)
        assert metrics["openclaw/max_identical_tool_call_turns"] == 3
        assert metrics["rollout/turns_hit_response_limit"] == 2
        assert metrics["rollout/trajectories_hit_response_limit"] == 1
        assert metrics["rollout/incomplete_tool_calls_at_limit"] == 1
        assert metrics["rollout/response_limit_tokens"] == 8192
        assert metrics["rollout/repeated_ngram_ratio"] == 0.25
        assert metrics["rollout/max_ngram_repeat"] == 128
        assert metrics["rollout/repetition_degenerated_trajectories"] == 1
        assert metrics["rollout/truncated"] == 1
        assert (
            metrics["rollout/termination/length_limited_incomplete_tool_call"]
            == 1
        )
    finally:
        rollout._TRAIN_METRICS = old_metrics


def test_train_metrics_expose_gateway_startup_phase_timings():
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1)
    old_metrics = rollout._TRAIN_METRICS
    try:
        rollout._TRAIN_METRICS = {
            "startup_attempts": 3,
            "startup_successful_attempts": 2,
            "startup_failed_attempts": 1,
            "startup_retries": 1,
            "startup_archive_bytes": 3000,
            "startup_archive_count": 2,
            "startup_queue_wait_seconds": [1.0, 2.0, 9.0],
            "startup_port_allocate_seconds": [0.5, 1.0, 1.5],
            "startup_health_ready_seconds": [10.0, 20.0, 120.0],
        }

        metrics = rollout._build_train_metrics(args, [])

        assert metrics["openclaw/startup_attempts"] == 3
        assert metrics["openclaw/startup_successful_attempts"] == 2
        assert metrics["openclaw/startup_failed_attempts"] == 1
        assert metrics["openclaw/startup_retries"] == 1
        assert metrics["openclaw/startup_archive_bytes_avg"] == 1500
        assert metrics["openclaw/startup_queue_wait_sec_avg"] == 4.0
        assert metrics["openclaw/startup_queue_wait_sec_p50"] == 2.0
        assert metrics["openclaw/startup_queue_wait_sec_p90"] == 9.0
        assert metrics["openclaw/startup_port_allocate_sec_avg"] == 1.0
        assert metrics["openclaw/startup_health_ready_sec_max"] == 120.0
        assert metrics["openclaw/startup_docker_run_sec_avg"] == 0.0
    finally:
        rollout._TRAIN_METRICS = old_metrics


def test_gateway_disables_background_exec_completion_notifications(tmp_path):
    config_dir = tmp_path / "config"
    workspace_dir = tmp_path / "workspace"
    config_dir.mkdir()
    workspace_dir.mkdir()
    runtime = rollout.ContainerRuntime(
        container_name="test",
        gateway_url="http://127.0.0.1:30001",
        token="test-token",
        config_dir=config_dir,
        workspace_dir=workspace_dir,
        session_id="session",
        model_id="main__session",
    )

    rollout._build_gateway_config(runtime, 30181)

    config = json.loads((config_dir / "openclaw.json").read_text())
    assert config["tools"]["exec"]["notifyOnExit"] is False
    assert config["agents"]["defaults"]["heartbeat"]["every"] == "35791m"


def test_termination_reason_distinguishes_length_limited_tool_call():
    assert rollout._classify_termination_reason(
        [
            {
                "hit_response_limit": True,
                "incomplete_tool_call_at_limit": True,
            }
        ],
        None,
    ) == "length_limited_incomplete_tool_call"
    assert rollout._classify_termination_reason(
        [], {"reason": "max_compaction_events"}
    ) == "session_compaction_limit"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("fixed_rollout_gate_for_test", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_environment_updates_import_time_chat_timeout(monkeypatch):
    gate = _load_gate_module()
    args = SimpleNamespace(
        proxy_port=30181,
        served_model="test-model",
        parallel=32,
        rollout_timeout=1800,
        tasks=Path("/tmp/tasks"),
        task_seed=0,
        remote_host="remote.invalid",
        remote_user="root",
        ssh_key=Path("/tmp/key"),
        remote_base="/root/test",
        ssh_pool_size=8,
        reverse_port=39181,
        gateway_port_base=30000,
        gateway_train_offset=1000,
        gateway_port_stride=2,
        output_dir=Path("/tmp/output"),
    )
    monkeypatch.setattr(gate.rollout, "_CHAT_TURN_TIMEOUT_SECONDS", 900)

    gate._configure_environment(args)

    assert gate.rollout._CHAT_TURN_TIMEOUT_SECONDS == 1800


def test_configure_environment_sets_eval_dataset_and_workers(monkeypatch):
    gate = _load_gate_module()
    args = SimpleNamespace(
        proxy_port=30181,
        served_model="test-model",
        parallel=32,
        rollout_timeout=1200,
        tasks=Path("/tmp/eval-tasks"),
        task_seed=0,
        remote_host="remote.invalid",
        remote_user="root",
        ssh_key=Path("/tmp/key"),
        remote_base="/root/test",
        ssh_pool_size=8,
        reverse_port=39181,
        gateway_port_base=30000,
        gateway_train_offset=1000,
        gateway_port_stride=2,
        output_dir=Path("/tmp/output"),
        eval_mode=True,
        samples_per_task=1,
    )

    gate._configure_environment(args)

    assert gate.os.environ["OPENCLAW_EVAL_TASKS"] == "/tmp/eval-tasks"
    assert gate.os.environ["OPENCLAW_EVAL_NUM_WORKERS"] == "32"
    assert gate.os.environ["OPENCLAW_EVAL_N_SAMPLES"] == "1"


def test_standalone_runtime_contract_provides_slime_save_attribute(tmp_path):
    gate = _load_gate_module()
    args = SimpleNamespace(output_dir=tmp_path)

    gate._set_standalone_runtime_contract(args)

    assert args.save == str(tmp_path)


def test_serper_missing_script_check_only_flags_canonical_mount(tmp_path):
    gate = _load_gate_module()
    log_dir = tmp_path / "openclaw_logs"
    log_dir.mkdir()
    log_path = log_dir / "agent.log"
    log_path.write_text(
        "--- Step 1 ---\n"
        "can't open file '/root/.openclaw/workspace/~/.openclaw/skills/serper/"
        "scripts/search.py': No such file or directory\n",
        encoding="utf-8",
    )
    assert gate._contains_missing_serper_script(tmp_path) is False

    log_path.write_text(
        "--- Step 1 ---\n"
        "can't open file '/root/.openclaw/skills/serper/scripts/search.py': "
        "No such file or directory\n",
        encoding="utf-8",
    )
    assert gate._contains_missing_serper_script(tmp_path) is True

    log_path.write_text(
        "--- Step 1 ---\n"
        "Action: exec {\"command\": \"cp "
        "/root/.openclaw/skills/serper/scripts/search.py "
        "/root/.openclaw/workspace/scripts/search.py\"}\n"
        "Observation: cp: cannot create regular file "
        "'/root/.openclaw/workspace/scripts/search.py': "
        "No such file or directory\n",
        encoding="utf-8",
    )
    assert gate._contains_missing_serper_script(tmp_path) is False


def test_called_process_error_does_not_serialize_secret_command():
    gate = _load_gate_module()
    secret = "gateway-secret-must-not-appear"
    exc = subprocess.CalledProcessError(
        28,
        ["curl", "-H", f"Authorization: Bearer {secret}", "https://example.invalid"],
    )

    text = gate._safe_exception_text(exc)

    assert text == "CalledProcessError: command failed with exit status 28"
    assert secret not in text


def test_eval_environment_error_does_not_serialize_secret_command():
    secret = "gateway-secret-must-not-appear"
    exc = subprocess.CalledProcessError(
        22,
        ["ssh", f"Authorization: Bearer {secret}"],
    )

    text = rollout._environment_error_summary(exc)

    assert text == "CalledProcessError: command failed with exit status 22"
    assert secret not in text


def test_build_eval_metrics_includes_paper_facing_score_and_accuracy():
    metrics = rollout._build_eval_metrics(
        {
            "clawgym_eval": {
                "rewards": [1.0, 0.75, 0.5, 0.0],
                "n_collected": 4,
                "n_skipped": 0,
                "avg_reward_collected_only": 0.5625,
            }
        }
    )

    assert metrics == {
        "eval/clawgym_eval-avg_reward": 0.5625,
        "eval/clawgym_eval-correct_count": 2,
        "eval/clawgym_eval-correct_rate": 0.5,
        "eval/clawgym_eval-task_count": 4,
        "eval/clawgym_eval-n_collected": 4,
        "eval/clawgym_eval-n_skipped": 0,
        "eval/clawgym_eval-avg_reward_collected_only": 0.5625,
    }


def test_log_contract_failure_happens_before_container_start(monkeypatch):
    container_started = False

    monkeypatch.setattr(rollout, "_register_session_metadata", lambda *args: None)

    def fail_log_root(*args):
        raise AttributeError("missing save")

    def mark_container_started(*args, **kwargs):
        nonlocal container_started
        container_started = True
        raise AssertionError("container must not start")

    monkeypatch.setattr(rollout, "_log_root", fail_log_root)
    monkeypatch.setattr(rollout, "_start_container", mark_container_started)

    with pytest.raises(AttributeError, match="missing save"):
        rollout._run_single_task(
            SimpleNamespace(), object(), 0, 0, "side", 1, 30181, None, "eval"
        )

    assert container_started is False


def test_ssh_pool_size_is_bounded(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REMOTE_SSH_POOL_SIZE", "8")
    assert rollout._ssh_pool_size() == 8

    monkeypatch.setenv("OPENCLAW_REMOTE_SSH_POOL_SIZE", "9")
    try:
        rollout._ssh_pool_size()
    except ValueError as exc:
        assert "between 1 and 8" in str(exc)
    else:
        raise AssertionError("oversized SSH pool should be rejected")


def test_remote_docker_start_semaphore_enforces_configured_limit(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "2")
    monkeypatch.setattr(rollout, "_REMOTE_DOCKER_START_LIMIT", None)
    monkeypatch.setattr(rollout, "_REMOTE_DOCKER_START_SEMAPHORE", None)
    semaphore = rollout._remote_docker_start_semaphore()

    assert semaphore.acquire(blocking=False)
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()
    semaphore.release()

    monkeypatch.setenv("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="must be positive"):
        rollout._remote_docker_start_semaphore()


def test_gateway_prewarm_waits_for_all_ready_tasks():
    barrier = rollout._GatewayPrewarmBarrier(target=2, timeout_seconds=2)
    first_done = threading.Event()
    follower_done = threading.Event()

    first = threading.Thread(
        target=lambda: (barrier.arrive_and_wait(10), first_done.set())
    )
    follower = threading.Thread(
        target=lambda: (barrier.wait_for_release(), follower_done.set())
    )
    first.start()
    follower.start()

    assert not first_done.wait(0.05)
    assert not follower_done.wait(0.05)
    barrier.arrive_and_wait(11)
    first.join(timeout=1)
    follower.join(timeout=1)

    assert first_done.is_set()
    assert follower_done.is_set()
    assert barrier.snapshot() | {"release_seconds": 0.0} == {
        "target": 2,
        "ready": 2,
        "released": True,
        "aborted": False,
        "reason": "ready",
        "release_seconds": 0.0,
    }


def test_gateway_prewarm_abort_releases_ready_and_following_tasks():
    barrier = rollout._GatewayPrewarmBarrier(target=2, timeout_seconds=2)
    ready_done = threading.Event()
    follower_done = threading.Event()
    ready = threading.Thread(
        target=lambda: (barrier.arrive_and_wait(10), ready_done.set())
    )
    follower = threading.Thread(
        target=lambda: (barrier.wait_for_release(), follower_done.set())
    )
    ready.start()
    follower.start()

    assert not ready_done.wait(0.05)
    barrier.abort("gateway failed health check")
    ready.join(timeout=1)
    follower.join(timeout=1)

    snapshot = barrier.snapshot()
    assert ready_done.is_set()
    assert follower_done.is_set()
    assert snapshot["ready"] == 1
    assert snapshot["released"] is True
    assert snapshot["aborted"] is True
    assert snapshot["reason"] == "abort:gateway failed health check"


def test_run_scoped_remote_gateway_ports_skip_busy_and_stay_unique(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REMOTE_GATEWAY_PORT_MODE", "run_scoped")
    monkeypatch.setenv("OPENCLAW_REMOTE_GATEWAY_PORT_MIN", "30000")
    monkeypatch.setenv("OPENCLAW_REMOTE_GATEWAY_PORT_MAX", "30100")
    monkeypatch.setenv("OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS", "8")
    monkeypatch.setenv("OPENCLAW_REMOTE_PROXY_PORT", "39999")
    monkeypatch.setattr(rollout, "_REMOTE_RUN_ID", "test-run")
    monkeypatch.setattr(rollout, "_REMOTE_GATEWAY_PORTS_IN_USE", set())
    probes = []

    def fake_remote_run(command, **kwargs):
        probes.append(command)
        stdout = b"LISTEN\n" if len(probes) == 1 else b""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(rollout, "_remote_run", fake_remote_run)

    first = rollout._allocate_remote_gateway_port(7, 0, True)
    second = rollout._allocate_remote_gateway_port(7, 1, True)

    assert first != second
    assert len(probes) == 3
    assert first in rollout._REMOTE_GATEWAY_PORTS_IN_USE
    assert second in rollout._REMOTE_GATEWAY_PORTS_IN_USE
    rollout._release_remote_gateway_port(first)
    rollout._release_remote_gateway_port(second)
    assert rollout._REMOTE_GATEWAY_PORTS_IN_USE == set()


def test_static_remote_gateway_port_mode_preserves_explicit_formula(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REMOTE_GATEWAY_PORT_MODE", "static")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_BASE", "37000")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET", "20000")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_STRIDE", "10")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT_MAX_SLOTS", "8")

    assert rollout._allocate_remote_gateway_port(100, 3, True) == 57030


def test_remote_chat_error_remains_recoverable_environment_error():
    exc = rollout.RemoteOpenClawChatError("401 Unauthorized")

    assert isinstance(exc, OSError)
    assert "401 Unauthorized" in rollout._environment_error_summary(exc)


def test_remote_gateway_startup_failure_cleans_up_and_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCLAW_REMOTE_START_ATTEMPTS", "3")
    monkeypatch.setattr(rollout, "_remote_enabled", lambda: True)
    startup_slot_depth = 0

    class StartupSlot:
        def __enter__(self):
            nonlocal startup_slot_depth
            startup_slot_depth += 1

        def __exit__(self, exc_type, exc, traceback):
            nonlocal startup_slot_depth
            startup_slot_depth -= 1

    monkeypatch.setattr(
        rollout, "_remote_docker_start_semaphore", lambda: StartupSlot()
    )
    monkeypatch.setattr(rollout, "_register_session_metadata", lambda *args: None)
    monkeypatch.setattr(rollout, "_log_root", lambda *args: tmp_path)

    runtimes = [
        SimpleNamespace(
            container_name=f"gateway-{i}",
            gateway_url=f"http://127.0.0.1:{45000 + i}",
            remote_root=f"/tmp/gateway-{i}",
        )
        for i in range(3)
    ]
    starts = []
    cleanups = []

    def fake_start(*args, **kwargs):
        runtime = runtimes[len(starts)]
        starts.append(runtime)
        return runtime

    def never_ready(runtime):
        assert startup_slot_depth == 1
        raise TimeoutError(f"{runtime.container_name} did not bind")

    monkeypatch.setattr(rollout, "_start_container", fake_start)
    monkeypatch.setattr(rollout, "_wait_for_chat_ready", never_ready)
    monkeypatch.setattr(
        rollout, "_stop_container", lambda runtime, log_dir: cleanups.append(runtime)
    )

    with pytest.raises(TimeoutError, match="gateway-2 did not bind"):
        rollout._run_single_task(
            SimpleNamespace(), object(), 7, 2, "main", 1, 30181, 0, "train"
        )

    assert starts == runtimes
    assert cleanups == runtimes
    assert startup_slot_depth == 0
    assert rollout._REMOTE_GATEWAY_PORTS_IN_USE == {45000, 45001, 45002}


def test_remote_gateway_readiness_rejects_exited_container(monkeypatch):
    commands = []

    def fake_remote_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 2, stdout=b"", stderr=b"")

    monkeypatch.setattr(rollout, "_remote_run", fake_remote_run)
    runtime = SimpleNamespace(
        container_name="gateway-dead",
        gateway_url="http://127.0.0.1:45010",
        remote_root="/tmp/gateway-dead",
    )

    with pytest.raises(TimeoutError, match="gateway-dead"):
        rollout._wait_for_chat_ready(runtime)

    assert len(commands) == 1
    assert "docker inspect" in commands[0]
    assert "gateway-dead" in commands[0]


def test_ssh_slots_are_assigned_round_robin_per_thread(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REMOTE_SSH_POOL_SIZE", "4")
    monkeypatch.setattr(rollout, "_SSH_SLOT_COUNTER", 0)
    monkeypatch.setattr(rollout, "_SSH_THREAD_STATE", threading.local())
    barrier = threading.Barrier(8)
    slots = []
    slots_lock = threading.Lock()

    def worker():
        barrier.wait()
        first = rollout._ssh_slot()
        second = rollout._ssh_slot()
        with slots_lock:
            slots.append((first, second))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(first == second for first, second in slots)
    assert sorted(first for first, _ in slots) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_ssh_master_start_retries_without_remote_command(monkeypatch, tmp_path):
    calls = []
    results = iter(
        [
            subprocess.CompletedProcess([], 255),
            subprocess.CompletedProcess([], 255, b"", b"kex reset"),
            subprocess.CompletedProcess([], 0, b"", b""),
        ]
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        return next(results)

    monkeypatch.setattr(rollout.subprocess, "run", fake_run)
    monkeypatch.setattr(rollout.time, "sleep", lambda _: None)
    monkeypatch.setattr(rollout, "_SSH_MASTER_LOCKS", {})

    rollout._ensure_ssh_master(
        "remote.invalid", "root", "/tmp/key", tmp_path / "master.sock"
    )

    assert len(calls) == 3
    assert "check" in calls[0]
    assert "ControlMaster=yes" in calls[1]
    assert "ControlMaster=yes" in calls[2]
    assert all("curl" not in argument for call in calls for argument in call)
