from types import SimpleNamespace

import pytest

from slime.router.router import SlimeRouter
import slime.router.router as router_module


WORKERS = [
    "http://127.0.0.1:10001",
    "http://127.0.0.1:10002",
]


def make_router(*, enabled=True, max_load_delta=2, mode="hash_bounded", migration_seconds=10.0):
    router = object.__new__(SlimeRouter)
    router.args = SimpleNamespace(
        slime_router_session_affinity=enabled,
        slime_router_affinity_max_load_delta=max_load_delta,
        slime_router_affinity_mode=mode,
        slime_router_affinity_samples_per_task=8,
        slime_router_affinity_migration_load_delta=4,
        slime_router_affinity_migration_sustained_seconds=migration_seconds,
    )
    router.worker_request_counts = {worker: 0 for worker in WORKERS}
    router.dead_workers = set()
    router.session_worker_bindings = {}
    router.session_migrated = set()
    router.worker_overload_since = {}
    return router


def test_same_session_stays_on_same_worker():
    router = make_router()
    first, first_routing = router._use_url("session-a")
    router._finish_url(first)
    second, second_routing = router._use_url("session-a")

    assert second == first
    assert first_routing["affinity_used"]
    assert second_routing["affinity_used"]


def test_affinity_falls_back_when_preferred_worker_is_overloaded():
    router = make_router(max_load_delta=2)
    preferred, _ = router._use_url("session-a")
    router._finish_url(preferred)
    other = next(worker for worker in WORKERS if worker != preferred)
    router.worker_request_counts[preferred] = 3

    selected, routing = router._use_url("session-a")

    assert selected == other
    assert not routing["affinity_used"]
    assert routing["affinity_fallback"]


def test_dead_worker_is_never_selected():
    router = make_router()
    preferred, _ = router._use_url("session-a")
    router._finish_url(preferred)
    router.dead_workers.add(preferred)

    selected, routing = router._use_url("session-a")

    assert selected != preferred
    assert selected not in router.dead_workers
    assert routing["affinity_used"]


def test_disabled_affinity_uses_least_loaded_worker():
    router = make_router(enabled=False)
    router.worker_request_counts[WORKERS[0]] = 2

    selected, routing = router._use_url("session-a")

    assert selected == WORKERS[1]
    assert not routing["affinity_used"]
    assert not routing["affinity_fallback"]


def test_no_healthy_worker_raises():
    router = make_router()
    router.dead_workers.update(WORKERS)

    with pytest.raises(RuntimeError, match="No healthy workers"):
        router._use_url("session-a")


def test_task_sample_fixed_assigns_exact_interleaved_four_plus_four():
    router = make_router(mode="task_sample_fixed")
    assignments = []

    for task_index in range(8):
        selected, routing = router._use_url(f"task-100-{task_index}")
        assignments.append(selected)
        router._finish_url(selected)
        assert routing["affinity_used"]
        assert not routing["affinity_fallback"]

    assert assignments == [WORKERS[index % 2] for index in range(8)]
    assert assignments.count(WORKERS[0]) == 4
    assert assignments.count(WORKERS[1]) == 4


def test_task_sample_fixed_staggers_same_sample_slot_across_task_groups():
    router = make_router(mode="task_sample_fixed")
    first_sample_assignments = []

    for task_group in range(8):
        task_index = task_group * 8
        selected, _ = router._use_url(f"task-100-{task_index}")
        first_sample_assignments.append(selected)
        router._finish_url(selected)

    assert first_sample_assignments == [WORKERS[index % 2] for index in range(8)]

    for task_group in range(8):
        assignments = [
            router._task_sample_preferred_worker(f"task-100-{task_group * 8 + slot}", WORKERS)
            for slot in range(8)
        ]
        assert assignments.count(WORKERS[0]) == 4
        assert assignments.count(WORKERS[1]) == 4


def test_task_sample_fixed_never_load_balances_an_existing_session():
    router = make_router(mode="task_sample_fixed")
    session = "task-100-0"
    first, _ = router._use_url(session)
    router._finish_url(first)
    other = next(worker for worker in WORKERS if worker != first)
    router.worker_request_counts[first] = 20

    selected, routing = router._use_url(session)

    assert selected == first
    assert selected != other
    assert routing["affinity_used"]
    assert not routing["affinity_migrated"]


def test_task_sample_sustained_does_not_migrate_on_short_spike(monkeypatch):
    router = make_router(mode="task_sample_sustained", migration_seconds=10.0)
    session = "task-100-0"
    first, _ = router._use_url(session)
    router._finish_url(first)
    router.worker_request_counts[first] = 8
    now = iter([100.0, 105.0])
    monkeypatch.setattr(router_module.time, "monotonic", lambda: next(now))

    selected1, routing1 = router._use_url(session)
    router._finish_url(selected1)
    selected2, routing2 = router._use_url(session)

    assert selected1 == first
    assert selected2 == first
    assert not routing1["affinity_migrated"]
    assert not routing2["affinity_migrated"]


def test_task_sample_sustained_migrates_once_and_stays_pinned(monkeypatch):
    router = make_router(mode="task_sample_sustained", migration_seconds=10.0)
    session = "task-100-0"
    first, _ = router._use_url(session)
    router._finish_url(first)
    other = next(worker for worker in WORKERS if worker != first)
    router.worker_request_counts[first] = 8
    now = iter([100.0, 111.0])
    monkeypatch.setattr(router_module.time, "monotonic", lambda: next(now))

    selected1, _ = router._use_url(session)
    router._finish_url(selected1)
    selected2, routing2 = router._use_url(session)
    router._finish_url(selected2)
    router.worker_request_counts[first] = 0
    router.worker_request_counts[other] = 20
    selected3, routing3 = router._use_url(session)

    assert selected1 == first
    assert selected2 == other
    assert routing2["affinity_migrated"]
    assert selected3 == other
    assert not routing3["affinity_migrated"]
    assert router.session_migrated == {session}
