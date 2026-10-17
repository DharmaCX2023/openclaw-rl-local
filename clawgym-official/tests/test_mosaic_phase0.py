from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "analyze_mosaic_phase0.py"
SPEC = importlib.util.spec_from_file_location("analyze_mosaic_phase0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mosaic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mosaic
SPEC.loader.exec_module(mosaic)


class MosaicPhase0Test(unittest.TestCase):
    def _task(self, root: Path) -> Path:
        task = root / "task_demo"
        reward = task / "reward"
        reward.mkdir(parents=True)
        (task / "data_entry.json").write_text(
            json.dumps(
                {
                    "task_id": "task_demo",
                    "user_query": "produce a.txt and b.txt",
                    "metadata": {},
                    "input_mount_dir": "/root/.openclaw/workspace",
                }
            ),
            encoding="utf-8",
        )
        (task / "input_files").mkdir()
        (reward / "check.py").write_text(
            """import json, pathlib, sys
w = pathlib.Path(sys.argv[1])
def ok(name, expected):
    p = w / name
    return p.is_file() and p.read_text() == expected
print(json.dumps({\"a_correct\": ok(\"a.txt\", \"A\"), \"b_correct\": ok(\"b.txt\", \"B\")}))
""",
            encoding="utf-8",
        )
        return task

    def _archive(self, root: Path, name: str, files: dict[str, str]) -> Path:
        workspace = root / f"{name}_workspace"
        workspace.mkdir()
        for relative, content in files.items():
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        archive = root / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for child in workspace.iterdir():
                handle.add(child, arcname=child.name)
        return archive

    def test_two_partial_failures_form_strict_super_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root)
            first = self._archive(root, "first", {"a.txt": "A"})
            second = self._archive(root, "second", {"b.txt": "B"})
            output = root / "analysis"
            output.mkdir()
            items = [
                {
                    "task_id": "task_demo",
                    "task_dir": str(task),
                    "task_index": 0,
                    "official_reward": 0.5,
                    "archive_path": first,
                },
                {
                    "task_id": "task_demo",
                    "task_dir": str(task),
                    "task_index": 1,
                    "official_reward": 0.5,
                    "archive_path": second,
                },
            ]
            result = mosaic._analyze_group(
                "task_demo",
                items,
                output,
                beam_width=8,
                max_queries=32,
                checker_timeout=10,
                max_file_bytes=1024 * 1024,
            )

            self.assertTrue(result["supported"])
            self.assertTrue(result["strict_super_parent"])
            self.assertTrue(result["cross_parent_complementarity"])
            self.assertEqual(result["best_parent_reward"], 0.5)
            self.assertEqual(result["best_mosaic_reward"], 1.0)
            self.assertEqual(result["minimum_source_cover"], 2)
            mosaic_workspace = Path(result["mosaic_workspace"])
            self.assertEqual((mosaic_workspace / "a.txt").read_text(), "A")
            self.assertEqual((mosaic_workspace / "b.txt").read_text(), "B")

    def test_one_parent_plus_initial_revert_is_not_cross_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root)
            (task / "input_files" / "b.txt").write_text("B", encoding="utf-8")
            first = self._archive(root, "first", {"a.txt": "A", "b.txt": "BAD"})
            second = self._archive(root, "second", {"b.txt": "BAD"})
            output = root / "analysis"
            output.mkdir()
            items = [
                {
                    "task_id": "task_demo",
                    "task_dir": str(task),
                    "task_index": 0,
                    "official_reward": 0.5,
                    "archive_path": first,
                },
                {
                    "task_id": "task_demo",
                    "task_dir": str(task),
                    "task_index": 1,
                    "official_reward": 0.0,
                    "archive_path": second,
                },
            ]
            result = mosaic._analyze_group(
                "task_demo",
                items,
                output,
                beam_width=8,
                max_queries=32,
                checker_timeout=10,
                max_file_bytes=1024 * 1024,
            )

            self.assertTrue(result["strict_super_parent"])
            self.assertEqual(result["minimum_source_cover"], 1)
            self.assertFalse(result["cross_parent_complementarity"])


if __name__ == "__main__":
    unittest.main()
