import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_matched_placebo_mapping.py"
SPEC = importlib.util.spec_from_file_location("build_matched_placebo_mapping", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class _AutoTokenizer:
    @staticmethod
    def from_pretrained(path, local_files_only):
        return _Tokenizer()


def _write_inputs(root: Path):
    anchors = root / "anchors.jsonl"
    rows = [
        {
            "task_id": f"task_{index}",
            "anchors": [
                {
                    "condition": f"condition {index}" + ("x" * extra),
                    "artifact": "output/result.json",
                    "verified_fact": "parseable",
                    "explanation": "required",
                }
            ],
        }
        for index, extra in enumerate([0, 0, 1, 1])
    ]
    anchors.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    data = root / "data"
    for index in range(4):
        task = data / f"task_{index}"
        task.mkdir(parents=True)
        (task / "data_entry.json").write_text(
            json.dumps({"user_query": f"unique query topic {index}"}),
            encoding="utf-8",
        )
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    return anchors, data, tokenizer


def test_mapping_is_deterministic_nonself_and_length_bounded(tmp_path, monkeypatch):
    anchors, data, tokenizer = _write_inputs(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )
    kwargs = dict(
        anchor_file=anchors,
        tokenizer_path=tokenizer,
        data_root=data,
        k=2,
        max_token_diff=1,
        max_query_jaccard=1.0,
        max_anchor_jaccard=1.0,
        seed=17,
    )

    first = MODULE.build_mapping(**kwargs)
    second = MODULE.build_mapping(**kwargs)

    assert first == second
    for task_id, row in first["tasks"].items():
        assert len(row["placebos"]) == 2
        assert all(item["task_id"] != task_id for item in row["placebos"])
        assert all(
            abs(item["token_difference"]) <= 1
            for item in row["placebos"]
        )
