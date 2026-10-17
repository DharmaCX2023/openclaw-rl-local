#!/usr/bin/env python3
"""Build deterministic length/structure-matched placebo Anchor assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))

from opsd_evidence import load_evidence_anchor_index, render_teacher_feedback


_WORD_RE = re.compile(r"[a-z0-9_./-]{3,}", re.IGNORECASE)


def _words(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _query_for_task(data_root: Path, task_id: str) -> str:
    path = data_root / task_id / "data_entry.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("user_query", ""))


def _stable_tie(seed: int, source: str, candidate: str) -> str:
    return hashlib.sha256(f"{seed}|{source}|{candidate}".encode()).hexdigest()


def _anchor_text(anchors: list[dict[str, str]]) -> str:
    return "\n".join(
        " ".join(
            [
                anchor.get("condition", ""),
                anchor.get("artifact", ""),
                anchor.get("verified_fact", ""),
                anchor.get("explanation", ""),
            ]
        )
        for anchor in anchors
    )


def build_mapping(
    *,
    anchor_file: Path,
    tokenizer_path: Path,
    data_root: Path,
    k: int,
    max_token_diff: int,
    max_query_jaccard: float,
    max_anchor_jaccard: float,
    seed: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    anchor_index = load_evidence_anchor_index(anchor_file)
    descriptors: dict[str, dict[str, Any]] = {}
    for task_id, anchors in sorted(anchor_index.items()):
        feedback = render_teacher_feedback(
            {
                "rollout_not_fully_successful": True,
                "evidence_anchors": anchors,
            }
        )
        query = _query_for_task(data_root, task_id)
        descriptors[task_id] = {
            "anchor_count": len(anchors),
            "feedback_tokens": len(
                tokenizer.encode(feedback, add_special_tokens=False)
            ),
            "query_words": _words(query),
            "anchor_words": _words(_anchor_text(anchors)),
        }

    tasks: dict[str, dict[str, Any]] = {}
    for task_id, source in descriptors.items():
        candidates = []
        for candidate_id, candidate in descriptors.items():
            if candidate_id == task_id:
                continue
            token_diff = candidate["feedback_tokens"] - source["feedback_tokens"]
            if abs(token_diff) > max_token_diff:
                continue
            query_jaccard = _jaccard(
                source["query_words"], candidate["query_words"]
            )
            anchor_jaccard = _jaccard(
                source["anchor_words"], candidate["anchor_words"]
            )
            if query_jaccard > max_query_jaccard:
                continue
            if anchor_jaccard > max_anchor_jaccard:
                continue
            candidates.append(
                (
                    abs(candidate["anchor_count"] - source["anchor_count"]),
                    abs(token_diff),
                    _stable_tie(seed, task_id, candidate_id),
                    {
                        "task_id": candidate_id,
                        "anchor_count": candidate["anchor_count"],
                        "feedback_tokens": candidate["feedback_tokens"],
                        "token_difference": token_diff,
                        "query_jaccard": query_jaccard,
                        "anchor_jaccard": anchor_jaccard,
                    },
                )
            )
        candidates.sort(key=lambda item: item[:3])
        selected = [item[3] for item in candidates[:k]]
        tasks[task_id] = {
            "status": "ok" if len(selected) == k else "insufficient_placebos",
            "real_anchor_count": source["anchor_count"],
            "real_feedback_tokens": source["feedback_tokens"],
            "placebos": selected,
        }

    complete = sum(row["status"] == "ok" for row in tasks.values())
    primary_differences = [
        abs(row["placebos"][0]["token_difference"])
        for row in tasks.values()
        if row["placebos"]
    ]
    return {
        "schema_version": "clawgym_matched_placebo_v1",
        "anchor_file": str(anchor_file.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "data_root": str(data_root.resolve()),
        "seed": seed,
        "k": k,
        "max_token_diff": max_token_diff,
        "max_query_jaccard": max_query_jaccard,
        "max_anchor_jaccard": max_anchor_jaccard,
        "summary": {
            "tasks": len(tasks),
            "complete_k": complete,
            "complete_k_fraction": complete / max(1, len(tasks)),
            "with_primary": len(primary_differences),
            "primary_exact_length": sum(x == 0 for x in primary_differences),
            "primary_within_8": sum(x <= 8 for x in primary_differences),
            "primary_max_abs_token_difference": (
                max(primary_differences) if primary_differences else None
            ),
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-token-diff", type=int, default=8)
    parser.add_argument("--max-query-jaccard", type=float, default=0.45)
    parser.add_argument("--max-anchor-jaccard", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.max_token_diff < 0:
        raise ValueError("--max-token-diff must be non-negative")

    payload = build_mapping(
        anchor_file=args.anchors,
        tokenizer_path=args.tokenizer,
        data_root=args.data_root,
        k=args.k,
        max_token_diff=args.max_token_diff,
        max_query_jaccard=args.max_query_jaccard,
        max_anchor_jaccard=args.max_anchor_jaccard,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
