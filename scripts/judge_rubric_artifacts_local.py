#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-evaluator-root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base-artifacts", type=Path, required=True)
    p.add_argument("--sft-artifacts", type=Path, required=True)
    p.add_argument("--base-code-report", type=Path, required=True)
    p.add_argument("--sft-code-report", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=540000)
    p.add_argument("--timeout", type=int, default=300)
    return p.parse_args()


def stable_seed(seed_base: int, model_label: str, task_id: str, attempt: int) -> int:
    material = f"{seed_base}:{model_label}:{task_id}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") & 0x7FFFFFFF


def load_code_scores(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = next(iter(payload["datasets"].values()))
    return {
        int(row["task_id"].rsplit("_", 1)[1]): float(row["reward"])
        for row in dataset["tasks"]
    }


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.official_evaluator_root))
    from internalbench.dataset import PayloadRecord
    from internalbench.judge import (
        _parse_judge_text,
        build_judge_messages,
        build_rubric_prompt,
        score_judge_scores,
    )

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {i for i, row in enumerate(rows) if row.get("rules")}
    if len(expected) != 46:
        raise ValueError(f"expected 46 released-data rubric tasks, got {len(expected)}")

    artifacts_by_model: dict[str, dict[int, dict]] = {}
    for label, directory in (("base", args.base_artifacts), ("sft", args.sft_artifacts)):
        artifacts = {}
        for path in sorted(directory.glob("task_*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            ordinal = int(item["benchmark_ordinal"])
            if item["prompt"] != rows[ordinal]["prompt"] or item["rules"] != rows[ordinal]["rules"]:
                raise ValueError(f"artifact/dataset mismatch: {label} ordinal={ordinal}")
            artifacts[ordinal] = item
        if set(artifacts) != expected:
            raise ValueError(f"{label} artifact set mismatch: missing={sorted(expected-set(artifacts))}")
        artifacts_by_model[label] = artifacts

    args.output_dir.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    results: dict[str, dict[str, dict]] = {"base": {}, "sft": {}}
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def judge_one(label: str, ordinal: int, artifact: dict) -> tuple[str, int, dict]:
        record = PayloadRecord(line_number=ordinal + 1, ordinal=ordinal + 1, raw=rows[ordinal])
        prompt = build_rubric_prompt(record, artifact["target_files"])
        attempts = []
        final_score = None
        final_parsed = {}
        final_normalization = {}
        final_text = ""
        for attempt in range(3):
            seed = stable_seed(args.seed_base, label, artifact["task_id"], attempt)
            request_payload = {
                "model": args.model,
                "messages": build_judge_messages(prompt),
                "temperature": 0.6,
                "max_tokens": 8192,
                "seed": seed,
            }
            started = time.time()
            response = requests.post(
                endpoint,
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                json=request_payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            parsed = _parse_judge_text(text)
            score, normalization = score_judge_scores(parsed, record.rules)
            attempts.append({
                "attempt": attempt,
                "seed": seed,
                "elapsed_seconds": time.time() - started,
                "score": score,
                "parsed": parsed,
                "score_normalization": normalization,
                "text": text,
                "usage": body.get("usage"),
            })
            final_score, final_parsed, final_normalization, final_text = score, parsed, normalization, text
            if score is not None and not normalization.get("missing"):
                break
        result = {
            "task_id": artifact["task_id"],
            "benchmark_ordinal": ordinal,
            "code_score": float(artifact["code_score"]),
            "rubric_score": final_score,
            "hybrid_score": (
                0.7 * float(artifact["code_score"]) + 0.3 * float(final_score)
                if final_score is not None else None
            ),
            "final_parsed": final_parsed,
            "final_score_normalization": final_normalization,
            "final_text": final_text,
            "attempts": attempts,
            "target_files": {
                path: {k: v for k, v in info.items() if k != "content"}
                for path, info in artifact["target_files"].items()
            },
        }
        return label, ordinal, result

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for label, artifacts in artifacts_by_model.items():
            for ordinal, artifact in artifacts.items():
                futures.append(executor.submit(judge_one, label, ordinal, artifact))
        for completed, future in enumerate(as_completed(futures), 1):
            label, ordinal, result = future.result()
            with lock:
                results[label][str(ordinal)] = result
                partial = {
                    "judge": {
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": 0.6,
                        "max_tokens": 8192,
                        "seed_base": args.seed_base,
                        "deterministic_per_request_seed": True,
                    },
                    "completed": completed,
                    "expected": len(futures),
                    "results": results,
                }
                (args.output_dir / "judge_results.partial.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            print(f"[{completed}/{len(futures)}] {label} task_{ordinal:04d} rubric={result['rubric_score']}", flush=True)

    full_code = {
        "base": load_code_scores(args.base_code_report),
        "sft": load_code_scores(args.sft_code_report),
    }
    summaries = {}
    for label in ("base", "sft"):
        judged = [results[label][str(i)] for i in sorted(expected)]
        if any(row["rubric_score"] is None for row in judged):
            raise RuntimeError(f"{label} has unscored rubric tasks")
        pure_ordinals = [i for i in range(len(rows)) if i not in expected]
        pure_sum = sum(full_code[label][i] for i in pure_ordinals)
        hybrid_sum = sum(row["hybrid_score"] for row in judged)
        summaries[label] = {
            "pure_code_task_count": len(pure_ordinals),
            "hybrid_task_count_released_data": len(expected),
            "rubric_mean_on_hybrid_tasks": sum(row["rubric_score"] for row in judged) / len(judged),
            "code_mean_on_rerun_hybrid_tasks": sum(row["code_score"] for row in judged) / len(judged),
            "hybrid_mean_on_hybrid_tasks": hybrid_sum / len(judged),
            "final_200_task_hybrid_mean": (pure_sum + hybrid_sum) / len(rows),
            "hybrid_correct_gt_0_5": sum(row["hybrid_score"] > 0.5 for row in judged),
            "judge_retries": sum(len(row["attempts"]) - 1 for row in judged),
        }
    final = {
        "method": {
            "judge_model": args.model,
            "judge_is_paper_gpt_5_4": False,
            "temperature": 0.6,
            "max_tokens": 8192,
            "per_request_seed_base": args.seed_base,
            "deterministic_inference_expected": True,
            "weights": {"code": 0.7, "rubric": 0.3},
            "released_data_note": "46 tasks contain rules although repository documentation says 44",
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        },
        "summaries": summaries,
        "results": results,
    }
    (args.output_dir / "judge_results.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
