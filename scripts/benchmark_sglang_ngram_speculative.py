#!/usr/bin/env python3
"""Benchmark SGLang NGRAM speculative decoding on saved real ClawGym prompts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any
import urllib.request

import torch


def wait_health(port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang port {port} did not become ready: {last_error}")


def post_json(port: int, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def output_ids(response: dict[str, Any]) -> list[int]:
    if isinstance(response.get("output_ids"), list):
        return [int(token) for token in response["output_ids"]]
    records = (response.get("meta_info") or {}).get("output_token_logprobs") or []
    return [int(record[1]) for record in records]


def launch(args: argparse.Namespace, mode: str, port: int, nccl_port: int):
    log_handle = (args.output_dir / f"sglang_{mode}.log").open("wb")
    command = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        f"ngram-benchmark-{mode}",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--nccl-port",
        str(nccl_port),
        "--tensor-parallel-size",
        "2",
        "--context-length",
        "65536",
        "--mem-fraction-static",
        "0.65",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--random-seed",
        "1234",
        "--trust-remote-code",
        "--tool-call-parser",
        "qwen",
    ]
    if mode == "ngram":
        command.extend(["--speculative-algorithm", "NGRAM"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return process, log_handle, command


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_request(args, port: int, prompt_ids: list[int], request_id: int) -> dict[str, Any]:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": args.request_seed + request_id,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
    }
    started = time.perf_counter()
    response = post_json(port, payload, args.request_timeout)
    elapsed = time.perf_counter() - started
    ids = output_ids(response)
    meta = response.get("meta_info") or {}
    spec = {
        key: value
        for key, value in meta.items()
        if "spec" in key.lower() and isinstance(value, (int, float, str, bool, type(None)))
    }
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    return {
        "request_id": request_id,
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(ids),
        "elapsed_seconds": elapsed,
        "token_sha256": digest,
        "finish_reason": meta.get("finish_reason"),
        "spec_metrics": spec,
    }


def benchmark_mode(args, mode: str, port: int, nccl_port: int, prompts: list[list[int]]):
    process, log_handle, command = launch(args, mode, port, nccl_port)
    try:
        wait_health(port, args.serve_timeout)
        post_json(
            port,
            {
                "input_ids": prompts[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            args.request_timeout,
        )
        jobs = [
            (prompt, repetition * len(prompts) + index)
            for repetition in range(args.repetitions)
            for index, prompt in enumerate(prompts)
        ]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_request, args, port, prompt, request_id) for prompt, request_id in jobs]
            records = [future.result() for future in futures]
        wall = time.perf_counter() - started
        total_output = sum(record["output_tokens"] for record in records)
        return {
            "mode": mode,
            "command": command,
            "wall_seconds": wall,
            "requests": len(records),
            "total_prompt_tokens": sum(record["prompt_tokens"] for record in records),
            "total_output_tokens": total_output,
            "output_tokens_per_second": total_output / wall,
            "mean_request_seconds": sum(record["elapsed_seconds"] for record in records) / len(records),
            "records": records,
        }
    finally:
        stop(process)
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--sample-indices", default="0,4,5,6,7,8,9")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--request-seed", type=int, default=20260820)
    parser.add_argument("--serve-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--base-port", type=int, default=30651)
    parser.add_argument("--base-nccl-port", type=int, default=31651)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.rollout_data, map_location="cpu", weights_only=False)
    samples = saved["samples"]
    indices = [int(index) for index in args.sample_indices.split(",") if index]
    prompts = []
    for index in indices:
        sample = samples[index]
        prompt_length = len(sample["tokens"]) - int(sample["response_length"])
        prompts.append([int(token) for token in sample["tokens"][:prompt_length]])

    results = []
    for offset, mode in enumerate(("baseline", "ngram")):
        result = benchmark_mode(
            args,
            mode,
            args.base_port + offset,
            args.base_nccl_port + offset,
            prompts,
        )
        results.append(result)
        (args.output_dir / f"{mode}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    baseline, ngram = results
    baseline_hashes = {record["request_id"]: record["token_sha256"] for record in baseline["records"]}
    matching = sum(
        baseline_hashes[record["request_id"]] == record["token_sha256"]
        for record in ngram["records"]
    )
    summary = {
        "sample_indices": indices,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "baseline_output_tokens_per_second": baseline["output_tokens_per_second"],
        "ngram_output_tokens_per_second": ngram["output_tokens_per_second"],
        "speedup": ngram["output_tokens_per_second"] / baseline["output_tokens_per_second"],
        "matching_token_sequences": matching,
        "compared_requests": len(ngram["records"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
