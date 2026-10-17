#!/usr/bin/env python3
"""Compare SGLang decode reproducibility with request seed/server determinism.

Runs three isolated TP servers in parallel:
  A: no request sampling seed, deterministic inference disabled
  B: fixed request sampling seed, deterministic inference disabled
  C: fixed request sampling seed, deterministic inference enabled

The prompt token IDs are taken from a saved real ClawGym rollout, avoiding any
tokenizer/rendering difference between repetitions.
"""

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
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def extract_output_ids(response: dict[str, Any]) -> list[int]:
    if isinstance(response.get("output_ids"), list):
        return [int(item) for item in response["output_ids"]]
    meta = response.get("meta_info") or {}
    records = meta.get("output_token_logprobs")
    if not isinstance(records, list):
        raise KeyError(f"No output token IDs in response keys={list(response)} meta_keys={list(meta)}")
    output_ids: list[int] = []
    for record in records:
        if isinstance(record, (list, tuple)) and len(record) >= 2:
            output_ids.append(int(record[1]))
        elif isinstance(record, dict):
            token_id = record.get("token_id", record.get("id"))
            if token_id is None:
                raise ValueError(f"Unrecognized token logprob record: {record}")
            output_ids.append(int(token_id))
        else:
            raise ValueError(f"Unrecognized token logprob record: {record}")
    return output_ids


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def sequence_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(item) for item in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def launch_server(
    args: argparse.Namespace,
    name: str,
    gpus: str,
    port: int,
    nccl_port: int,
    deterministic: bool,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_handle = (args.output_dir / f"sglang_{name}.log").open("wb")
    command = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        f"decode-determinism-{name}",
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
        str(args.engine_seed),
        "--trust-remote-code",
        "--tool-call-parser",
        "qwen",
        "--disable-cuda-graph",
    ]
    if deterministic:
        command.append("--enable-deterministic-inference")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return process, log_handle


def run_group(
    args: argparse.Namespace,
    name: str,
    port: int,
    prompt_ids: list[int],
    request_seed: int | None,
) -> dict[str, Any]:
    outputs: list[list[int]] = []
    records: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        sampling_params: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
        }
        if request_seed is not None:
            sampling_params["sampling_seed"] = request_seed
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
        }
        started = time.time()
        response = post_json(port, payload, args.request_timeout)
        output_ids = extract_output_ids(response)
        outputs.append(output_ids)
        meta = response.get("meta_info") or {}
        records.append(
            {
                "repetition": repetition,
                "num_output_tokens": len(output_ids),
                "token_sha256": sequence_hash(output_ids),
                "finish_reason": meta.get("finish_reason"),
                "elapsed_seconds": time.time() - started,
                "output_ids": output_ids,
            }
        )

    reference = outputs[0]
    for record, output_ids in zip(records, outputs, strict=True):
        lcp = common_prefix_length(reference, output_ids)
        record["common_prefix_with_run0"] = lcp
        record["first_divergence_from_run0"] = (
            None if output_ids == reference else lcp
        )

    hashes = [record["token_sha256"] for record in records]
    return {
        "group": name,
        "request_sampling_seed": request_seed,
        "repetitions": args.repetitions,
        "unique_token_sequences": len(set(hashes)),
        "all_identical": len(set(hashes)) == 1,
        "records": records,
    }


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--request-seed", type=int, default=424242)
    parser.add_argument("--engine-seed", type=int, default=12345)
    parser.add_argument("--serve-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.rollout_data, map_location="cpu", weights_only=False)
    sample = saved["samples"][args.sample_index]
    prompt_length = len(sample["tokens"]) - int(sample["response_length"])
    prompt_ids = [int(item) for item in sample["tokens"][:prompt_length]]

    specs = {
        "A": {"gpus": "0,1", "port": 30210, "nccl_port": 31210, "deterministic": False, "seed": None},
        "B": {"gpus": "2,3", "port": 30220, "nccl_port": 31220, "deterministic": False, "seed": args.request_seed},
        "C": {"gpus": "4,5", "port": 30230, "nccl_port": 31230, "deterministic": True, "seed": args.request_seed},
    }
    metadata = {
        "model": str(args.model),
        "rollout_data": str(args.rollout_data),
        "sample_index": args.sample_index,
        "task_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": sequence_hash(prompt_ids),
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0},
        "max_new_tokens": args.max_new_tokens,
        "engine_seed": args.engine_seed,
        "request_seed_B_C": args.request_seed,
        "groups": specs,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    servers: list[tuple[subprocess.Popen[bytes], Any]] = []
    try:
        for name, spec in specs.items():
            servers.append(
                launch_server(
                    args,
                    name,
                    spec["gpus"],
                    spec["port"],
                    spec["nccl_port"],
                    spec["deterministic"],
                )
            )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(wait_health, spec["port"], args.serve_timeout) for spec in specs.values()]
            for future in futures:
                future.result()

        with ThreadPoolExecutor(max_workers=3) as pool:
            future_map = {
                name: pool.submit(run_group, args, name, spec["port"], prompt_ids, spec["seed"])
                for name, spec in specs.items()
            }
            results = {name: future.result() for name, future in future_map.items()}

        output = {"metadata": metadata, "results": results}
        (args.output_dir / "results.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            name: {
                "unique_token_sequences": result["unique_token_sequences"],
                "all_identical": result["all_identical"],
                "first_divergences_from_run0": [
                    record["first_divergence_from_run0"] for record in result["records"]
                ],
            }
            for name, result in results.items()
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        for process, log_handle in servers:
            stop_process(process)
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
