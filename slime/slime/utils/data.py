import itertools
import json
import logging
import os
import random
import re

import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor and any(s.multimodal_inputs is not None for s in origin_samples):
        filtered_samples = []
        for sample in origin_samples:
            multimodal_inputs = sample.multimodal_inputs or {"images": None, "videos": None}
            processor_output = processor(text=sample.prompt, **multimodal_inputs)
            input_ids = processor_output["input_ids"][0]
            if len(input_ids) <= max_length:
                filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return prompt


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor:
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    return len(get_capacity_constrained_micro_batches(total_lengths, max_tokens_per_gpu))


def get_capacity_constrained_micro_batches(total_lengths, max_tokens_per_gpu, target_num_micro_batches=None):
    """Pack sample indices with deterministic best-fit decreasing.

    Samples longer than ``max_tokens_per_gpu`` remain valid singleton
    micro-batches. They cannot be combined with another sample. When data
    parallel ranks require the same number of micro-batches, ``target`` may be
    larger than the local minimum; existing bins are then split without
    violating capacity.
    """
    if max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")

    lengths = [int(length) for length in total_lengths]
    if any(length < 0 for length in lengths):
        raise ValueError("total_lengths must be non-negative")

    # Each entry is [current token sum, [original sample indices]]. Sorting by
    # (-length, index) makes packing reproducible across runs and Python builds.
    bins = []
    for index, length in sorted(enumerate(lengths), key=lambda item: (-item[1], item[0])):
        if length > max_tokens_per_gpu:
            bins.append([length, [index]])
            continue

        candidates = [
            (max_tokens_per_gpu - (token_sum + length), bin_index)
            for bin_index, (token_sum, _) in enumerate(bins)
            if token_sum <= max_tokens_per_gpu and token_sum + length <= max_tokens_per_gpu
        ]
        if candidates:
            _, bin_index = min(candidates)
            bins[bin_index][0] += length
            bins[bin_index][1].append(index)
        else:
            bins.append([length, [index]])

    if target_num_micro_batches is not None:
        target = int(target_num_micro_batches)
        if target < len(bins):
            raise ValueError(f"target_num_micro_batches={target} is below packed minimum={len(bins)}")
        if target > len(lengths):
            raise ValueError(f"target_num_micro_batches={target} exceeds num_samples={len(lengths)}")

        while len(bins) < target:
            # Split the fullest splittable bin first. Pop its smallest item so
            # both resulting bins remain capacity-safe.
            splittable = [
                (len(indices), token_sum, -bin_index, bin_index)
                for bin_index, (token_sum, indices) in enumerate(bins)
                if len(indices) > 1
            ]
            if not splittable:
                raise ValueError(f"cannot expand {len(bins)} bins to target={target}")
            _, _, _, bin_index = max(splittable)
            index = bins[bin_index][1].pop()
            length = lengths[index]
            bins[bin_index][0] -= length
            bins.append([length, [index]])

    return [indices for _, indices in bins]


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = list(rollout_data.pop("partition"))
    rollout_data["_partition"] = partition
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
