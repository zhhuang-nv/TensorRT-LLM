#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate JoyAI-LLM-Flash on GSM8K with TensorRT-LLM."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

INVALID = -9999999
OFFICIAL_GSM8K_ACCURACY = 0.9583
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
BASIC_CHAT_TEMPLATE = "{%- for message in messages -%}{{- message['content'] -}}{%- endfor -%}"
JOYAI_DEFAULT_SYSTEM = (
    "You are JoyAI , a large language model trained by JD（京东）that can interact with a "
    "computer to solve tasks. Answer as concisely as possible."
)


def _download(url: str, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {output_path}", flush=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def _read_jsonl(path: Path) -> Iterable[dict[str, str]]:
    with path.open() as jsonl:
        for line in jsonl:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def load_gsm8k_data(cache_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_path = _download(GSM8K_TRAIN_URL, cache_dir / "train.jsonl")
    test_path = _download(GSM8K_TEST_URL, cache_dir / "test.jsonl")
    return list(_read_jsonl(train_path)), list(_read_jsonl(test_path))


def _numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))


def _to_number(text: str) -> int | float:
    try:
        value = float(text)
    except ValueError:
        return INVALID
    if abs(value - round(value)) < 1e-6:
        return int(round(value))
    return value


def extract_answer(text: str) -> int | float:
    """Extract a numeric answer from JoyAI/GSM8K-style output."""
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"(?i)(?:final answer|answer)\s*[:=]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if not matches:
            continue
        numbers = _numbers(matches[-1].group(1))
        if numbers:
            return _to_number(numbers[-1])

    numbers = _numbers(text)
    return _to_number(numbers[-1]) if numbers else INVALID


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    conversation: list[dict[str, str]],
    chat_template: str,
) -> str:
    if chat_template == "joyai":
        system_prompt = JOYAI_DEFAULT_SYSTEM
        for message in conversation:
            if message["role"] == "system":
                system_prompt = message["content"]
                break

        prompt_parts = [f"{tokenizer.bos_token or ''}{system_prompt}"]
        last_was_user = False
        for message in conversation:
            if message["role"] == "system":
                continue
            if message["role"] == "user":
                prompt_parts.append(f"<|User|>{message['content']}")
                last_was_user = True
            elif message["role"] == "assistant":
                if last_was_user:
                    prompt_parts.append("<|Assistant|>")
                prompt_parts.append(
                    f"<|end_of_thought|>{message.get('content', '')}{tokenizer.eos_token or ''}"
                )
                last_was_user = False

        prompt_parts.append("<|Assistant|><|end_of_thought|>")
        return "".join(prompt_parts)

    if chat_template == "basic":
        template = BASIC_CHAT_TEMPLATE
    elif chat_template == "auto":
        template = None
    else:
        template_path = Path(chat_template)
        template = template_path.read_text() if template_path.exists() else chat_template

    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
    )


def build_prompts(
    tokenizer: PreTrainedTokenizerBase,
    cache_dir: Path,
    num_questions: int,
    num_shots: int,
    chat_template: str,
) -> tuple[list[str], list[int | float]]:
    train_data, test_data = load_gsm8k_data(cache_dir)
    num_questions = min(num_questions, len(test_data))

    few_shot_prompt = "".join(
        f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['answer']}\n\n"
        for i in range(num_shots)
    )

    conversations = [
        [
            {
                "role": "user",
                "content": few_shot_prompt + f"Question: {test_data[i]['question']}\nAnswer:",
            }
        ]
        for i in range(num_questions)
    ]
    labels = [extract_answer(test_data[i]["answer"]) for i in range(num_questions)]
    prompts = [
        _apply_chat_template(tokenizer, conversation, chat_template)
        for conversation in conversations
    ]
    return prompts, labels


def _load_existing_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    records = {}
    with path.open() as jsonl:
        for line in jsonl:
            record = json.loads(line)
            records[int(record["idx"])] = record
    return records


def _append_records(path: Path, records: list[dict]) -> None:
    with path.open("a") as jsonl:
        for record in records:
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()


def _score(records: list[dict], total_questions: int) -> tuple[int, int, int, int]:
    correct = sum(int(record["correct"]) for record in records)
    invalid = sum(int(record["pred"] == INVALID) for record in records)
    maxed = sum(int(record["output_tokens"] >= record["max_tokens"]) for record in records)
    total_output_tokens = sum(int(record["output_tokens"]) for record in records)

    if len(records) != total_questions:
        print(
            f"WARNING scored {len(records)} records but expected {total_questions}",
            flush=True,
        )
    return correct, invalid, maxed, total_output_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/scratch.zhhuang_sw/aq_runs/joyai_native")
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--num-shots", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--moe-expert-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--kv-cache-free-gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--enable-attention-dp", action="store_true")
    parser.add_argument("--output-dir", default="./joyai_gsm8k_trtllm")
    parser.add_argument("--data-cache-dir", default="./joyai_gsm8k_data")
    parser.add_argument("--chat-template", default="joyai")
    parser.add_argument("--official-accuracy", type=float, default=OFFICIAL_GSM8K_ACCURACY)
    parser.add_argument("--warn-threshold", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--fix-mistral-regex", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.set_level("info")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stop_suffix = "_stop" if args.stop else ""
    run_name = (
        f"{args.dtype}_{args.chat_template}{stop_suffix}_"
        f"4shot_{args.max_tokens}_n{args.num_questions}"
    )
    summary_path = output_dir / f"{run_name}_summary.json"
    predictions_path = output_dir / f"{run_name}_predictions.jsonl"

    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    prompts, labels = build_prompts(
        tokenizer=tokenizer,
        cache_dir=Path(args.data_cache_dir),
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        chat_template=args.chat_template,
    )

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction)

    if args.dtype == "bf16":
        moe_config = MoeConfig(backend="AUTO")
    else:
        moe_config = MoeConfig(backend="TRTLLM")
    llm_kwargs = {
        "max_seq_len": args.max_model_len,
        "max_batch_size": args.max_batch_size,
        "kv_cache_config": kv_cache_config,
        "cuda_graph_config": None,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "moe_expert_parallel_size": args.moe_expert_parallel_size,
        "moe_config": moe_config,
        "enable_attention_dp": args.enable_attention_dp,
    }

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
        stop=args.stop,
        add_special_tokens=False,
    )

    if args.no_resume and predictions_path.exists():
        predictions_path.unlink()
    existing_records = {} if args.no_resume else _load_existing_records(predictions_path)
    pending_indices = [idx for idx in range(len(prompts)) if idx not in existing_records]

    print(
        "RUN_BEGIN "
        f"questions={len(prompts)} pending={len(pending_indices)} "
        f"shots={args.num_shots} max_tokens={args.max_tokens} chat_template={args.chat_template}",
        flush=True,
    )

    start_time = time.perf_counter()
    if pending_indices:
        llm = LLM(
            model=args.model,
            tokenizer=tokenizer,
            trust_remote_code=args.trust_remote_code,
            **llm_kwargs,
        )

        pending_prompts = [prompts[idx] for idx in pending_indices]
        outputs = llm.generate(
            pending_prompts,
            sampling_params=sampling_params,
            use_tqdm=not args.no_tqdm,
        )

        new_records = []
        for idx, output in zip(pending_indices, outputs, strict=True):
            completion = output.outputs[0]
            text = completion.text
            output_tokens = len(completion.token_ids)
            pred = extract_answer(text)
            label = labels[idx]
            new_records.append(
                {
                    "idx": idx,
                    "label": label,
                    "pred": pred,
                    "correct": bool(pred == label),
                    "output_tokens": output_tokens,
                    "max_tokens": args.max_tokens,
                    "prompt_tokens": len(tokenizer.encode(prompts[idx], add_special_tokens=False)),
                    "output": text,
                }
            )
        _append_records(predictions_path, new_records)

        print(
            f"RUN_PROGRESS completed_new={len(new_records)}/{len(pending_indices)} "
            f"total_records={len(existing_records) + len(new_records)}/{len(prompts)}",
            flush=True,
        )

    latency = time.perf_counter() - start_time
    records_by_idx = _load_existing_records(predictions_path)
    records = [records_by_idx[idx] for idx in sorted(records_by_idx)]
    correct, invalid, maxed, total_output_tokens = _score(records, len(prompts))
    accuracy = correct / len(prompts) if prompts else 0.0

    summary = {
        "model": args.model,
        "chat_template": args.chat_template,
        "num_questions": len(prompts),
        "num_shots": args.num_shots,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "moe_backend": moe_config.backend,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "moe_expert_parallel_size": args.moe_expert_parallel_size,
        "enable_attention_dp": args.enable_attention_dp,
        "max_tokens": args.max_tokens,
        "stop": args.stop,
        "correct": correct,
        "accuracy": accuracy,
        "official_accuracy": args.official_accuracy,
        "diff_abs": accuracy - args.official_accuracy,
        "warn_threshold": args.warn_threshold,
        "invalid": invalid,
        "invalid_rate": invalid / len(prompts) if prompts else 0.0,
        "maxed": maxed,
        "latency": latency,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": total_output_tokens / latency if latency > 0 else 0.0,
        "timestamp": time.time(),
        "predictions_path": str(predictions_path),
    }
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    if abs(summary["diff_abs"]) > args.warn_threshold:
        print("RUN_WARNING accuracy differs from official target beyond threshold", flush=True)
    print("RUN_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("RUN_OUTPUTS", summary_path, predictions_path, flush=True)


if __name__ == "__main__":
    main()
