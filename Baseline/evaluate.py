"""Pass@1 evaluation for math/code GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

try:
    from .data import build_eval_datasets
    from .verify_code import verify_code_completion
    from .verify_math import verify_math_answer
except ImportError:  # pragma: no cover - used when running as a script.
    from data import build_eval_datasets
    from verify_code import verify_code_completion
    from verify_math import verify_math_answer


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["model_name_or_path"] = args.checkpoint

    eval_sets = build_eval_datasets(config)
    if not eval_sets:
        raise SystemExit("No evaluation datasets were built.")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on server env.
        raise SystemExit(
            "Transformers is required for evaluation. Install Baseline/requirements.txt "
            "or run on the prepared server."
        ) from exc

    model_name = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if config.get("bf16", False) else "auto",
        device_map=config.get("eval_device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()

    all_results: dict[str, Any] = {}
    for name, dataset in eval_sets.items():
        result = evaluate_dataset(
            model,
            tokenizer,
            dataset,
            config=config,
            allow_code_execution=args.allow_code_execution or bool(config.get("allow_code_execution")),
            max_samples=args.max_samples,
        )
        all_results[name] = result
        print(f"{name}: {result}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    *,
    config: Mapping[str, Any],
    allow_code_execution: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    total = 0
    correct = 0
    code_timeouts = 0
    code_errors: dict[str, int] = {}

    batch_size = int(config.get("eval_batch_size", 1))
    rows = [dataset[idx] for idx in range(len(dataset))]
    if max_samples is not None:
        rows = rows[:max_samples]

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompt_texts = [
            render_prompt(tokenizer, row["prompt"], enable_thinking=bool(config.get("enable_thinking")))
            for row in batch
        ]
        completions = generate_batch(model, tokenizer, prompt_texts, config)
        for row, completion in zip(batch, completions):
            total += 1
            if row["task"] == "math":
                passed = verify_math_answer(completion, row["answer"]).passed
            elif row["task"] == "code":
                if not allow_code_execution:
                    raise RuntimeError(
                        "Code evaluation requires --allow_code_execution because it runs "
                        "model-generated Python."
                    )
                code_result = verify_code_completion(
                    completion,
                    row["tests"],
                    entry_point=row.get("entry_point") or None,
                    timeout_seconds=float(config.get("code_timeout_seconds", 5.0)),
                )
                passed = code_result.passed
                if code_result.timed_out:
                    code_timeouts += 1
                if code_result.error_type != "none":
                    code_errors[code_result.error_type] = code_errors.get(code_result.error_type, 0) + 1
            else:
                raise ValueError(f"Unknown task: {row['task']}")
            correct += int(passed)

    return {
        "samples": total,
        "correct": correct,
        "pass_at_1": correct / total if total else 0.0,
        "code_timeouts": code_timeouts,
        "code_errors": code_errors,
    }


def generate_batch(model, tokenizer, prompt_texts: list[str], config: Mapping[str, Any]) -> list[str]:
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.get("max_prompt_length"),
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(config.get("eval_max_new_tokens", config.get("max_completion_length", 512))),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(output[input_width:], skip_special_tokens=True)
        for output in output_ids
    ]


def render_prompt(tokenizer, messages, *, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Model/checkpoint path to evaluate.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--max_samples", type=int, help="Limit examples per eval dataset.")
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow evaluation to execute generated Python code.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()

