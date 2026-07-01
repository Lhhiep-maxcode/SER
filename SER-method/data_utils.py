"""Dataset loading and collation for the scratch GRPO baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_from_disk


def load_processed_dataset(path: str | Path, *, max_samples: int | None = None) -> Dataset:
    dataset_path = Path(str(path))
    suffix = dataset_path.suffix.lower()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset path does not exist: {dataset_path}")

    if dataset_path.is_file() and suffix in {".json", ".jsonl"}:
        dataset = Dataset.from_json(str(dataset_path), cache_dir=str(_json_cache_dir()))
    elif dataset_path.is_file() and suffix == ".arrow":
        dataset = Dataset.from_file(str(dataset_path))
    elif dataset_path.is_dir():
        if _is_saved_dataset_dir(dataset_path):
            dataset = load_from_disk(str(dataset_path))
        else:
            arrow_files = sorted(dataset_path.glob("*.arrow"))
            jsonl_files = sorted(dataset_path.glob("*.jsonl"))
            json_files = [
                path
                for path in sorted(dataset_path.glob("*.json"))
                if path.name not in {"dataset_info.json", "dataset_dict.json", "state.json"}
            ]

            if arrow_files:
                shards = [Dataset.from_file(str(path)) for path in arrow_files]
                dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
            elif jsonl_files:
                dataset = Dataset.from_json(
                    [str(path) for path in jsonl_files],
                    cache_dir=str(_json_cache_dir()),
                )
            elif json_files:
                dataset = Dataset.from_json(
                    [str(path) for path in json_files],
                    cache_dir=str(_json_cache_dir()),
                )
            else:
                dataset = load_from_disk(str(dataset_path))
    else:
        raise ValueError(
            f"Unsupported processed dataset format: {dataset_path}. "
            "Use a Hugging Face save_to_disk directory, .arrow file, .jsonl file, or .json file."
        )

    if max_samples is not None and int(max_samples) < len(dataset):
        dataset = dataset.select(range(int(max_samples)))
    return dataset


def _is_saved_dataset_dir(path: Path) -> bool:
    return (path / "state.json").exists() and (
        (path / "dataset_info.json").exists() or any(path.glob("*.arrow"))
    )


def render_prompt(tokenizer, prompt: list[dict[str, str]], *, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

def render_full_message(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def _json_cache_dir() -> Path:
    path = Path(".cache") / "baseline_grpo_json"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "source": str(row.get("source") or ""),
        "task": str(row.get("task") or ""),
        "prompt": row.get("prompt") or [],
        "answer": str(row.get("answer") or ""),
        "tests": row.get("tests") or [],
        "entry_point": str(row.get("entry_point") or ""),
        "test_type": str(row.get("test_type") or ""),
        "difficulty": str(row.get("difficulty") or ""),
    }


class ProcessedDataCollator:
    def __init__(self, tokenizer, *, max_prompt_length: int, enable_thinking: bool):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.enable_thinking = enable_thinking

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [normalize_row(example) for example in examples]
        prompt_texts = [
            render_prompt(self.tokenizer, row["prompt"], enable_thinking=self.enable_thinking)
            for row in rows
        ]
        tokenized = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "rows": rows,
        }
