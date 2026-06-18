"""Processed dataset loading for the math+code GRPO baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_from_disk

try:
    from .formatting import code_prompt, math_prompt
except ImportError:  # pragma: no cover - used when running as a script.
    from formatting import code_prompt, math_prompt


DEFAULT_EVAL_NAMES = ("gsm8k", "math500", "mbpp", "humaneval")


def build_train_dataset(config: Mapping[str, Any]) -> Dataset:
    if config.get("use_synthetic_data", False):
        return Dataset.from_list(_synthetic_train_records())

    path = _config_get(config, "processed_train_path")
    if not path:
        raise ValueError(
            "processed_train_path is required. Run Baseline/preprocess_datasets.py "
            "to build the DAPO-Math + TACO training dataset first."
        )

    dataset = load_from_disk(str(path))
    max_samples = config.get("max_train_samples")
    if max_samples is not None and int(max_samples) < len(dataset):
        dataset = dataset.select(range(int(max_samples)))
    return dataset


def build_eval_datasets(config: Mapping[str, Any]) -> dict[str, Dataset]:
    if config.get("use_synthetic_data", False):
        return {
            "synthetic_math": Dataset.from_list(_synthetic_math_records()),
            "synthetic_code": Dataset.from_list(_synthetic_code_records()),
        }

    eval_sets: dict[str, Dataset] = {}
    eval_paths = _config_get(config, "eval_dataset_paths") or {}
    for name, path in eval_paths.items():
        if path:
            eval_sets[name] = load_from_disk(str(path))

    eval_dir = _config_get(config, "processed_eval_dir")
    if eval_dir:
        root = Path(str(eval_dir))
        names = _config_get(config, "eval_dataset_names") or DEFAULT_EVAL_NAMES
        for name in names:
            if name in eval_sets:
                continue
            path = root / str(name)
            if path.exists():
                eval_sets[str(name)] = load_from_disk(str(path))

    if not eval_sets:
        raise ValueError(
            "No processed eval datasets found. Set processed_eval_dir or "
            "eval_dataset_paths after running Baseline/preprocess_datasets.py."
        )
    return eval_sets


def _synthetic_train_records() -> list[dict[str, Any]]:
    return _synthetic_math_records() + _synthetic_code_records()


def _synthetic_math_records() -> list[dict[str, Any]]:
    examples = [
        ("What is 2 + 3?", "5"),
        ("If x = 7, what is x * x?", "49"),
    ]
    return [
        {
            "id": f"synthetic-math-{idx}",
            "source": "synthetic",
            "task": "math",
            "prompt": math_prompt(question),
            "answer": answer,
            "tests": [],
            "entry_point": "",
            "test_type": "",
            "difficulty": "",
        }
        for idx, (question, answer) in enumerate(examples)
    ]


def _synthetic_code_records() -> list[dict[str, Any]]:
    examples = [
        (
            "Write a function add_one(x) that returns x plus one.",
            ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
            "add_one",
        ),
        (
            "Write a function square(x) that returns x squared.",
            ["assert square(4) == 16", "assert square(-3) == 9"],
            "square",
        ),
    ]
    return [
        {
            "id": f"synthetic-code-{idx}",
            "source": "synthetic",
            "task": "code",
            "prompt": code_prompt(problem, tests, entry_point=entry_point),
            "answer": "",
            "tests": tests,
            "entry_point": entry_point,
            "test_type": "assert",
            "difficulty": "SYNTHETIC",
        }
        for idx, (problem, tests, entry_point) in enumerate(examples)
    ]


def _config_get(config: Mapping[str, Any], key: str):
    if key in config:
        return config[key]
    datasets_cfg = config.get("datasets", {})
    if isinstance(datasets_cfg, Mapping):
        return datasets_cfg.get(key)
    return None

