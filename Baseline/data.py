"""Dataset builders for the math+code multi-task GRPO baseline."""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from typing import Any, Iterable, Mapping, Optional

from datasets import Dataset, load_dataset

try:
    from .verify_code import infer_entry_point_from_tests
    from .verify_math import extract_gsm8k_answer
except ImportError:  # pragma: no cover - used when running as a script.
    from verify_code import infer_entry_point_from_tests
    from verify_math import extract_gsm8k_answer


MATH_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. Solve the problem step by step, "
    "then put the final answer in \\boxed{...}."
)

CODE_SYSTEM_PROMPT = (
    "You are a precise Python programmer. Return only valid Python code that "
    "satisfies the requested function and tests."
)


def build_train_dataset(config: Mapping[str, Any]) -> Dataset:
    if config.get("use_synthetic_data", False):
        return build_mixed_dataset_from_records(
            _synthetic_math_records(),
            _synthetic_code_records(),
            task_weights=config.get("task_weights", {"math": 0.5, "code": 0.5}),
            max_samples=config.get("max_train_samples", 8),
            seed=int(config.get("seed", 42)),
        )

    datasets_cfg = config.get("datasets", {})
    limits = config.get("source_limits", {})
    math_records = load_gsm8k_records(
        datasets_cfg,
        split=datasets_cfg.get("gsm8k_train_split", "train"),
        max_records=limits.get("math_train"),
    )
    code_records = load_mbpp_records(
        datasets_cfg,
        train=True,
        max_records=limits.get("code_train"),
    )
    return build_mixed_dataset_from_records(
        math_records,
        code_records,
        task_weights=config.get("task_weights", {"math": 0.5, "code": 0.5}),
        max_samples=config.get("max_train_samples"),
        seed=int(config.get("seed", 42)),
    )


def build_eval_datasets(config: Mapping[str, Any]) -> dict[str, Dataset]:
    if config.get("use_synthetic_data", False):
        return {
            "synthetic_math": Dataset.from_list(_synthetic_math_records()),
            "synthetic_code": Dataset.from_list(_synthetic_code_records()),
        }

    datasets_cfg = config.get("datasets", {})
    limits = config.get("source_limits", {})
    eval_sets: dict[str, Dataset] = {}

    gsm8k_eval = load_gsm8k_records(
        datasets_cfg,
        split=datasets_cfg.get("gsm8k_eval_split", "test"),
        max_records=limits.get("math_eval"),
    )
    if gsm8k_eval:
        eval_sets["gsm8k"] = Dataset.from_list(gsm8k_eval)

    math500 = load_math500_records(datasets_cfg, max_records=limits.get("math500_eval"))
    if math500:
        eval_sets["math500"] = Dataset.from_list(math500)

    mbpp_eval = load_mbpp_records(
        datasets_cfg,
        train=False,
        max_records=limits.get("code_eval"),
    )
    if mbpp_eval:
        eval_sets["mbpp_heldout"] = Dataset.from_list(mbpp_eval)

    humaneval = load_humaneval_records(datasets_cfg, max_records=limits.get("humaneval_eval"))
    if humaneval:
        eval_sets["humaneval"] = Dataset.from_list(humaneval)

    return eval_sets


def build_mixed_dataset_from_records(
    math_records: Iterable[Mapping[str, Any]],
    code_records: Iterable[Mapping[str, Any]],
    *,
    task_weights: Mapping[str, float],
    max_samples: Optional[int],
    seed: int,
) -> Dataset:
    records_by_task = {
        "math": [dict(record) for record in math_records],
        "code": [dict(record) for record in code_records],
    }
    mixed = weighted_mix_records(records_by_task, task_weights, max_samples=max_samples, seed=seed)
    return Dataset.from_list(mixed)


def weighted_mix_records(
    records_by_task: Mapping[str, list[Mapping[str, Any]]],
    task_weights: Mapping[str, float],
    *,
    max_samples: Optional[int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    available_tasks = [
        task
        for task, records in records_by_task.items()
        if records and float(task_weights.get(task, 0.0)) > 0.0
    ]
    if not available_tasks:
        raise ValueError("No records available for any task with non-zero sampling weight.")

    if max_samples is None:
        records = [dict(record) for task in available_tasks for record in records_by_task[task]]
        rng.shuffle(records)
        for idx, record in enumerate(records):
            record["mix_index"] = idx
        return records

    weights = [float(task_weights[task]) for task in available_tasks]
    pools = {task: [dict(record) for record in records_by_task[task]] for task in available_tasks}
    cursors = {task: 0 for task in available_tasks}
    for pool in pools.values():
        rng.shuffle(pool)

    mixed: list[dict[str, Any]] = []
    for mix_index in range(int(max_samples)):
        task = rng.choices(available_tasks, weights=weights, k=1)[0]
        pool = pools[task]
        cursor = cursors[task]
        if cursor > 0 and cursor % len(pool) == 0:
            rng.shuffle(pool)
        record = deepcopy(pool[cursor % len(pool)])
        record["mix_index"] = mix_index
        mixed.append(record)
        cursors[task] = cursor + 1

    return mixed


def load_gsm8k_records(
    datasets_cfg: Mapping[str, Any],
    *,
    split: str,
    max_records: Optional[int] = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        datasets_cfg.get("gsm8k_name", "openai/gsm8k"),
        datasets_cfg.get("gsm8k_config", "main"),
        split=split,
    )
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        question = str(_row_get(row, "question", "problem", default="")).strip()
        answer = extract_gsm8k_answer(str(_row_get(row, "answer", default="")).strip())
        if not question or not answer:
            continue
        records.append(
            {
                "id": f"gsm8k-{split}-{idx}",
                "source": "gsm8k",
                "task": "math",
                "prompt": _math_prompt(question),
                "answer": answer,
                "tests": [],
                "entry_point": "",
            }
        )
    return records


def load_math500_records(
    datasets_cfg: Mapping[str, Any],
    *,
    max_records: Optional[int] = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        datasets_cfg.get("math500_name", "HuggingFaceH4/MATH-500"),
        split=datasets_cfg.get("math500_split", "test"),
    )
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        problem = str(_row_get(row, "problem", "question", default="")).strip()
        answer = str(_row_get(row, "answer", "solution", default="")).strip()
        if not problem or not answer:
            continue
        records.append(
            {
                "id": f"math500-{idx}",
                "source": "math500",
                "task": "math",
                "prompt": _math_prompt(problem),
                "answer": answer,
                "tests": [],
                "entry_point": "",
            }
        )
    return records


def load_mbpp_records(
    datasets_cfg: Mapping[str, Any],
    *,
    train: bool,
    max_records: Optional[int] = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        datasets_cfg.get("mbpp_name", "google-research-datasets/mbpp"),
        split=datasets_cfg.get("mbpp_split", "train"),
    )
    eval_ratio = float(datasets_cfg.get("mbpp_eval_ratio", 0.1))
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        task_id = str(_row_get(row, "task_id", default=idx))
        heldout = _stable_fraction(task_id) < eval_ratio
        if heldout == train:
            continue
        setup = _coerce_tests(_row_get(row, "test_setup_code", default=[]))
        tests = setup + _coerce_tests(_row_get(row, "test_list", "tests", default=[]))
        text = str(_row_get(row, "text", "prompt", default="")).strip()
        if not text or not tests:
            continue
        entry_point = str(_row_get(row, "entry_point", default="")).strip()
        if not entry_point:
            entry_point = infer_entry_point_from_tests(tests) or ""
        records.append(
            {
                "id": f"mbpp-{task_id}",
                "source": "mbpp",
                "task": "code",
                "prompt": _code_prompt(text, tests, entry_point=entry_point),
                "answer": "",
                "tests": tests,
                "entry_point": entry_point,
            }
        )
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


def load_humaneval_records(
    datasets_cfg: Mapping[str, Any],
    *,
    max_records: Optional[int] = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        datasets_cfg.get("humaneval_name", "openai/openai_humaneval"),
        split=datasets_cfg.get("humaneval_split", "test"),
    )
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        prompt = str(_row_get(row, "prompt", default="")).strip()
        entry_point = str(_row_get(row, "entry_point", default="")).strip()
        test = str(_row_get(row, "test", default="")).strip()
        if not prompt or not entry_point or not test:
            continue
        tests = [test, f"check({entry_point})"]
        records.append(
            {
                "id": str(_row_get(row, "task_id", default=f"humaneval-{idx}")),
                "source": "humaneval",
                "task": "code",
                "prompt": _code_prompt(prompt, tests, entry_point=entry_point),
                "answer": "",
                "tests": tests,
                "entry_point": entry_point,
            }
        )
    return records


def _math_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _code_prompt(problem: str, tests: Iterable[str], *, entry_point: str = "") -> list[dict[str, str]]:
    test_text = "\n".join(str(test) for test in tests)
    signature = f"\nFunction name: {entry_point}" if entry_point else ""
    user = (
        f"{problem.strip()}{signature}\n\n"
        f"Reference tests:\n{test_text}\n\n"
        "Return a complete Python solution. Do not include Markdown fences."
    )
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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
            "prompt": _math_prompt(question),
            "answer": answer,
            "tests": [],
            "entry_point": "",
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
            "prompt": _code_prompt(problem, tests, entry_point=entry_point),
            "answer": "",
            "tests": tests,
            "entry_point": entry_point,
        }
        for idx, (problem, tests, entry_point) in enumerate(examples)
    ]


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _coerce_tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _row_get(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _limit(dataset: Iterable[Mapping[str, Any]], max_records: Optional[int]):
    for idx, row in enumerate(dataset):
        if max_records is not None and idx >= int(max_records):
            break
        yield row
