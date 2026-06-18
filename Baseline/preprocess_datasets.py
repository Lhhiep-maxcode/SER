"""Preprocess raw HF datasets into the baseline GRPO schema.

Training data:
  - math: BytedTsinghua-SIA/DAPO-Math-17k
  - code: BAAI/TACO

Evaluation data:
  - GSM8K, MATH-500, MBPP, HumanEval
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

try:
    from .formatting import code_prompt, math_prompt
    from .verify_code import infer_entry_point_from_tests
    from .verify_math import extract_gsm8k_answer
except ImportError:  # pragma: no cover - used when running as a script.
    from formatting import code_prompt, math_prompt
    from verify_code import infer_entry_point_from_tests
    from verify_math import extract_gsm8k_answer


SCHEMA_COLUMNS = (
    "id",
    "source",
    "task",
    "prompt",
    "answer",
    "tests",
    "entry_point",
    "test_type",
    "difficulty",
)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and converting DAPO-Math...")
    dapo_source_limit = args.max_dapo_records
    if dapo_source_limit is None:
        dapo_source_limit = max(
            args.train_size,
            int(args.train_size * max(args.math_weight, 0.1) * 3),
        )
        print(f"Using --max_dapo_records {dapo_source_limit} by default for preprocessing speed.")

    math_records = load_dapo_math_records(
        args.dapo_name,
        split=args.dapo_split,
        max_records=dapo_source_limit,
    )
    print(f"Converted {len(math_records)} DAPO-Math records.")

    print("Loading and converting TACO...")
    code_records = load_taco_records(
        args.taco_name,
        split=args.taco_split,
        max_records=args.max_taco_records,
        max_tests_per_sample=args.taco_max_tests_per_sample,
        trust_remote_code=args.taco_trust_remote_code,
    )
    print(f"Converted {len(code_records)} TACO records.")

    difficulty_weights = parse_weight_mapping(args.code_difficulty_weights)
    train_records = sample_train_records(
        math_records,
        code_records,
        train_size=args.train_size,
        math_weight=args.math_weight,
        code_weight=args.code_weight,
        code_difficulty_weights=difficulty_weights,
        rng=rng,
    )
    save_records(train_records, output_dir / "train")

    eval_root = output_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_sets = build_eval_sets(args)
    for name, records in eval_sets.items():
        save_records(records, eval_root / name)

    metadata = {
        "seed": args.seed,
        "train_size": len(train_records),
        "math_weight": args.math_weight,
        "code_weight": args.code_weight,
        "code_difficulty_weights": difficulty_weights,
        "train_counts": dict(Counter(record["task"] for record in train_records)),
        "code_difficulty_counts": dict(
            Counter(record["difficulty"] for record in train_records if record["task"] == "code")
        ),
        "eval_counts": {name: len(records) for name, records in eval_sets.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="Baseline/processed/dapo_taco")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--math_weight", type=float, default=0.5)
    parser.add_argument("--code_weight", type=float, default=0.5)
    parser.add_argument(
        "--code_difficulty_weights",
        default="EASY=0.5,MEDIUM=0.5",
        help="Comma-separated TACO difficulty weights, e.g. EASY=0.4,MEDIUM=0.4,HARD=0.2.",
    )

    parser.add_argument("--dapo_name", default="BytedTsinghua-SIA/DAPO-Math-17k")
    parser.add_argument("--dapo_split", default="train")
    parser.add_argument("--max_dapo_records", type=int)

    parser.add_argument("--taco_name", default="BAAI/TACO")
    parser.add_argument("--taco_split", default="train")
    parser.add_argument("--max_taco_records", type=int)
    parser.add_argument("--taco_max_tests_per_sample", type=int)
    parser.add_argument("--no_taco_trust_remote_code", dest="taco_trust_remote_code", action="store_false")
    parser.set_defaults(taco_trust_remote_code=True)

    parser.add_argument("--gsm8k_name", default="openai/gsm8k")
    parser.add_argument("--gsm8k_config", default="main")
    parser.add_argument("--gsm8k_split", default="test")
    parser.add_argument("--max_gsm8k_eval", type=int)

    parser.add_argument("--math500_name", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500_split", default="test")
    parser.add_argument("--max_math500_eval", type=int)

    parser.add_argument("--mbpp_name", default="google-research-datasets/mbpp")
    parser.add_argument("--mbpp_split", default="train")
    parser.add_argument("--max_mbpp_eval", type=int)

    parser.add_argument("--humaneval_name", default="openai/openai_humaneval")
    parser.add_argument("--humaneval_split", default="test")
    parser.add_argument("--max_humaneval_eval", type=int)
    return parser.parse_args()


def load_dapo_math_records(
    name: str,
    *,
    split: str,
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    dataset = load_dataset(name, split=split)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        prompt_text = _extract_dapo_prompt_text(_row_get(row, "prompt", default=""))
        problem = _strip_dapo_instruction(prompt_text)
        reward_model = _row_get(row, "reward_model", default={}) or {}
        answer = str(reward_model.get("ground_truth", "")).strip()
        extra_info = _row_get(row, "extra_info", default={}) or {}
        item_id = str(extra_info.get("index") or idx)
        if not problem or not answer:
            continue
        records.append(
            normalize_record(
                {
                    "id": f"dapo-{item_id}",
                    "source": "dapo_math",
                    "task": "math",
                    "prompt": math_prompt(problem),
                    "answer": answer,
                    "tests": [],
                    "entry_point": "",
                    "test_type": "",
                    "difficulty": "",
                }
            )
        )
    return records


def load_taco_records(
    name: str,
    *,
    split: str,
    max_records: Optional[int],
    max_tests_per_sample: Optional[int],
    trust_remote_code: bool,
) -> list[dict[str, Any]]:
    dataset = load_taco_dataset(name, split=split, trust_remote_code=trust_remote_code)

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        question = str(_row_get(row, "question", default="")).strip()
        difficulty = str(_row_get(row, "difficulty", default="UNKNOWN")).upper() or "UNKNOWN"
        input_output = _parse_jsonish(_row_get(row, "input_output", default={}))
        if not question or not isinstance(input_output, Mapping):
            continue

        inputs = input_output.get("inputs") or input_output.get("input") or []
        outputs = input_output.get("outputs") or input_output.get("output") or []
        entry_point = str(input_output.get("fn_name") or "").strip()
        starter_code = str(_row_get(row, "starter_code", default="")).strip()

        if entry_point:
            tests = _function_assert_tests_from_io(
                entry_point,
                inputs,
                outputs,
                max_tests=max_tests_per_sample,
            )
            test_type = "assert"
        else:
            tests = _stdin_stdout_tests_from_io(inputs, outputs, max_tests=max_tests_per_sample)
            test_type = "stdin_stdout"

        if not tests:
            continue
        records.append(
            normalize_record(
                {
                    "id": f"taco-{split}-{idx}",
                    "source": "taco",
                    "task": "code",
                    "prompt": code_prompt(
                        question,
                        tests,
                        entry_point=entry_point,
                        test_type=test_type,
                        starter_code=starter_code,
                    ),
                    "answer": "",
                    "tests": tests,
                    "entry_point": entry_point,
                    "test_type": test_type,
                    "difficulty": difficulty,
                }
            )
        )
    return records


def load_taco_dataset(name: str, *, split: str, trust_remote_code: bool):
    local_path = Path(name)
    if local_path.exists():
        return load_taco_arrow_split(local_path, split)

    kwargs: dict[str, Any] = {"split": split}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        return load_dataset(name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        try:
            return load_dataset(name, **kwargs)
        except RuntimeError as exc:
            return _load_taco_after_script_error(name, split, exc)
    except RuntimeError as exc:
        return _load_taco_after_script_error(name, split, exc)


def _load_taco_after_script_error(name: str, split: str, exc: RuntimeError):
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        snapshot_path = find_hf_snapshot(name)
        if snapshot_path is None:
            raise RuntimeError(
                "TACO uses a legacy dataset script that this datasets version refuses. "
                "Download the raw TACO dataset repo locally and pass "
                "`--taco_name /path/to/BAAI_TACO`, or downgrade datasets to a version "
                "that still supports dataset scripts."
            ) from exc
        print(f"Falling back to direct TACO Arrow loading from {snapshot_path}")
        return load_taco_arrow_split(snapshot_path, split)


def load_taco_arrow_split(root: Path, split: str):
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Could not find TACO split directory {split_dir}. Expected raw repo layout "
            "with files like train/data-00000-of-00009.arrow."
        )

    arrow_files = sorted(split_dir.glob("*.arrow"))
    if arrow_files:
        shards = [Dataset.from_file(str(path)) for path in arrow_files]
        return concatenate_datasets(shards) if len(shards) > 1 else shards[0]

    parquet_files = sorted(split_dir.glob("*.parquet"))
    if parquet_files:
        return load_dataset(
            "parquet",
            data_files={split: [str(path) for path in parquet_files]},
            split=split,
        )

    json_files = sorted(split_dir.glob("*.json")) + sorted(split_dir.glob("*.jsonl"))
    if json_files:
        return load_dataset(
            "json",
            data_files={split: [str(path) for path in json_files]},
            split=split,
        )

    raise FileNotFoundError(f"No Arrow, Parquet, JSON, or JSONL files found in {split_dir}.")


def find_hf_snapshot(repo_id: str) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_files_only=True)
        if path:
            return Path(path)
    except Exception:
        pass

    candidates = []
    repo_cache_name = "datasets--" + repo_id.replace("/", "--")
    env_roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path.home() / ".cache" / "huggingface"),
    ]
    for root in env_roots:
        if not root:
            continue
        base = Path(root)
        candidates.extend((base / "hub" / repo_cache_name / "snapshots").glob("*"))
        candidates.extend((base / repo_cache_name / "snapshots").glob("*"))

    valid = [path for path in candidates if (path / "train").exists() or (path / "test").exists()]
    if not valid:
        return None
    valid.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return valid[0]


def build_eval_sets(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    print("Converting eval datasets...")
    return {
        "gsm8k": load_gsm8k_eval_records(
            args.gsm8k_name,
            config_name=args.gsm8k_config,
            split=args.gsm8k_split,
            max_records=args.max_gsm8k_eval,
        ),
        "math500": load_math500_eval_records(
            args.math500_name,
            split=args.math500_split,
            max_records=args.max_math500_eval,
        ),
        "mbpp": load_mbpp_eval_records(
            args.mbpp_name,
            split=args.mbpp_split,
            max_records=args.max_mbpp_eval,
        ),
        "humaneval": load_humaneval_eval_records(
            args.humaneval_name,
            split=args.humaneval_split,
            max_records=args.max_humaneval_eval,
        ),
    }


def load_gsm8k_eval_records(
    name: str,
    *,
    config_name: str,
    split: str,
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    dataset = load_dataset(name, config_name, split=split)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        question = str(_row_get(row, "question", "problem", default="")).strip()
        answer = extract_gsm8k_answer(str(_row_get(row, "answer", default="")).strip())
        if not question or not answer:
            continue
        records.append(
            normalize_record(
                {
                    "id": f"gsm8k-{split}-{idx}",
                    "source": "gsm8k",
                    "task": "math",
                    "prompt": math_prompt(question),
                    "answer": answer,
                    "tests": [],
                    "entry_point": "",
                    "test_type": "",
                    "difficulty": "",
                }
            )
        )
    return records


def load_math500_eval_records(
    name: str,
    *,
    split: str,
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    dataset = load_dataset(name, split=split)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        problem = str(_row_get(row, "problem", "question", default="")).strip()
        answer = str(_row_get(row, "answer", "solution", default="")).strip()
        if not problem or not answer:
            continue
        records.append(
            normalize_record(
                {
                    "id": f"math500-{idx}",
                    "source": "math500",
                    "task": "math",
                    "prompt": math_prompt(problem),
                    "answer": answer,
                    "tests": [],
                    "entry_point": "",
                    "test_type": "",
                    "difficulty": "",
                }
            )
        )
    return records


def load_mbpp_eval_records(
    name: str,
    *,
    split: str,
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    dataset = load_dataset(name, split=split)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        task_id = str(_row_get(row, "task_id", default=idx))
        setup = _coerce_tests(_row_get(row, "test_setup_code", default=[]))
        tests = setup + _coerce_tests(_row_get(row, "test_list", "tests", default=[]))
        text = str(_row_get(row, "text", "prompt", default="")).strip()
        if not text or not tests:
            continue
        entry_point = str(_row_get(row, "entry_point", default="")).strip()
        if not entry_point:
            entry_point = infer_entry_point_from_tests(tests) or ""
        records.append(
            normalize_record(
                {
                    "id": f"mbpp-{task_id}",
                    "source": "mbpp",
                    "task": "code",
                    "prompt": code_prompt(text, tests, entry_point=entry_point),
                    "answer": "",
                    "tests": tests,
                    "entry_point": entry_point,
                    "test_type": "assert",
                    "difficulty": "",
                }
            )
        )
    return records


def load_humaneval_eval_records(
    name: str,
    *,
    split: str,
    max_records: Optional[int],
) -> list[dict[str, Any]]:
    dataset = load_dataset(name, split=split)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(_limit(dataset, max_records)):
        prompt = str(_row_get(row, "prompt", default="")).strip()
        entry_point = str(_row_get(row, "entry_point", default="")).strip()
        test = str(_row_get(row, "test", default="")).strip()
        if not prompt or not entry_point or not test:
            continue
        tests = [test, f"check({entry_point})"]
        records.append(
            normalize_record(
                {
                    "id": str(_row_get(row, "task_id", default=f"humaneval-{idx}")),
                    "source": "humaneval",
                    "task": "code",
                    "prompt": code_prompt(prompt, tests, entry_point=entry_point),
                    "answer": "",
                    "tests": tests,
                    "entry_point": entry_point,
                    "test_type": "assert",
                    "difficulty": "",
                }
            )
        )
    return records


def sample_train_records(
    math_records: list[dict[str, Any]],
    code_records: list[dict[str, Any]],
    *,
    train_size: int,
    math_weight: float,
    code_weight: float,
    code_difficulty_weights: Mapping[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not math_records:
        raise ValueError("No DAPO-Math records are available for training.")
    if not code_records:
        raise ValueError("No TACO records are available for training.")

    math_pool = ShuffledPool(math_records, rng)
    code_by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in code_records:
        difficulty = str(record.get("difficulty") or "UNKNOWN").upper()
        if float(code_difficulty_weights.get(difficulty, 0.0)) > 0.0:
            code_by_difficulty[difficulty].append(record)

    if not code_by_difficulty:
        available = sorted({str(record.get("difficulty") or "UNKNOWN") for record in code_records})
        raise ValueError(
            "No TACO records match code_difficulty_weights. "
            f"Available difficulties: {available}"
        )

    code_pools = {
        difficulty: ShuffledPool(records, rng)
        for difficulty, records in code_by_difficulty.items()
    }
    difficulties = sorted(code_pools)
    difficulty_weights = [float(code_difficulty_weights[difficulty]) for difficulty in difficulties]

    task_names = ["math", "code"]
    task_weights = [float(math_weight), float(code_weight)]
    if sum(task_weights) <= 0.0:
        raise ValueError("math_weight and code_weight cannot both be zero.")

    sampled: list[dict[str, Any]] = []
    for mix_index in range(int(train_size)):
        task = rng.choices(task_names, weights=task_weights, k=1)[0]
        if task == "math":
            record = dict(math_pool.next())
        else:
            difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
            record = dict(code_pools[difficulty].next())
        record["mix_index"] = mix_index
        sampled.append(record)
    return sampled


class ShuffledPool:
    def __init__(self, records: list[dict[str, Any]], rng: random.Random):
        self.records = [dict(record) for record in records]
        self.rng = rng
        self.cursor = 0
        self.rng.shuffle(self.records)

    def next(self) -> dict[str, Any]:
        if self.cursor > 0 and self.cursor % len(self.records) == 0:
            self.rng.shuffle(self.records)
        record = self.records[self.cursor % len(self.records)]
        self.cursor += 1
        return dict(record)


def save_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list([normalize_record(record) for record in records])
    dataset.save_to_disk(str(path))
    print(f"Saved {len(dataset)} records to {path}")


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {column: record.get(column) for column in SCHEMA_COLUMNS}
    normalized["id"] = str(normalized["id"] or "")
    normalized["source"] = str(normalized["source"] or "")
    normalized["task"] = str(normalized["task"] or "")
    normalized["answer"] = str(normalized["answer"] or "")
    normalized["entry_point"] = str(normalized["entry_point"] or "")
    normalized["test_type"] = str(normalized["test_type"] or "")
    normalized["difficulty"] = str(normalized["difficulty"] or "")
    normalized["tests"] = _coerce_tests(normalized["tests"])
    normalized["prompt"] = normalized["prompt"] or []
    return normalized


def parse_weight_mapping(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item {item!r}; expected NAME=VALUE.")
        name, value = item.split("=", 1)
        weights[name.strip().upper()] = float(value)
    if not weights:
        raise ValueError("At least one code difficulty weight is required.")
    return weights


def _extract_dapo_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, Mapping) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
        if prompt and isinstance(prompt[0], Mapping):
            return str(prompt[0].get("content", "")).strip()
    return str(prompt).strip()


def _strip_dapo_instruction(text: str) -> str:
    content = text.strip()
    if "\n\n" in content and content.lower().startswith("solve the following math problem"):
        content = content.split("\n\n", 1)[1].strip()
    marker = "\n\nRemember to put your answer"
    if marker in content:
        content = content.split(marker, 1)[0].strip()
    return content


def _stdin_stdout_tests_from_io(
    inputs: Any,
    outputs: Any,
    *,
    max_tests: Optional[int],
) -> list[str]:
    return [
        json.dumps({"input": str(test_input), "output": str(expected_output)})
        for test_input, expected_output in _paired_io(inputs, outputs, max_tests=max_tests)
    ]


def _function_assert_tests_from_io(
    entry_point: str,
    inputs: Any,
    outputs: Any,
    *,
    max_tests: Optional[int],
) -> list[str]:
    tests: list[str] = []
    for raw_input, raw_output in _paired_io(inputs, outputs, max_tests=max_tests):
        parsed_input = _parse_jsonish(raw_input)
        parsed_output = _parse_jsonish(raw_output)
        if isinstance(parsed_input, tuple):
            args = list(parsed_input)
        elif isinstance(parsed_input, list):
            args = parsed_input
        else:
            args = [parsed_input]
        args_text = ", ".join(repr(arg) for arg in args)
        tests.append(f"assert {entry_point}({args_text}) == {repr(parsed_output)}")
    return tests


def _paired_io(inputs: Any, outputs: Any, *, max_tests: Optional[int]) -> list[tuple[Any, Any]]:
    input_items = inputs if isinstance(inputs, list) else [inputs]
    output_items = outputs if isinstance(outputs, list) else [outputs]
    pairs = list(zip(input_items, output_items))
    if max_tests is not None:
        pairs = pairs[: int(max_tests)]
    return pairs


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


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


if __name__ == "__main__":
    main()
