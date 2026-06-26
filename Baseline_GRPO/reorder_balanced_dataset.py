"""Reorder a processed math/code dataset for balanced no-shuffle training.

The output is another Hugging Face ``save_to_disk`` dataset. Use it with
``shuffle_dataset: false`` in ``Baseline_GRPO/grpo_baseline.py`` so contiguous
DataLoader batches contain the intended math/code mix.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Baseline_GRPO.data_utils import load_processed_dataset  # noqa: E402


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset = load_processed_dataset(args.input_path)
    records = list(dataset)
    math_records = [record for record in records if record.get("task") == "math"]
    code_records = [record for record in records if record.get("task") == "code"]

    if not math_records or not code_records:
        counts = Counter(str(record.get("task")) for record in records)
        raise ValueError(f"Need both math and code records. Found task counts: {dict(counts)}")

    rng.shuffle(math_records)
    rng.shuffle(code_records)

    ordered = build_balanced_order(
        math_records,
        code_records,
        batch_size=args.batch_size,
        drop_unbalanced_tail=args.drop_unbalanced_tail,
    )
    output = Dataset.from_list(ordered)

    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    output.save_to_disk(str(output_path))
    write_metadata(output_path, args, records, output)

    print(f"Input records: {len(records)}")
    print(f"Input task counts: {dict(Counter(str(record.get('task')) for record in records))}")
    print(f"Output records: {len(output)}")
    print(f"Output task counts: {dict(Counter(str(record.get('task')) for record in output))}")
    print(f"Saved balanced dataset to: {output_path}")


def build_balanced_order(
    math_records: list[dict[str, Any]],
    code_records: list[dict[str, Any]],
    *,
    batch_size: int,
    drop_unbalanced_tail: bool,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    math_pos = 0
    code_pos = 0
    ordered: list[dict[str, Any]] = []
    batch_index = 0

    while math_pos < len(math_records) and code_pos < len(code_records):
        math_needed, code_needed = task_counts_for_batch(batch_size, batch_index)
        if math_pos + math_needed > len(math_records) or code_pos + code_needed > len(code_records):
            break

        batch = []
        for _ in range(math_needed):
            batch.append(math_records[math_pos])
            math_pos += 1
        for _ in range(code_needed):
            batch.append(code_records[code_pos])
            code_pos += 1

        # Interleave within each batch so prompts alternate task as much as possible.
        ordered.extend(interleave_batch(batch, math_needed, code_needed))
        batch_index += 1

    if not drop_unbalanced_tail:
        ordered.extend(math_records[math_pos:])
        ordered.extend(code_records[code_pos:])

    return ordered


def task_counts_for_batch(batch_size: int, batch_index: int) -> tuple[int, int]:
    if batch_size == 1:
        return (1, 0) if batch_index % 2 == 0 else (0, 1)

    math_count = batch_size // 2
    code_count = batch_size - math_count
    if batch_size % 2 == 1 and batch_index % 2 == 1:
        math_count, code_count = code_count, math_count
    return math_count, code_count


def interleave_batch(batch: list[dict[str, Any]], math_count: int, code_count: int) -> list[dict[str, Any]]:
    math_items = batch[:math_count]
    code_items = batch[math_count : math_count + code_count]
    result = []
    while math_items or code_items:
        if math_items:
            result.append(math_items.pop(0))
        if code_items:
            result.append(code_items.pop(0))
    return result


def write_metadata(output_path: Path, args: argparse.Namespace, input_records, output: Dataset) -> None:
    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "drop_unbalanced_tail": args.drop_unbalanced_tail,
        "input_count": len(input_records),
        "output_count": len(output),
        "input_task_counts": dict(Counter(str(record.get("task")) for record in input_records)),
        "output_task_counts": dict(Counter(str(record.get("task")) for record in output)),
    }
    (output_path / "balance_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_unbalanced_tail", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
