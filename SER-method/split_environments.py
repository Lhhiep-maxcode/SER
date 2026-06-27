"""Split a processed math/code RLVR dataset into separate environments."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Baseline_GRPO.data_utils import load_processed_dataset  # noqa: E402


def main() -> None:
    args = parse_args()
    dataset = load_processed_dataset(args.input_path, max_samples=args.max_samples)
    records = list(dataset)
    rng = random.Random(args.seed)

    math_records = [record for record in records if record.get("task") == "math"]
    code_records = [record for record in records if record.get("task") == "code"]
    if args.shuffle:
        rng.shuffle(math_records)
        rng.shuffle(code_records)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_environment(math_records, output_dir / "math")
    save_environment(code_records, output_dir / "code")

    metadata = {
        "input_path": str(args.input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "shuffle": args.shuffle,
        "max_samples": args.max_samples,
        "counts": dict(Counter(str(record.get("task")) for record in records)),
        "math_count": len(math_records),
        "code_count": len(code_records),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved math environment: {output_dir / 'math'} ({len(math_records)} records)")
    print(f"Saved code environment: {output_dir / 'code'} ({len(code_records)} records)")


def save_environment(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        raise ValueError(f"No records to save for {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
