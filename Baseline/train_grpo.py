"""Config-driven TRL GRPO training entrypoint for the math+code baseline."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import yaml

try:
    from .data import build_eval_datasets, build_train_dataset
    from .rewards import RewardStats, build_reward_functions
except ImportError:  # pragma: no cover - used by `accelerate launch Baseline/train_grpo.py`.
    from data import build_eval_datasets, build_train_dataset
    from rewards import RewardStats, build_reward_functions


BASELINE_ONLY_CONFIG_KEYS = {
    "allow_code_execution",
    "datasets",
    "enable_thinking",
    "eval_max_new_tokens",
    "eval_batch_size",
    "max_train_samples",
    "model_name_or_path",
    "source_limits",
    "task_weights",
    "use_synthetic_data",
    "code_timeout_seconds",
}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    allow_code_execution = bool(config.get("allow_code_execution", False)) or args.allow_code_execution
    config["allow_code_execution"] = allow_code_execution

    if _uses_code_task(config) and not allow_code_execution:
        raise SystemExit(
            "This baseline includes code RLVR rewards. Re-run with --allow_code_execution "
            "or set allow_code_execution: true in the config."
        )

    train_dataset = build_train_dataset(config)
    eval_datasets = build_eval_datasets(config) if eval_is_enabled(config) else None
    print(f"Built train dataset with {len(train_dataset)} mixed samples.")
    if eval_datasets:
        print("Built eval datasets:", {name: len(ds) for name, ds in eval_datasets.items()})

    if args.dry_run:
        print("Dry run requested; skipping TRL imports and training.")
        return

    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover - depends on server training env.
        raise SystemExit(
            "TRL is required for training. Install dependencies from Baseline/requirements.txt "
            "or run on the prepared B200 server."
        ) from exc

    stats = RewardStats()
    reward_funcs, stats = build_reward_functions(
        allow_code_execution=allow_code_execution,
        code_timeout_seconds=float(config.get("code_timeout_seconds", 5.0)),
        stats=stats,
    )
    training_args = build_grpo_config(GRPOConfig, config)

    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
    )
    add_log_sanitizer_callback(trainer)
    add_stats_callback(trainer, stats)

    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(config.get("output_dir"))
    print("Final verifier stats:", stats.log_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow reward functions to execute model-generated Python code.",
    )
    parser.add_argument("--model_name_or_path", help="Override the configured model path.")
    parser.add_argument("--output_dir", help="Override the configured output directory.")
    parser.add_argument("--max_steps", type=int, help="Override max_steps.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build datasets and validate config without importing TRL or training.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "model_name_or_path" not in config:
        raise ValueError("Config must define model_name_or_path.")
    if "output_dir" not in config:
        raise ValueError("Config must define output_dir.")
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("model_name_or_path", "output_dir", "max_steps"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value


def build_grpo_config(grpo_config_cls, config: dict[str, Any]):
    valid_keys = set(inspect.signature(grpo_config_cls).parameters)
    kwargs: dict[str, Any] = {}
    ignored: list[str] = []

    for key, value in config.items():
        if key in BASELINE_ONLY_CONFIG_KEYS:
            continue
        if key in valid_keys:
            kwargs[key] = value
        else:
            ignored.append(key)

    if "chat_template_kwargs" in valid_keys and "chat_template_kwargs" not in kwargs:
        kwargs["chat_template_kwargs"] = {
            "enable_thinking": bool(config.get("enable_thinking", False))
        }

    if ignored:
        print("Ignoring config keys not accepted by this TRL GRPOConfig:", sorted(ignored))

    return grpo_config_cls(**kwargs)


def add_stats_callback(trainer, stats: RewardStats) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
        from transformers import TrainerCallback
    except Exception:
        return

    class VerifierStatsCallback(TrainerCallback):
        def __init__(self):
            self.writer = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.writer = SummaryWriter(log_dir=args.logging_dir)

        def on_log(self, args, state, control, logs=None, **kwargs):
            stat_logs = stats.log_dict()
            if logs is not None:
                logs.update(stat_logs)
            if self.writer is not None:
                for key, value in stat_logs.items():
                    self.writer.add_scalar(key, value, state.global_step)
                self.writer.flush()

        def on_train_end(self, args, state, control, **kwargs):
            if self.writer is not None:
                self.writer.close()

    trainer.add_callback(VerifierStatsCallback())


def add_log_sanitizer_callback(trainer) -> None:
    try:
        from transformers import TrainerCallback
    except Exception:
        return

    class DropNoneLogsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            for key in list(logs):
                if logs[key] is None:
                    del logs[key]

    # Put this before TensorBoardCallback so non-scalar None values are removed
    # before Hugging Face tries to write them as scalars.
    trainer.callback_handler.callbacks.insert(0, DropNoneLogsCallback())


def _uses_code_task(config: dict[str, Any]) -> bool:
    return float(config.get("task_weights", {}).get("code", 0.0)) > 0.0


def eval_is_enabled(config: dict[str, Any]) -> bool:
    value = config.get("eval_strategy", "no")
    return str(value).lower() not in {"no", "false", "none", "0"}


if __name__ == "__main__":
    main()
