# Multi-Task GRPO Baseline

This directory contains the math+code RLVR baseline for SER comparisons. It is
intentionally plain GRPO: every sampled completion is fully generated and every
completion is verified by the task-specific verifier.

## Train

```bash
accelerate launch Baseline/train_grpo.py \
  --config Baseline/configs/grpo_math_code_b200.yaml \
  --allow_code_execution
```

The default B200 config targets `Qwen/Qwen3-8B` with Qwen thinking mode enabled.
Override `model_name_or_path` in the YAML or CLI for a later Qwen3.5 8B server
checkpoint.

## Local Smoke

```bash
accelerate launch Baseline/train_grpo.py \
  --config Baseline/configs/grpo_math_code_smoke.yaml \
  --allow_code_execution
```

The smoke config uses synthetic math/code rows so it can validate the pipeline
without downloading GSM8K, MBPP, MATH-500, or HumanEval.

## Evaluate

```bash
python Baseline/evaluate.py \
  --config Baseline/configs/grpo_math_code_b200.yaml \
  --checkpoint Baseline/outputs/grpo_math_code_qwen3_8b \
  --allow_code_execution \
  --output Baseline/outputs/eval_results.json
```

Code rewards and code evaluation execute model-generated Python in a temporary
subprocess with a timeout. Use `--allow_code_execution` only in a trusted
environment.

