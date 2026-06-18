# Multi-Task GRPO Baseline

This directory contains the math+code RLVR baseline for SER comparisons. It is
plain GRPO: every sampled completion is fully generated and then verified.

Training data is preprocessed once into a shared schema:

- Math train: `BytedTsinghua-SIA/DAPO-Math-17k`
- Code train: `BAAI/TACO`
- Eval only: GSM8K, MATH-500, MBPP, HumanEval

## Preprocess

Run this first on the server after the Hugging Face datasets have been
downloaded or made available to `load_dataset`.

```bash
python Baseline/preprocess_datasets.py \
  --output_dir Baseline/processed/dapo_taco \
  --train_size 20000 \
  --math_weight 0.5 \
  --code_weight 0.5 \
  --code_difficulty_weights EASY=0.5,MEDIUM=0.5 \
  --taco_max_tests_per_sample 8
```

The script writes:

```text
Baseline/processed/dapo_taco/train
Baseline/processed/dapo_taco/eval/gsm8k
Baseline/processed/dapo_taco/eval/math500
Baseline/processed/dapo_taco/eval/mbpp
Baseline/processed/dapo_taco/eval/humaneval
Baseline/processed/dapo_taco/metadata.json
```

To include harder TACO tasks:

```bash
python Baseline/preprocess_datasets.py \
  --output_dir Baseline/processed/dapo_taco \
  --train_size 50000 \
  --math_weight 0.5 \
  --code_weight 0.5 \
  --code_difficulty_weights EASY=0.3,MEDIUM=0.4,MEDIUM_HARD=0.2,HARD=0.1
```

## Train

```bash
accelerate launch \
  --config_file Baseline/configs/accelerate_deepspeed_zero3_b200.yaml \
  Baseline/train_grpo.py \
  --config Baseline/configs/grpo_math_code_b200.yaml \
  --allow_code_execution
```

The default B200 config targets `Qwen/Qwen3-8B` with Qwen thinking mode enabled.
Override `model_name_or_path` in the YAML or CLI for another checkpoint.

## Local Smoke

```bash
accelerate launch Baseline/train_grpo.py \
  --config Baseline/configs/grpo_math_code_smoke.yaml \
  --allow_code_execution
```

The smoke config uses synthetic math/code rows and does not require the
preprocessed dataset.

## Evaluate

```bash
python Baseline/evaluate.py \
  --config Baseline/configs/grpo_math_code_b200.yaml \
  --checkpoint Baseline/outputs/grpo_dapo_taco_qwen3_8b \
  --allow_code_execution \
  --output Baseline/outputs/eval_results.json
```

Code rewards and code evaluation execute model-generated Python in a temporary
subprocess with a timeout. Use `--allow_code_execution` only in a trusted
environment.
