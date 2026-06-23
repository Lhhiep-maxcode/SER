# Baseline GRPO From Scratch

This folder implements a non-speculative GRPO baseline without TRL. It is meant
to be compared directly with `FastGRPO/grpo_speculative.py`.

The implementation mirrors the FastGRPO training style:

- PEFT LoRA target training.
- Standard autoregressive `model.generate`, no speculative decoding.
- `repeated_generate_nums` completions per prompt.
- Rewards normalized within each prompt group.
- Zero-variance reward groups skipped.
- Clipped GRPO objective with KL against the base model by disabling adapters.
- Same processed data produced by `Baseline/preprocess_datasets.py`.

## Run

```bash
python Baseline_GRPO/grpo_baseline.py \
  --model_dir Qwen/Qwen3-8B \
  --dataset_path Baseline/processed/dapo_taco/train \
  --output_dir Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b \
  --log_file Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b/train.jsonl \
  --batch_size 1 \
  --num_epochs 1 \
  --accumulation_steps 8 \
  --target_lr 1e-6 \
  --repeated_generate_nums 8 \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 2048 \
  --max_prompt_length 1024 \
  --max_training_token 3072 \
  --max_training_padding_gap 256 \
  --beta 0.01 \
  --epsilon 0.1 \
  --enable_thinking \
  --allow_code_execution
```

Use `CUDA_VISIBLE_DEVICES=0` if you want the same single-GPU launch style as
the current FastGRPO script.

## Resume

```bash
python Baseline_GRPO/grpo_baseline.py \
  --model_dir Qwen/Qwen3-8B \
  --dataset_path Baseline/processed/dapo_taco/train \
  --resume_from_checkpoint Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b/step500 \
  --output_dir Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b \
  --log_file Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b/train.jsonl \
  --allow_code_execution
```

