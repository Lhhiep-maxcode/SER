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

Preferred config-driven launch:

```bash
python Baseline_GRPO/grpo_baseline.py \
  --config Baseline_GRPO/configs/grpo_dapo_taco_qwen3_8b.yaml
```

Command-line arguments override YAML values, for example:

```bash
python Baseline_GRPO/grpo_baseline.py \
  --config Baseline_GRPO/configs/grpo_dapo_taco_qwen3_8b.yaml \
  --max_steps 1000 \
  --tensorboard_log_dir Baseline_GRPO/outputs/debug_tensorboard
```

Use `CUDA_VISIBLE_DEVICES=0` if you want the same single-GPU launch style as
the current FastGRPO script.

The provided config enables code execution because code rewards run generated
Python tests. Use it only in a trusted training environment.

## TensorBoard

```bash
tensorboard \
  --logdir Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b/tensorboard \
  --port 6006
```

The JSONL training log is still written to `log_file`. TensorBoard receives the
same numeric training metrics under `train/*`, including reward, loss, KL,
timing, sequence counts, verifier calls, and domain accuracies.

## Resume

```bash
python Baseline_GRPO/grpo_baseline.py \
  --config Baseline_GRPO/configs/grpo_dapo_taco_qwen3_8b.yaml \
  --resume_from_checkpoint Baseline_GRPO/outputs/grpo_dapo_taco_qwen3_8b/step500 \
  --max_steps 2000
```
