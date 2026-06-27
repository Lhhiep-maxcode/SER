# SER Method

This folder contains a first implementation of Speculative Environment Rollouts
for the existing math/code RLVR setup.

Implemented components:

- Separate math and code environments.
- vLLM/OpenAI-compatible trajectory critic for speculative early accept/reject.
- Environment-aware scheduling using utility/cost ratios.
- Environment-specific updates as soon as a selected environment finishes,
  without waiting for the other environment to complete a synchronized batch.
- Same processed data schema, LoRA setup, rewards, and verifiers as
  `Baseline_GRPO`.

## Split Environments

Run this once after building the DAPO/TACO processed dataset:

```bash
python SER-method/split_environments.py \
  --input_path Baseline_GRPO/processed/dapo_taco/train \
  --output_dir SER-method/processed/dapo_taco \
  --shuffle \
  --overwrite
```

This writes:

```text
SER-method/processed/dapo_taco/math
SER-method/processed/dapo_taco/code
SER-method/processed/dapo_taco/metadata.json
```

## Critic Server

Start your Qwen3-235B critic with a vLLM OpenAI-compatible server, for example:

```bash
vllm serve /workspace/storage-shared/models/Qwen3-235B-A22B \
  --host 0.0.0.0 \
  --port 8000
```

Then set the same endpoint/model in:

```yaml
critic:
  enabled: true
  base_url: http://127.0.0.1:8000
  model: /workspace/storage-shared/models/Qwen3-235B-A22B
```

The critic receives only the task prompt and partial assistant trajectory. It is
asked to return:

```json
{"success_probability": 0.73}
```

## Train

```bash
python SER-method/train_ser.py \
  --config SER-method/configs/ser_qwen3_8b_math_code.yaml \
  --allow_code_execution
```

`--allow_code_execution` is needed when unresolved code trajectories reach full
verification. Early accepted/rejected code trajectories skip verifier execution.

## Thresholds

The default code acceptance threshold is intentionally high:

```yaml
thresholds:
  math:
    accept: 0.9
    reject: 0.1
  code:
    accept: 0.98
    reject: 0.05
```

Use a stricter code `accept` threshold because code can fail from formatting,
stdin/stdout, function signatures, or hidden tests even when the reasoning looks
promising.

## Budget Allocation

The scheduler maintains per-environment moving reward and moving rollout cost.
It uses:

```text
R_e = max(delta_reward, utility_floor) / max(cost_seconds, cost_floor)
```

and samples the next environment according to normalized `R_e`, with a minimum
probability floor so neither math nor code starves.

Logged metrics include:

```text
budget/math_probability
budget/code_probability
budget/*_moving_reward
budget/*_moving_cost_seconds
early_accepts
early_rejects
verification_fraction
rollout_fraction
critic_calls
critic_errors
```

## TensorBoard

```bash
tensorboard \
  --logdir SER-method/outputs/ser_qwen3_8b_math_code/tensorboard \
  --port 6006
```

