# SER Method

This folder contains a first implementation of Speculative Environment Rollouts
for the existing math/code RLVR setup.

Implemented components:

- Separate math and code environments.
- vLLM/OpenAI-compatible trajectory critic for speculative early accept/reject.
- Environment-aware batch allocation using utility/cost ratios.
- Every SER iteration can include all environments, with integer sample counts
  such as math/code `4/4`, `5/3`, or `6/2`.
- Mixed-environment rollout generation: active trajectories from all allocated
  environments share the same generation micro-batches.
- Concurrent critic requests to a separately hosted vLLM critic.
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
  concurrency: 16
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

`rollout_generation_batch_size` controls how many active trajectories are passed
to `model.generate` at once. This pool is mixed across environments, so if one
iteration allocates math/code `5/3` with `repeated_generate_nums: 8`, the active
rollout pool contains `64` trajectories and generation chunks are taken from
that shared pool.

## EAGLE Speculative Generation

EAGLE rollout acceleration is optional and disabled by default. To enable it:

```yaml
speculative:
  enabled: true
  train_draft: true
  draft_adapter_path: ""
  allow_scratch_draft: true
  draft_warmup_steps: 256
  draft_warmup_accumulation_steps: 16
  draft_warmup_max_length: 4096
  draft_warmup_generate_target_responses: true
  draft_warmup_generate_missing_answers: true
  draft_warmup_teacher_max_new_tokens: 512
  draft_warmup_include_prompt_only: false
  draft_train_from_target_hidden: true
```

If `draft_adapter_path` is empty, the draft model starts from scratch and is
first aligned to the target with a draft warmup pass over the processed SER
datasets, then trained online from rollout traces. Checkpoints save
`speculative.pt` beside the target adapter checkpoint, including the draft model
and draft optimizer state. By default, warmup asks the target model to generate
assistant responses for the processed prompts, then computes target hidden
states on the resulting prompt+response sequences. Online draft training also
recomputes target hidden states on accepted rollout sequences by default.
Prompt-only draft warmup is disabled unless `draft_warmup_include_prompt_only`
is set to `true`.
Speculative metrics are logged under `speculative/*` in JSONL and TensorBoard.

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

The allocator maintains per-environment moving reward and moving rollout cost.
For every SER iteration, it converts environment probabilities into integer
sample counts for the total `batch_size`.

With:

```yaml
batch_size: 8
ensure_all_envs_per_step: true
```

the initial equal allocation is math/code `4/4`. If math receives a larger
budget probability, the next mixed batches can become `5/3`, then `6/2`.

By default, the utility uses moving reward:

```text
R_e = max(moving_reward, utility_floor) / max(cost_seconds, cost_floor)
```

If you want the older gain-style rule, set:

```yaml
budget:
  utility_mode: gain
```

which uses:

```text
R_e = max(delta_reward, utility_floor) / max(cost_seconds, cost_floor)
```

The final integer allocation also uses a probability floor and, when
`ensure_all_envs_per_step` is true, at least one sample for each environment.

Logged metrics include:

```text
budget/math_probability
budget/code_probability
allocation/math
allocation/code
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
