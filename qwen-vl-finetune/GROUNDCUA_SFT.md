# GroundCUA SFT for Qwen3-VL-8B-Instruct

This fork adds a GroundCUA supervised fine-tuning path intended to precede the
EasyR1/GRPO stage in `Qwen3-VL-GUI-Specialization`.

## Pipeline

```text
GroundCUA prepared EasyR1 parquet
        ↓
prepare_groundcua_sft.py
        ↓
Qwen3-VL conversation JSONL
        ↓
qwen-vl-finetune full-parameter SFT
        ↓
Hugging Face checkpoint
        ↓
EasyR1 GroundCUA RL/GRPO
```

The SFT target deliberately uses the same output protocol as the RL reward:

```xml
<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[x,y]}}</tool_call>
```

Coordinates are normalized to `[0,1000] x [0,1000]` and use the target bbox
center.

## 1. Prepare JSONL

From `qwen-vl-finetune/`:

```bash
python tools/prepare_groundcua_sft.py \
  --input /home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/easyr1/train.parquet \
  --output /home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/sft/qwen3vl_train.jsonl \
  --strict
```

Create a small smoke-test dataset:

```bash
python tools/prepare_groundcua_sft.py \
  --input /home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/easyr1/train.parquet \
  --output /home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/sft/qwen3vl_train_smoke.jsonl \
  --limit 1000 \
  --strict
```

Each conversion also writes `<output>.manifest.json`.

## 2. Smoke SFT

The smoke run performs 100 full-parameter optimizer steps on 4 GPUs using
DeepSpeed ZeRO-3.

```bash
bash scripts/sft_qwen3_8b_groundcua_smoke.sh
```

Check:

- loss decreases / stays finite
- no OOM
- sec/step and GPU utilization
- checkpoint can be loaded with Qwen3-VL
- checkpoint can be passed to EasyR1 as `worker.actor.model.model_path`

## 3. Full SFT

```bash
bash scripts/sft_qwen3_8b_groundcua_full.sh
```

Default recipe:

- Qwen3-VL-8B-Instruct
- full parameter: vision + multimodal merger + language model
- DeepSpeed ZeRO-3
- bf16
- FlashAttention-2 (trainer default)
- 2 epochs
- learning rate `3e-6`
- cosine schedule
- warmup ratio `0.05`
- global batch 128 on 4 GPUs: `1 x 4 x grad_accum(32)`
- gradient checkpointing enabled
- image pixel range `262144..2359296`

The LR, epochs and effective global batch follow the GroundCUA/GroundNext SFT
recipe as closely as practical. The default pixel budget is intentionally
aligned with the later EasyR1 GroundCUA GRPO stage rather than GroundCUA's
larger original SFT `image_max_pixels=12845056`. To test the original upper
budget, override it explicitly:

```bash
MAX_PIXELS=12845056 bash scripts/sft_qwen3_8b_groundcua_full.sh
```

Do not start the multi-day full run before the 100-step smoke run is verified.
