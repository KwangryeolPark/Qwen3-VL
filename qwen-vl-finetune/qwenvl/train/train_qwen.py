# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import math
import os
import pathlib
import shutil
import sys
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dumps it to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _hardlink_or_copy(src: Path, dst: Path):
    """Hard-link a file when possible, falling back to a normal copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class EvalSnapshotCallback(TrainerCallback):
    """Trigger milestone saves and preserve model-only evaluation snapshots.

    Saves are triggered at 25/50/75/100% of the actual Trainer max_steps.  Trainer
    writes a full resumable DeepSpeed checkpoint, while this callback hard-links
    only the reloadable HF model/tokenizer/config files into a persistent snapshot
    directory.  With save_total_limit=1, only the latest large resume checkpoint is
    retained, while all four lightweight evaluation snapshots remain available.
    """

    _EXCLUDED_FILES = {
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "trainer_state.json",
        "training_args.bin",
    }

    def __init__(self, processor, snapshot_root: str, fractions=(0.25, 0.50, 0.75, 1.0)):
        self.processor = processor
        self.snapshot_root = Path(snapshot_root)
        self.fractions = tuple(fractions)
        self._targets = None

    def _target_steps(self, state):
        if self._targets is None:
            self._targets = {
                min(state.max_steps, max(1, math.ceil(state.max_steps * f)))
                for f in self.fractions
            }
        return self._targets

    def on_train_begin(self, args, state, control, **kwargs):
        targets = sorted(self._target_steps(state))
        if args.should_save:
            print(f"Evaluation milestone steps: {targets}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self._target_steps(state):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save or state.global_step not in self._target_steps(state):
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            logging.warning("Snapshot source checkpoint does not exist: %s", checkpoint_dir)
            return control

        epoch = float(state.epoch) if state.epoch is not None else -1.0
        snapshot_dir = self.snapshot_root / f"epoch-{epoch:.2f}-step-{state.global_step}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        copied = []
        for src in checkpoint_dir.iterdir():
            if not src.is_file() or src.name in self._EXCLUDED_FILES:
                continue
            if src.name.startswith("rng_state"):
                continue
            _hardlink_or_copy(src, snapshot_dir / src.name)
            copied.append(src.name)

        self.processor.save_pretrained(snapshot_dir)

        metadata = {
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "progress_fraction": float(state.global_step / state.max_steps),
            "epoch": epoch,
            "source_checkpoint": str(checkpoint_dir),
            "copied_files": sorted(copied),
            "optimizer_state_included": False,
        }
        with open(snapshot_dir / "snapshot_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved evaluation snapshot: {snapshot_dir}")
        return control


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")
        for p in model.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)
        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if training_args.save_eval_snapshots:
        snapshot_root = training_args.eval_snapshot_dir or str(Path(training_args.output_dir) / "eval_snapshots")
        trainer.add_callback(EvalSnapshotCallback(processor, snapshot_root))
        rank0_print(f"Evaluation snapshots enabled: {snapshot_root}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if training_args.skip_final_save:
        rank0_print("Skipping trainer state/model/processor save for benchmark run.")
        return

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
