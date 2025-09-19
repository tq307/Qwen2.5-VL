"""Training entry-point for the dual-stream + VE-Gate Qwen2.5-VL model."""

import logging
import os
import pathlib
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
import sys

sys.path.append(str(project_root))

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.models import DualStreamConfig, Qwen2_5_VLDualStreamForConditionalGeneration
from qwenvl.train.argument import (
    DataArguments,
    DualStreamArguments,
    ModelArguments,
    TrainingArguments,
)
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def build_dual_stream_config(args: DualStreamArguments) -> DualStreamConfig:
    return DualStreamConfig(
        dinov3_modelscope_model_id=args.dinov3_modelscope_model_id,
        dinov3_modelscope_revision=args.dinov3_modelscope_revision,
        dinov3_modelscope_device_map=args.dinov3_modelscope_device_map,
        dinov3_modelscope_trust_remote_code=args.dinov3_modelscope_trust_remote_code,
        dinov3_repo_or_dir=args.dinov3_repo_or_dir,
        dinov3_model_name=args.dinov3_model_name,
        dinov3_checkpoint_path=args.dinov3_checkpoint_path,
        dinov3_image_size=args.dinov3_image_size,
        freeze_semantic_stream=args.freeze_semantic_stream,
        freeze_high_fidelity_stream=args.freeze_high_fidelity_stream,
        fusion_hidden_ratio=args.fusion_hidden_ratio,
        fusion_dropout=args.fusion_dropout,
        ve_gate_hidden_ratio=args.ve_gate_hidden_ratio,
        ve_gate_dropout=args.ve_gate_dropout,
        ve_gate_temperature=args.ve_gate_temperature,
    )


def set_trainable_params(model_args: ModelArguments, model: Qwen2_5_VLDualStreamForConditionalGeneration):
    dual = model.model.dual_vision

    if model_args.tune_mm_vision:
        for param in dual.semantic_encoder.parameters():
            param.requires_grad = True
        for param in dual.high_fidelity_encoder.parameters():
            param.requires_grad = True
    else:
        for param in dual.semantic_encoder.parameters():
            param.requires_grad = False
        for param in dual.high_fidelity_encoder.parameters():
            param.requires_grad = False

    if model_args.tune_mm_mlp:
        for param in dual.high_proj.parameters():
            param.requires_grad = True
        for param in dual.fusion_mlp.parameters():
            param.requires_grad = True
        model.model.ve_gate.requires_grad_(True)
    else:
        for param in dual.high_proj.parameters():
            param.requires_grad = False
        for param in dual.fusion_mlp.parameters():
            param.requires_grad = False
        model.model.ve_gate.requires_grad_(False)

    if model_args.tune_mm_llm:
        for param in model.model.language_model.parameters():
            param.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for param in model.model.language_model.parameters():
            param.requires_grad = False
        model.lm_head.requires_grad = False


def log_trainable_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")


def train(attn_implementation: str = "flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DualStreamArguments)
    )
    model_args, data_args, training_args, dual_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    dual_config = build_dual_stream_config(dual_args)

    model = Qwen2_5_VLDualStreamForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        dual_stream_config=dual_config,
    )
    processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_trainable_params(model_args, model)
    log_trainable_parameters(model)

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        **data_module,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
