"""
patched_arguments.py — DUAL-TRACK modified copy of  src/stage2/arguments.py
============================================================================
Adds cot_w / ans_w to ModelArguments (the visible-CoT and answer CE weights).
Everything else is byte-for-byte the original.
"""

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    latent_model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier for latent model"}
    )
    ce_w: float = field(default=1.0)          # legacy (unused by dual-track forward)
    kl_w: float = field(default=1.0)
    cot_w: float = field(default=1.0, metadata={"help": "DUAL-TRACK: visible-CoT CE weight"})
    ans_w: float = field(default=1.0, metadata={"help": "DUAL-TRACK: answer CE weight"})
    bfloat16: bool = field(default=True)
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "DUAL-TRACK: keep False — SDPA is required for the 4-D bottleneck mask"},
    )
    topk_interpolation: int = field(
        default=5, metadata={"help": "The k value for topk interpolation"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "Path to train data"})
    train_latent_soft_label_path: str = field(metadata={"help": "Path to train latent state chunks"})
    add_gumbel_noise: bool = field(default=False, metadata={"help": "Add Gumbel noise to soft labels"})
    gumbel_temperature: float = field(default=1.0, metadata={"help": "Temperature for Gumbel-softmax"})
    noise_scale: float = field(default=1.0, metadata={"help": "Scale factor for Gumbel noise"})


@dataclass
class Stage2TrainingArguments(TrainingArguments):
    lora_tune: bool = field(default=True, metadata={"help": "Whether to use lora"})
    lora_path: str = field(default=None, metadata={"help": "Lora path"})
    lora_rank: int = field(default=32, metadata={"help": "Lora rank"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Lora dropout"})
    training: bool = field(default=True, metadata={"help": "Whether to training"})
