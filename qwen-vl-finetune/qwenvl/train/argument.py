import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class DualStreamArguments:
    freeze_high_fidelity_stream: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the high-fidelity (DINO) stream when using the dual-stream "
                "vision encoder. Defaults to False so the DINO branch is unfrozen and can adapt "
                "to industrial scenarios. Set to true to keep the branch frozen."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


@dataclass
class DualStreamArguments:
    dinov3_modelscope_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "ModelScope model identifier or local directory passed to AutoModel.from_pretrained "
                "to initialise the DINOv3 branch. When set it takes precedence over the torch.hub configuration."
            )
        },
    )
    dinov3_modelscope_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional revision to load from ModelScope when using a remote checkpoint.",
        },
    )
    dinov3_modelscope_device_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device map forwarded to ModelScope AutoModel for loading the DINOv3 encoder (e.g. 'auto').",
        },
    )
    dinov3_modelscope_trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable trust_remote_code when downloading the DINOv3 checkpoint from ModelScope.",
        },
    )
    dinov3_repo_or_dir: str = field(
        default="facebookresearch/dinov3",
        metadata={
            "help": "Torch Hub repository (or local directory) that provides the DINOv3 implementation."
        },
    )
    dinov3_model_name: str = field(
        default="dinov3_vith16plus",
        metadata={
            "help": "Backbone name resolved by torch.hub.load (e.g. 'dinov3_vith16plus')."
        },
    )
    dinov3_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional checkpoint path for the DINOv3 backbone. Can be a file "
                "or directory containing weights such as 'model.safetensors' or "
                "'pytorch_model.bin'."
            )
        },
    )
    dinov3_image_size: int = field(
        default=518,
        metadata={"help": "Square input resolution for the DINOv3 backbone."},
    )
    freeze_semantic_stream: bool = field(
        default=False,
        metadata={"help": "Freeze the original Qwen2.5-VL vision tower."},
    )
    freeze_high_fidelity_stream: bool = field(
        default=True,
        metadata={"help": "Freeze the DINOv3 branch during finetuning."},
    )
    fusion_hidden_ratio: float = field(
        default=2.0,
        metadata={"help": "Hidden size multiplier used inside the fusion MLP."},
    )
    fusion_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability applied within the fusion MLP."},
    )
    ve_gate_hidden_ratio: int = field(
        default=4,
        metadata={"help": "Reduction ratio used by the VE-Gate MLP."},
    )
    ve_gate_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability inside the VE-Gate."},
    )
    ve_gate_temperature: float = field(
        default=1.0,
        metadata={"help": "Initial temperature scaling for the VE-Gate sigmoid."},
    )
