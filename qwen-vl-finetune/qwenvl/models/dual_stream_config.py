"""Configuration dataclass for the dual-stream visual encoder stack."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class DualStreamConfig:
    """Holds hyper-parameters for the dual-stream vision encoder and VE-Gate.

    Attributes:
        dinov3_modelscope_model_id: Optional ModelScope model or directory that
            should be loaded via :func:`modelscope.AutoModel.from_pretrained`.
            When provided the ModelScope implementation is preferred over
            Torch Hub.
        dinov3_modelscope_revision: Optional revision identifier passed to
            ModelScope when ``dinov3_modelscope_model_id`` references a remote
            model.
        dinov3_modelscope_device_map: Optional device map forwarded to
            :func:`AutoModel.from_pretrained` so the ModelScope model can be
            loaded on a specific device (e.g. ``"auto"``).
        dinov3_modelscope_trust_remote_code: Whether to enable
            ``trust_remote_code`` when initialising the ModelScope checkpoint.
        dinov3_repo_or_dir: Torch Hub repository or local directory hosting the
            DINOv3 implementation. Defaults to the official
            ``facebookresearch/dinov3`` repo. Only used when the ModelScope
            configuration is absent or fails.
        dinov3_model_name: Name of the high-fidelity vision backbone that will
            be instantiated via :func:`torch.hub.load` as a fallback.
        dinov3_checkpoint_path: Optional checkpoint file or directory containing
            weights for the DINOv3 backbone. When ``None`` the pretrained weights
            bundled with ``dinov3_repo_or_dir`` will be downloaded automatically.
        dinov3_image_size: Input spatial resolution expected by the DINOv3
            backbone. Input images will be resized to this square resolution
            before being fed to the high-fidelity encoder.
        freeze_semantic_stream: If ``True`` the original Qwen2.5-VL vision tower
            is frozen during training.
        freeze_high_fidelity_stream: If ``True`` the DINOv3 branch is kept
            frozen during training.
        fusion_hidden_ratio: Multiplicative factor controlling the hidden size
            of the fusion MLP. A value of ``2.0`` doubles the hidden size.
        fusion_dropout: Dropout probability applied inside the fusion MLP.
        ve_gate_hidden_ratio: Reduction ratio used by the VE-Gate MLP. Larger
            values make the gating network smaller.
        ve_gate_dropout: Dropout probability inside the VE-Gate MLP.
        ve_gate_temperature: Initial temperature used to sharpen or smooth the
            gating sigmoid. The parameter is learned during training but can be
            initialised here.
    """

    dinov3_modelscope_model_id: str | None = None
    dinov3_modelscope_revision: str | None = None
    dinov3_modelscope_device_map: str | None = None
    dinov3_modelscope_trust_remote_code: bool = True
    dinov3_repo_or_dir: str = "facebookresearch/dinov3"
    dinov3_model_name: str = "dinov3_vith16plus"
    dinov3_checkpoint_path: str | None = None
    dinov3_image_size: int = 518
    freeze_semantic_stream: bool = False
    freeze_high_fidelity_stream: bool = True
    fusion_hidden_ratio: float = 2.0
    fusion_dropout: float = 0.1
    ve_gate_hidden_ratio: int = 4
    ve_gate_dropout: float = 0.1
    ve_gate_temperature: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration into a serialisable dictionary."""

        return asdict(self)


__all__ = ["DualStreamConfig"]
