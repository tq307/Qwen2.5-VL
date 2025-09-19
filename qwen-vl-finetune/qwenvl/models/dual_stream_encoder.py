"""Dual-stream visual encoder that augments Qwen2.5-VL with a DINOv3 branch."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:  # pragma: no cover - optional dependency during linting
    import timm
except ImportError as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TIMM_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency during linting
    from modelscope import AutoModel
except ImportError as exc:  # pragma: no cover
    AutoModel = None
    _MODELSCOPE_IMPORT_ERROR = exc
else:  # pragma: no cover
    _MODELSCOPE_IMPORT_ERROR = None

from .dual_stream_config import DualStreamConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class DualStreamVisionOutput:
    """Container returned by :class:`DualStreamVisionEncoder`.

    Attributes:
        semantic: List of tensors produced by the original Qwen2.5-VL vision
            tower. Each tensor corresponds to a different image placeholder and
            has shape ``(num_tokens, hidden_size)``.
        high_fidelity: List of tensors containing the projected DINOv3 features
            that are aligned with the semantic stream.
        fused: List of tensors that result from fusing the semantic and
            high-fidelity streams.
    """

    semantic: List[torch.Tensor]
    high_fidelity: List[torch.Tensor]
    fused: List[torch.Tensor]

    def flatten(self, attr: str) -> torch.Tensor:
        """Concatenate a list attribute into a single tensor."""

        tensors: Sequence[torch.Tensor] = getattr(self, attr)
        if not tensors:
            raise ValueError(f"Attribute '{attr}' is empty and cannot be flattened.")
        return torch.cat(tensors, dim=0)


class DualStreamVisionEncoder(nn.Module):
    """Wraps Qwen2.5-VL's vision tower with a high-fidelity DINOv3 branch.

    The encoder keeps the original Qwen tower for semantic alignment while a
    second DINOv3 stream preserves rich geometric and textural cues. Both
    streams are fused through a learnable MLP and exposed via
    :class:`DualStreamVisionOutput`.
    """

    def __init__(self, semantic_encoder: nn.Module, config: DualStreamConfig):
        super().__init__()

        self.semantic_encoder = semantic_encoder
        self.config = config

        self.hidden_size = getattr(semantic_encoder.config, "hidden_size", None)
        if self.hidden_size is None:
            raise AttributeError("semantic_encoder.config.hidden_size must be defined")

        self.spatial_merge_size = getattr(semantic_encoder, "spatial_merge_size", 1)

        checkpoint_path = self._resolve_checkpoint(config.dinov3_checkpoint_path)
        self.high_fidelity_encoder = self._build_high_fidelity_encoder(
            config, checkpoint_path
        )
        if checkpoint_path:
            LOGGER.info("Loaded DINOv3 weights from %s", checkpoint_path)
        if hasattr(self.high_fidelity_encoder, "reset_classifier"):
            self.high_fidelity_encoder.reset_classifier(0)

        self.high_dim = getattr(self.high_fidelity_encoder, "num_features", None)
        if self.high_dim is None:
            self.high_dim = getattr(self.high_fidelity_encoder, "embed_dim", None)
        if self.high_dim is None:
            raise AttributeError("Could not infer hidden size of the DINOv3 encoder")

        patch_size = getattr(self.high_fidelity_encoder.patch_embed, "patch_size", None)
        if patch_size is None:
            raise AttributeError("DINOv3 encoder must expose patch_embed.patch_size")
        self.register_buffer(
            "high_patch_size",
            torch.tensor(patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size), dtype=torch.long),
            persistent=False,
        )
        self.base_num_patches = getattr(
            self.high_fidelity_encoder.patch_embed, "num_patches", None
        )
        self.has_cls_token = hasattr(self.high_fidelity_encoder, "cls_token")

        fusion_hidden = int(self.hidden_size * config.fusion_hidden_ratio)
        self.high_proj = nn.Linear(self.high_dim, self.hidden_size, bias=False)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, fusion_hidden),
            nn.SiLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(fusion_hidden, self.hidden_size),
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_size)

        if config.freeze_high_fidelity_stream:
            for param in self.high_fidelity_encoder.parameters():
                param.requires_grad = False
            self.high_fidelity_encoder.eval()
        if config.freeze_semantic_stream:
            for param in self.semantic_encoder.parameters():
                param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> DualStreamVisionOutput:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw must be provided for dual-stream encoding")

        split_sizes = self._compute_split_sizes(image_grid_thw)
        target_hw = self._target_hw(image_grid_thw)

        # Extract high-fidelity features before the semantic stream touches the
        # pixels so that DINOv3 always sees the raw image tensor without the
        # preprocessing side effects introduced by the original vision tower.
        high_chunks_raw = self._encode_high_fidelity(pixel_values, target_hw)
        high_chunks = [self.high_proj(chunk) for chunk in high_chunks_raw]

        semantic_tokens = self.semantic_encoder(pixel_values, grid_thw=image_grid_thw)
        semantic_chunks = list(torch.split(semantic_tokens, split_sizes))

        fused_chunks = []
        for semantic, high in zip(semantic_chunks, high_chunks):
            fusion_input = torch.cat([semantic, high], dim=-1)
            fused = self.fusion_mlp(fusion_input)
            fused = self.fusion_norm(fused + semantic)
            fused_chunks.append(fused)

        return DualStreamVisionOutput(
            semantic=semantic_chunks, high_fidelity=high_chunks, fused=fused_chunks
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _build_high_fidelity_encoder(
        self, config: DualStreamConfig, checkpoint_path: Optional[str]
    ) -> nn.Module:
        """Instantiate the high-fidelity branch with ModelScope or fall back to Torch Hub."""

        modelscope_encoder = self._try_build_modelscope_encoder(config)
        if modelscope_encoder is not None:
            LOGGER.info(
                "Initialised DINOv3 encoder from ModelScope target '%s'",
                config.dinov3_modelscope_model_id,
            )
            return modelscope_encoder

        hub_kwargs = dict(
            repo_or_dir=config.dinov3_repo_or_dir,
            model=config.dinov3_model_name,
            pretrained=checkpoint_path is None,
        )
        if checkpoint_path is not None:
            hub_kwargs["weights"] = checkpoint_path

        try:
            encoder = torch.hub.load(**hub_kwargs)
            LOGGER.info(
                "Initialised DINOv3 encoder %s from %s",
                config.dinov3_model_name,
                config.dinov3_repo_or_dir,
            )
            return encoder
        except Exception as hub_error:  # pragma: no cover - defensive fallback
            if timm is None:
                raise RuntimeError(
                    "torch.hub.load failed and timm is unavailable to build the DINOv3 encoder"
                ) from hub_error

            LOGGER.warning(
                "torch.hub.load failed for %s (repo %s): %s. Falling back to timm.",
                config.dinov3_model_name,
                config.dinov3_repo_or_dir,
                hub_error,
            )
            return timm.create_model(
                config.dinov3_model_name,
                pretrained=checkpoint_path is None,
                checkpoint_path=checkpoint_path or "",
            )

    def _try_build_modelscope_encoder(
        self, config: DualStreamConfig
    ) -> Optional[nn.Module]:  # pragma: no cover - network/environment dependent
        if not config.dinov3_modelscope_model_id:
            return None

        if AutoModel is None:
            LOGGER.warning(
                "ModelScope is not available (import error: %s). Falling back to torch.hub.",
                _MODELSCOPE_IMPORT_ERROR,
            )
            return None

        load_target = config.dinov3_modelscope_model_id
        load_kwargs = {
            "trust_remote_code": config.dinov3_modelscope_trust_remote_code,
        }
        if config.dinov3_modelscope_revision:
            load_kwargs["revision"] = config.dinov3_modelscope_revision
        if config.dinov3_modelscope_device_map:
            load_kwargs["device_map"] = config.dinov3_modelscope_device_map

        try:
            encoder = AutoModel.from_pretrained(load_target, **load_kwargs)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load DINOv3 via ModelScope target '%s': %s. Falling back to torch.hub.",
                load_target,
                exc,
            )
            return None

        unwrapped = self._unwrap_modelscope_model(encoder)
        if unwrapped is None:
            LOGGER.warning(
                "ModelScope model '%s' did not expose a usable encoder. Falling back to torch.hub.",
                load_target,
            )
            return None

        return unwrapped

    def _compute_split_sizes(self, image_grid_thw: torch.LongTensor) -> List[int]:
        merge_sq = self.spatial_merge_size ** 2
        split_sizes = (image_grid_thw.prod(-1) // merge_sq).tolist()
        if any(size <= 0 for size in split_sizes):
            raise ValueError(
                "image_grid_thw produced non-positive split sizes. Check preprocessing."
            )
        return split_sizes

    def _target_hw(self, image_grid_thw: torch.LongTensor) -> List[Tuple[int, int]]:
        target_hw = []
        merge = self.spatial_merge_size
        for grid in image_grid_thw.tolist():
            _, h, w = grid
            target_hw.append((max(1, h // merge), max(1, w // merge)))
        return target_hw

    def _resolve_checkpoint(self, user_path: Optional[str]) -> Optional[str]:
        if user_path is None:
            return None

        path = Path(user_path).expanduser()
        if path.is_file():
            return str(path)

        if path.is_dir():
            preferred = (
                "model.safetensors",
                "pytorch_model.bin",
                "model.pt",
                "model.pth",
                "checkpoint.pth",
                "weights.pt",
                "weights.pth",
            )
            for name in preferred:
                candidate = path / name
                if candidate.exists():
                    return str(candidate)

            # Fall back to the first checkpoint-like file we can find.
            for suffix in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
                matches = sorted(path.glob(suffix))
                if matches:
                    return str(matches[0])

            raise FileNotFoundError(
                f"Could not find a checkpoint file in directory '{path}'."
            )

        raise FileNotFoundError(f"Provided DINOv3 checkpoint path '{path}' is invalid")

    def _unwrap_modelscope_model(self, encoder: nn.Module) -> Optional[nn.Module]:
        if not isinstance(encoder, nn.Module):
            return None

        queue: List[nn.Module] = [encoder]
        visited = {id(encoder)}
        fallback: Optional[nn.Module] = None

        while queue:
            candidate = queue.pop(0)

            if self._has_implemented_method(candidate, "forward_features"):
                return candidate

            if self._has_implemented_method(candidate, "forward") and fallback is None:
                fallback = candidate

            for attr in ("model", "module", "backbone", "encoder", "net"):
                child = getattr(candidate, attr, None)
                if isinstance(child, nn.Module) and id(child) not in visited:
                    visited.add(id(child))
                    queue.append(child)

        return fallback

    @staticmethod
    def _has_implemented_method(module: nn.Module, name: str) -> bool:
        if not hasattr(module, name):
            return False

        method = getattr(module, name)
        if not callable(method):
            return False

        base = getattr(nn.Module, name, None)
        return getattr(type(module), name, None) is not base

    def _encode_high_fidelity(
        self,
        pixel_values: torch.Tensor,
        target_hw: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        resize_size = (self.config.dinov3_image_size, self.config.dinov3_image_size)
        inputs = F.interpolate(
            pixel_values.float(), size=resize_size, mode="bicubic", align_corners=False
        )
        features = self._forward_high_fidelity_features(inputs)
        tokens = self._extract_patch_tokens(features)
        batch, length, channels = tokens.shape
        grid_h = int(round(math.sqrt(length)))
        grid_w = grid_h
        if grid_h * grid_w != length:
            if grid_h == 0:
                raise ValueError("DINOv3 produced zero patch tokens")
            grid_w = length // grid_h
            if grid_h * grid_w != length:
                raise ValueError(
                    "DINOv3 patch tokens cannot be reshaped into a 2D grid"
                )
        tokens = tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)

        outputs: List[torch.Tensor] = []
        for idx, (target_h, target_w) in enumerate(target_hw):
            resized = F.interpolate(
                tokens[idx : idx + 1], size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            outputs.append(resized.flatten(2).transpose(1, 2).squeeze(0))
        return outputs

    def _forward_high_fidelity_features(self, inputs: torch.Tensor):
        encoder = self.high_fidelity_encoder

        if hasattr(encoder, "forward_features") and callable(getattr(encoder, "forward_features")):
            return encoder.forward_features(inputs)

        try:
            return encoder(pixel_values=inputs)
        except TypeError:
            try:
                return encoder(inputs)
            except TypeError as exc:
                raise TypeError(
                    "High-fidelity encoder does not accept 'pixel_values' or positional inputs"
                ) from exc

    def _extract_patch_tokens(self, features: torch.Tensor | dict) -> torch.Tensor:
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        elif hasattr(features, "token_embeddings"):
            features = features.token_embeddings

        if isinstance(features, (list, tuple)) and features:
            features = features[0]

        if isinstance(features, dict):
            for key in (
                "x_norm_patchtokens",
                "token_embeddings",
                "last_hidden_state",
                "x_norm_clstoken",
            ):
                if key in features:
                    features = features[key]
                    break
            else:  # pragma: no cover - defensive branch
                raise KeyError("Unsupported DINOv3 forward_features output format")

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "High-fidelity encoder produced an output that could not be converted to a tensor"
            )

        if features.ndim != 3:
            raise ValueError("Expected features with shape (batch, tokens, dim)")

        if self.has_cls_token and features.shape[1] > 0:
            total_tokens = features.shape[1]
            if (
                self.base_num_patches is not None
                and total_tokens == self.base_num_patches + 1
            ):
                features = features[:, 1:, :]
            elif total_tokens > 1:
                maybe_grid = int(round(math.sqrt(total_tokens - 1)))
                if maybe_grid * maybe_grid == total_tokens - 1:
                    features = features[:, 1:, :]
        return features


__all__ = ["DualStreamVisionEncoder", "DualStreamVisionOutput"]
