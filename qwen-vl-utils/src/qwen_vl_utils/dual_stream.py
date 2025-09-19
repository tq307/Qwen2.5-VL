from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass
class DualStreamConfig:
    """Configuration container for :class:`DualStreamEncoder`."""

    dinov3_image_size: int = 336
    """Target resolution for the longer side of each image."""

    dinov3_keep_aspect_ratio: bool = False
    """Whether to keep the input aspect ratio when resizing for DINOv3."""

    dinov3_mean: Sequence[float] = (0.485, 0.456, 0.406)
    dinov3_std: Sequence[float] = (0.229, 0.224, 0.225)
    dinov3_interpolation: InterpolationMode = InterpolationMode.BICUBIC
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32


class DualStreamEncoder:
    """Utility to encode images for the dual-stream inference pipeline."""

    def __init__(
        self,
        config: DualStreamConfig,
        dinov3_model: torch.nn.Module | None = None,
    ) -> None:
        self.config = config
        self.dinov3_model = dinov3_model
        if self.dinov3_model is not None:
            self.dinov3_model.to(self.config.device, dtype=self.config.dtype)
            self.dinov3_model.eval()

    def _resize_for_high_fidelity(self, image: Image.Image) -> Image.Image:
        """Resize the input image following the configuration policy."""

        longer_side = self.config.dinov3_image_size
        width, height = image.size

        if self.config.dinov3_keep_aspect_ratio:
            if height >= width:
                new_height = longer_side
                new_width = max(1, round(width * longer_side / max(height, 1)))
            else:
                new_width = longer_side
                new_height = max(1, round(height * longer_side / max(width, 1)))
            size = (new_height, new_width)
        else:
            size = (longer_side, longer_side)

        return F.resize(image, size, interpolation=self.config.dinov3_interpolation, antialias=True)

    def _preprocess_high_fidelity(self, image: Image.Image) -> torch.Tensor:
        resized = self._resize_for_high_fidelity(image.convert("RGB"))
        tensor = F.to_tensor(resized)
        return F.normalize(tensor, self.config.dinov3_mean, self.config.dinov3_std)

    def _encode_high_fidelity(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """Encode a collection of images using the high-fidelity DINOv3 branch."""

        batch = torch.stack([self._preprocess_high_fidelity(img) for img in images])
        batch = batch.to(self.config.device, dtype=self.config.dtype)

        if self.dinov3_model is None:
            return batch

        with torch.no_grad():
            features = self.dinov3_model(batch)
        return features


__all__ = ["DualStreamConfig", "DualStreamEncoder"]
