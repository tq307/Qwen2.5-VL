"""Configuration dataclasses for dual-stream vision encoders."""

from dataclasses import dataclass, field


@dataclass
class DualStreamConfig:
    """Configuration for the dual-stream visual encoder.

    Attributes:
        freeze_high_fidelity_stream: Whether to keep the high-fidelity stream (DINO branch)
            frozen. Set this to ``False`` (the default) to let the branch adapt to the target
            domain during fine-tuning. We recommend unfreezing it in industrial scenarios so the
            DINO stream can learn domain-specific features.
    """

    freeze_high_fidelity_stream: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the high-fidelity (DINO) stream. We recommend leaving this "
                "stream unfrozen in industrial fine-tuning so it can adapt to domain-specific "
                "data."
            )
        },
    )


__all__ = ["DualStreamConfig"]
