"""Visual evidence gating module for adaptive text-vision fusion."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class VisualEvidenceGate(nn.Module):
    """Computes a confidence-controlled gate for visual evidence.

    The module compares semantic and high-fidelity visual tokens while
    conditioning on the textual context. The output gate scales the fused visual
    representation before it is injected into the language model, reducing the
    influence of ambiguous or low-confidence visual cues.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: Optional[int] = None,
        hidden_ratio: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        context_size = context_size or hidden_size
        reduced = max(1, hidden_size // hidden_ratio)

        self.visual_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(context_size),
            nn.Linear(context_size, hidden_size),
            nn.SiLU(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size * 2, reduced),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        high_fidelity_tokens: torch.Tensor,
        fused_tokens: torch.Tensor,
        text_context: torch.Tensor,
        prior_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gate values for each image placeholder.

        Args:
            semantic_tokens: Tokens from the original Qwen2.5-VL vision stream.
            high_fidelity_tokens: Tokens coming from the DINOv3 branch projected
                into the Qwen embedding space.
            fused_tokens: Tokens produced after combining both visual streams.
                They serve as a reference for scaling the final embeddings.
            text_context: Per-sample textual summaries used to moderate the
                influence of linguistic priors.
            prior_logits: Optional prior logits that will be added before the
                sigmoid, allowing external signals (e.g. heuristic confidence
                scores) to modulate the gate.

        Returns:
            gate: Values in ``[0, 1]`` representing how much of the fused visual
                token should be trusted.
            logits: Raw logits before the sigmoid activation, useful for
                monitoring and auxiliary losses.
        """

        if not (semantic_tokens.shape == high_fidelity_tokens.shape == fused_tokens.shape):
            raise ValueError("All visual tensors must share the same shape")
        if text_context.shape != semantic_tokens.shape:
            raise ValueError("text_context must match the visual token shape")

        visual_features = torch.cat(
            [semantic_tokens, high_fidelity_tokens, fused_tokens - semantic_tokens], dim=-1
        )
        visual_features = self.visual_mlp(visual_features)
        context_features = self.context_mlp(text_context)
        gate_input = torch.cat([visual_features, context_features], dim=-1)
        logits = self.gate_head(gate_input)

        if prior_logits is not None:
            logits = logits + prior_logits

        temperature = torch.clamp(self.temperature, min=1e-2)
        gate = torch.sigmoid(logits / temperature)
        return gate, logits


__all__ = ["VisualEvidenceGate"]
