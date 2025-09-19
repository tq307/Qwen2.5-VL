"""Integration of the dual-stream encoder and VE-Gate into Qwen2.5-VL."""

from __future__ import annotations

from typing import Optional, Union

import torch

from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

from .dual_stream_config import DualStreamConfig
from .dual_stream_encoder import DualStreamVisionEncoder
from .ve_gate import VisualEvidenceGate


class DualStreamQwen25VLModel(Qwen2_5_VLModel):
    """Extends :class:`Qwen2_5_VLModel` with dual-stream vision and VE-Gate."""

    def __init__(self, config, dual_stream_config: Optional[DualStreamConfig] = None):
        self.dual_stream_config = dual_stream_config or DualStreamConfig()
        super().__init__(config)
        self.dual_vision = DualStreamVisionEncoder(self.visual, self.dual_stream_config)
        self.ve_gate = VisualEvidenceGate(
            hidden_size=self.dual_vision.hidden_size,
            context_size=self.dual_vision.hidden_size,
            hidden_ratio=self.dual_stream_config.ve_gate_hidden_ratio,
            dropout=self.dual_stream_config.ve_gate_dropout,
            temperature=self.dual_stream_config.ve_gate_temperature,
        )
        self.register_to_config(mm_dual_stream=self.dual_stream_config.to_dict())
        self.last_gating_metadata: Optional[dict] = None
        self.last_text_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Vision feature extraction helpers
    # ------------------------------------------------------------------
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None
    ):
        outputs = self.dual_vision(pixel_values, image_grid_thw=image_grid_thw)
        return outputs.fused

    def get_dual_stream_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None
    ):
        """Return raw semantic, high-fidelity and fused tokens for analysis."""

        outputs = self.dual_vision(pixel_values, image_grid_thw=image_grid_thw)
        return outputs

    # ------------------------------------------------------------------
    # Forward override with VE-Gate
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        gating_metadata = None
        image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
        video_mask_for_text = torch.zeros_like(inputs_embeds, dtype=torch.bool)

        if pixel_values is not None:
            dual_outputs = self.dual_vision(pixel_values, image_grid_thw=image_grid_thw)
            fused_tokens = dual_outputs.flatten("fused").to(inputs_embeds.device, inputs_embeds.dtype)
            semantic_tokens = dual_outputs.flatten("semantic").to(inputs_embeds.device, inputs_embeds.dtype)
            high_tokens = dual_outputs.flatten("high_fidelity").to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, video_mask_for_text = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=fused_tokens
            )
            placeholder_indices = image_mask[..., 0].nonzero(as_tuple=False)
            if placeholder_indices.shape[0] != fused_tokens.shape[0]:
                raise ValueError("Mismatch between visual tokens and image placeholders")

            text_context = self._summarise_text_context(inputs_embeds, image_mask, video_mask_for_text)
            text_tokens = text_context[placeholder_indices[:, 0]]

            original_placeholders = inputs_embeds[placeholder_indices[:, 0], placeholder_indices[:, 1]]
            gate, gate_logits = self.ve_gate(
                semantic_tokens, high_tokens, fused_tokens, text_tokens
            )
            gated_tokens = gate * fused_tokens + (1.0 - gate) * original_placeholders
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, gated_tokens)

            gating_metadata = {
                "gate": gate.detach(),
                "logits": gate_logits.detach(),
                "indices": placeholder_indices.detach(),
            }

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            video_mask_for_text = video_mask

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        if gating_metadata is not None:
            output.visual_evidence = gating_metadata
        text_mask = (~image_mask[..., 0] & ~video_mask_for_text[..., 0]).detach()
        output.text_mask = text_mask
        self.last_gating_metadata = gating_metadata
        self.last_text_mask = text_mask
        return output if return_dict else output.to_tuple()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _summarise_text_context(
        self,
        inputs_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = ~(image_mask[..., 0] | video_mask[..., 0])
        summaries = []
        for batch_idx in range(inputs_embeds.shape[0]):
            valid = mask[batch_idx]
            if valid.any():
                summaries.append(inputs_embeds[batch_idx, valid].mean(dim=0))
            else:
                summaries.append(inputs_embeds.new_zeros(inputs_embeds.shape[-1]))
        return torch.stack(summaries, dim=0)


class Qwen2_5_VLDualStreamForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Drop-in replacement that exposes the dual-stream architecture."""

    def __init__(self, config, dual_stream_config: Optional[DualStreamConfig] = None):
        self.dual_stream_config = dual_stream_config or DualStreamConfig()
        super().__init__(config)
        self.model = DualStreamQwen25VLModel(config, self.dual_stream_config)
        self.register_to_config(mm_dual_stream=self.dual_stream_config.to_dict())

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        gating_metadata = getattr(self.model, "last_gating_metadata", None)
        text_mask = getattr(self.model, "last_text_mask", None)
        if isinstance(outputs, Qwen2_5_VLCausalLMOutputWithPast):
            outputs.visual_evidence = gating_metadata
            outputs.text_mask = text_mask
        return outputs


__all__ = ["DualStreamQwen25VLModel", "Qwen2_5_VLDualStreamForConditionalGeneration"]
