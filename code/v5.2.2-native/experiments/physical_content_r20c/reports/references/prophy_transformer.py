import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from prophy.models.refinement_expert_block import RefinementExpertBlock
from prophy.models.semantic_expert_block import SemanticExpertBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PhysTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        moe_ffn_dim: Optional[int] = None,
        num_expert: Optional[int] = 32,
        moe_topk: Optional[int] = 4,
        moe_capacity_factor: float = 1.25,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
        apply_physical_moe: bool = False,
    ):
        super().__init__()

        # 1. Input projection
        self.proj_in = None
        if apply_input_projection:
            self.proj_in = nn.Linear(dim, dim)

        # 2. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=WanAttnProcessor(),
        )

        # 3. Cross-attention
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanAttnProcessor(),
        )

        # 4. MoE feed-forward
        if apply_physical_moe:
            self.moe_norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
            self.moe_router = RefinementExpertBlock(dim, num_expert)
            self.moe_topk = moe_topk
            self.moe_capacity_factor = moe_capacity_factor
            self.moe_ffn = nn.ModuleList(
                [
                    FeedForward(dim, inner_dim=moe_ffn_dim, activation_fn="gelu-approximate")
                    for _ in range(num_expert)
                ]
            )

        # 5. Feed-forward
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        # 6. Output projection
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)
            # Zero convolution: initialize to output zeros at training start
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

        self.apply_physical_moe = apply_physical_moe
        self.scale_shift_table = nn.Parameter(torch.randn(1, 9, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            p_shift_msa,
            p_scale_msa,
            p_gate_msa,
        ) = (self.scale_shift_table + temb.float()).chunk(9, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(control_hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        control_hidden_states = (control_hidden_states.float() + attn_output * gate_msa).type_as(
            control_hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(control_hidden_states.float()).type_as(
            control_hidden_states
        )
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        control_hidden_states = control_hidden_states + attn_output

        # 3. Physical Mixture-of-Experts feed-forward
        if self.apply_physical_moe:
            # Normalization with adaptive modulation
            norm_hidden_states = (
                self.moe_norm(control_hidden_states.float()) * (1 + p_scale_msa) + p_shift_msa
            ).type_as(control_hidden_states)

            num_experts = len(self.moe_ffn)
            # Get expert routing logits (NOT probabilities for better gradient flow)
            # Using logits as input to expert_adapter provides:
            # 1. Higher variance input (std ~2.0 vs ~0.08 for softmax)
            # 2. Stronger gradients (8-10x compared to softmax)
            # 3. Better ability to map to unconstrained output range [0, 1.5]
            expert_probs = self.moe_router(
                norm_hidden_states, return_probs=False
            )  # [B, S, E] - logits!
            k = min(self.moe_topk, expert_probs.size(-1))
            expert_probs_for_loss = expert_probs
            # Apply noise to expert_probs
            if self.training:
                expert_noise = torch.randn_like(expert_probs) / num_experts**2
                expert_probs = expert_probs + expert_noise
            topk_probs, topk_indices = torch.topk(expert_probs, k=k, dim=-1)
            topk_probs = F.softmax(topk_probs.float(), dim=-1).to(topk_probs.dtype)

            # Reshape for token-level processing
            bsz, seq_len, hidden_dim = norm_hidden_states.shape
            num_tokens = bsz * seq_len
            capacity = max(
                int(math.ceil(self.moe_capacity_factor * num_tokens * k / num_experts)),
                1,
            )
            flat_states = norm_hidden_states.reshape(num_tokens, hidden_dim)
            flat_topk_indices = topk_indices.reshape(num_tokens, k)
            flat_topk_probs = topk_probs.reshape(num_tokens, k)

            # Initialize output buffer (use float32 for numerical stability)
            moe_output = torch.zeros(
                num_tokens, hidden_dim, dtype=torch.float32, device=flat_states.device
            )
            token_gate_sums = torch.zeros(
                num_tokens, dtype=torch.float32, device=flat_states.device
            )

            # Process each expert
            for expert_id in range(num_experts):
                # Find tokens that selected this expert
                expert_mask = flat_topk_indices == expert_id  # [num_tokens, k]

                if not expert_mask.any():
                    continue

                # Compute gating weights: sum probabilities for this expert across top-k
                expert_gate = (flat_topk_probs * expert_mask.float()).sum(dim=-1)  # [num_tokens]
                token_indices = expert_gate.nonzero(as_tuple=True)[0]

                if token_indices.numel() == 0:
                    continue

                expert_gate_values = expert_gate.index_select(0, token_indices).float()

                if self.training and expert_gate_values.numel() > capacity:
                    _, top_positions = torch.topk(
                        expert_gate_values, capacity, largest=True, sorted=False
                    )
                    token_indices = token_indices.index_select(0, top_positions)
                    expert_gate_values = expert_gate_values.index_select(0, top_positions)

                expert_input = flat_states.index_select(0, token_indices)
                expert_output = self.moe_ffn[expert_id](expert_input)
                expert_weights = expert_gate_values.unsqueeze(-1)

                weighted_output = expert_output.to(moe_output.dtype) * expert_weights.to(
                    moe_output.dtype
                )
                moe_output.index_add_(0, token_indices, weighted_output)
                token_gate_sums.index_add_(0, token_indices, expert_gate_values)

            # Reshape back to [B, S, D] and convert to original dtype
            valid_mask = token_gate_sums > 0
            if valid_mask.any():
                norm_values = token_gate_sums[valid_mask].unsqueeze(-1)
                moe_output[valid_mask] = moe_output[valid_mask] / norm_values

            moe_output = moe_output.view(bsz, seq_len, hidden_dim).to(control_hidden_states.dtype)
            control_hidden_states = (
                control_hidden_states.float() + moe_output.float() * p_gate_msa
            ).type_as(control_hidden_states)
        else:
            expert_probs_for_loss = None

        # 4. Standard feed-forward
        norm_hidden_states = (
            self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(control_hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        control_hidden_states = (
            control_hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(control_hidden_states)

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states, expert_probs_for_loss


class WanVACETransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "physical_patch_embedding",
        "condition_embedder",
        "physical_condition_embedder",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock", "WanVACETransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,  # 1.3B -- 12 heads
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,  # 1.3B -- 8960
        num_layers: int = 40,  # 1.3B -- 30 layers
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        physical_layers: List[int] = [0, 5, 10, 15, 20, 25, 30, 35],
        physical_in_channels: int = 96,
        # For MoE embedding
        num_experts: int = 32,
        expert_thw: Tuple[int] = (21, 60, 104),
        router_temperature: float = 4.0,
        # For local MoE
        local_num_experts: int = 32,
        local_topk: int = 4,
        local_moe_ffn_dim: Optional[int] = None,
        local_physical_classes: int = 23,
        local_moe_capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        if local_moe_ffn_dim is None:
            local_moe_ffn_dim = inner_dim

        if max(physical_layers) >= num_layers:
            raise ValueError(
                f"Physical layers {physical_layers} exceed the number of transformer layers {num_layers}."
            )
        if 0 not in physical_layers:
            raise ValueError("Physical layers must include layer 0.")

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )
        self.physical_patch_embedding = nn.Conv3d(
            physical_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # Physical-branch condition embedder with expanded time projection
        self.physical_condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 9,  # Expanded from 6 to 9 for additional parameters
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.physical_blocks = nn.ModuleList(
            [
                PhysTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    local_moe_ffn_dim,
                    local_num_experts,
                    local_topk,
                    local_moe_capacity_factor,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    apply_input_projection=i == 0,
                    apply_output_projection=True,
                    apply_physical_moe=i == len(physical_layers) - 1,
                )
                for i in range(len(physical_layers))
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # Global expert embeddings and routing network
        self.experts = nn.Parameter(torch.randn(num_experts, physical_in_channels, *expert_thw))
        self.gating_network = SemanticExpertBlock(
            num_experts=num_experts,
            dim=inner_dim,
            text_embed_dim=text_dim,
        )
        self.global_proj = nn.Linear(num_experts, num_experts)

        # Adapter to map local expert LOGITS to physical class space
        # Note: Input is now logits (range ~[-10, 10]) instead of probabilities ([0, 1])
        # Output is expected to match attention map intensity range [0, ~1.5]
        self.expert_adapter = nn.Sequential(
            nn.Linear(local_num_experts, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, local_physical_classes),
        )

        # Adjust initialization for logits input (which has much larger range than probabilities)
        # With logits input, standard Xavier initialization produces reasonable range
        # Keep default initialization (no scaling) for balanced initial output
        with torch.no_grad():
            # No scaling needed - standard Xavier initialization works well with logits input
            # self.expert_adapter[-1].weight.data *= 1.0  # Keep as is
            if self.expert_adapter[-1].bias is not None:
                # Initialize bias to small positive values to help initial learning
                self.expert_adapter[-1].bias.data.uniform_(0.0, 0.1)

        self.gradient_checkpointing = False

    @classmethod
    def from_config(cls, config, **kwargs):
        """Support loading old checkpoints that saved vace_* in config."""
        config = dict(config)
        if "vace_layers" in config and "physical_layers" not in config:
            config["physical_layers"] = config.pop("vace_layers")
        if "vace_in_channels" in config and "physical_in_channels" not in config:
            config["physical_in_channels"] = config.pop("vace_in_channels")
        return super().from_config(config, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Support loading old checkpoints that saved vace_* attribute names."""
        state_dict = dict(state_dict)
        for old_key in list(state_dict.keys()):
            if old_key.startswith("vace_blocks."):
                state_dict["physical_blocks." + old_key[11:]] = state_dict.pop(old_key)
            elif old_key.startswith("vace_patch_embedding."):
                state_dict["physical_patch_embedding." + old_key[22:]] = state_dict.pop(old_key)
            elif old_key.startswith("vace_condition_embedder."):
                state_dict["physical_condition_embedder." + old_key[24:]] = state_dict.pop(old_key)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        control_hidden_states_scale: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Choose expert condition first
        encoder_hidden_states_2 = (
            encoder_hidden_states if encoder_hidden_states_2 is None else encoder_hidden_states_2
        )

        # Use gating network and activation
        experts_logits = self.gating_network(encoder_hidden_states_2, return_probs=False)  # [B, E]
        combine_weights = F.softmax(experts_logits, dim=-1)
        experts_logits = self.global_proj(experts_logits)
        control_hidden_states = torch.einsum(
            "b e, e c t h w -> b c t h w", combine_weights, self.experts
        )  # [B, C, T, H, W]

        if control_hidden_states_scale is None:
            control_hidden_states_scale = control_hidden_states.new_ones(
                len(self.config.physical_layers)
            )
        control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
        if len(control_hidden_states_scale) != len(self.config.physical_layers):
            raise ValueError(
                f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
                f"equal to {len(self.config.physical_layers)}."
            )

        # 1. Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, S, D]

        control_hidden_states = self.physical_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)  # [B, S, D]

        control_hidden_states_padding = control_hidden_states.new_zeros(
            batch_size,
            hidden_states.size(1) - control_hidden_states.size(1),
            control_hidden_states.size(2),
        )
        control_hidden_states = torch.cat(
            [control_hidden_states, control_hidden_states_padding], dim=1
        )

        # 3. Time embedding
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Image embedding
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # Physical branch: time embedding for encoder_hidden_states_2 with expanded projection
        _, timestep_proj_2, encoder_hidden_states_2, encoder_hidden_states_image = (
            self.physical_condition_embedder(
                timestep, encoder_hidden_states_2, encoder_hidden_states_image
            )
        )
        timestep_proj_2 = timestep_proj_2.unflatten(1, (9, -1))  # 9 parameters for Physical blocks

        if encoder_hidden_states_image is not None:
            encoder_hidden_states_2 = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states_2], dim=1
            )

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            control_hidden_states_list = []
            local_expert_probs_list = []
            for i, physical_block in enumerate(self.physical_blocks):
                conditioning_states, control_hidden_states, local_expert_probs = (
                    self._gradient_checkpointing_func(
                        physical_block,
                        hidden_states,
                        encoder_hidden_states_2,
                        control_hidden_states,
                        timestep_proj_2,
                        rotary_emb,
                    )
                )
                control_hidden_states_list.append(
                    (conditioning_states, control_hidden_states_scale[i])
                )
                local_expert_probs_list.append(local_expert_probs)
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
                if i in self.config.physical_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale
        else:
            control_hidden_states_list = []
            local_expert_probs_list = []
            for i, physical_block in enumerate(self.physical_blocks):
                conditioning_states, control_hidden_states, local_expert_probs = physical_block(
                    hidden_states,
                    encoder_hidden_states_2,
                    control_hidden_states,
                    timestep_proj_2,
                    rotary_emb,
                )
                control_hidden_states_list.append(
                    (conditioning_states, control_hidden_states_scale[i])
                )
                local_expert_probs_list.append(local_expert_probs)
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if i in self.config.physical_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale

        # 6. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        # Aggregate local expert probabilities from last physical block and map to physical class space
        # Reshape from flat sequence to spatial dimensions [B, T, H, W, E]
        source_local_expert_probs = local_expert_probs_list[-1]
        source_local_expert_probs = source_local_expert_probs.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )

        # Apply adapter to map from local expert space to physical class space
        local_expert_probs = self.expert_adapter(source_local_expert_probs)

        # Permute to standard [B, C, T, H, W] format
        source_local_expert_probs = source_local_expert_probs.permute(0, 4, 1, 2, 3)
        local_expert_probs = local_expert_probs.permute(0, 4, 1, 2, 3)

        if not return_dict:
            # source_local_expert_probs: [B, E, T, H, W]
            # local_expert_probs: [B, P, T, H, W]
            return (
                output,
                experts_logits,
                source_local_expert_probs,
                local_expert_probs,
            )

        return Transformer2DModelOutput(sample=output)


# Alias for paper-neutral naming (backward compat: checkpoints may still use WanVACETransformer3DModel)
WanPhysicalTransformer3DModel = WanVACETransformer3DModel
