# Adapt from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/models/motion_module.py
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.nn import Fold, Unfold
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def forward(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        attention_mask=None,
        anchor_frame_idx=None,
        **kwargs
    ):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, **kwargs
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
                # check resolution
        resolu = (hidden_states.shape[-2], hidden_states.shape[-1])
        trajs = {}
        trajs["traj"] = kwargs["trajs"]["traj{}".format(resolu)]
        trajs["mask"] = kwargs["trajs"]["mask{}".format(resolu)]
        trajs["t"] = kwargs["t"]
        trajs["old_qk"] = kwargs["old_qk"]
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                size=(height, width),
                **trajs
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        size=None,
        **kwargs
    ):

        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                    size=size,
                    traj=kwargs['traj'],
                    mask=kwargs['mask']
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(Attention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )
        self.set_processor(LocalAttnProcessor())

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Optional[Callable] = None,
    ):
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            # XFormersAttnProcessor corrupts video generation and work with Pytorch 1.13.
            # Pytorch 2.0.1 AttnProcessor works the same as XFormersAttnProcessor in Pytorch 1.13.
            # You don't need XFormersAttnProcessor here.
            # processor = XFormersAttnProcessor(
            #     attention_op=attention_op,
            # )
            processor = LocalAttnProcessor()
        else:
            processor = LocalAttnProcessor()

        self.set_processor(processor)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        size=None,
        traj=None,
        mask=None,
        **cross_attention_kwargs,
    ):
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]  # d means HxW
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )
        else:
            raise NotImplementedError
        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            traj=traj,
            mask=mask,
            video_length=video_length,
            size=size,
            **cross_attention_kwargs,
        )

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class LocalAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, window_size=3):
        self.window_size = 3
        # define unfold and fold operation
        self.unfold = Unfold(kernel_size=self.window_size, padding=self.window_size//2)
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        height, width = kwargs.get("size")
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # local temporal attention
        video_length = sequence_length

        # Reshape input to (b*f, c, h, w) for unfolding
        key_unfolded = self.unfold(rearrange(key, "(b h w) f c -> (b f) c h w", h=height, w=width, f=video_length))
        value_unfolded = self.unfold(rearrange(value, "(b h w) f c -> (b f) c h w", h=height, w=width, f=video_length))
        # reshape unfolded patches to (b*h*w, f, c*window_size*window_size)
        key_unfolded = rearrange(key_unfolded, "(b f) c (h w) -> (b h w) f c", h=height, w=width, f=video_length)
        value_unfolded = rearrange(value_unfolded, "(b f) c (h w) -> (b h w) f c", h=height, w=width, f=video_length)
        # reshape to temporal dim
        key_unfolded = rearrange(key_unfolded, "(b h w) f (c s) -> (b h w) (f s) c", h=height, w=width, f=video_length, s=self.window_size**2)
        value_unfolded = rearrange(value_unfolded, "(b h w) f (c s) -> (b h w) (f s) c", h=height, w=width, f=video_length, s=self.window_size**2)
        

        query = attn.head_to_batch_dim(query)
        key_unfolded = attn.head_to_batch_dim(key_unfolded)
        value_unfolded = attn.head_to_batch_dim(value_unfolded)

        attention_probs = attn.get_attention_scores(query, key_unfolded, attention_mask)
        hidden_states = torch.bmm(attention_probs, value_unfolded)
        hidden_states = attn.batch_to_head_dim(hidden_states)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states





def get_window_indices(t_inds, x_inds, y_inds, height, width, window_size=4):
    half_window = window_size // 2
    windows = []

    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            x_window_inds = torch.clamp(x_inds + i, 0, height - 1)
            y_window_inds = torch.clamp(y_inds + j, 0, width - 1)
            windows.append((t_inds, x_window_inds, y_window_inds))
    
    return windows



class FlowGuidedAttnProcessor(AttnProcessor):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self):
        super().__init__()

    def reshape_heads_to_batch_dim3(self, tensor, head_size):
        batch_size1, batch_size2, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size1, batch_size2, seq_len, head_size, dim // head_size)
        tensor = rearrange(tensor, "b1 b2 n m c -> (b1 b2) n m c")
        return tensor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        traj=None,
        mask=None,
        video_length=None,
        size=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        height, width = size
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # query = attn.head_to_batch_dim(query)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)

        traj = rearrange(traj, '(f n) l d -> f n l d', f=video_length)
        mask = rearrange(mask, '(f n) l -> f n l', f=video_length)
        mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask[:, :, -video_length+1:]], dim=-1)

        traj_key_sequence_inds = torch.cat([traj[:, :, 0, :].unsqueeze(-2), traj[:, :, -video_length+1:, :]], dim=-2)
        
        t_inds = traj_key_sequence_inds[:, :, :, 0]
        x_inds = traj_key_sequence_inds[:, :, :, 1]
        y_inds = traj_key_sequence_inds[:, :, :, 2]        


        query_tempo = rearrange(query, '(b h w) f d -> (b f) (h w) 1 d', h=height, w=width, f=video_length)
        _key = rearrange(key, '(b h w) f d -> b f h w d', h=height, w=width, f=video_length)
        _value = rearrange(value, '(b h w) f d -> b f h w d', h=height, w=width, f=video_length)
        window_size = 2
        windows = get_window_indices(t_inds, x_inds, y_inds, height, width, window_size=window_size)
        key_windows = []
        value_windows = []
        w = len(windows)
        for t_ind, x_ind, y_ind in windows:
            key_windows.append(_key[:, t_ind, x_ind, y_ind])
            value_windows.append(_value[:, t_ind, x_ind, y_ind])
        
        key_windows = torch.stack(key_windows, dim=-2)
        value_windows = torch.stack(value_windows, dim=-2)

        key_tempo = rearrange(key_windows, 'b f n l w d -> (b f) n (l w) d', f=video_length, w=w)
        value_tempo = rearrange(value_windows, 'b f n l w d -> (b f) n (l w) d', f=video_length, w=w)

        if mask.shape[0] != query_tempo.shape[0]:
            mask = rearrange(torch.stack([mask, mask]),  'b f n l -> (b f) n l') # for classifier-free guidance?
        mask = rearrange(mask.unsqueeze(-1).repeat(1, 1, 1, w), "(b f) n l w -> (b f) n (l w)", f=video_length, w=w)
        mask = mask[:,None].repeat(1, attn.heads, 1, 1).unsqueeze(-2)
        attn_bias = torch.zeros_like(mask, dtype=key_tempo.dtype) # regular zeros_like
        attn_bias[~mask] = -torch.inf
        # flow attention
        query_tempo = self.reshape_heads_to_batch_dim3(query_tempo, attn.heads)
        key_tempo = self.reshape_heads_to_batch_dim3(key_tempo, attn.heads)
        value_tempo = self.reshape_heads_to_batch_dim3(value_tempo, attn.heads)

        out = xformers.ops.memory_efficient_attention(
            query_tempo, key_tempo, value_tempo,
        ).squeeze()
        # breakpoint()
        # attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(query_tempo.size(-1)) + attn_bias
        # attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
        # out = (attn_matrix2@value_tempo).squeeze(-2)
        hidden_states = rearrange(out,'(b f h w) k d -> (b h w) f (k d)', f=video_length, h=height, w=width)

        # # attention processing
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states