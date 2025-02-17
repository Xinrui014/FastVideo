from typing import Any, Dict, List, Optional, Tuple, Union
from abc import abstractmethod
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange

from fastvideo.models.hunyuan.modules.posemb_layers import \
    get_nd_rotary_pos_embed
from fastvideo.utils.parallel_states import nccl_info

from .activation_layers import get_activation_layer
from .attenion import parallel_attention
from .embed_layers import PatchEmbed, TextProjection, TimestepEmbedder
from .mlp_layers import MLP, FinalLayer, MLPEmbedder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .norm_layers import get_norm_layer
from .posemb_layers import apply_rotary_emb
from .token_refiner import SingleTokenRefiner
from fastvideo.models.ldm.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
)
from fastvideo.models.ldm.attention import SpatialTransformer


# add upsampler
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        # if self.dims == 3:
        #     self.deconv = nn.ConvTranspose3d(
        #         channels, self.out_channels, kernel_size=4, stride=2, padding=1
        #     )
        # else:
        #     self.deconv = nn.ConvTranspose2d(
        #             channels, self.out_channels, kernel_size=4, stride=2, padding=1
        #         )
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            # x = self.deconv(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with separate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size,
                                      hidden_size * 3,
                                      bias=qkv_bias,
                                      **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.img_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.img_attn_proj = nn.Linear(hidden_size,
                                       hidden_size,
                                       bias=qkv_bias,
                                       **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size,
                                      hidden_size * 3,
                                      bias=qkv_bias,
                                      **factory_kwargs)
        self.txt_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.txt_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.txt_attn_proj = nn.Linear(hidden_size,
                                       hidden_size,
                                       bias=qkv_bias,
                                       **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated,
                                 shift=img_mod1_shift,
                                 scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv,
                                        "B L (K H D) -> K B L H D",
                                        K=3,
                                        H=self.heads_num)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:

            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                return encoder_state.narrow(
                    dim, nccl_info.rank_within_group * local_heads,
                    local_heads)

            freqs_cis = (
                shrink_head(freqs_cis[0], dim=0),
                shrink_head(freqs_cis[1], dim=0),
            )

            img_qq, img_kk = apply_rotary_emb(img_q,
                                              img_k,
                                              freqs_cis,
                                              head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated,
                                 shift=txt_mod1_shift,
                                 scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv,
                                        "B L (K H D) -> K B L H D",
                                        K=3,
                                        H=self.heads_num)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
        )

        # attention computation end

        img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

        # Calculate the img blocks.
        img = img + apply_gate(self.img_attn_proj(img_attn),
                               gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img),
                         shift=img_mod2_shift,
                         scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        # Calculate the txt blocks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn),
                               gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(self.txt_norm2(txt),
                         shift=txt_mod2_shift,
                         scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim,
                                 **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size,
                                 **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                       if qk_norm else nn.Identity())
        self.k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                       if qk_norm else nn.Identity())

        self.pre_norm = nn.LayerNorm(hidden_size,
                                     elementwise_affine=False,
                                     eps=1e-6,
                                     **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod),
                               [3 * self.hidden_size, self.mlp_hidden_dim],
                               dim=-1)

        q, k, v = rearrange(qkv,
                            "B L (K H D) -> K B L H D",
                            K=3,
                            H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(
                dim, nccl_info.rank_within_group * local_heads, local_heads)

        freqs_cis = (shrink_head(freqs_cis[0],
                                 dim=0), shrink_head(freqs_cis[1], dim=0))

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q,
                                          img_k,
                                          freqs_cis,
                                          head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
        )

        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformer_upsampler(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        rope_theta: int = 256,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(self.patch_size, self.in_channels,
                                 self.hidden_size, **factory_kwargs)

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.config.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.config.text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size,
                                        get_activation_layer("silu"),
                                        **factory_kwargs)

        # text modulation
        self.vector_in = MLPEmbedder(self.config.text_states_dim_2,
                                     self.hidden_size, **factory_kwargs)

        # guidance modulation
        self.guidance_in = (TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
                            if guidance_embed else None)

        # double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            ) for _ in range(mm_double_blocks_depth)
        ])

        # single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                **factory_kwargs,
            ) for _ in range(mm_single_blocks_depth)
        ])

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )


        # add upsampler after double bolcks
        self.feature_upsamplers = nn.ModuleList([
            TimestepEmbedSequential(
                ResBlock(common_input_dim, time_embed_dim, dropout,
                         dims=dims, use_checkpoint=use_checkpoint, up=True,
                         use_scale_shift_norm=use_scale_shift_norm),
                ResBlock(common_input_dim, time_embed_dim, dropout,
                         dims=dims, use_checkpoint=use_checkpoint,
                         use_scale_shift_norm=use_scale_shift_norm)
            )
            for _ in range(20)
        ])

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [
                head_dim // target_ndim for _ in range(target_ndim)
            ]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin
        # x: torch.Tensor,
        # t: torch.Tensor,  # Should be in range(0, 1000).
        # text_states: torch.Tensor = None,
        # text_mask: torch.Tensor = None,  # Now we don't use it.
        # text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        # guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        # return_dict: bool = True,

    def forward(
        self,
        hidden_states: torch.Tensor,
        lr_fea: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        guidance=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor([6016.0],
                                    device=hidden_states.device,
                                    dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = encoder_hidden_states[:, 1:]
        text_states_2 = encoder_hidden_states[:, 0, :self.config.
                                              text_states_dim_2]
        _, _, ot, oh, ow = x.shape  # codespell:ignore
        tt, th, tw = (
            ot // self.patch_size[0],  # codespell:ignore
            oh // self.patch_size[1],  # codespell:ignore
            ow // self.patch_size[2],  # codespell:ignore
        )
        original_tt = nccl_info.sp_size * tt
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((original_tt, th, tw))
        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        lr_fea = self.img_in(lr_fea)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t,
                              text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}")

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        lr_fea_seq_len = lr_fea.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        # add upsampler in double blocks
        for i, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis, text_mask]
            double_block_lr_fea_args = [lr_fea, txt, vec, freqs_cis, text_mask]

            img, txt = block(*double_block_args)
            lr_fea, _ = block(*double_block_lr_fea_args)
            hr_fea = self.feature_upsamplers[i](lr_fea)
            img = img + hr_fea


        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if output_features:
            features_list = []
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                ]

                x = block(*single_block_args)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img,
                               vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        assert not return_dict, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
        else:
            features_list = None
        return (img, features_list)

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double":
            sum([
                sum(p.numel() for p in block.img_attn_qkv.parameters()) +
                sum(p.numel() for p in block.img_attn_proj.parameters()) +
                sum(p.numel() for p in block.img_mlp.parameters()) +
                sum(p.numel() for p in block.txt_attn_qkv.parameters()) +
                sum(p.numel() for p in block.txt_attn_proj.parameters()) +
                sum(p.numel() for p in block.txt_mlp.parameters())
                for block in self.double_blocks
            ]),
            "single":
            sum([
                sum(p.numel() for p in block.linear1.parameters()) +
                sum(p.numel() for p in block.linear2.parameters())
                for block in self.single_blocks
            ]),
            "total":
            sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
