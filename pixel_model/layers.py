import math
from typing import Union, Callable, Type
from operator import attrgetter
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


def shift_backwards_3d(input, size=1):
    '''
    If an image is given by (b c d h w)
    Front-Pads the image in the h dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [0,0]],
     [[1,2],
      [3,4]]]
    '''
    assert size >= 1
    b, c, d, h, w = input.shape
    return F.pad(input, (0, 0, 0, 0, size, 0))[..., :-size, :, :]


def shift_forwards_3d(input, size=1):
    '''
    If an image is given by (b c d h w)
    Front-Pads the image in the h dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [0,0]],
     [[1,2],
      [3,4]]]
    '''
    assert size >= 1
    b, c, d, h, w = input.shape
    return F.pad(input, (0, 0, 0, 0, size, 0))[..., size:, :, :]


def shift_down_3d(input, size=1):
    ''''
    If an image is given by (b c d h w)
    Top-Pads the image in the h dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [1,2]],
     [[0,0],
      [5,6]]]
    '''
    assert size >= 1
    return F.pad(input, (0, 0, size, 0, 0, 0))[..., :-size, :]

def shift_up_3d(input, size=1):
    ''''
    If an image is given by (b c d h w)
    Top-Pads the image in the h dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,0],
      [1,2]],
     [[0,0],
      [5,6]]]
    '''
    assert size >= 1
    return F.pad(input, (0, 0, 0, size, 0, 0))[..., size:, :]

def shift_right_3d(input, size=1):
    '''
    If an image is given by (b c d h w)
    Left-Pads the image in the w dimension by `size`
    [[[1,2],
      [3,4]],
     [[5,6],
      [7,8]]]
    --> becomes
    [[[0,1],
      [0,3]],
     [[0,5],
      [0,7]]]
    '''
    assert size >= 1
    return F.pad(input, (size, 0, 0, 0, 0, 0))[..., :-size]


def input_to_stack(input: torch.Tensor) -> torch.Tensor:
    return repeat(input, 'b c d h w -> dim b c d h w', dim=3)

def stack_to_output(stack: torch.Tensor) -> torch.Tensor:
    return stack.sum(dim=0)

def restack(depth, height, width):
    return rearrange([depth, height, width], 'dim b c d h w -> dim b c d h w')

class ConcatActivation(nn.Module):
    def __init__(self, activation: nn.Module, dim: int):
        super().__init__()
        self.activation = activation()
        self.dim = dim

    def forward(self, x):
        return torch.cat([self.activation(x), -self.activation(-x)], dim=self.dim)


class CausalConv3dAdd(nn.Module):
    r'''

    Layer that convolves a stack of three tensors in a causal manner.
    The stack of three tensors should be in the order:
    1. depth-wise
    2. height-wise
    3. width-wise
    Practically, this means the shape of the input of the forward function should be:
    `3 x batch x channels x depth x height x width`

    Warning:
    - The assumptions this module makes are quite subtle, and great care
      should be taken that none of these assumptions are broken in a different part
      of the user's code.
    - If causality is broken such that information flows directly from the
      input pixel to the output pixel, validation loss will go to zero almost immediately.
    - If the causality is not perfectly managed, the model as a whole will underperform.

    Note:
    - The implementation of the `kernel_size` argument is also non-trivial.
      Please check the example below.

    Some examples are provides as to how causailty is assumed in this method.
    These examples are done in 2D, but the extension to 3D is straightforward.

    Example 1:
    Assuming the stack consists of [height, width]:
    height: [[1, 2],  width: [[5, 6],
             [3, 4]]          [7, 8]]
    - The information at position 1 is always the information for position 7
    If mask == 'A':
    - The information at position 5 is the information for position 6
    - The information at position 7 is the information for position 8
    - The information at positions 6 & 8 are discarded (in the width part of the stack)
    If mask == 'B':
    - The information at position 5 is the information for position 5 etc.

    Example 2:
    If mask == 'B', the pixels marked 'x' contain information for pixel 'y'.
    pixel 'y' also contains information for pixel 'y'.
    pixel 'o' doesn't contain information for 'y'
    [[x, x, x], <- height stack
     [x, x, x], <- height stack
     [x, y, o]] <- width stack
    If mask == 'A', the pixel 'y' doesn't contain information for pixel 'y'.

    Example 3:
    If `kernel_size = 3`, the kernel sizes for the (depth, height, width) convs are as follows:
    - depth_conv = (2, 3, 3)
    - height_conv = (1, 2, 3)
    - width_conv = (1, 1, 2) if mask == 'B', else (1, 1, 1)
    The size of the width_conv is related to example 2.
    '''
    def __init__(
        self,
        mask: str = 'B',
        **conv_kwargs,
    ):
        super().__init__()
        assert 'padding' not in conv_kwargs
        self.padding_mode = 'constant'

        assert mask in ('A', 'B')
        self.mask = mask

        kernel_size = conv_kwargs.pop('kernel_size')
        assert kernel_size > 0
        assert kernel_size % 2 == 1, "even kernel sizes are not supported"

        # Making sure kernel size is at least 1
        depth_size  = max(kernel_size-1, 1)
        height_size = max(kernel_size-1, 1)
        width_size  = max(kernel_size//2 + (1 if mask == 'B' else 0), 1)

        # Split depth, height, width conv into three, allowing the receptive field to grow without blindspot
        # See https://papers.nips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf figure 1.
        self.depth_conv = nn.Conv3d(kernel_size=(depth_size, kernel_size, kernel_size), **conv_kwargs)
        self.height_conv = nn.Conv3d(kernel_size=(1, height_size, kernel_size), **conv_kwargs)
        self.width_conv = nn.Conv3d(kernel_size=(1, 1, width_size), **conv_kwargs)

        # padding is always (*width, *height, *depth), i.e. (left, right, top, bottom, front, back)
        half_kernel = kernel_size // 2
        self.depth_pad = (half_kernel, half_kernel, half_kernel, half_kernel, depth_size-1, 0)
        self.height_pad = (half_kernel, half_kernel, height_size-1, 0, 0, 0)
        self.width_pad = (width_size-1, 0, 0, 0, 0, 0)

    def forward(self, stack):
        depth, height, width = stack

        if self.mask == 'A':
            depth = shift_backwards_3d(depth)
            height = shift_down_3d(height)
            width = shift_right_3d(width)

        # The padding below only works with the mask 'A' padding done beforehand
        depth = self.depth_conv(F.pad(depth, pad=self.depth_pad, mode=self.padding_mode))
        height = self.height_conv(F.pad(height, pad=self.height_pad, mode=self.padding_mode))
        width = self.width_conv(F.pad(width, pad=self.width_pad, mode=self.padding_mode))

        return restack(depth, height, width)


class ExpandRFConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels*2,
            kernel_size=1,
        )

        self.height_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
        )

    def forward(self, stack):
        depth, height, width = stack

        depth_conv_height, depth_conv_width = torch.chunk(self.depth_conv(depth), chunks=2, dim=1)

        width = width + self.height_conv(height) + depth_conv_width
        height = height + depth_conv_height

        return restack(depth, height, width)


class FixupCausalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask: str = 'B',  # options are ('A', 'B')
        out: bool = False,
        activation: Type[nn.Module] = nn.ELU,
        use_dropout: bool = True,
        dropout_prob: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__()

        self.out = out

        self.bias1a, self.bias1b, self.bias2a, self.bias2b = (
            nn.Parameter(torch.zeros(1)) for _ in range(4)
        )
        self.scale = nn.Parameter(torch.ones(1))

        branch_channels = max(in_channels, out_channels)
        self.branch_conv1 = CausalConv3dAdd(
            in_channels=in_channels, out_channels=branch_channels, kernel_size=kernel_size, mask=mask, bias=False
        )

        # second conv in the branch is always ok to be 'B'
        self.branch_conv2 = CausalConv3dAdd(
            in_channels=branch_channels, out_channels=out_channels, kernel_size=kernel_size, mask='B', bias=False
        )

        # 1x1x1 conv for channel dim projection
        self.skip_conv = CausalConv3dAdd(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, mask=mask, bias=True
        ) if (in_channels != out_channels or mask == 'A') else None

        self.activation = activation()

        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, stack):
        out = self.branch_conv1(stack + self.bias1a)
        out = self.activation(out + self.bias1b)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.branch_conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out = out + (stack if self.skip_conv is None else self.skip_conv(stack))

        if not self.out:
            out = self.activation(out)

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):

        weight_getter = attrgetter('depth_conv.weight', 'height_conv.weight', 'width_conv.weight')

        # branch_conv1
        for weight in weight_getter(self.branch_conv1):
            torch.nn.init.normal_(
                weight,
                mean=0,
                std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
            )

        # branch_conv2
        for weight in weight_getter(self.branch_conv2):
            torch.nn.init.constant_(weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            for weight in weight_getter(self.skip_conv):
                init_method = torch.nn.init.kaiming_normal_ if not self.out else torch.nn.init.xavier_normal_
                # init_method = torch.nn.init.kaiming_normal_
                init_method(weight)
            bias_getter = attrgetter('depth_conv.bias', 'height_conv.bias', 'width_conv.bias')
            for bias in bias_getter(self.skip_conv):
                torch.nn.init.constant_(tensor=bias, val=0)


class PreActFixupCausalResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask: str = 'B',  # options are ('A', 'B')
        condition_dim: int = 0, #
        condition_kernel_size: int = 1, # ignored if condition_dim == 0
        activation: Type[nn.Module] = nn.ELU,
        dropout_prob: float = 0.5,
        bottleneck_divisor: int = 4, # set to 1 to disable bottlenecking
        concat_activation: bool = False,
        aux = False,
        #Catch spurious (keyword-)arguments
        *args,
        **kwargs
    ):
        super().__init__()

        self.bias1a, self.bias1b, self.bias2a, self.bias2b, self.bias3a, self.bias3b, self.bias4 = (
            nn.Parameter(torch.zeros(1)) for _ in range(7)
        )
        self.scale = nn.Parameter(torch.ones(1))

        groups = 2 if concat_activation else 1

        branch_channels = max(
            max(in_channels, out_channels) // bottleneck_divisor,
            groups
        )

        self.branch_conv1 = CausalConv3dAdd(
            in_channels=in_channels*groups,
            out_channels=branch_channels,
            kernel_size=1,
            mask=mask,
            bias=False,
            groups=groups
        )

        # second conv in the branch is always ok to be 'B'
        self.branch_conv2 = CausalConv3dAdd(
            in_channels=branch_channels*groups,
            out_channels=branch_channels,
            kernel_size=kernel_size,
            mask='B',
            bias=False,
            groups=groups
        )

        # same for third
        self.branch_conv3 = CausalConv3dAdd(
            in_channels=branch_channels*groups,
            out_channels=out_channels,
            kernel_size=1,
            mask='B',
            bias=False,
            groups=groups
        )

        self.expand_rf = ExpandRFConv(branch_channels*groups)

        # 1x1x1 conv for channel dim projection
        # Also needed if mask == 'A', since otherwise causality is broken
        self.skip_conv = CausalConv3dAdd(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, mask=mask, bias=True
        ) if (in_channels != out_channels or mask == 'A') else None

        self.condition = nn.Conv3d(
            in_channels=condition_dim,
            out_channels=branch_channels,
            kernel_size=condition_kernel_size,
            padding=condition_kernel_size // 2,
            bias=True
        ) if condition_dim > 0 else None

        self.aux = CausalConv3dAdd(
            in_channels=branch_channels,
            out_channels=branch_channels,
            kernel_size=1,
            bias=True
        ) if aux else None

        self.activation = activation() if not concat_activation else ConcatActivation(activation, dim=2)

        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, stack: torch.Tensor, aux: torch.Tensor = None, condition: torch.Tensor = None, condition_cache = None):
        out = stack

        out = self.activation(out + self.bias1a)
        out = self.branch_conv1(out + self.bias1b)

        out = self.expand_rf(out)

        if aux is not None:
            assert self.aux is not None
            out = out + self.aux(self.activation(aux))

        out = self.activation(out + self.bias2a)
        out = self.branch_conv2(out + self.bias2b)

        if self.dropout is not None:
            out = restack(*map(self.dropout, out))

        if not (condition is None and condition_cache is None):
            # deliberately prefer condition_cache over computing condition again
            if condition_cache is not None:
                condition = condition_cache.popleft()
            else:
                assert self.condition is not None, 'Condition projection matrix not initialised!'
                condition = self.condition(condition)

            condition = condition[(..., *(slice(d) for d in out.shape[-3:]))]

            # without this check, accidental broadcasts may happen if batch_size == 1
            assert torch.eq(*map(torch.as_tensor, (out.shape[1:], condition.shape))).all()

            # add condition equally and volumetrically to all stacks
            out = out + condition

        out = self.activation(out + self.bias3a)
        out = self.branch_conv3(out + self.bias3b)

        out = out * self.scale + self.bias4

        out = out + (stack if self.skip_conv is None else self.skip_conv(stack))

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):

        weight_getter = attrgetter('depth_conv.weight', 'height_conv.weight', 'width_conv.weight')

        # branch_conv1
        for weight in weight_getter(self.branch_conv1):
            torch.nn.init.normal_(
                weight,
                mean=0,
                std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
            )

        for weight in weight_getter(self.branch_conv2):
            torch.nn.init.kaiming_normal_(weight)

        # branch_conv2 & branch_conv3
        for weight in weight_getter(self.branch_conv3):
            torch.nn.init.constant_(weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            for weight in weight_getter(self.skip_conv):
                # init_method = torch.nn.init.kaiming_normal_ if not self.out else torch.nn.init.xavier_normal_
                init_method = torch.nn.init.xavier_normal_
                init_method(weight)
            bias_getter = attrgetter('depth_conv.bias', 'height_conv.bias', 'width_conv.bias')
            for bias in bias_getter(self.skip_conv):
                torch.nn.init.constant_(tensor=bias, val=0)


def tanh_glu(x: torch.Tensor, dim: int):
    a, b = torch.chunk(x, chunks=2, dim=dim)
    return torch.tanh(a) * torch.sigmoid(b)

class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        mask: str = 'B',  # options are ('A', 'B')
        condition_dim: int = 0, #
        condition_kernel_size: int = 1, # ignored if condition_dim == 0
        dropout_prob: float = 0.5,
        #Catch spurious (keyword-)arguments
        *args,
        **kwargs
    ):
        super().__init__()


        self.causal_conv = CausalConv3dAdd(
            in_channels=in_channels,
            out_channels=in_channels*2,
            kernel_size=3,
            mask=mask,
            bias=True,
        )

        self.depth_conv = nn.Conv3d(
            in_channels=in_channels*2,
            out_channels=in_channels*4,
            kernel_size=1,
            groups=2
        )

        self.height_conv = nn.Conv3d(
            in_channels=in_channels*2,
            out_channels=in_channels*2,
            kernel_size=1,
        )

        self.res_convs = nn.ModuleList(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            ) for _ in range(3)
        )

        self.condition_convs = nn.ModuleList(
            nn.Conv3d(
                in_channels=condition_dim,
                out_channels=in_channels*2,
                kernel_size=condition_kernel_size,
                padding=condition_kernel_size // 2,
            ) for _ in range(3)
        ) if condition_dim > 0 else None

        # 1x1x1 conv for channel dim projection
        # Also needed if mask == 'A', since otherwise causality is broken
        self.skip_conv = CausalConv3dAdd(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, mask=mask
        ) if mask == 'A' else None

        # self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, stack: torch.Tensor, condition: torch.Tensor = None, condition_cache = None):
        depth, height, width = self.causal_conv(stack)

        depth_cond_height, depth_cond_width = torch.chunk(self.depth_conv(depth), chunks=2, dim=1)

        # height, width = (
        #     layer + layer_cond for layer, layer_cond in zip(
        #         height_width,
        #         torch.chunk(self.depth_conv(depth), chunks=2, dim=1)
        #     )
        # )

        # since height is about to shift down, depth_cond_height needs to shift up
        height = height + shift_backwards_3d(depth_cond_height)

        width = width + shift_down_3d(self.height_conv(height)) + shift_down_3d(shift_backwards_3d(depth_cond_width))

        if condition is not None or condition_cache is not None:
            # deliberately prefer condition_cache over computing condition again
            if condition_cache is not None:
                conditions = condition_cache.popleft()
            else:
                assert self.condition_convs is not None, 'Condition projection matrix not initialised!'
                conditions = (condition_conv(condition) for condition_conv in self.condition_convs)

            depth, height, width = (
                stack_item + condition_item[(..., *(slice(dim) for dim in stack.shape[-3:]))]
                for stack_item, condition_item
                in zip((depth, height, width), conditions)
            )

        depth, height, width = map(partial(tanh_glu, dim=1), (depth, height, width))

        stack = rearrange([
            stack_item + res_conv(item)
            for stack_item, res_conv, item
            in zip(
                stack if self.skip_conv is None else self.skip_conv(stack),
                self.res_convs,
                (depth, height, width)
            )
        ], 'dim b c d h w -> dim b c d h w')

        return stack, condition, condition_cache


class CausalAttention(nn.Module):
    def __init__(self, dropout_prob=0.5, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, keys, queries, values, attn_mask):
        stack_dim, b, num_keys, *dims = keys.shape
        num_values = values.shape[2]

        nh = self.num_heads
        assert num_values % nh == 0
        assert num_keys % nh == 0

        embed_dim = math.prod(dims)
        # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
        flat_q = queries.reshape(stack_dim, b, nh, num_keys//nh, embed_dim) * (num_keys//nh) ** -0.5
        flat_k = keys.reshape(stack_dim, b, nh, num_keys//nh, embed_dim)
        flat_v = values.reshape(stack_dim, b, nh, num_values//nh, embed_dim)

        logits = torch.matmul(flat_q.transpose(3,4), flat_k)            # (B,nh,HW,dq) dot (B,nh,dq,HW) = (B,nh,HW,HW)

        if self.dropout.training:
            logits = self.dropout(logits)

            # replace dropped values with a very large negative number instead of -inf to prevent nans in the output
            logits = logits.masked_fill(logits == 0, -1e3)

        logits = logits.masked_fill(~attn_mask, float('-inf'))

        weights = F.softmax(logits, -1)

        attn_out = torch.matmul(weights, flat_v.transpose(3,4))           # (B,nh,HW,HW) dot (B,nh,HW,dvh) = (B,nh,HW,dvh)
        attn_out = attn_out.transpose(3,4)                                # (B,nh,dvh,HW)
        return attn_out.reshape(stack_dim, b, -1, *dims)                              # (B,dv,H,W)


class CausalAttentionPixelBlock(nn.Module):
    '''3D attention with causal convs'''
    def __init__(
        self,
        in_channels: int,
        bottleneck_divisor: int,
        num_layers: int,
        causal_conv: nn.Module,
        # attention specific
        num_heads: int = 8,
        attention_dropout_prob = 0.5
    ):
        super().__init__()
        # stack + out + background
        branch_channels = in_channels // bottleneck_divisor
        self.key_value_proj = CausalConv3dAdd(
            in_channels=(in_channels * 2 + 3),
            out_channels=(branch_channels * 2),
            kernel_size=1
        )
        # stack + background
        self.query_proj = CausalConv3dAdd(
            in_channels=(in_channels + 3),
            out_channels=branch_channels,
            kernel_size=1
        )

        self.causal_layers = nn.ModuleList([causal_conv() for _ in range(num_layers)])

        self.causal_attention = CausalAttention(dropout_prob=attention_dropout_prob, num_heads=num_heads)

        self.out_proj = causal_conv(aux=True)

    def forward(self, stack, background, attn_mask, condition = None, condition_cache = None):

        out = stack

        for layer in self.causal_layers:
            out = layer(out, condition=condition, condition_cache=condition)

        # stack, out, and background should have the same shape
        keys, values = torch.chunk(self.key_value_proj(torch.cat([stack, out, background], dim=2)), chunks=2, dim=2)
        queries = self.query_proj(torch.cat([out, background], dim=2))

        attn_out = self.causal_attention(queries, keys, values, attn_mask=attn_mask)

        out = self.out_proj(
            out,
            aux=attn_out,
            condition=condition,
            condition_cache=condition_cache
        )

        return out
