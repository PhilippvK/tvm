# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D schedule on x86"""

import logging

import tvm
from tvm import te
from tvm import autotvm
from tvm.contrib import dnnl
from .. import nn
from ..generic import schedule_extern
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.utils import get_pad_tuple
from ..utils import get_const_tuple, traverse_inline
from . import conv2d_avx_1x1, conv2d_avx_common

logger = logging.getLogger("topi")


def _get_default_config(
    cfg, data, kernel, strides, padding, dilation, out_dtype, is_depthwise=False, layout="NCHW"
):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype)
        from .depthwise_conv2d import _fallback_schedule

        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype, layout)
        is_kernel_1x1 = wkl.kernel_h == 1 and wkl.kernel_w == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)


@conv2d_infer_layout.register("cpu")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, _, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    out_channel, _, k_height, k_width = kernel[1]
    idxdiv = tvm.tir.indexdiv

    pt, pl, pb, pr = get_pad_tuple(padding, (k_height, k_width))
    hdilation, wdilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    dilated_kernel_h = (k_height - 1) * hdilation + 1
    dilated_kernel_w = (k_width - 1) * wdilation + 1
    out_height = idxdiv(in_height + pt + pb - dilated_kernel_h, strides[0]) + 1
    out_width = idxdiv(in_width + pl + pr - dilated_kernel_w, strides[1]) + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, idxdiv(in_channel, tile_ic), in_height, in_width, tile_ic)
    in_layout = f"NCHW{tile_ic}c"
    out_shape = (batch_size, idxdiv(out_channel, tile_oc), out_height, out_width, tile_oc)
    out_layout = f"NCHW{tile_oc}c"
    return ((in_shape, in_layout),), ((out_shape, out_layout),)


def schedule_conv2d_nhwc(outs):
    """Create schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _callback(op):
        if "conv2d_nhwc" in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            s[C].vectorize(c)

            O = output_op.output(0)
            if len(O.op.axis) == 4:  # schedule bias + bn + relu
                n, h, w, c = O.op.axis
                fused = s[O].fuse(n, h, w)
                s[O].parallel(fused)
                channels = int(O.shape[-1])
                if channels % 64 == 0:
                    c, ci = s[O].split(c, 64)
                    s[O].vectorize(ci)
                if C != O:
                    s[C].compute_at(s[O], c)

    traverse_inline(s, output_op, _callback)
    return s


def conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype):
    layout = "NCHW"
    packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, layout, out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHWc(outs)


def _pack_data(cfg, data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    # Handle dynamic shape to pass tuning dispatch.
    if isinstance(n, tvm.tir.Any):
        n = tvm.te.size_var("n")
    if isinstance(ih, tvm.tir.Any):
        ih = tvm.te.size_var("ih")
    if isinstance(iw, tvm.tir.Any):
        iw = tvm.te.size_var("iw")
    if isinstance(ic, tvm.tir.Any):
        raise RuntimeError("Dynamic input channel is not supported for conv2d.")

    data = te.compute(
        (n, ic_chunk, ih, iw, ic_bn),
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
    )

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb: kernel[occ * oc_bn + ocb, icc * ic_bn + icb, k_h, k_w],
        name="kernel_vec",
    )

    return data, kernel


@autotvm.register_topi_compute("conv2d_NCHWc.x86")
def conv2d_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape
        )
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    if isinstance(ow, (tvm.tir.IntImm, int)):
        cfg.define_split(
            "tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64, policy="verbose"
        )
    if is_kernel_1x1:
        if isinstance(oh, (tvm.tir.IntImm, int)):
            cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            dilation,
            out_dtype,
        )

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, in_channel // cfg["tile_ic"].size[-1], ih, iw, cfg["tile_ic"].size[-1])
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                num_filter // cfg["tile_oc"].size[-1],
                in_channel // cfg["tile_ic"].size[-1],
                kernel_height,
                kernel_width,
                cfg["tile_ic"].size[-1],
                cfg["tile_oc"].size[-1],
            )
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)


@autotvm.register_topi_schedule("conv2d_NCHWc.x86")
def schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            (_, _, kh, kw, _, _) = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchw_dnnl.x86")
def conv2d_nchw_dnnl(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d in NCHW format using dnnl."""
    groups = 1
    _out = dnnl.dnnl_conv2d(data, kernel, strides, padding, dilation, groups, False, out_dtype)
    return _out


@autotvm.register_topi_schedule("conv2d_nchw_dnnl.x86")
def schedule_conv2d_nchw_dnnl(_, outs):
    """Create schedule for conv2d_nchw_dnnl"""
    return schedule_extern(outs)


@autotvm.register_topi_compute("conv2d_nhwc_dnnl.x86")
def conv2d_nhwc_dnnl(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d in NHWC format using dnnl."""
    groups = 1
    _out = dnnl.dnnl_conv2d(data, kernel, strides, padding, dilation, groups, True, out_dtype)
    return _out


@autotvm.register_topi_schedule("conv2d_nhwc_dnnl.x86")
def schedule_conv2d_nhwc_dnnl(_, outs):
    """Create schedule for conv2d_nhwc_dnnl"""
    return schedule_extern(outs)

# NEW
from tvm import autotvm
from tvm.autotvm.task import deserialize_args
from tvm import te
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm.tir.expr import Mul

import random
import string
from typing import Callable, Tuple, Union

import tvm
from tvm import te
from tvm.tir import indexdiv, indexmod
from tvm.topi.utils import traverse_inline
from tvm.topi.nn.pad import pad

def conv2d_nhwc_hwoi_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TODO"""
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_size, in_height, in_width, in_channels = data.shape
    kernel_h, kernel_w, out_channels, _ = kernel.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    rc = te.reduce_axis((0, in_channels), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    conv = te.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: te.sum(
            padded_data[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * kernel[ry, rx, ff, rc].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="conv2d",
        tag="conv2d_nhwc",
    )

    ###########################
    # Config Space Definition #
    ###########################
    # n, oh, ow, co = (
    #     cfg.axis(batch_size.value),
    #     cfg.axis(out_height.value),
    #     cfg.axis(out_width.value),
    #     cfg.axis(out_channels.value),
    # )
    # kh, kw, ci = (
    #     cfg.reduce_axis(kernel_h.value),
    #     cfg.reduce_axis(kernel_w.value),
    #     cfg.reduce_axis(in_channels.value),
    # )

    # owo, owi = cfg.define_split("tile_ow", ow, policy="factors", num_outputs=2)
    # cio, cii = cfg.define_split(
    #     "tile_ci",
    #     ci,
    #     policy="factors",
    #     num_outputs=2,
    #     # TODO: check case with in_channels.value % 4 != 0 with AutoTVM
    #     filter=None if cfg.is_fallback else lambda x: x.size[-1] % 4 == 0,
    # )
    # coo, coi = cfg.define_split("tile_co", co, policy="factors", num_outputs=2)

    # cfg.define_reorder(
    #     "reorder_0_simd",
    #     [n, oh, owo, owi, coo, coi, kh, kw, cio, cii],
    #     policy="candidate",
    #     candidate=[
    #         [n, oh, kh, kw, owo, coo, cio, owi, coi, cii],
    #         [n, oh, kh, kw, coo, owo, cio, owi, coi, cii],
    #         [n, kh, kw, oh, owo, coo, cio, owi, coi, cii],
    #         [n, kh, kw, oh, coo, owo, cio, owi, coi, cii],
    #     ],
    # )

    # cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    # cfg.define_knob("unroll_explicit", [0, 1])

    # if cfg.is_fallback:
    #     cfg.fallback_split("tile_ow", [-1, out_width.value])
    #     cfg.fallback_split("tile_ci", [-1, in_channels.value])
    #     cfg.fallback_split("tile_co", [-1, out_channels.value])

    return conv


def conv2d_nhwc_hwoi_schedule(cfg, outs):
    """TODO"""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc" not in op.tag:
            return

    traverse_inline(sched, outs[-1].op, _callback)
    return sched

def conv2d_nhwc_hwoi(*args, **kwargs):
    """TODO"""
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    layout = args[-2]
    cfg = autotvm.get_config()
    args = [cfg] + args
    assert layout == "NHWC"
    conv = conv2d_nhwc_hwoi_compute(*args)
    sched = conv2d_nhwc_hwoi_schedule(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


conv2d_nhwc_hwoi.template_key = "dsp"
conv2d_nhwc_hwoi.default_data_layout = "NHWC"
conv2d_nhwc_hwoi.default_kernel_layout = "HWOI"

@autotvm.register_topi_compute("conv2d_nhwc_hwoi.x86")
def conv2d_nhwc_hwoi(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TODO"""
    return conv2d_nhwc_hwoi_compute(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_hwoi.x86")
def schedule_conv2d_nhwc_hwoi(cfg, outs):
    """TODO"""
    return conv2d_nhwc_hwoi_schedule(cfg, outs)


def conv2d_nhwc_ohwi_schedule(cfg, outs):
    """TODO"""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc" not in op.tag:
            return

    traverse_inline(sched, outs[-1].op, _callback)
    return sched

def _unpack_2d_argument(argument: Union[int, Tuple]) -> Tuple:
    if isinstance(argument, int):
        return (argument, argument)
    assert len(argument) == 2
    return argument


def _check_no_dilation(dilation: Union[int, Tuple]) -> None:
    """Takes a dilation argument as an integer or tuple, and makes sure both dimensions are 1.
    Dilation prevents us from using DSP instructions, so this schedule can't work (aside from the
    niche case where dilation_h == stride_h and dilation_w == stride_w, which is rare enough we
    probably don't need to support it)."""

    dilation_h, dilation_w = _unpack_2d_argument(dilation)
    assert dilation_h == dilation_w == 1


def _unpack_padding(padding: Tuple) -> Tuple:
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        (pad_up, pad_down), (pad_left, pad_right) = padding
    else:
        pad_up, pad_left, pad_down, pad_right = padding
    return pad_up, pad_left, pad_down, pad_right


def _pad_if_needed(data: te.tensor.Tensor, layout: str, padding: Tuple) -> te.tensor.Tensor:
    """Performs padding on a te.tensor.Tensor object if necessary. If padding = (0, 0, 0, 0), the
    input tensor is returned unmodified. We only care about tuples here - "VALID" and "SAME" padding
    will be converted by the importer TFLite importer if present."""

    pad_up, pad_left, pad_down, pad_right = padding
    if not any(padding):
        return data

    # We want to pad the "H" and "W" columns, and their position depends on the layout
    pad_before, pad_after = [0, 0, 0, 0], [0, 0, 0, 0]
    pad_before[layout.index("H")] = pad_up
    pad_before[layout.index("W")] = pad_left
    pad_after[layout.index("H")] = pad_down
    pad_after[layout.index("W")] = pad_right
    return pad(data, pad_before, pad_after, name="padded_data")


def _compute_output_dim(
    data_dim: int, kernel_dim: int, pad_before: int, pad_after: int, stride: int
) -> int:
    """Computes an output dimension of a convolution, given the data dimension, kernel dimension,
    padding, and stride along that axis. Note that when stride > 1, this division will often not
    be perfectly even."""
    return (data_dim + pad_before + pad_after - kernel_dim) // stride + 1


def _wrap_te_compute(
    shape: Tuple,
    fcompute: Callable[[int, int, int, int], tvm.ir.PrimExpr],
    desired_out_layout: str,
    current_out_layout: str = "NHWC",
    **kwargs,
) -> te.tensor.Tensor:
    """Wrapper over te.compute that allows the output layout to be easily changed."""
    assert current_out_layout.isalpha() and desired_out_layout.isalpha()
    assert sorted(current_out_layout) == sorted(desired_out_layout)
    forward_order = (current_out_layout.index(c) for c in desired_out_layout)
    reverse_order = (desired_out_layout.index(c) for c in current_out_layout)

    return te.compute(
        tuple(shape[i] for i in forward_order),
        lambda *args: fcompute(*(args[i] for i in reverse_order)),
        **kwargs,
    )


def _get_suffix() -> str:
    """Returns a random eight-character string to append to C function names. Prevents accidental
    re-definition of functions if the same operator appears twice in a Relay graph."""
    return "".join(random.choices(string.ascii_uppercase, k=8))


def conv2d_nhwc_ohwi_compute(
    _cfg, data, kernel, strides, padding, dilation, out_layout, out_dtype
):
    """TODO"""

    stride_h, stride_w = _unpack_2d_argument(strides)
    pad_up, pad_left, pad_down, pad_right = _unpack_padding(padding)
    _check_no_dilation(dilation)
    # TODO: support dilation

    batch_size, data_h, data_w, in_channels = data.shape
    output_channels, kernel_h, kernel_w, _ = kernel.shape
    assert kernel.shape[3] == in_channels

    output_h = _compute_output_dim(data_h, kernel_h, pad_up, pad_down, stride_h)
    output_w = _compute_output_dim(data_w, kernel_w, pad_left, pad_right, stride_w)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    kc_i = te.reduce_axis((0, in_channels), name="rc")

    padded_data = _pad_if_needed(data, "NHWC", (pad_up, pad_left, pad_down, pad_right))
    # return _wrap_te_compute(
    #     (batch_size, output_h, output_w, output_channels),
    #     lambda n, y, x, c: te.sum(
    #         padded_data[n, y * stride_h + kh_i, x * stride_w + kw_i, kc_i].astype(out_dtype)
    #         * kernel[c, kh_i, kw_i, kc_i].astype(out_dtype),
    #         axis=(kh_i, kw_i, kc_i),
    #     ),
    #     out_layout,
    #     name="conv2d",
    #     tag="conv2d_nhwc_ohwi_dsp",
    # )
    conv = te.compute(
        (batch_size, output_h, output_w, output_channels),
        lambda n, y, x, c: te.sum(
            padded_data[
                n, y * stride_h + kh_i, x * stride_w + kw_i * 1, kc_i
            ].astype(out_dtype)
            * kernel[c, kh_i, kw_i, kc_i].astype(out_dtype),
            axis=[kh_i, kw_i, kc_i],
        ),
        name="conv2d",
        tag="conv2d_nhwc_ohwi",
    )
    return conv


@autotvm.register_topi_compute("conv2d_nhwc_ohwi.x86")
def conv2d_nhwc_ohwi(cfg, data, kernel, strides, padding, dilation, out_layout, out_dtype):
    """TODO"""
    return conv2d_nhwc_ohwi_compute(
        cfg, data, kernel, strides, padding, dilation, out_layout, out_dtype
    )


@autotvm.register_topi_schedule("conv2d_nhwc_ohwi.x86")
def schedule_conv2d_nhwc_ohwi(cfg, outs):
    """TODO"""
    return conv2d_nhwc_ohwi_schedule(cfg, outs)


# FIXME - https://github.com/apache/tvm/issues/4122
# _declaration_conv_nhwc_pack expects kernel layout to be HWOI. However, the tests use HWIO
# layout. Commenting until we have clarity about the nhwc_pack implementation from the author.
# elif layout == 'NHWC' and kh == 1 and kw == 1 and kernel.dtype == "int8":
#     if cfg.is_fallback:
#         _get_default_config(cfg, data, kernel, strides, padding, out_dtype, False, layout)
#     # specialize for INT8 1X1 conv on X86
#     return conv2d_avx_1x1._declaration_conv_nhwc_pack(cfg, data, kernel, strides,
#                                                       padding, dilation, out_dtype)
