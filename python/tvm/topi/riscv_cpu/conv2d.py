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
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument, import-outside-toplevel
"""Conv2D schedule for RISC-V CPU"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from tvm import autotvm
import tvm.contrib.nnpack

from ..utils import traverse_inline, get_const_tuple
from .. import nn
from ..nn.utils import get_const_int, get_pad_tuple
# from .conv2d_spatial_pack import (
#     conv2d_spatial_pack_nchw,
#     conv2d_spatial_pack_nhwc,
#     schedule_conv2d_spatial_pack_nchw,
#     schedule_conv2d_spatial_pack_nhwc,
# )
from .extensions.pext.conv2d import (
    conv2d_nhwc_pext_compute,
    conv2d_nhwc_pext_schedule,
)
from .extensions.vext.conv2d import (
    conv2d_nhwc_vext_compute,
    conv2d_nhwc_vext_schedule,
)


# @autotvm.register_topi_compute("conv2d_nchw_spatial_pack.riscv_cpu")
# def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
#     """Compute conv2d with NCHW layout"""
#     return conv2d_spatial_pack_nchw(
#         cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=2
#     )


# @autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.riscv_cpu")
# def schedule_conv2d_nchw_spatial_pack(cfg, outs):
#     """Create schedule for conv2d_nchw"""
#     s = te.create_schedule([x.op for x in outs])
# 
#     def _callback(op):
#         # schedule conv2d
#         if "spatial_conv2d_output" in op.tag:
#             output = op.output(0)
#             conv = op.input_tensors[0]
# 
#             data_vec = conv.op.input_tensors[0]
#             data_pad = data_vec.op.input_tensors[0]
#             s[data_pad].compute_inline()
# 
#             kernel_vec = conv.op.input_tensors[1]
#             if kernel_vec.op.name == "kernel_vec":
#                 kernel = kernel_vec.op.input_tensors[0]
#             else:
#                 kernel = kernel_vec
#             if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
#                 s[kernel].compute_inline()
# 
#             schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec, conv, output, outs[0])
# 
#     traverse_inline(s, outs[0].op, _callback)
#     return s


# @autotvm.register_topi_compute("conv2d_nhwc_spatial_pack.riscv_cpu")
# def conv2d_nhwc_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
#     """Compute conv2d with NHWC layout"""
#     return conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype)


# @autotvm.register_topi_schedule("conv2d_nhwc_spatial_pack.riscv_cpu")
# def schedule_conv2d_nhwc_spatial_pack(cfg, outs):
#     """Create schedule for conv2d_nhwc"""
#     s = te.create_schedule([x.op for x in outs])
#
#     def _callback(op):
#         if "spatial_conv_output_NHWC" in op.tag:
#             schedule_conv2d_spatial_pack_nhwc(cfg, s, op, outs[0])
#
#     traverse_inline(s, outs[0].op, _callback)
#     return s


@autotvm.register_topi_compute("conv2d_nhwc_pext.riscv_cpu")
def conv2d_nhwc_pext(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nhwc with packed extension."""
    return conv2d_nhwc_pext_compute(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_pext.riscv_cpu")
def schedule_conv2d_nhwc_pext(cfg, outs):
    """Create schedule for conv2d_nhwc_pext"""
    return conv2d_nhwc_pext_schedule(cfg, outs)

@autotvm.register_topi_compute("conv2d_nhwc_vext.riscv_cpu")
def conv2d_nhwc_vext(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d_nhwc with vector extension."""
    return conv2d_nhwc_vext_compute(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_vext.riscv_cpu")
def schedule_conv2d_nhwc_vext(cfg, outs):
    """Create schedule for conv2d_nhwc_vext"""
    return conv2d_nhwc_vext_schedule(cfg, outs)
