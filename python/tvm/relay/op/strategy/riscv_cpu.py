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
"""Definition of RISC-V CPU operator strategy."""
import logging

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re

from tvm import relay, topi

from ....auto_scheduler import is_auto_scheduler_enabled
from ....meta_schedule import is_meta_schedule_enabled
from ....target import riscv_isa
from ....topi.generic import conv2d as conv2d_generic
from .. import op as _op
from .generic import *

logger = logging.getLogger("strategy")


# TODO: test the following strategies (ARM vs. Intel vs. Generic)
# @schedule_reduce.register("riscv_cpu")
# def schedule_reduce_riscv_cpu(attrs, outs, target):
#     """schedule reduction ops for riscv cpu"""
#     with target:
#         return topi.x86.schedule_reduce(outs)
#
#
# @schedule_injective.register("riscv_cpu")
# def schedule_injective_riscv_cpu(_, outs, target):
#     """schedule injective ops for riscv cpu"""
#     with target:
#         return topi.riscv_cpu.schedule_injective(outs)
#
#
# @schedule_concatenate.register("riscv_cpu")
# def schedule_concatenate_riscv_cpu(_, outs, target):
#     """schedule concatenate for riscv cpu"""
#     with target:
#         return topi.riscv_cpu.schedule_concatenate(outs)


@schedule_pool.register(["riscv_cpu"])
def schedule_pool_riscv_cpu(attrs, outs, target):
    """schedule pooling ops riscv cpu"""
    layout = attrs.layout
    isa = riscv_isa.IsaAnalyzer(target)
    avg_pool = isinstance(attrs, relay.op.op_attrs.AvgPool2DAttrs)
    with target:
        if (
            avg_pool
            and isa.has_vext
            and layout in ("NCW", "NCHW")
            or not avg_pool
            and isa.has_vext
            and layout in ("NWC", "NHWC")
        ):
            return topi.riscv_cpu.schedule_pool_vext(outs, layout)
        elif (
            avg_pool
            and isa.has_pext
            and layout in ("NCW", "NCHW")
            or not avg_pool
            and isa.has_pext
            and layout in ("NWC", "NHWC")
        ):
            return topi.riscv_cpu.schedule_pool_pext(outs, layout)
        logger.warning("pool is not optimized for riscv cpu.")
        return topi.generic.schedule_pool(outs, layout)


@conv2d_strategy.register("riscv_cpu")
def conv2d_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """conv2d riscv cpu strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    isa = riscv_isa.IsaAnalyzer(target)

    if groups == 1:
        if layout == "NCHW":
            # Only tested cases are commented in
            if kernel_layout == "OIHW":
                if (
                    topi.riscv_cpu.is_int8_hw_support(data.dtype, kernel.dtype)  # TODO: or isa.has_pext
                    and kernel.shape[1] >= 64  # TODO:  why?
                ):
                    pass
                    # strategy.add_implementation(
                    #     wrap_compute_conv2d(topi.riscv_cpu.conv2d_nchw_int8),
                    #     wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nchw_int8),
                    #     name="conv2d_nchw_int8.riscv_cpu",
                    #     plevel=15,
                    # )
                else:
                    pass
                    # ARM conv2d spatial pack schedule.
                    # strategy.add_implementation(
                    #     wrap_compute_conv2d(topi.riscv_cpu.conv2d_nchw_spatial_pack),
                    #     wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nchw_spatial_pack),
                    #     name="conv2d_nchw_spatial_pack.riscv_cpu",
                    #     plevel=10,
                    # )

                    # strategy.add_implementation(
                    #     wrap_compute_conv2d(topi.x86.conv2d_nchw),
                    #     wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
                    #     name="conv2d_nchw.x86",
                    # )
                    # TODO: implement winograd for riscv
            elif re.match(r"OIHW\d*o", kernel_layout):
                pass
                # strategy.add_implementation(
                #     wrap_compute_conv2d(topi.riscv_cpu.conv2d_nchw_spatial_pack),
                #     wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nchw_spatial_pack),
                #     name="conv2d_nchw_spatial_pack.riscv_cpu",
                # )
            else:
                raise RuntimeError(
                    "Unsupported weight layout {} for conv2d NCHW".format(kernel_layout)
                )
        elif layout == "HWCN":
            pass
            # assert kernel_layout == "HWIO"
            # logger.warning("conv2d_hwcn is not optimized for riscv cpu.")
            # strategy.add_implementation(
            #     wrap_compute_conv2d(topi.nn.conv2d_hwcn),
            #     wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
            #     name="conv2d_hwcn.generic",
            # )
        elif layout == "NHWC":
            if isa.has_vext and kernel_layout == "HWOI":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv_cpu.conv2d_nhwc_vext),
                    wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nhwc_vext),
                    name="conv2d_nhwc_vext.riscv_cpu",
                )
            elif isa.has_pext and kernel_layout == "HWOI":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv_cpu.conv2d_nhwc_pext),
                    wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nhwc_pext),
                    name="conv2d_nhwc_pext.riscv_cpu",
                )
            elif kernel_layout == "HWIO":
                pass
                # is_riscv64 = topi.riscv_cpu.riscv_utils.is_riscv64()
                # has_dot_prod = topi.riscv_cpu.riscv_utils.is_dotprod_available()
                # if has_dot_prod and data.dtype in ["int8", "uint8"]:
                #     strategy.add_implementation(
                #         wrap_compute_conv2d(topi.riscv_cpu.compute_conv2d_NHWC_quantized_native),
                #         wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_NHWC_quantized_native),
                #         name="conv2d_NHWC_quantized_native.riscv_cpu",
                #     )
                # if is_aarch64 and data.dtype in ["int8", "uint8"]:
                #     strategy.add_implementation(
                #         wrap_compute_conv2d(topi.riscv_cpu.compute_conv2d_NHWC_quantized_interleaved),
                #         wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_NHWC_quantized_interleaved),
                #         name="conv2d_NHWC_quantized_interleaved.riscv_cpu",
                #     )
                # if (not is_aarch64) or (data.dtype not in ["int8", "uint8"]):
                #     # TODO(@giuseros)
                #     # This strategy errors out for quantized data types when tuning.
                #     # Let's use this only for non-aarch64 or non-quantized cases
                #     strategy.add_implementation(
                #         wrap_compute_conv2d(topi.riscv_cpu.conv2d_nhwc_spatial_pack),
                #         wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nhwc_spatial_pack),
                #         name="conv2d_nhwc_spatial_pack.riscv_cpu",
                #     )
            else:
                raise RuntimeError(
                    "Unsupported kernel layout {} for conv2d NHWC".format(kernel_layout)
                )

        else:
            raise RuntimeError("Unsupported conv2d layout {} for riscv cpu".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW" or re.match(r"OIHW\d*o", kernel_layout)
            # RISC-V conv2d depthwise schedule
            if kernel_layout == "OIHW":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv_cpu.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.riscv_cpu.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.riscv_cpu",
                )
            else:
                logger.warning(f"depthwise_conv2d with data layout NCHW and kernel layout {kernel_layout} is not optimized for riscv cpu.")
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                    wrap_topi_schedule(conv2d_generic.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nhwc.generic",
                )

            # Intel x86 depthwise conv2d schedule.
            # channel_multiplier = get_const_tuple(inputs[1].shape)[1]
            # if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
            #     strategy.add_implementation(
            #         wrap_compute_conv2d(topi.x86.depthwise_conv2d_nchw),
            #         wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_nchw),
            #         name="depthwise_conv2d_nchw.x86",
            #     )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            is_riscv64 = topi.riscv_cpu.riscv_utils.is_riscv64()
            if is_riscv64 or "+?" in target.mattr:
                pass
                # strategy.add_implementation(
                #     wrap_compute_conv2d(topi.riscv_cpu.compute_depthwise_conv2d_nhwc),
                #     wrap_topi_schedule(topi.riscv_cpu.schedule_depthwise_conv2d_nhwc),
                #     name="depthwise_conv2d_nhwc.riscv_cpu",
                # )

            # Optimized special case depthwiseConv2D operation. Requires a 3x3 kernel, a
            # NHWC layout, a HWOI kernel layout (which we rearrange), no dilation, int8 inputs,
            # int32 output, the same number of input and output channels, and for that channel
            # count to be divisible by 4. Additional work could remove these restrictions.

            elif (
                isa.has_pext
                and kernel.shape[0] == kernel.shape[1] == 3
                and dilation_w == dilation_h == 1
                and kernel.shape[3] == 1  # channel_multiplier == 1
                and data.dtype == "int8"
                and out_type.dtype == "int32"
                and data.shape[3] % 4 == 0
                and (padding != "SAME" or data.shape[1] % stride_h == data.shape[2] % stride_w == 0)
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv_cpu.depthwise_conv2d_nhwc_pext),
                    wrap_topi_schedule(topi.riscv_cpu.schedule_depthwise_conv2d_nhwc_pext),
                    name="depthwise_conv2d_nhwc_pext.riscv_cpu",
                )

            else:
                logger.warning("depthwise_conv2d with layout NHWC is not optimized for riscv cpu.")
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                    wrap_topi_schedule(conv2d_generic.schedule_depthwise_conv2d_nhwc),
                    name="depthwise_conv2d_nhwc.generic",
                )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {} for riscv cpu".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            # TODO
            # strategy.add_implementation(
            #     wrap_compute_conv2d(topi.riscv_cpu.group_conv2d_nchw, has_groups=True),
            #     wrap_topi_schedule(topi.riscv_cpu.schedule_group_conv2d_nchw),
            #     name="group_conv2d_nchw.riscv_cpu",
            # )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            logger.warning("group_conv2d with layout NHWC is not optimized for riscv cpu.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {} for riscv cpu".format(layout))
    return strategy


@conv2d_NCHWc_strategy.register("riscv_cpu")
def conv2d_NCHWc_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc adopted from x86"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    if topi.riscv_cpu.is_int8_hw_support(data.dtype, kernel.dtype):
        pass
        # strategy.add_implementation(
        #     wrap_compute_conv2d(topi.riscv_cpu.conv2d_NCHWc_int8, True, True),
        #     wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_NCHWc_int8),
        #     name="conv2d_NCHWc_int8.riscv_cpu",
        # )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.x86.conv2d_NCHWc, True, True),
            wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.x86",
        )
    return strategy


@depthwise_conv2d_NCHWc_strategy.register("riscv_cpu")
def depthwise_conv2d_NCHWc_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d_NCHWc adopted from x86"""
    # TODO: int8 hw support?
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.x86.depthwise_conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_NCHWc),
        name="depthwise_conv2d_NCHWc.x86",
    )
    return strategy


def wrap_compute_conv2d_gemm(topi_compute):
    """wrap topi compute for conv2d_gemm"""

    def _compute_conv2d_gemm(attrs, inputs, out_type):
        padding = attrs.get_int_tuple("padding")
        strides = attrs.get_int_tuple("strides")
        dilation = attrs.get_int_tuple("dilation")
        out_dtype = attrs.get_str("out_dtype")
        channels = attrs["channels"]
        kernel_size = attrs["kernel_size"]
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        return [
            topi_compute(
                inputs[0], inputs[1], strides, padding, dilation, out_dtype, kernel_size, channels
            )
        ]

    return _compute_conv2d_gemm


@conv2d_gemm_without_weight_transform_strategy.register("riscv_cpu")
def conv2d_gemm_without_weight_transform_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom riscv cpu strategy"""
    layout = attrs.data_layout
    data = inputs[0]
    strategy = _op.OpStrategy()

    interleaved_compute = topi.riscv_cpu.compute_conv2d_NHWC_quantized_interleaved_without_transform
    native_compute = topi.riscv_cpu.compute_conv2d_NHWC_quantized_native_without_transform
    if layout == "NHWC" and data.dtype in ["int8", "uint8"]:
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(native_compute),
            wrap_topi_schedule(
                topi.riscv_cpu.schedule_conv2d_NHWC_quantized_native_without_transform
            ),
            name="conv2d_NHWC_quantized_native_without_transform.riscv_cpu",
        )
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(interleaved_compute),
            wrap_topi_schedule(
                topi.riscv_cpu.schedule_conv2d_NHWC_quantized_interleaved_without_transform
            ),
            name="conv2d_NHWC_quantized_interleaved_without_transform.riscv_cpu",
        )
    else:
        raise RuntimeError(
            "Unsupported conv2d_NHWC_quantized_without_transform layout {0}"
            "with datatype {1}".format(layout, data.dtype)
        )
    return strategy


# @conv2d_transpose_strategy.register("riscv_cpu")
# def conv2d_transpose_strategy_riscv_cpu(attrs, inputs, out_type, target):
#     """conv2d_transpose riscv cpu strategy"""
#     layout = attrs.data_layout
#     dilation = get_const_tuple(attrs.dilation)
#     groups = attrs.groups
#     assert layout == "NCHW", "only support nchw for now"
#     assert dilation == (1, 1), "not support dilate now"
#     assert groups == 1, "only support groups == 1 for now"
#     strategy = _op.OpStrategy()
#     strategy.add_implementation(
#         wrap_compute_conv2d_transpose(topi.riscv_cpu.conv2d_transpose_nchw),
#         wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_transpose_nchw),
#         name="conv2d_tranpose_nchw.riscv_cpu",
#     )
#     return strategy


@dense_strategy.register(["riscv_cpu"])
def schedule_dense_riscv_cpu(attrs, inputs, out_type, target):
    """dense riscv cpu strategy"""
    strategy = _op.OpStrategy()
    isa = riscv_isa.IsaAnalyzer(target)
    if isa.has_vext:
        strategy.add_implementation(
            wrap_compute_dense(topi.riscv_cpu.dense_vext),
            wrap_topi_schedule(topi.riscv_cpu.schedule_dense_vext),
            name="dense_vext.riscv_cpu",
        )
    elif isa.has_pext:
        strategy.add_implementation(
            wrap_compute_dense(topi.riscv_cpu.dense_pext),
            wrap_topi_schedule(topi.riscv_cpu.schedule_dense_pext),
            name="dense_pext.riscv_cpu",
        )
    else:
        logger.warning("dense is not optimized for riscv cpu.")
        strategy.add_implementation(
            wrap_compute_dense(
                topi.nn.dense,
                need_auto_scheduler_layout=is_auto_scheduler_enabled(),
                need_meta_schedule_layout=is_meta_schedule_enabled(),
            ),
            wrap_topi_schedule(topi.generic.schedule_dense),
            name="dense.generic",
        )
    return strategy


# @conv1d_strategy.register("riscv_cpu")
# def conv1d_strategy_riscv_cpu(attrs, inputs, out_type, target):
#     """conv1d strategy"""
#     strategy = _op.OpStrategy()
#     layout = attrs.data_layout
#     kernel_layout = attrs.kernel_layout
#     dilation = get_const_tuple(attrs.dilation)
#     if dilation[0] < 1:
#         raise ValueError("dilation should be a positive value")
#
#     isa = riscv_isa.IsaAnalyzer(target)
#
#     if kernel_layout == "WOI":
#         if layout == "NWC" and isa.has_pext:
#             strategy.add_implementation(
#                 wrap_compute_conv1d(topi.riscv_cpu.conv1d_nwc_pext),
#                 wrap_topi_schedule(topi.riscv_cpu.schedule_conv1d_nwc_pext),
#                 name="conv1d_pext",
#             )
#         else:
#             raise RuntimeError(
#                 "Unsupported kernel layout {} for conv1d {} for riscv cpu.".format(
#                     kernel_layout, layout
#                 )
#             )
#     elif layout == "NCW":
#         logger.warning("conv1d with layout %s is not optimized for riscv cpu.", layout)
#         strategy.add_implementation(
#             wrap_compute_conv1d(topi.nn.conv1d_ncw),
#             wrap_topi_schedule(topi.generic.schedule_conv1d_ncw),
#             name="conv1d_ncw.generic",
#         )
#     elif layout == "NWC":
#         logger.warning("conv1d with layout %s is not optimized for riscv cpu.", layout)
#         strategy.add_implementation(
#             wrap_compute_conv1d(topi.nn.conv1d_nwc),
#             wrap_topi_schedule(topi.generic.schedule_conv1d_nwc),
#             name="conv1d_nwc.generic",
#         )
#     else:
#         raise RuntimeError(
#             "Unsupported kernel layout {} for conv1d {} for riscv cpu.".format(kernel_layout, layout)
#         )
#     return strategy
