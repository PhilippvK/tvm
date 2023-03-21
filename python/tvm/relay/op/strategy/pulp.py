import re
import logging

from .generic import *
from tvm import tir, topi
from .. import op as _op
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.meta_schedule import is_meta_schedule_enabled

logger = logging.getLogger(__name__)

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")

# @conv2d_strategy.register("pulp")
# def conv2d_strategy(attrs, inputs, out_type, target):
#     print("conv2d_strategy")
#     """conv2d pulp strategy"""
#
#     logger.info("Registering strategy for conv2d")
#     logger.info("attrs:")
#     for k in attrs.keys():
#         logger.info("  %s: %s", str(k), str(attrs[k]))
#     logger.info("inputs: %s", str(inputs))
#     logger.info("out_type: %s", str(out_type))
#     logger.info("target: %s", str(target))
#
#     strategy = _op.OpStrategy()
#     data, kernel = inputs
#     dilation = get_const_tuple(attrs.dilation)
#     groups = attrs.groups
#     layout = attrs.data_layout
#     kernel_layout = attrs.kernel_layout
#     (dilation_h, dilation_w) = dilation
#     if dilation_h < 1 or dilation_w < 1:
#         raise ValueError("dilation should be positive value")
#
#     if groups == 1:
#         # if layout == "NCHW":
#         #     assert kernel_layout == "OIHW"
#         #     print("A1")
#         #     strategy.add_implementation(
#         #         wrap_compute_conv2d(topi.pulp.conv2d_nchw),
#         #         wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw),
#         #         name="conv2d_nchw.pulp",
#         #     )
#         if layout == "NCHW":
#             assert kernel_layout == "OIHW"
#             # if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
#             if False:
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.pulp.conv2d_nchw_int8),
#                     wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw_int8),
#                     name="conv2d_nchw_int8.pulp",
#                 )
#             else:
#                 print("oooooooo")
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.pulp.conv2d_nchw),
#                     wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw),
#                     name="conv2d_nchw.pulp",
#                 )
#                 print("?.")
#         elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
#             print("match NCHWxc")
#             assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
#             return conv2d_NCHWc_strategy_pulp(attrs, inputs, out_type, target)
#         elif layout == "NHWC":
#             raise NotImplementedError
#             assert kernel_layout == "HWIO" or kernel_layout == "OHWI"
#             if kernel_layout == "HWIO":
#                 print("A2a")
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.pulp.conv2d_nhwc),
#                     wrap_topi_schedule(topi.pulp.schedule_conv2d_nhwc),
#                     name="conv2d_nhwc.pulp",
#                 )
#             else:
#                 print("A2b")
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.pulp.conv2d_nhwc_ohwi),
#                     wrap_topi_schedule(topi.pulp.schedule_conv2d_nhwc_ohwi),
#                     name="conv2d_nhwc_ohwi.pulp"
#                 )
#         elif layout == "HWCN":
#             raise NotImplementedError
#             assert kernel_layout == "HWIO"
#             print("A3")
#             strategy.add_implementation(
#                 wrap_compute_conv2d(topi.nn.conv2d_hwcn),
#                 wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
#                 name="conv2d_hwcn.generic",
#             )
#         else:
#             print("A4")
#             raise RuntimeError("Unsupported conv2d layout {}".format(layout))
#     elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
#         if layout == "NCHW":
#             # raise NotImplementedError
#             # print("B1")
#             # assert kernel_layout == "OIHW"
#             # strategy.add_implementation(
#             #     wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
#             #     wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
#             #     name="depthwise_conv2d_nchw.generic",
#             # )
#             assert kernel_layout == "OIHW"
#             channel_multiplier = get_const_tuple(inputs[1].shape)[1]
#             if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.pulp.depthwise_conv2d_nchw),
#                     wrap_topi_schedule(topi.pulp.schedule_depthwise_conv2d_nchw),
#                     name="depthwise_conv2d_nchw.pulp",
#                 )
#             else:
#                 logger.warning(
#                     "For pulp target, depthwise_conv2d with channel "
#                     "multiplier greater than 1 is not optimized"
#                 )
#                 strategy.add_implementation(
#                     wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
#                     wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
#                     name="depthwise_conv2d_nchw.generic",
#                 )
#         elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
#             assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
#             return depthwise_conv2d_NCHWc_strategy_pulp(attrs, inputs, out_type, target)
#         elif layout == "NHWC":
#             raise NotImplementedError
#             print("B2")
#             assert kernel_layout == "HWOI"
#             strategy.add_implementation(
#                 wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
#                 wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc),
#                 name="depthwise_conv2d_nhwc.generic",
#             )
#         else:
#             print("B3")
#             raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
#     else:  # group_conv2d
#         if layout == "NCHW":
#             raise NotImplementedError
#             assert kernel_layout == "OIHW"
#             print("C1")
#             strategy.add_implementation(
#                 wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
#                 wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw),
#                 name="group_conv2d_nchw.generic",
#             )
#         elif layout == "NHWC":
#             raise NotImplementedError
#             assert kernel_layout == "HWIO"
#             print("C2")
#             strategy.add_implementation(
#                 wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
#                 wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
#                 name="group_conv2d_nhwc.generic",
#             )
#         else:
#             print("C3")
#             raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
#     return strategy
#
# @conv2d_NCHWc_strategy.register("pulp")
# def conv2d_NCHWc_strategy_pulp(attrs, inputs, out_type, target):
#     print("pulp: conv2d_NCHWc_strategy_pulp")
#     """conv2d_NCHWc pulp strategy"""
#     strategy = _op.OpStrategy()
#     data, kernel = inputs
#     # if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
#     if False:
#         print("pulp: is_int8_hw_support")
#         strategy.add_implementation(
#             wrap_compute_conv2d(
#                 topi.pulp.conv2d_NCHWc_int8, need_data_layout=True, need_out_layout=True
#             ),
#             wrap_topi_schedule(topi.pulp.schedule_conv2d_NCHWc_int8),
#             name="conv2d_NCHWc_int8.pulp",
#         )
#     else:
#         strategy.add_implementation(
#             wrap_compute_conv2d(topi.pulp.conv2d_NCHWc, need_data_layout=True, need_out_layout=True),
#             wrap_topi_schedule(topi.pulp.schedule_conv2d_NCHWc),
#             name="conv2d_NCHWc.pulp",
#         )
#     return strategy
#
#
# @depthwise_conv2d_NCHWc_strategy.register("pulp")
# def depthwise_conv2d_NCHWc_strategy_pulp(attrs, inputs, out_type, target):
#     """depthwise_conv2d pulp strategy"""
#     strategy = _op.OpStrategy()
#     strategy.add_implementation(
#         wrap_compute_conv2d(
#             topi.pulp.depthwise_conv2d_NCHWc, need_data_layout=True, need_out_layout=True
#         ),
#         wrap_topi_schedule(topi.pulp.schedule_depthwise_conv2d_NCHWc),
#         name="depthwise_conv2d_NCHWc.pulp",
#     )
#     return strategy
#
#
# @dense_strategy.register("pulp")
# def dense_strategy_pulp(attrs, inputs, out_type, target):
#     """dense x86 strategy"""
#
#     strategy = _op.OpStrategy()
#     # For dynamic matrix-vector multiply we use a hand written kernel.
#     if (
#         isinstance(inputs[0].shape[0], (int, tir.IntImm))
#         and inputs[0].shape[0] == 1
#         and (
#             topi.utils.is_dynamic_shape(inputs[0].shape)
#             or topi.utils.is_dynamic_shape(inputs[1].shape)
#         )
#     ):
#         strategy.add_implementation(
#             wrap_compute_dense(topi.x86.dense_dynamic),
#             wrap_topi_schedule(topi.x86.schedule_dense_dynamic),
#             name="dense_dynamic.x86",
#             plevel=20,
#         )
#         return strategy
#
#     same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
#     dtype = inputs[0].dtype
#     u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
#     strategy.add_implementation(
#         wrap_compute_dense(topi.x86.dense_nopack),
#         wrap_topi_schedule(topi.x86.schedule_dense_nopack),
#         name="dense_nopack.x86",
#         plevel=5,
#     )
#
#     strategy.add_implementation(
#         wrap_compute_dense(topi.x86.dense_pack),
#         wrap_topi_schedule(topi.x86.schedule_dense_pack),
#         name="dense_pack.x86",
#         plevel=10,
#     )
#
#     need_auto_scheduler_layout = is_auto_scheduler_enabled()
#     need_meta_schedule_layout = is_meta_schedule_enabled()
#
#     if need_auto_scheduler_layout or need_meta_schedule_layout:
#         strategy.add_implementation(
#             wrap_compute_dense(
#                 topi.nn.dense,
#                 need_auto_scheduler_layout=need_auto_scheduler_layout,
#                 need_meta_schedule_layout=need_meta_schedule_layout,
#             ),
#             naive_schedule,
#             name="dense.generic",
#             plevel=11,
#         )
#
#     if "cblas" in target.libs:
#         with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
#             strategy.add_implementation(
#                 wrap_compute_dense(topi.x86.dense_cblas),
#                 wrap_topi_schedule(topi.x86.schedule_dense_cblas),
#                 name="dense_cblas.x86",
#                 plevel=13,
#             )
#     if "mkl" in target.libs:
#         with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
#             strategy.add_implementation(
#                 wrap_compute_dense(topi.x86.dense_mkl),
#                 wrap_topi_schedule(topi.x86.schedule_dense_mkl),
#                 name="dense_mkl.x86",
#                 plevel=14,
#             )
#     if "dnnl" in target.libs:
#         with SpecializedCondition(same_type and dtype == "float32"):
#             strategy.add_implementation(
#                 wrap_compute_dense(topi.x86.dense_dnnl),
#                 wrap_topi_schedule(topi.x86.schedule_dense_dnnl),
#                 name="dense_dnnl.x86",
#                 plevel=15,
#             )
#     return strategy
#
#
# @dense_pack_strategy.register("pulp")
# def dense_pack_strategy_pulp(attrs, inputs, out_type, target):
#     """dense_pack x86 strategy"""
#     strategy = _op.OpStrategy()
#     if (
#         inputs[0].dtype == "uint8"
#         and inputs[1].dtype == "int8"
#         and out_type.dtype == "int32"
#         and attrs["weight_layout"] == "NC16n4c"
#     ):
#         strategy.add_implementation(
#             wrap_compute_dense(topi.x86.dense_int8),
#             wrap_topi_schedule(topi.x86.schedule_dense_int8),
#             name="dense_int8.x86",
#             plevel=13,
#         )
#     else:
#         strategy.add_implementation(
#             wrap_compute_dense(topi.x86.dense_pack),
#             wrap_topi_schedule(topi.x86.schedule_dense_pack),
#             name="dense_pack.x86",
#             plevel=10,
#         )
#
#     return strategy
@conv2d_strategy.register(["pulp"])
def conv2d_strategy_pulp(attrs, inputs, out_type, target):
    """conv2d pulp strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            print("target.kind.name", target.kind.name, data.dtype, kernel.dtype)
            if (
                # (target.kind.name in ["pulp"])
                (target.kind.name in ["llvm"])
                and data.dtype in ("int8", "uint8")
                and kernel.dtype in ("int8", "uint8")
            ):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.pulp",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.conv2d_nchw),
                    wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw),
                    name="conv2d_nchw.pulp",
                )
            N, _, H, W = get_const_tuple(data.shape)
            CO, CI, KH, KW = get_const_tuple(kernel.shape)
            # (_, _, judge_winograd_auto_scheduler) = judge_winograd(
            #     N,
            #     H,
            #     W,
            #     KH,
            #     KW,
            #     CI,
            #     CO,
            #     padding,
            #     stride_h,
            #     stride_w,
            #     dilation_h,
            #     dilation_w,
            #     data.dtype,
            #     kernel.dtype,
            #     pre_flag=False,
            # )
            # if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
            #     strategy.add_implementation(
            #         wrap_compute_conv2d(topi.nn.conv2d_winograd_nchw),
            #         naive_schedule,  # this implementation should never be picked by autotvm
            #         name="conv2d_nchw_winograd.pulp",
            #         plevel=15,
            #     )
            # elif (
            #     (2 < KH < 8 and 2 < KW < 8 and KH == KW)
            #     and (stride_h == 1 and stride_w == 1)
            #     and (dilation_h == 1 and dilation_w == 1)
            # ):
            #     strategy.add_implementation(
            #         wrap_compute_conv2d(topi.pulp.conv2d_nchw_winograd),
            #         wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw_winograd),
            #         name="conv2d_nchw_winograd.pulp",
            #         plevel=5,
            #     )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.pulp.conv2d_hwcn),
                wrap_topi_schedule(topi.pulp.schedule_conv2d_hwcn),
                name="conv2d_hwcn.pulp",
            )
        elif layout == "NHWC" and kernel_layout == "HWIO":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                name="conv2d_nhwc.gpu",
            )

            N, H, W, _ = get_const_tuple(data.shape)
            KH, KW, CI, CO = get_const_tuple(kernel.shape)
            # Winograd shape related judgment
            (
                judge_winograd_tensorcore,
                judge_winograd_autotvm,
                judge_winograd_auto_scheduler,
            ) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if judge_winograd_autotvm:
                if (
                    target.kind.name == "pulp"
                    and nvcc.have_tensorcore(target=target)
                    and judge_winograd_tensorcore
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.pulp.conv2d_nhwc_winograd_tensorcore),
                        wrap_topi_schedule(topi.pulp.schedule_conv2d_nhwc_winograd_tensorcore),
                        name="conv2d_nhwc_winograd_tensorcore.pulp",
                        plevel=5,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.pulp.conv2d_nhwc_winograd_direct),
                        wrap_topi_schedule(topi.pulp.schedule_conv2d_nhwc_winograd_direct),
                        name="conv2d_nhwc_winograd_direct.pulp",
                        plevel=5,
                    )
            if (
                target.kind.name == "pulp"
                and not is_auto_scheduler_enabled()
                and not is_meta_schedule_enabled()
                and nvcc.have_tensorcore(target=target)
                and (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.conv2d_nhwc_tensorcore),
                    wrap_topi_schedule(topi.pulp.schedule_conv2d_nhwc_tensorcore),
                    name="conv2d_nhwc_tensorcore.pulp",
                    plevel=20,
                )

            # register auto-scheduler implementations
            if is_auto_scheduler_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
            # register meta-schedule implementations
            if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )

        elif layout == "HWNC":
            assert kernel_layout in ["HWOI", "HWOI16o16i", "HWOI8o32i", "HWOI32o16i"]
            _, _, N, in_channels = get_const_tuple(data.shape)
            pre_computed = len(kernel.shape) == 6
            if pre_computed:
                _, _, oc_chunk, _, oc_block_factor, _ = get_const_tuple(kernel.shape)
                out_channels = oc_chunk * oc_block_factor
            else:
                _, _, out_channels, _ = get_const_tuple(kernel.shape)

            tensorcore_dtypes = ["int4", "uint4", "int8", "uint8"]
            if (
                target.kind.name == "pulp"
                and nvcc.have_tensorcore(target=target)
                and kernel.dtype in tensorcore_dtypes
                and (
                    (
                        data.dtype in ["int4", "uint4"]
                        and N % 8 == 0
                        and in_channels % 32 == 0
                        and out_channels % 8 == 0
                    )
                    or (
                        data.dtype in ["int8", "uint8"]
                        and N % 8 == 0
                        and in_channels % 16 == 0
                        and out_channels % 32 == 0
                    )
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.conv2d_hwnc_tensorcore),
                    wrap_topi_schedule(topi.pulp.schedule_conv2d_hwnc_tensorcore),
                    name="conv2d_hwnc_tensorcore_direct.pulp",
                    plevel=20,
                )
            else:
                raise RuntimeError(
                    "Unsupported shape for conv2d HWNC.\
                                    Need to satisfy tensor core schedule."
                )
        elif (
            (target.kind.name in ["pulp", "vulkan", "rocm"])
            and layout == "NCHW4c"
            and data.dtype in ["int8", "uint8"]
        ):
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.pulp.conv2d_NCHWc_int8, need_data_layout=True),
                wrap_topi_schedule(topi.pulp.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.pulp",
            )
        elif is_auto_scheduler_enabled() or is_meta_schedule_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.nn.conv, need_data_layout=True, need_kernel_layout=True, has_groups=True
                ),
                naive_schedule,
                name="conv2d.pulp",
                plevel=15,
            )
        elif target.kind.name == "pulp" and "cudnn" not in target.libs:
            # No TVM native kernel applicable
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))

        if (
            target.kind.name == "pulp"
            and "cudnn" in target.libs
            and layout in ["NCHW", "NHWC"]
            and padding[0] == padding[2]
            and padding[1] == padding[3]
            and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
        ):
            # add cudnn implementation
            if layout == "NHWC":
                assert kernel_layout == "OHWI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.pulp.conv2d_cudnn, need_data_layout=True, has_groups=True),
                wrap_topi_schedule(topi.pulp.schedule_conv2d_cudnn),
                name="conv2d_cudnn.pulp",
                plevel=25,
            )

    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups) and (
        layout == "NCHW" or "cudnn" not in target.libs
    ):  # cuDNN requires a different kernel layout for NHWC inputs.
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.pulp.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.pulp.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.pulp",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.pulp.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.pulp",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        # add cudnn implementation, if any
        cudnn_impl = False
        if target.kind.name == "pulp" and "cudnn" in target.libs:
            if (
                layout in ["NCHW", "NHWC"]
                and padding[0] == padding[2]
                and padding[1] == padding[3]
                and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.pulp.conv2d_cudnn, need_data_layout=True, has_groups=True
                    ),
                    wrap_topi_schedule(topi.pulp.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.pulp",
                    plevel=25,
                )
                cudnn_impl = True

        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            _, channels, _, _ = get_const_tuple(data.shape)
            out_channels, in_channels, _, _ = get_const_tuple(kernel.shape)
            oc_chunk = out_channels // 4
            ic_chunk = in_channels // 4

            if (
                (target.kind.name in ["pulp", "vulkan", "rocm"])
                and data.dtype in ["int8", "uint8"]
                and kernel.dtype in ["int8", "uint8"]
                and channels % groups == 0
                and out_channels % groups == 0
                and channels % 4 == 0
                and out_channels % 4 == 0
                and groups <= oc_chunk
                and groups <= ic_chunk
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.group_conv2d_nchw_int8, has_groups=True),
                    wrap_topi_schedule(topi.pulp.schedule_group_conv2d_nchw_int8),
                    name="group_conv2d_nchw_int8.pulp",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.pulp.group_conv2d_nchw, has_groups=True),
                    wrap_topi_schedule(topi.pulp.schedule_group_conv2d_nchw),
                    name="group_conv2d_nchw.pulp",
                )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.pulp.group_conv2d_NCHWc_int8, has_groups=True),
                wrap_topi_schedule(topi.pulp.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.pulp",
            )
        elif not cudnn_impl:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy
