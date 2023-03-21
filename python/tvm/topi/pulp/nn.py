import logging

import tvm
from tvm import autotvm, relay, te, tir

from .. import nn
from .. import utils
from ..nn import conv2d_legalize
from ..utils import get_const_tuple, is_target
# from .conv2d_winograd import _infer_tile_size
# from .tensorcore_alter_op import pad_to_tensorcore

logger = logging.getLogger("topi")

def schedule_injective_from_existing(sch, out):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """

    def find_nearest_small_factor(num, target):
        """Find the nearest factor of the given number that is smaller than the target."""
        for i in range(target, 0, -1):
            if num % i == 0:
                return i
        # Unreachable because i=1 must hold.
        return -1

    fused = sch[out].fuse(*sch[out].op.axis)
    # num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
    num_thread = 1
    max_block = 256

    # Vectorize on fp16 data type to enable half2 for better memory bandwidth utilization.
    vector_width = 2 if out.dtype == "float16" else 1

    is_dynamic_output = False
    for dim in out.shape:
        if not isinstance(dim, tvm.tir.IntImm):
            is_dynamic_output = True
            break

    out_len = utils.prod(out.shape)

    try:
        const_size = utils.get_const_int(out_len)

        # Adjust block and thread to make sure they are dividable so that vectorize can be
        # correctly applied.
        if vector_width > 1 and const_size % vector_width == 0:
            remain_total_size = const_size // vector_width
            cand_sizes = []
            for max_size in [num_thread, max_block]:
                cand_sizes.append(
                    max_size
                    if remain_total_size % max_size == 0
                    else find_nearest_small_factor(remain_total_size, max_size)
                )
                remain_total_size //= cand_sizes[-1]

            # If the product of candidate dividable (block * thread) is too small,
            # then the performance may be worse even half2 is enabled. Note that 0.7
            # is just a heuristic ratio and may not be optimal for all workloads.
            if np.prod(cand_sizes) / (max_block * num_thread) >= 0.7:
                num_thread, max_block = cand_sizes

        need_block_split = const_size > max_block * num_thread * vector_width
    except ValueError:
        need_block_split = False
        const_size = 0

    if vector_width > 1:
        fused, v = sch[out].split(fused, vector_width)
        # sch[out].vectorize(v)

    if need_block_split:
        xo, xi = sch[out].split(fused, factor=num_thread * max_block)
        bx, tx = sch[out].split(xi, factor=num_thread)
        sch[out].reorder(bx, tx, xo)
        # sch[out].bind(bx, te.thread_axis("blockIdx.x"))
        # sch[out].bind(tx, te.thread_axis("threadIdx.x"))
    else:
        # Use less threads for dynamic shape ops to avoid runtime error.
        if is_dynamic_output:
            num_thread //= 2
        if const_size != 0 and const_size < num_thread:
            bx, tx = sch[out].split(fused, factor=const_size)
        else:
            bx, tx = sch[out].split(fused, factor=num_thread)
        # sch[out].bind(tx, te.thread_axis("threadIdx.x"))
        # sch[out].bind(bx, te.thread_axis("blockIdx.x"))

    return sch

@nn.conv2d_alter_layout.register(["pulp"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    # if not is_target(["vulkan", "rocm", "cuda"]):
    #     return None
    dispatch_ctx = autotvm.task.DispatchContext.current

    new_attrs = {k: attrs[k] for k in attrs.keys()}
    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

    impl, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
    )
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template.
        # It may be from the auto-scheduler

        if impl.name.find("winograd") != -1:
            if dilation != (1, 1):
                logger.warning("Does not support weight pre-transform for dilated convolution.")
                return None

            if data_layout == "NHWC" and kernel_layout == "HWIO":
                N, H, W, CI = get_const_tuple(data.shape)
                KH, KW, _, CO = get_const_tuple(kernel.shape)
                # Pre-compute weight transformation in winograd
                tile_size = _infer_tile_size(tinfos[0], tinfos[1], layout="NHWC")
                # HWIO -> OIHW
                kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
                # alpha, alpha, CO, CI
                weight = relay.nn.contrib_conv2d_winograd_weight_transform(
                    kernel_transform, tile_size=tile_size
                )
                new_attrs["tile_size"] = tile_size
                new_attrs["channels"] = CO
                return relay.nn.contrib_conv2d_winograd_without_weight_transform(
                    inputs[0], weight, **new_attrs
                )
            elif data_layout == "NCHW" and kernel_layout == "OIHW":
                N, CI, H, W = get_const_tuple(data.shape)
                CO, _, KH, KW = get_const_tuple(kernel.shape)
                # Pre-compute weight transformation in winograd
                tile_size = _infer_tile_size(tinfos[0], tinfos[1], layout="NCHW")
                # alpha, alpha, CO, CI
                weight = relay.nn.contrib_conv2d_winograd_weight_transform(
                    inputs[1], tile_size=tile_size
                )
                # alpha, alpha, CI, CO
                weight = relay.transpose(weight, axes=[0, 1, 3, 2])
                new_attrs["tile_size"] = tile_size
                new_attrs["channels"] = CO
                return relay.nn.contrib_conv2d_winograd_without_weight_transform(
                    inputs[0], weight, **new_attrs
                )

        return None

    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        do_new_layout = False
        # if is_target(["vulkan", "rocm"]):
        if is_target(["vulkan", "rocm"]):  # ?
            # do_new_layout = "+dotprod" in target.mattr or target.supports_integer_dot_product
            do_new_layout = "+xcorevmac" in target.mattr or target.supports_integer_dot_product
        if not do_new_layout:
            return None

    topi_tmpl = workload[0]
    if topi_tmpl == "conv2d_NCHWc_int8.pulp":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        assert CO % 4 == 0, "Number of output channels should be multiple of 4"
        new_layout = "NCHW4c"
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "OIHW4o4i"
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CO // oc_block_factor,
                CI // ic_block_factor,
                KH,
                KW,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_layout, out_dtype],
            "conv2d_NCHWc_int8.pulp",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd.pulp":
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        # pre-compute weight transformation in winograd
        tile_size = _infer_tile_size(tinfos[0], tinfos[1])

        weight = relay.nn.contrib_conv2d_winograd_weight_transform(inputs[1], tile_size=tile_size)
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO

        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, CI, CO), dtype=kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_weight, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_without_weight_transform.pulp",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs
        )

    if topi_tmpl in ("conv2d_nhwc_winograd_direct.pulp", "conv2d_nhwc_winograd_tensorcore.pulp"):
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, _, CO = get_const_tuple(kernel.shape)

        # Pre-compute weight transformation in winograd
        tile_size = _infer_tile_size(data, kernel, layout="NHWC")
        kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
        weight = relay.nn.contrib_conv2d_winograd_weight_transform(
            kernel_transform, tile_size=tile_size
        )
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO
        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, CI, CO), dtype=kernel.dtype
        )
        if topi_tmpl == "conv2d_nhwc_winograd_direct.pulp":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_direct_without_weight_transform.pulp",
            )
        elif topi_tmpl == "conv2d_nhwc_winograd_tensorcore.pulp":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_tensorcore_without_weight_transform.pulp",
            )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs
        )

    if topi_tmpl == "group_conv2d_NCHWc_int8.pulp":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_layout = "NCHW4c"
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "OIHW4o4i"
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CO // oc_block_factor,
                CI // ic_block_factor // groups,
                KH,
                KW,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, groups, out_dtype],
            "group_conv2d_NCHWc_int8.pulp",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_HWNCnc_tensorcore.pulp":
        assert data_layout == "HWNC" and kernel_layout == "HWOI"
        # assert float(tvm.cuda(0).compute_version) >= 7.5
        H, W, N, CI = get_const_tuple(data.shape)
        KH, KW, CO, _ = get_const_tuple(kernel.shape)

        if (
            kernel.dtype in ["int4", "uint4"]
            and (CI % 32 != 0 or CO % 8 != 0)
            or kernel.dtype in ["int8", "uint8"]
            and (CI % 16 != 0 or CO % 32 != 0)
        ):
            return relay.nn.conv2d(*inputs, **new_attrs)

        new_attrs["channels"] = CO
        if kernel.dtype in ["int4", "uint4"]:
            new_attrs["kernel_layout"] = "HWOI8o32i"
            ic_block_factor = 32
            oc_block_factor = 8
        else:
            new_attrs["kernel_layout"] = "HWOI32o16i"
            ic_block_factor = 16
            oc_block_factor = 32

        new_kernel = te.placeholder(
            (
                KH,
                KW,
                CO // oc_block_factor,
                CI // ic_block_factor,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )

        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_HWNCnc_tensorcore.pulp",
        )

        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    return None


def _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, db), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, do), (0, 0)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


def _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, db), (0, 0), (0, 0), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, di), (0, 0)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, do)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import OtherOptionEntity
from tvm.contrib import cudnn

from .. import nn, generic
from ..nn.utils import get_pad_tuple
from ..utils import get_const_tuple, traverse_inline
# from .conv2d_direct import schedule_direct_cuda

def schedule_direct_pulp(cfg, s, conv):
    """schedule optimized for batch size = 1"""
    pass

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # # fallback support
    # if cfg.is_fallback:
    #     ref_log = autotvm.tophub.load_reference_log(
    #         target.kind.name, target.model, "conv2d_nchw.pulp"
    #     )
    #     cfg.fallback_with_reference_log(ref_log)
    # ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        # s[conv].set_scope("local")
        OL = conv

    # # create cache stage
    AA = s.cache_read(pad_data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    # s[output].bind(bf, te.thread_axis("blockIdx.z"))
    # s[output].bind(by, te.thread_axis("blockIdx.y"))
    # s[output].bind(bx, te.thread_axis("blockIdx.x"))
    # s[output].bind(vf, te.thread_axis("vthread"))
    # s[output].bind(vy, te.thread_axis("vthread"))
    # s[output].bind(vx, te.thread_axis("vthread"))
    # s[output].bind(tf, te.thread_axis("threadIdx.z"))
    # s[output].bind(ty, te.thread_axis("threadIdx.y"))
    # s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        # s[load].bind(tz, te.thread_axis("threadIdx.z"))
        # s[load].bind(ty, te.thread_axis("threadIdx.y"))
        # s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, KH, KW, CI = get_const_tuple(kernel.shape)

    # if isinstance(N, int):
    #     cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

@autotvm.register_topi_compute("conv2d_nchw.pulp")
def conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d with NCHW layout"""
    return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nchw.pulp")
def schedule_conv2d_nchw(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_nchw":
            schedule_direct_pulp(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


import tvm
from tvm import te
from tvm import autotvm

# from .injective import schedule_injective_from_existing
# from .tensor_intrin import dp4a
from ..nn.pad import pad
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.utils import get_pad_tuple
from ..utils import get_const_tuple, traverse_inline


def conv2d_nchw_int8(data, kernel, strides, padding, dilation, out_dtype="int32"):
    """Compute conv2d internally using conv2d_nchwc layout for int8 dtype"""
    assert data.dtype in ("int8", "uint8")
    assert kernel.dtype in ("int8", "uint8")
    assert data.dtype == kernel.dtype
    packed_out = conv2d_NCHWc_int8(data, kernel, strides, padding, dilation, "NCHW", out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw_int8(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHWc_int8(outs)


@autotvm.register_topi_compute("conv2d_NCHWc_int8.pulp")
def conv2d_NCHWc_int8(cfg, data, kernel, stride, padding, dilation, layout, out_dtype):
    """Convolution operator in NCHW[x]c layout for int8.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    assert layout in ["NCHW", "NCHW4c"]
    ic_block_factor = 4
    oc_block_factor = 4

    pre_computed = len(kernel.shape) == 6
    if not pre_computed:
        batch, channels, height, width = get_const_tuple(data.shape)
        print("data.shape", data.shape)
        assert (
            channels % ic_block_factor == 0
        ), "Number of input channels should be multiple of {}".format(ic_block_factor)
        packed_data = te.compute(
            (batch, channels // ic_block_factor, height, width, ic_block_factor),
            lambda n, c, h, w, vc: data[n, c * ic_block_factor + vc, h, w],
            name="packed_data",
        )

        out_channels, in_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)
        assert (
            out_channels % oc_block_factor == 0
        ), "Number of output channels should be multiple of {}".format(oc_block_factor)
        packed_kernel = te.compute(
            (
                out_channels // oc_block_factor,
                in_channels // ic_block_factor,
                kernel_h,
                kernel_w,
                oc_block_factor,
                ic_block_factor,
            ),
            lambda oc_chunk, ic_chunk, kh, kw, oc_block, ic_block: kernel[
                oc_chunk * oc_block_factor + oc_block, ic_chunk * ic_block_factor + ic_block, kh, kw
            ],
            name="packed_kernel",
        )

    else:
        packed_data = data
        packed_kernel = kernel

    batch, ic_chunk, in_height, in_width, ic_block = get_const_tuple(packed_data.shape)
    oc_chunk, ic_chunk, kernel_h, kernel_w, oc_block, ic_block = get_const_tuple(
        packed_kernel.shape
    )

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    pad_data = pad(packed_data, pad_before, pad_after, name="pad_data")

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    oshape = (batch, oc_chunk, out_height, out_width, oc_block)

    icc = te.reduce_axis((0, ic_chunk), name="ic_chunk")
    icb = te.reduce_axis((0, ic_block), name="ic_block")
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")

    packed_kernel_dtype = packed_kernel.dtype
    packed_dtype = "int32" if packed_kernel_dtype == "int8" else "uint32"
    conv = te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            pad_data[
                n, icc, oh * stride_h + kh * dilation_h, ow * stride_w + kw * dilation_w, icb
            ].astype(packed_dtype)
            * packed_kernel[oc_chunk, icc, kh, kw, oc_block, icb].astype(packed_dtype),
            axis=[icc, kh, kw, icb],
        ),
    )

    output = te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: conv[n, oc_chunk, oh, ow, oc_block].astype(out_dtype),
        tag="conv2d_NCHWc_int8",
    )

    # num flop
    num_flop = (
        batch
        * oc_chunk
        * oc_block
        * out_height
        * out_width
        * ic_chunk
        * ic_block
        * kernel_h
        * kernel_w
        * 2
    )
    cfg.add_flop(num_flop)

    return output


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.pulp")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Schedule conv2d int8 NCHWc template"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_NCHWc_int8":
            _schedule_conv2d_NCHWc_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

def dp4a(x_scope="local", y_scope="local", z_scope="local", dtypes=("int8", "int8")):
    """
    Int8 dot product reduced by every 4 elements using __dp4a

    Parameters
    ----------
    x_scope : str, optional
        The storage scope of buffer for lhs
    y_scope : str, optional
        The storage scope of buffer for rhs
    z_scope : str, optional
        The storage scope of buffer for result
    dtypes:  tuple of strs, optional
        The dtype of x and y

    Returns
    -------
    intrin : TensorIntrin
        The dp4a TensorIntrin that can be used in tensorizing schedule.
    """

    n = 4  # dp4a requires operands packed by 4
    result_dtype = "int32" if dtypes[1] == "int8" else "uint32"

    x = te.placeholder((n,), name="x", dtype=dtypes[0])
    y = te.placeholder((n,), name="y", dtype=dtypes[1])

    k = te.reduce_axis((0, n), name="rc")

    z = te.compute(
        (1,), lambda i: te.sum(x[k].astype(result_dtype) * y[k].astype(result_dtype), axis=[k])
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            xx, yy = ins
            zz = outs[0]
            zz_dtype = zz.dtype
            print("zz_dtype", zz_dtype)

            if index == 1:
                return zz.vstore(0, tvm.tir.const(0, zz_dtype))

            ib = tvm.tir.ir_builder.create()

            vec_x_dtype = "int8x4" if xx.dtype == "int8" else "uint8x4"
            vec_y_dtype = "int8x4" if yy.dtype == "int8" else "uint8x4"

            vec_x = xx.vload(0, dtype=vec_x_dtype)
            vec_y = yy.vload(0, dtype=vec_y_dtype)
            prev_z = 0 if index == 0 else zz.vload(0)

            # if is_target("rocm"):
            if False:
                # TODO(masahi): Here we are assuming that we are compiling for gfx10 or later
                # We can refine the specification for dot product on rocm if needed later.

                # We can just use "llvm.amdgcn.udot4" for u8u8u32, but it is not tested.
                assert (
                    dtypes[0] == "int8" and dtypes[0] == "int8"
                ), "u8u8u32 dot product for rocm not supported yet"

                new_z = tvm.tir.call_llvm_pure_intrin(
                    zz_dtype,
                    "llvm.amdgcn.sdot4",
                    tvm.tir.const(4, "uint32"),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    prev_z,
                    True,
                )
            else:
                # new_z = tvm.tir.call_pure_extern(zz_dtype, "__dp4a", vec_x, vec_y, prev_z)
                intrinsic_name = "llvm.riscv.corev.macs"
                intrinsic_name2 = "llvm.riscv.corev.machhs"
                new_z = tir.call_llvm_pure_intrin(
                    "int32",
                    intrinsic_name,
                    # 3,
                    tvm.tir.const(3, "uint32"),
                    # aaval, bbval, ccval
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    prev_z,
                )
                new_z = tir.call_llvm_pure_intrin(
                    "int32",
                    intrinsic_name2,
                    # 3,
                    tvm.tir.const(3, "uint32"),
                    # aaval, bbval, ccval
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    new_z,
                )
                new_z = tir.call_llvm_pure_intrin(
                    "int32",
                    intrinsic_name,
                    # 3,
                    tvm.tir.const(3, "uint32"),
                    # aaval, bbval, ccval
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    new_z,
                )
                new_z = tir.call_llvm_pure_intrin(
                    "int32",
                    intrinsic_name2,
                    # 3,
                    tvm.tir.const(3, "uint32"),
                    # aaval, bbval, ccval
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    new_z,
                )
                # ).astype(out_dtype)

            ib.emit(zz.vstore(0, new_z))

            return ib.get()

        return _instr(0), _instr(1), _instr(2)  # body, reset, update

    default_buffer_params = {"data_alignment": 4, "offset_factor": 1}
    scopes = {x: x_scope, y: y_scope, z: z_scope}
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, t.op.name, scope=scopes[t], **default_buffer_params
        )
        for t in [x, y, z]
    }

    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds=binds, default_buffer_params=default_buffer_params
    )

def _schedule_conv2d_NCHWc_int8(cfg, s, output):
    conv = output.op.input_tensors[0]
    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.te.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # skip this part during tuning to make recrods accurate
        # this part will be pre-computed during NNVM's pre-compute optimization pass
        s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
        s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
    else:
        if isinstance(packed_kernel.op, tvm.te.ComputeOp) and packed_kernel.name == "packed_kernel":
            # data and kernel are not pre-computed, schedule layout transform here
            pass
            schedule_injective_from_existing(s, packed_data)
            schedule_injective_from_existing(s, packed_kernel)

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [conv])
    WW = s.cache_read(packed_kernel, "shared", [conv])

    s[conv].set_scope("local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    # tile and bind spatial axes
    if len(s[output].op.axis) == 5:
        n, f, y, x, c = s[output].op.axis
    else:
        # For task extraction of auto-tuning, the expected output is 4D.  Since auto-tuning tasks
        # are created from scratch, therefore the real auto-tuning will still happen on 5D output.
        n, f, y, x = s[output].op.axis

    cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
    cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    # s[output].bind(bn, te.thread_axis("blockIdx.z"))
    # s[output].bind(bf, te.thread_axis("blockIdx.y"))
    # s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    # s[output].bind(vn, te.thread_axis("vthread"))
    # s[output].bind(vf, te.thread_axis("vthread"))
    # s[output].bind(vy, te.thread_axis("vthread"))
    # s[output].bind(vx, te.thread_axis("vthread"))

    # cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    # if cfg["fuse_yx"].val:
    #     s[output].bind(tn, te.thread_axis("threadIdx.z"))
    #     s[output].bind(tf, te.thread_axis("threadIdx.y"))
    #     tyx = s[output].fuse(ty, tx)
    #     s[output].bind(tyx, te.thread_axis("threadIdx.x"))
    #     s[conv].compute_at(s[output], tyx)

    #     # number of threads
    #     n_tz = cfg["tile_n"].size[2]
    #     n_ty = cfg["tile_f"].size[2]
    #     n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    # else:
    #     s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
    #     s[output].bind(ty, te.thread_axis("threadIdx.y"))
    #     s[output].bind(tx, te.thread_axis("threadIdx.x"))
    #     s[conv].compute_at(s[output], tx)

    #     # number of threads
    #     n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
    #     n_ty = cfg["tile_y"].size[2]
    #     n_tx = cfg["tile_x"].size[2]

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis

    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg["tile_rc"].apply(s, conv, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)

    cfg.define_reorder("reorder_inner", [rco, ryo, rxo], policy="all")
    cfg["reorder_inner"].apply(s, conv, [rco, ryo, rxo])
    cfg["reorder_inner"].apply(s, conv, [rci, ryi, rxi])

    _, rc_block = s[conv].split(rc_block, factor=4)
    target = tvm.target.Target.current(allow_none=False)
    # do_tensorize = "+dotprod" in target.mattr or target.supports_integer_dot_product
    do_tensorize = "+xcorevmac" in target.mattr or target.supports_integer_dot_product
    # do_tensorize = False
    print("do_tensorize", do_tensorize)

    if do_tensorize:
        dtypes = (pad_data.dtype, packed_kernel.dtype)
        s[conv].tensorize(rc_block, dp4a("shared", "shared", "local", dtypes))

    cache_loc = [rco, ryo, rxo][cfg["reorder_inner"].perm[-1]]
    s[AA].compute_at(s[conv], cache_loc)
    s[WW].compute_at(s[conv], cache_loc)

    # cooperative fetching
    for load in [AA, WW]:
        c = s[load].op.axis[-1]
        c_outer, c = s[load].split(c, factor=4)
        # s[load].vectorize(c)
        fused = s[load].op.axis[:-1] + [c_outer]
        fused = s[load].fuse(*fused)

        # fused, tx = s[load].split(fused, factor=n_tx)
        # fused, ty = s[load].split(fused, factor=n_ty)
        # fused, tz = s[load].split(fused, factor=n_tz)
        # s[load].bind(tz, te.thread_axis("threadIdx.z"))
        # s[load].bind(ty, te.thread_axis("threadIdx.y"))
        # s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # double buffer
    # cfg.define_knob("AA_double_buffer", [0, 1])
    # cfg.define_knob("WW_double_buffer", [0, 1])
    # if cfg["AA_double_buffer"].val:
    #     s[AA].double_buffer()
    # if cfg["WW_double_buffer"].val:
    #     s[WW].double_buffer()

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", False)

    return s
#
#
# import tvm
# from tvm import te
# from tvm import autotvm
# from ..utils import traverse_inline
# from .. import tag
# from .. import nn

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
@autotvm.register_topi_compute("depthwise_conv2d_nchw.pulp")
def depthwise_conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute depthwise_conv2d with NCHW layout."""
    return nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("depthwise_conv2d_nchw.pulp")
def schedule_depthwise_conv2d_nchw(cfg, outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "depthwise_conv2d_nchw":
            pass
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            cfg.define_split("tile_f", f, num_outputs=4)
            cfg.define_split("tile_y", y, num_outputs=4)
            cfg.define_split("tile_x", x, num_outputs=4)
            cfg.define_knob("auto_unroll_max_step", [0, 256, 1500])

            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            # # fallback support
            # if cfg.is_fallback:
            #     ref_log = autotvm.tophub.load_reference_log(
            #         target.kind.name, target.model, "depthwise_conv2d_nchw.pulp"
            #     )
            #     cfg.fallback_with_reference_log(ref_log)
            #     # TODO(lmzheng): A bug here, set unroll_explicit to False as workaround
            #     cfg["unroll_explicit"].val = 0
            # ##### space definition end #####

            s[pad_data].compute_inline()
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            # create cache stage
            AA = s.cache_read(pad_data, "shared", [OL])
            WW = s.cache_read(kernel, "shared", [OL])
            AL = s.cache_read(AA, "local", [OL])
            WL = s.cache_read(WW, "local", [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            kernel_scope, n = s[output].split(n, nparts=1)
            bf = s[output].fuse(n, bf)
            # s[output].bind(bf, te.thread_axis("blockIdx.z"))
            # s[output].bind(by, te.thread_axis("blockIdx.y"))
            # s[output].bind(bx, te.thread_axis("blockIdx.x"))
            # s[output].bind(vf, te.thread_axis("vthread"))
            # s[output].bind(vy, te.thread_axis("vthread"))
            # s[output].bind(vx, te.thread_axis("vthread"))
            # s[output].bind(tf, te.thread_axis("threadIdx.z"))
            # s[output].bind(ty, te.thread_axis("threadIdx.y"))
            # s[output].bind(tx, te.thread_axis("threadIdx.x"))
            s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
            s[OL].compute_at(s[output], tx)

            # # cooperative fetching
            # s[AA].compute_at(s[output], bx)
            # s[WW].compute_at(s[output], bx)
            # s[AL].compute_at(s[output], tx)
            # s[WL].compute_at(s[output], tx)

            for load in [AA, WW]:
                pass
                # fused = s[load].fuse(*list(s[load].op.axis))
                # fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
                # fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
                # fused, tz = s[load].split(fused, cfg["tile_f"].size[2])
                # s[load].bind(tz, te.thread_axis("threadIdx.z"))
                # s[load].bind(ty, te.thread_axis("threadIdx.y"))
                # s[load].bind(tx, te.thread_axis("threadIdx.x"))

            # s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            # s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, db), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, do), (0, 0)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


def _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, db), (0, 0), (0, 0), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, di), (0, 0)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, do)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


@conv2d_legalize.register(["pulp"])
def _conv2d_legalize(attrs, inputs, arg_types):
    """Legalizes Conv2D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    if not is_target(["pulp"]):
        return None
    # Dilation not supported yet. Return None if dilation is not (1, 1)
    dilation = attrs.get_int_tuple("dilation")
    if not (dilation[0] == 1 and dilation[1] == 1):
        return None

    # No legalization for depthwise convolutions yet.
    groups = attrs.get_int("groups")
    if groups != 1:
        return None

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Get data layout. Return None if not NCHW
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    # Pad input and output channels to use int8 schedule.
    if data_dtype in ["int8", "uint8"]:
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            oc_modified = False
            in_channel = data_tensor.shape[1].value
            out_channel = kernel_tensor.shape[0].value

            # Pad input channel
            if in_channel % 4 != 0:
                new_in_channel = ((in_channel + 4) // 4) * 4
                diff = new_in_channel - in_channel
                pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
                data = relay.nn.pad(data, pad_width=pad_width)
                kernel = relay.nn.pad(kernel, pad_width=pad_width)

            # Pad output channel
            new_out_channel = out_channel
            if out_channel % 4 != 0:
                new_out_channel = ((out_channel + 4) // 4) * 4
                diff = new_out_channel - out_channel
                kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
                oc_modified = True

            if oc_modified:
                new_attrs["channels"] = new_out_channel
                out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
                original_out_shape = [x.value for x in output_tensor.shape]
                out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
            else:
                out = relay.nn.conv2d(data, kernel, **new_attrs)
            return out

        if data_layout == "NHWC" and kernel_layout == "HWIO":
            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

        if data_layout == "HWNC" and kernel_layout == "HWOI":
            batch = data_tensor.shape[2].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[2].value

            if batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0:
                return None

            candidates = [(8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None
            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    elif data_dtype in ["float16"]:
        if data_layout == "NHWC" and kernel_layout == "HWIO":
            if isinstance(data_tensor.shape[0], tvm.tir.expr.Any):
                # Skip legalize when the batch size is dynamic
                return None

            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    elif data_dtype in ["int4", "uint4"]:
        if data_layout == "NHWC" and kernel_layout == "HWIO":
            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

        if data_layout == "HWNC" and kernel_layout == "HWOI":
            batch = data_tensor.shape[2].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[2].value

            if batch % 8 == 0 and in_channel % 32 == 0 and out_channel % 8 == 0:
                return None

            candidates = [(8, 32, 8)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None
            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    return None
