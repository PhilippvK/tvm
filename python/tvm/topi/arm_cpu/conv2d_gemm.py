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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""GEMM Convolution schedule on ARM"""

import tvm
from tvm.target import Target
from tvm import te
from tvm.topi import nn
from tvm.topi.arm_cpu import arm_utils
from tvm.autotvm.task.space import AnnotateEntity, ReorderEntity, OtherOptionEntity
from ..utils import get_const_tuple, get_const_int
from ..nn.utils import get_pad_tuple
from .tensor_intrin import (
    gemm_4x4_int8_int8_int32,
    gemm_acc_4x4_int8_int8_int32,
    gemm_acc_nx16_int8_int8_int32,
    gemm_acc_2x2_int8_int8_int32,
)

# START

from typing import Optional, Sequence, Tuple

import tvm
from tvm import te, tir
from tvm.topi.utils import get_const_tuple


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    assert len(value) == 2
    return int(value[0]), int(value[1])


def _quad_padding(padding) -> Tuple[int, int, int, int]:
    """Return top, left, bottom, right padding."""
    if isinstance(padding, int):
        return padding, padding, padding, padding

    if len(padding) == 2:
        pad_h, pad_w = padding
        return int(pad_h), int(pad_w), int(pad_h), int(pad_w)

    assert len(padding) == 4
    pad_top, pad_left, pad_bottom, pad_right = padding
    return (
        int(pad_top),
        int(pad_left),
        int(pad_bottom),
        int(pad_right),
    )


def _choose_conv_ki(
    K: int,
    K_MAX: int,
    K_STEP: int,
) -> int:
    print("_choose_conv_ki", K, K_MAX, K_STEP)
    """Choose the largest legal KI not exceeding K_MAX.

    KI must:
      - divide the complete reduction extent K;
      - be a multiple of the IME K step;
      - not exceed K_MAX.

    For K=576:
      K_MAX=576 -> KI=576
      K_MAX=512 -> KI=288
      K_MAX=256 -> KI=192
      K_MAX=128 -> KI=96
    """
    assert K > 0
    assert K_MAX >= K_STEP
    assert K % K_STEP == 0

    upper = min(K, K_MAX)
    upper -= upper % K_STEP
    print("upper", upper)

    for ki in range(upper, K_STEP - 1, -K_STEP):
        if K % ki == 0:
            print("ki", ki)
            return ki

    raise ValueError(f"Could not find KI dividing K={K}, " f"with KI <= {K_MAX} and KI % {K_STEP} == 0")


# def _spatial_pair(value, layout="NHWC", name="value"):
#     """Normalize a scalar, 2-D pair, or layout-aware 4-D tuple to (h, w)."""
#     if isinstance(value, (int, tir.IntImm)):
#         value = int(value)
#         return value, value
#
#     value = tuple(int(x) for x in value)
#
#     if len(value) == 2:
#         return value
#
#     if len(value) == 4:
#         if layout == "NHWC":
#             # [N, H, W, C]
#             n, h, w, c = value
#             if n != 1 or c != 1:
#                 raise ValueError(f"{name}={value} is invalid for NHWC; " "batch and channel components must be 1")
#             return h, w
#
#         if layout == "NCHW":
#             # [N, C, H, W]
#             n, c, h, w = value
#             if n != 1 or c != 1:
#                 raise ValueError(f"{name}={value} is invalid for NCHW; " "batch and channel components must be 1")
#             return h, w
#
#     raise ValueError(
#         f"Unsupported {name}={value} for layout={layout!r}; "
#         "expected a scalar, a 2-element spatial tuple, or a 4-element layout tuple"
#     )


def conv2d_nhwc_hwoi_ime_packed_compute(
    cfg,
    data: te.Tensor,
    weight: te.Tensor,
    # bias: Optional[te.Tensor] = None,
    strides: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0, 0, 0),
    dilation: Sequence[int] = (1, 1),
    out_dtype: str = "int32",
    MI: int = 8,
    NI: int = 8,
    # K_MAX: int = 576,
    K_MAX: int = 1024,
    K_STEP: int = 8,
    name: str = "conv2d_nhwc_hwoi_ime",
):
    print(
        "IME conv2d args:",
        "strides=",
        strides,
        "padding=",
        padding,
        "dilation=",
        dilation,
        "stride_len=",
        len(strides) if hasattr(strides, "__len__") else None,
    )
    """IME-compatible packed NHWC/HWOI int8 convolution.

    Input:
      data:   [batch, input_height, input_width, input_channels]
      weight: [kernel_height, kernel_width, output_channels, input_channels]
      bias:   [output_channels], optional

    Output:
      [batch, output_height, output_width, output_channels]

    Logical GEMM:
      M = batch * output_height * output_width
      N = output_channels
      K = kernel_height * kernel_width * input_channels

    Physical packed layout:
      A_pack[MO, KO, KB, MT, 4, K_STEP]
      B_pack[NO, KO, KB, NT, 4, K_STEP]
      C_pack[MO, NO, MT, NT, 4, 4]

    One microkernel sees:
      A[KB, MT, 4, K_STEP]
      B[KB, NT, 4, K_STEP]
      C[MT, NT, 4, 4]

    where:
      MT = MI // 4
      NT = NI // 4
      KB = KI // K_STEP
    """
    assert data.dtype == "int8"
    assert weight.dtype == "int8"
    assert out_dtype == "int32"

    assert MI in (4, 8)
    assert NI in (4, 8)
    assert MI % 4 == 0
    assert NI % 4 == 0
    assert K_STEP == 8

    batch, input_h, input_w, input_c = get_const_tuple(data.shape)
    kernel_h, kernel_w, output_c, weight_input_c = get_const_tuple(weight.shape)

    assert input_c == weight_input_c
    assert output_c % NI == 0, (
        f"OC={output_c} must be divisible by NI={NI}; " "add output-channel padding for tail support"
    )

    stride_h, stride_w = _pair(strides)
    dilation_h, dilation_w = _pair(dilation)
    pad_top, pad_left, pad_bottom, pad_right = _quad_padding(padding)
    # stride_h, stride_w = _spatial_pair(
    #     strides,
    #     layout="NHWC",
    #     name="strides",
    # )
    # dilation_h, dilation_w = _spatial_pair(
    #     dilation,
    #     layout="NHWC",
    #     name="dilation",
    # )
    # pad_top, pad_left, pad_bottom, pad_right = _normalize_padding(padding)

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    output_h = (input_h + pad_top + pad_bottom - dilated_kernel_h) // stride_h + 1
    output_w = (input_w + pad_left + pad_right - dilated_kernel_w) // stride_w + 1

    M = batch * output_h * output_w
    N = output_c
    K = kernel_h * kernel_w * input_c

    assert M % MI == 0, (
        f"M=N_batch*OH*OW={M} must be divisible by MI={MI}; " "add output-position padding for tail support"
    )
    assert K % K_STEP == 0, f"K=KH*KW*IC={K} must be divisible by K_STEP={K_STEP}"

    KI = _choose_conv_ki(K, K_MAX, K_STEP)

    MO = M // MI
    NO = N // NI
    KO = K // KI

    MT = MI // 4
    NT = NI // 4
    KB = KI // K_STEP

    pack_attrs = {
        "ime_explicit_pack": True,
        "meta_schedule.no_random_compute_location": True,
    }

    def unpack_m(m):
        """Map flattened GEMM M to batch/output-height/output-width."""
        batch_index = m // (output_h * output_w)
        output_index = m % (output_h * output_w)
        output_y = output_index // output_w
        output_x = output_index % output_w
        return batch_index, output_y, output_x

    def unpack_k(k):
        """Map flattened GEMM K to kernel-height/kernel-width/input-channel."""
        input_channel = k % input_c
        kernel_index = k // input_c
        kernel_x = kernel_index % kernel_w
        kernel_y = kernel_index // kernel_w
        return kernel_y, kernel_x, input_channel

    # Fused im2col and IME packing.
    #
    # For one microkernel tile:
    #   A_tile[KB, MT, 4, 8]
    #
    # Flattened order inside one tile is:
    #   kb -> mt -> mi4 -> kk
    def compute_a_pack(mo, ko, kb, mt, mi4, kk):
        m = mo * MI + mt * 4 + mi4
        k = ko * KI + kb * K_STEP + kk

        batch_index, output_y, output_x = unpack_m(m)
        kernel_y, kernel_x, input_channel = unpack_k(k)

        input_y = output_y * stride_h + kernel_y * dilation_h - pad_top
        input_x = output_x * stride_w + kernel_x * dilation_w - pad_left

        in_bounds = tir.all(
            input_y >= 0,
            input_y < input_h,
            input_x >= 0,
            input_x < input_w,
        )

        return tir.if_then_else(
            in_bounds,
            data[batch_index, input_y, input_x, input_channel],
            tir.const(0, data.dtype),
        )

    A_pack = te.compute(
        (MO, KO, KB, MT, 4, K_STEP),
        compute_a_pack,
        name="A_pack",
        attrs=pack_attrs,
    )

    # Weight packing for HWOI:
    #   weight[kh, kw, output_channel, input_channel]
    #
    # For one microkernel tile:
    #   B_tile[KB, NT, 4, 8]
    def compute_b_pack(no, ko, kb, nt, ni4, kk):
        output_channel = no * NI + nt * 4 + ni4
        k = ko * KI + kb * K_STEP + kk

        kernel_y, kernel_x, input_channel = unpack_k(k)

        return weight[
            kernel_y,
            kernel_x,
            output_channel,
            input_channel,
        ]

    B_pack = te.compute(
        (NO, KO, KB, NT, 4, K_STEP),
        compute_b_pack,
        name="B_pack",
        attrs=pack_attrs,
    )

    reduction_attrs = {
        "ime_layout": "microtile_major_4x4_k8",
        "ime_m_tile": MI,
        "ime_n_tile": NI,
        "ime_k_tile": KI,
        "ime_k_step": K_STEP,
        "ime_m_micro_tile": 4,
        "ime_n_micro_tile": 4,
        "ime_m_micro_tiles": MT,
        "ime_n_micro_tiles": NT,
        "ime_k_micro_tiles": KB,
        "ime_k_max": K_MAX,
        # "meta_schedule.no_random_compute_location": True,
        # TODO: test
        "layout_free_placeholders": [B_pack],
    }

    # Keep KO outside the tensorized microkernel.  One ukernel consumes KI
    # reduction elements, represented by KB x K_STEP.
    rkb = te.reduce_axis((0, KB), name="rkb")
    rkk = te.reduce_axis((0, K_STEP), name="rkk")

    if KO == 1:
        C_pack = te.compute(
            (MO, NO, MT, NT, 4, 4),
            lambda mo, no, mt, nt, mi4, ni4: te.sum(
                A_pack[
                    mo,
                    0,
                    rkb,
                    mt,
                    mi4,
                    rkk,
                ].astype(out_dtype)
                * B_pack[
                    no,
                    0,
                    rkb,
                    nt,
                    ni4,
                    rkk,
                ].astype(out_dtype),
                axis=[rkb, rkk],
            ),
            name="C_pack",
            attrs=reduction_attrs,
        )
    else:
        rko = te.reduce_axis((0, KO), name="rko")

        C_pack = te.compute(
            (MO, NO, MT, NT, 4, 4),
            lambda mo, no, mt, nt, mi4, ni4: te.sum(
                A_pack[
                    mo,
                    rko,
                    rkb,
                    mt,
                    mi4,
                    rkk,
                ].astype(out_dtype)
                * B_pack[
                    no,
                    rko,
                    rkb,
                    nt,
                    ni4,
                    rkk,
                ].astype(out_dtype),
                axis=[rko, rkb, rkk],
            ),
            name="C_pack",
            attrs=reduction_attrs,
        )

    # Convert packed GEMM output back to NHWC.
    def unpack_output(batch_index, output_y, output_x, output_channel):
        m = batch_index * output_h * output_w + output_y * output_w + output_x

        value = C_pack[
            m // MI,
            output_channel // NI,
            (m % MI) // 4,
            (output_channel % NI) // 4,
            m % 4,
            output_channel % 4,
        ]

        # if bias is not None:
        #     value = value + bias[output_channel].astype(out_dtype)

        return value

    output = te.compute(
        (batch, output_h, output_w, output_c),
        unpack_output,
        name=name,
    )

    return output


# END


def configure_knobs(cfg, M, K, target):
    """Configure auto-tuning knobs for the interleaved strategy"""

    x, y = cfg.axis(M // 4), cfg.axis(K // 16)
    cfg.define_reorder("reorder_gemm", [x, y], policy="candidate", candidate=[[x, y], [y, x]])

    outer_loop, inner_loop = cfg.axis(4), cfg.axis(16)
    cfg.define_annotate(
        "A_interleaved_unroll_vec", [outer_loop, inner_loop], policy="try_unroll_vec"
    )

    # Fallback configuration
    if cfg.is_fallback:
        cfg["reorder_gemm"] = ReorderEntity([0, 1])
        cfg["A_interleaved_unroll_vec"] = AnnotateEntity(["unroll", "vec"])

    if not target.features.has_dotprod:
        cfg.define_knob("gemm_quantized_unroll", [True, False])
        if cfg.is_fallback:
            cfg["gemm_quantized_unroll"] = OtherOptionEntity(False)


# Compute function
def compute_conv2d_gemm_without_weight_transform(
    cfg,
    data,
    B_interleaved_t,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size,
    output_channels,
    interleave_A,
    use_scalable_vectors=False,
    use_sme=False,
):
    """Compute conv2d by transforming the input,
    executing GEMM and transforming the output back"""
    batches, IH, IW, IC = get_const_tuple(data.shape)
    in_dtype = data.dtype

    KH, KW = get_const_tuple(kernel_size)
    OC = get_const_int(output_channels)
    kernel_area = KH * KW

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = get_const_tuple(dilation)

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    # Input padding (if necessary)
    if pad_top or pad_left or pad_down or pad_right:
        data_pad = nn.pad(
            data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0], name="data_pad"
        )
    else:
        data_pad = data

    # Im2col transformation
    M = OH * OW
    K = IC * kernel_area
    N = OC

    A_shape = (batches, M, K)
    if kernel_area == 1:
        A = tvm.topi.reshape(data_pad, A_shape)
    else:
        A = te.compute(
            A_shape,
            lambda n, x, y: data_pad[
                n,
                HSTR * (x // OW) + dilation_h * ((y // IC) // KW),
                WSTR * (x % OW) + dilation_w * ((y // IC) % KW),
                y % IC,
            ],
            name="data_im2col",
        )

    # Select the tiling strategy for A and B
    tile_M, tile_K_A = arm_utils.get_tiling_A(interleave_A, in_dtype, use_sme)
    tile_N, tile_K_B = arm_utils.get_tiling_B_transformed(
        interleave_A,
        in_dtype,
        use_scalable_vectors,
        use_sme,
    )

    # Pad to tiles (if necessary)
    use_explicit_predication = use_sme and in_dtype == "float32"
    if not use_explicit_predication:
        pad_M, pad_K = arm_utils.get_conv2d_im2col_padding(M, K, tile_M, tile_K_A)
        pad_N, _ = arm_utils.get_conv2d_weights_padding(N, K, tile_N, tile_K_B)

        M_padded = M + pad_M
        K_padded = K + pad_K
        N_padded = N + pad_N

        pad_before = (0, 0, 0)
        pad_after = (0, pad_M, pad_K)

        if pad_K != 0:
            A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded_K")
        elif pad_M != 0:
            A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded_M")

    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, K if use_explicit_predication else K_padded), "k")

    # Determine matrix multiplication compute definition
    target = Target.current(allow_none=False)
    if in_dtype in ["int8", "uint8"]:
        assert len(B_interleaved_t.shape) == 4
        if interleave_A:
            # Configuration space
            configure_knobs(cfg, M_padded, K_padded, target)

            # Pack the input data
            A_interleaved = te.compute(
                (
                    batches,
                    M_padded // tile_M,
                    K_padded // tile_K_A,
                    tile_M,
                    tile_K_A,
                ),
                lambda b, x, y, z, w: A[b, z + tile_M * x, w + tile_K_A * y],
                name="A_interleaved",
            )
            N_transformed = B_interleaved_t.shape[0]
            if target.features.has_matmul_i8:
                # Execute GEMM. In the case of mmla, we need to enforce the tiling
                # from the compute. This is because mmla is doing a tiled computation
                # as well. So we have a big 8x12 tile, with small 2x2 sub-tiles
                # generated by mmla. In theory we could make the tile 2x2 and
                # fuse and split during scheduling, but this would not work
                # because of possible padding
                C_interleaved = te.compute(
                    (
                        batches,
                        M_padded // tile_M,
                        N_transformed,
                        tile_M // 2,
                        tile_N // 2,
                        2,
                        2,
                    ),
                    lambda b, x, y, w, z, s, t: te.sum(
                        A_interleaved[b, x, k // tile_K_A, 2 * w + s, idxm(k, tile_K_A)].astype(
                            "int32"
                        )
                        * B_interleaved_t[y, k // tile_K_B, 2 * z + t, idxm(k, tile_K_B)].astype(
                            "int32"
                        ),
                        axis=k,
                    ),
                    name="C_interleaved",
                )
                # Ensure the padding needed for tensorize does not get removed during tir passes
                # by adding a dummy reference to the specific padded area of the result
                zero = (
                    tvm.tir.const(1, C_interleaved.dtype)
                    * C_interleaved[
                        batches - 1,
                        M // tile_M,
                        N_transformed - 1,
                        idxm(M, tile_M) // 2,
                        tile_N // 2 - 1,
                        1,
                        1,
                    ]
                    - tvm.tir.const(1, C_interleaved.dtype)
                    * C_interleaved[
                        batches - 1,
                        M // tile_M,
                        N_transformed - 1,
                        idxm(M, tile_M) // 2,
                        tile_N // 2 - 1,
                        1,
                        1,
                    ]
                )
                # Unpack the result
                C = te.compute(
                    (batches, M, N),
                    lambda b, x, y: (
                        C_interleaved[
                            b,
                            x // tile_M,
                            y // tile_N,
                            idxm(x, tile_M) // 2,
                            idxm(y, tile_N) // 2,
                            idxm(idxm(x, tile_M), 2),
                            idxm(idxm(y, tile_N), 2),
                        ]
                        + zero
                    ).astype(out_dtype),
                    name="C",
                )
            else:
                # Execute GEMM
                C_interleaved = te.compute(
                    (batches, M_padded // tile_M, N_transformed, tile_M, tile_N),
                    lambda b, x, y, w, z: te.sum(
                        A_interleaved[b, x, k // tile_K_A, w, idxm(k, tile_K_A)].astype("int32")
                        * B_interleaved_t[y, k // tile_K_B, z, idxm(k, tile_K_B)].astype("int32"),
                        axis=k,
                    ),
                    name="C_interleaved",
                )
                # Unpack the result
                C = te.compute(
                    (batches, M, N),
                    lambda b, x, y: C_interleaved[
                        b,
                        x // tile_M,
                        y // tile_N,
                        idxm(x, tile_M),
                        idxm(y, tile_N),
                    ].astype(out_dtype),
                    name="C",
                )
            zero = tvm.tir.const(0)
        else:
            # No need to pack/unpack, execute GEMM directly
            C = te.compute(
                (batches, M_padded, N_padded),
                lambda b, x, y: te.sum(
                    A[b, x, k].astype("int32")
                    * B_interleaved_t[
                        y // tile_N,
                        k // tile_K_B,
                        idxm(y, tile_N),
                        idxm(k, tile_K_B),
                    ].astype("int32"),
                    axis=k,
                ),
                name="C",
            )

            # We need to ensure that infer bound pass does not remove the padding
            # which is necessary for the tensorizations to work. So we need to
            # add a dummy reference to the padding area of the result
            zero = (
                tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
                - tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
            )
    elif use_sme and in_dtype == "float16" and out_dtype == "float32":
        assert len(B_interleaved_t.shape) == 2
        C = te.compute(
            (batches, M_padded, N_padded),
            lambda b, x, y: te.sum(
                A[b, x, k].astype(out_dtype) * B_interleaved_t[y, k].astype(out_dtype),
                axis=k,
            ),
            name="C",
        )
        zero = tvm.tir.const(0)
    elif use_explicit_predication:
        assert len(B_interleaved_t.shape) == 2
        C = te.compute(
            (batches, M, N),
            lambda b, x, y: te.sum(
                A[b, x, k].astype(in_dtype) * B_interleaved_t[k, y].astype(in_dtype),
                axis=k,
            ),
            name="C",
        )
        zero = tvm.tir.const(0)
    elif use_scalable_vectors:
        assert len(B_interleaved_t.shape) == 2
        C = te.compute(
            (batches, M_padded, N_padded),
            lambda b, x, y: te.sum(
                A[b, x, k].astype(in_dtype) * B_interleaved_t[k, y].astype(in_dtype),
                axis=k,
            ),
            name="C",
        )
        # Ensure padding on the N axis does not get removed during tir passes
        # by adding a dummy reference to the specific padded area of the result
        zero = (
            tvm.tir.const(1, C.dtype) * C[0, 0, N_padded - 1]
            - tvm.tir.const(1, C.dtype) * C[0, 0, N_padded - 1]
        )
    else:
        assert len(B_interleaved_t.shape) == 4
        C = te.compute(
            (batches, M_padded, N_padded),
            lambda b, x, y: te.sum(
                A[b, x, k].astype(in_dtype)
                * B_interleaved_t[
                    y // tile_N,
                    k // tile_K_B,
                    idxm(k, tile_K_B),
                    idxm(y, tile_N),
                ].astype(in_dtype),
                axis=k,
            ),
            name="C",
        )
        # Ensure padding on the N axis does not get removed during tir passes
        # by adding a dummy reference to the specific padded area of the result
        if in_dtype == "float16" and target.features.has_fp16_simd:
            zero = (
                tvm.tir.const(1, C.dtype) * C[0, 0, N_padded - 1]
                - tvm.tir.const(1, C.dtype) * C[0, 0, N_padded - 1]
            )
        else:
            zero = tvm.tir.const(0)

    # Reshape the result into a convolution output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(
        out_shape,
        lambda b, x, y, z: (C(b, y + OW * x, z) + zero).astype(out_dtype),
        name="conv2d_gemm_output",
        attrs={"use_scalable_vectors": use_scalable_vectors, "use_sme": use_sme},
    )
    return out


def schedule_conv2d_gemm_interleaved(cfg, s, out, final_out):
    """Schedule the conv2d_gemm interleaved strategy"""
    C = out.op.input_tensors[0]
    C_interleaved = C.op.input_tensors[0]
    A_interleaved = C_interleaved.op.input_tensors[0]
    in_type = A_interleaved.dtype
    tile_M, tile_K = arm_utils.get_tiling_A(True, in_type)

    # Input transform
    A_interleaved_input = A_interleaved.op.input_tensors[0]
    if A_interleaved_input.op.name == "A_padded_K" or A_interleaved_input.op.name == "A_padded_M":
        s[A_interleaved_input].compute_at(s[A_interleaved], A_interleaved.op.axis[3])
        s[A_interleaved_input].vectorize(A_interleaved_input.op.axis[2])
        s[A_interleaved_input].compute_inline()
        data_im2col = A_interleaved_input.op.input_tensors[0]
    else:
        data_im2col = A_interleaved_input

    b, m, n = data_im2col.op.axis
    if data_im2col.op.name == "data_im2col":
        n_size = data_im2col.shape[2]
        if n_size % 16 == 0:
            split_factor = 16
        else:
            split_factor = 8
        n_outer, n_inner = s[data_im2col].split(n, split_factor)
        s[data_im2col].unroll(n_outer)
        s[data_im2col].vectorize(n_inner)
        b_m_fused = s[data_im2col].fuse(b, m)
        s[data_im2col].parallel(b_m_fused)
    else:
        s[data_im2col].compute_inline()

    # Computation(through tensorize)
    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
    outer_gemm, inner_gemm = cfg["reorder_gemm"].apply(s, C_interleaved, [xo, yo])

    b_outer_gemm_fused = s[C_interleaved].fuse(b, outer_gemm)
    s[C_interleaved].parallel(b_outer_gemm_fused)
    s[A_interleaved].compute_at(s[C_interleaved], b_outer_gemm_fused)
    _, _, _, outer_A_interleaved, inner_A_interleaved = A_interleaved.op.axis
    cfg["A_interleaved_unroll_vec"].apply(
        s, A_interleaved, [outer_A_interleaved, inner_A_interleaved]
    )

    k = C_interleaved.op.reduce_axis[0]
    _, M, N = C.shape
    if in_type in ["int8", "uint8"]:
        target = Target.current(allow_none=False)
        if target.features.has_matmul_i8:
            gemm_acc = gemm_acc_2x2_int8_int8_int32(in_type)
            xi_inner, yi_inner = C_interleaved.op.axis[-2:]
            k_outer, k_inner = s[C_interleaved].split(k, tile_K)
            s[C_interleaved].reorder(
                b_outer_gemm_fused, inner_gemm, k_outer, xi, yi, xi_inner, yi_inner, k_inner
            )
            s[C_interleaved].tensorize(xi_inner, gemm_acc)
            s[C_interleaved].unroll(xi)
            s[C_interleaved].unroll(yi)
        elif target.features.has_dotprod:
            gemm_acc = gemm_acc_4x4_int8_int8_int32(in_type)
            xi_outer, yi_outer, xi_inner, yi_inner = s[C_interleaved].tile(
                xi, yi, x_factor=tile_M, y_factor=4
            )
            k_outer, k_inner = s[C_interleaved].split(k, tile_K)
            xi_inner_outer, xi_inner_inner = s[C_interleaved].split(xi_inner, 4)
            s[C_interleaved].reorder(
                b_outer_gemm_fused,
                inner_gemm,
                xi_outer,
                yi_outer,
                k_outer,
                xi_inner_outer,
                xi_inner_inner,
                yi_inner,
                k_inner,
            )
            s[C_interleaved].tensorize(xi_inner_inner, gemm_acc)
            s[C_interleaved].unroll(xi_inner_outer)

        elif target.features.has_asimd:
            s[C_interleaved].reorder(yi, xi)
            K = A_interleaved_input.shape[2]
            assert in_type in ["int8", "uint8"], "Only int8 and uint8 gemm are supported"
            unroll = cfg["gemm_quantized_unroll"].val
            gemm = gemm_4x4_int8_int8_int32(M, N, K, unroll, in_type)
            s[C_interleaved].tensorize(yi, gemm)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[C].compute_at(s[out], inner)
        s[out].vectorize(inner)
    return s


def schedule_conv2d_gemm_native(cfg, s, out, final_out):
    """Schedule the conv2d_gemm hybrid strategy"""
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    in_type = A.dtype
    use_scalable_vectors = bool(out.op.attrs["use_scalable_vectors"])
    tile_M, tile_K = arm_utils.get_tiling_A(False, in_type)
    tile_N, _ = arm_utils.get_tiling_B_transformed(False, in_type, use_scalable_vectors)

    # Computation
    b, x, y = C.op.axis
    (k,) = C.op.reduce_axis

    if in_type in ["int8", "uint8"]:
        k_outer, k_inner = s[C].split(k, tile_K)
        x_outer, y_outer, x_inner, y_inner = s[C].tile(x, y, x_factor=tile_M, y_factor=tile_N)
        s[C].reorder(b, x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)
        gemm_acc = gemm_acc_nx16_int8_int8_int32(in_type, rows=1)
        s[C].unroll(x_inner)
        s[C].tensorize(y_inner, gemm_acc)
        s[C].parallel(x_outer)
    elif use_scalable_vectors:
        k_outer, k_inner = s[C].split(k, factor=tile_K)
        x_outer, x_inner = s[C].split(x, factor=tile_M)
        y_outer, y_inner = s[C].split(y, factor=tile_N, disable_predication=use_scalable_vectors)
        b_x_outer_fused = s[C].fuse(b, x_outer)
        s[C].parallel(b_x_outer_fused)
        s[C].reorder(
            b_x_outer_fused,
            y_outer,
            k_outer,
            k_inner,
            x_inner,
            y_inner,
        )
        s[C].unroll(x_inner)
        s[C].vectorize(y_inner)
    else:
        k_outer, k_inner = s[C].split(k, factor=tile_K)
        x_outer, x_inner = s[C].split(x, factor=tile_M)
        y_outer, y_inner = s[C].split(y, factor=tile_N)
        y_inner_outer, y_inner_inner = s[C].split(y_inner, nparts=4)
        b_x_outer_fused = s[C].fuse(b, x_outer)
        s[C].parallel(b_x_outer_fused)
        s[C].reorder(
            b_x_outer_fused,
            y_outer,
            k_outer,
            k_inner,
            y_inner_outer,
            x_inner,
            y_inner_inner,
        )
        s[C].unroll(y_inner_outer)
        s[C].unroll(x_inner)
        s[C].vectorize(y_inner_inner)

    # Input transform
    if A.op.name == "A_padded_K" or A.op.name == "A_padded_M":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if data_im2col.op.name == "data_im2col":
        # Either only pad_K or both pad_K and pad_M applied
        if A.op.name == "A_padded_K":
            s[data_im2col].compute_at(s[A], A.op.axis[1])
            s[A].parallel(A.op.axis[1])
        # Only pad_M applied
        elif A.op.name == "A_padded_M":
            s[data_im2col].parallel(m)
            s[A].parallel(A.op.axis[1])
        # No padding
        else:
            s[data_im2col].parallel(m)

        split_factor = 16
        n_size = data_im2col.shape[2]
        if n_size % 16 == 0:
            split_factor = 16
        elif n_size % 8 == 0:
            split_factor = 8
        else:
            # Split by kernel area (KH * KW) to ensure proper vectorization
            ic = data_im2col.op.input_tensors[0].shape[3]
            split_factor = n_size // ic

        n_outer, n_inner = s[data_im2col].split(n, split_factor)
        s[data_im2col].unroll(n_outer)
        s[data_im2col].vectorize(n_inner)
    elif padding_A:
        s[data_im2col].compute_inline()
        _, n_inner = s[A].split(A.op.axis[2], tile_N)
        s[A].vectorize(n_inner)
        s[A].compute_at(s[C], x_inner)
    else:
        s[data_im2col].compute_at(s[C], x_inner)

    A_pad = data_im2col.op.input_tensors[0]
    if A_pad.op.name == "data_pad":
        n, h, w, c = A_pad.op.axis
        n_h_fused = s[A_pad].fuse(n, h)
        s[A_pad].parallel(n_h_fused)
        s[A_pad].vectorize(c)

    # Weight transform
    if use_scalable_vectors:
        B_pad = C.op.input_tensors[1]
        s[B_pad].parallel(B_pad.op.axis[0])
        B_flat = B_pad.op.input_tensors[0]
        s[B_flat].compute_inline()

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[out].vectorize(inner)
    return s
