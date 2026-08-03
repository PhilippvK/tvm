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
"""GeMM dense schedule on AArch64"""
import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.arm_cpu.arm_utils import get_tiling_A, get_tiling_B_transformed, pad_dim_to_multiple
from ..utils import get_const_tuple, traverse_inline
from .. import tag

NO_RANDOM_COMPUTE_LOCATION = {
    "meta_schedule.no_random_compute_location": True,
}


def _choose_ki(K: int, K_MAX: int, K_MIN: int) -> int:
    """Choose the largest valid KI <= min(K, K_MAX).

    K_MIN is the required alignment / minimum reduction tile.
    K_MAX is the maximum reduction span supported by the implementation.
    """
    assert K_MIN > 0
    assert K_MAX >= K_MIN
    assert K % K_MIN == 0, f"K={K} must be divisible by K_MIN={K_MIN}"
    assert K_MAX % K_MIN == 0, f"K_MAX={K_MAX} must be divisible by K_MIN={K_MIN}"

    upper = min(K, K_MAX)

    # Prefer the largest KI that divides K and is aligned to K_MIN.
    for ki in range(upper, K_MIN - 1, -K_MIN):
        if K % ki == 0:
            return ki

    raise ValueError(f"Could not find valid KI for K={K}, K_MAX={K_MAX}, K_MIN={K_MIN}")


def dense_ime_packed_compute(
    cfg,
    data,
    weight,
    bias=None,
    out_dtype="int32",
    # MI=4,
    # NI=4,
    MI=8,
    NI=8,
    K_MAX=512,
    K_MIN=8,
    name="T_matmul_NT",
):
    """IME-friendly packed dense compute.

    Logical dense convention:
      data:   [M, K]
      weight: [N, K]
      output: [M, N]

    Microtile-major packed layout:

      A_pack[MO, KO, KB, MT, 4, 8]
      B_pack[NO, KO, KB, NT, 4, 8]
      C_pack[MO, NO, MT, NT, 4, 4]

    where:
      MT = MI // 4
      NT = NI // 4
      KB = KI // 8

    This layout keeps every 4xKx4 subtile contiguous, so the same
    packing can support 4xKx4, 8xKx4, 4xKx8, and 8xKx8 ukernels.

    K_MAX:
      maximum K span consumed by one tensorized microkernel.

    K_MIN:
      minimum/alignment unit, e.g. 8 for IME dot granularity.
    """
    assert data.dtype == "int8"
    assert weight.dtype == "int8"
    assert out_dtype == "int32"

    assert MI in [4, 8], f"Unsupported MI={MI}"
    assert NI in [4, 8], f"Unsupported NI={NI}"
    assert MI % 4 == 0
    assert NI % 4 == 0

    K_STEP = 8
    assert K_MIN == K_STEP or K_MIN % K_STEP == 0
    assert K_MAX % K_STEP == 0

    M, K = get_const_tuple(data.shape)
    N, K2 = get_const_tuple(weight.shape)
    assert K == K2

    assert M % MI == 0, f"M={M} must be divisible by MI={MI}"
    assert N % NI == 0, f"N={N} must be divisible by NI={NI}"
    assert K % K_MIN == 0, f"K={K} must be divisible by K_MIN={K_MIN}"

    KI = _choose_ki(K, K_MAX, K_MIN)
    assert KI % K_STEP == 0, f"KI={KI} must be divisible by K_STEP={K_STEP}"

    MO = M // MI
    NO = N // NI
    KO = K // KI

    MT = MI // 4
    NT = NI // 4
    KB = KI // K_STEP
    print("KB", KB)
    print("KB", KB)

    pack_attrs = {
        "meta_schedule.no_random_compute_location": True,
    }
    b_pack_attrs = {
        **pack_attrs,
        "layout_free_placeholders": [weight],
    }

    # A_pack layout:
    #
    #   [mo, ko, kb, mt, mi4, kk]
    #
    # For each K_STEP=8 chunk, each 4x8 A microtile is contiguous.
    # Flat order inside one macro A tile:
    #
    #   kb0, mt0, 4x8
    #   kb0, mt1, 4x8
    #   kb1, mt0, 4x8
    #   kb1, mt1, 4x8
    #   ...
    #
    # This matches ukernel offsets:
    #
    #   a_off = kt * (MT * 32) + mt * 32
    #
    if KO == 1 and KB == 1:
        if MO == 1:
            A_pack = te.compute(
                (MT, 4, K_STEP),
                lambda mt, mi4, kk: data[
                    0 * MI + mt * 4 + mi4,
                    0 * KI + 0 * K_STEP + kk,
                ],
                name="A_pack",
                attrs=pack_attrs,
            )
        else:
            A_pack = te.compute(
                (MO, MT, 4, K_STEP),
                lambda mo, mt, mi4, kk: data[
                    mo * MI + mt * 4 + mi4,
                    0 * KI + 0 * K_STEP + kk,
                ],
                name="A_pack",
                attrs=pack_attrs,
            )
    else:
        A_pack = te.compute(
            (MO, KO, KB, MT, 4, K_STEP),
            lambda mo, ko, kb, mt, mi4, kk: data[
                mo * MI + mt * 4 + mi4,
                ko * KI + kb * K_STEP + kk,
            ],
            name="A_pack",
            attrs=pack_attrs,
        )

    # B_pack layout:
    #
    #   [no, ko, kb, nt, ni4, kk]
    #
    # For each K_STEP=8 chunk, each 4x8 B microtile is contiguous.
    # This matches ukernel offsets:
    #
    #   b_off = kt * (NT * 32) + nt * 32
    #
    if KO == 1 and KB == 1:
        if MO == 1:
            B_pack = te.compute(
                (NT, 4, K_STEP),
                lambda nt, ni4, kk: weight[
                    0 * NI + nt * 4 + ni4,
                    0 * KI + 0 * K_STEP + kk,
                ],
                name="B_pack",
                attrs=b_pack_attrs,
            )
        else:
            B_pack = te.compute(
                (NO, NT, 4, K_STEP),
                lambda no, nt, ni4, kk: weight[
                    no * NI + nt * 4 + ni4,
                    0 * KI + 0 * K_STEP + kk,
                ],
                name="B_pack",
                attrs=b_pack_attrs,
            )
    else:
        B_pack = te.compute(
            (NO, KO, KB, NT, 4, K_STEP),
            lambda no, ko, kb, nt, ni4, kk: weight[
                no * NI + nt * 4 + ni4,
                ko * KI + kb * K_STEP + kk,
            ],
            name="B_pack",
            attrs=b_pack_attrs,
        )

    rko = te.reduce_axis((0, KO), "rko")
    rkb = te.reduce_axis((0, KB), "rkb")
    rkk = te.reduce_axis((0, K_STEP), "rkk")

    reduction_attrs = {
        # "meta_schedule.no_random_compute_location": True,
        # "layout_free_placeholders": [B_pack],
        "layout_free_placeholders": [weight],
    }

    # C_pack layout:
    #
    #   [mo, no, mt, nt, mi4, ni4]
    #
    # Each 4x4 C microtile is contiguous.
    #
    # For MI=NI=8, the four C microtiles are stored at flat byte offsets:
    #
    #   mt=0, nt=0 ->   0
    #   mt=0, nt=1 ->  64
    #   mt=1, nt=0 -> 128
    #   mt=1, nt=1 -> 192
    #
    # i.e. microtile-major, not row-major 8x8.
    #
    # C_pack = te.compute(
    #     (MO, NO, MT, NT, 4, 4),
    #     lambda mo, no, mt, nt, mi4, ni4: te.sum(
    #         A_pack[mo, rko, rkb, mt, mi4, rkk].astype(out_dtype)
    #         * B_pack[no, rko, rkb, nt, ni4, rkk].astype(out_dtype),
    #         axis=[rko, rkb, rkk],
    #     ),
    #     name="C_pack",
    #     attrs=reduction_attrs,
    # )

    if KO == 1 and KB == 1:
        # C_pack = te.compute(
        #     (MO, NO, MT, NT, 4, 4),
        #     lambda mo, no, mt, nt, mi4, ni4: te.sum(
        #         A_pack[mo, 0, 0, mt, mi4, rkk].astype(out_dtype) * B_pack[no, 0, 0, nt, ni4, rkk].astype(out_dtype),
        #         axis=[rkk],
        #     ),
        #     name="C_pack",
        #     attrs=reduction_attrs,
        # )
        if MO == 1 and NO == 1:
            # C_pack = te.compute(
            #     (MT, NT, 4, 4),
            #     lambda mt, nt, mi4, ni4: te.sum(
            #         A_pack[0, mt, mi4, rkk].astype(out_dtype) * B_pack[0, nt, ni4, rkk].astype(out_dtype),
            #         axis=[rkk],
            #     ),
            #     name="C_pack",
            #     attrs=reduction_attrs,
            # )
            C_pack = te.compute(
                (MT, NT, 4, 4),
                lambda mt, nt, mi4, ni4: te.sum(
                    A_pack[mt, mi4, rkk].astype(out_dtype) * B_pack[nt, ni4, rkk].astype(out_dtype),
                    axis=[rkk],
                ),
                name="C_pack",
                attrs=reduction_attrs,
            )
        else:
            C_pack = te.compute(
                (MO, NO, MT, NT, 4, 4),
                lambda mo, no, mt, nt, mi4, ni4: te.sum(
                    A_pack[mo, mt, mi4, rkk].astype(out_dtype) * B_pack[no, nt, ni4, rkk].astype(out_dtype),
                    axis=[rkk],
                ),
                name="C_pack",
                attrs=reduction_attrs,
            )
    elif KO == 1:
        rkb = te.reduce_axis((0, KB), "rkb")
        C_pack = te.compute(
            (MO, NO, MT, NT, 4, 4),
            lambda mo, no, mt, nt, mi4, ni4: te.sum(
                A_pack[mo, 0, rkb, mt, mi4, rkk].astype(out_dtype) * B_pack[no, 0, rkb, nt, ni4, rkk].astype(out_dtype),
                axis=[rkb, rkk],
            ),
            name="C_pack",
            attrs=reduction_attrs,
        )
    elif KB == 1:
        rko = te.reduce_axis((0, KO), "rko")
        C_pack = te.compute(
            (MO, NO, MT, NT, 4, 4),
            lambda mo, no, mt, nt, mi4, ni4: te.sum(
                A_pack[mo, rko, 0, mt, mi4, rkk].astype(out_dtype) * B_pack[no, rko, 0, nt, ni4, rkk].astype(out_dtype),
                axis=[rko, rkk],
            ),
            name="C_pack",
            attrs=reduction_attrs,
        )
    else:
        rko = te.reduce_axis((0, KO), "rko")
        rkb = te.reduce_axis((0, KB), "rkb")
        C_pack = te.compute(
            (MO, NO, MT, NT, 4, 4),
            lambda mo, no, mt, nt, mi4, ni4: te.sum(
                A_pack[mo, rko, rkb, mt, mi4, rkk].astype(out_dtype)
                * B_pack[no, rko, rkb, nt, ni4, rkk].astype(out_dtype),
                axis=[rko, rkb, rkk],
            ),
            name="C_pack",
            attrs=reduction_attrs,
        )
    if MO == 1 and NO == 1:
        C = te.compute(
            (M, N),
            lambda m, n: C_pack[
                # m // MI,
                # n // NI,
                (m % MI) // 4,
                (n % NI) // 4,
                m % 4,
                n % 4,
            ],
            name=name,
        )
    else:
        C = te.compute(
            (M, N),
            lambda m, n: C_pack[
                m // MI,
                n // NI,
                (m % MI) // 4,
                (n % NI) // 4,
                m % 4,
                n % 4,
            ],
            name=name,
        )

    if bias is not None:
        C = te.compute(
            (M, N),
            lambda m, n: C[m, n] + bias[n].astype(out_dtype),
            name=name + "_bias",
        )

    return C


# def dense_ime_packed_compute(
#     cfg,
#     data,
#     weight,
#     bias=None,
#     out_dtype="int32",
#     MI=4,
#     NI=4,
#     K_MAX=512,
#     K_MIN=8,
#     name="T_matmul_NT",
# ):
#     """IME-friendly packed dense compute.
#
#     Logical dense convention:
#       data:   [M, K]
#       weight: [N, K]
#       output: [M, N]
#
#     Packed layout:
#       A_pack[MO, KO, MI, KI]
#       B_pack[NO, KO, NI, KI]
#       C_pack[MO, NO, MI, NI]
#
#     K_MAX:
#       maximum K span consumed by one tensorized microkernel.
#
#     K_MIN:
#       minimum/alignment unit, e.g. 8 for the IME dot granularity.
#     """
#     assert data.dtype == "int8"
#     assert weight.dtype == "int8"
#     assert out_dtype == "int32"
#
#     M, K = get_const_tuple(data.shape)
#     N, K2 = get_const_tuple(weight.shape)
#     assert K == K2
#
#     assert M % MI == 0, f"M={M} must be divisible by MI={MI}"
#     assert N % NI == 0, f"N={N} must be divisible by NI={NI}"
#     assert K % K_MIN == 0, f"K={K} must be divisible by K_MIN={K_MIN}"
#
#     KI = _choose_ki(K, K_MAX, K_MIN)
#
#     MO = M // MI
#     NO = N // NI
#     KO = K // KI
#
#     A_pack = te.compute(
#         (MO, KO, MI, KI),
#         lambda mo, ko, mi, ki: data[mo * MI + mi, ko * KI + ki],
#         name="A_pack",
#     )
#
#     B_pack = te.compute(
#         (NO, KO, NI, KI),
#         lambda no, ko, ni, ki: weight[no * NI + ni, ko * KI + ki],
#         name="B_pack",
#     )
#
#     rko = te.reduce_axis((0, KO), "rko")
#     rki = te.reduce_axis((0, KI), "rki")
#
#     C_pack = te.compute(
#         (MO, NO, MI, NI),
#         lambda mo, no, mi, ni: te.sum(
#             A_pack[mo, rko, mi, rki].astype(out_dtype)
#             * B_pack[no, rko, ni, rki].astype(out_dtype),
#             axis=[rko, rki],
#         ),
#         name="C_pack",
#         attrs={
#             "ime_m_tile": MI,
#             "ime_n_tile": NI,
#             "ime_k_tile": KI,
#             "ime_k_min": K_MIN,
#             "ime_k_max": K_MAX,
#         },
#     )
#
#     C = te.compute(
#         (M, N),
#         lambda m, n: C_pack[m // MI, n // NI, m % MI, n % NI],
#         name=name,
#     )
#
#     if bias is not None:
#         C = te.compute(
#             (M, N),
#             lambda m, n: C[m, n] + bias[n].astype(out_dtype),
#             name=name + "_bias",
#         )
#
#     return C


# def dense_ime_packed_compute(cfg, data, weight, bias=None, out_dtype=None, MI=4, NI=4, KI=8):
# def dense_ime_packed_compute(cfg, data, weight, bias=None, out_dtype=None, MI=4, NI=4, K_MAX=512, K_MIN=8):
#     # print("dense_ime_packed_compute", cfg, data, weight, bias, out_dtype, MI, NI, KI)
#     print("dense_ime_packed_compute", cfg, data, weight, bias, out_dtype, MI, NI, K_MAX, K_MIN)
#     M, K = get_const_tuple(data.shape)
#     print("M", M)
#     print("K", K)
#     N, K2 = get_const_tuple(weight.shape)
#     print("N", N)
#     print("K2", K2)
#     assert K == K2
#     KI = min(K_MAX, K)
#     print("KI", KI)
#     assert KI % K_MIN == 0
#     assert M % MI == 0
#     assert N % NI == 0
#     assert K % KI == 0
#
#     MO = M // MI
#     NO = N // NI
#     KO = K // KI
#
#     A_pack = te.compute(
#         (MO, KO, MI, KI),
#         lambda mo, ko, mi, ki: data[mo * MI + mi, ko * KI + ki],
#         name="A_pack",
#     )
#
#     B_pack = te.compute(
#         (NO, KO, NI, KI),
#         lambda no, ko, ni, ki: weight[no * NI + ni, ko * KI + ki],
#         name="B_pack",
#     )
#
#     rko = te.reduce_axis((0, KO), "rko")
#     rki = te.reduce_axis((0, KI), "rki")
#
#     C_pack = te.compute(
#         (MO, NO, MI, NI),
#         lambda mo, no, mi, ni: te.sum(
#             A_pack[mo, rko, mi, rki].astype(out_dtype)
#             * B_pack[no, rko, ni, rki].astype(out_dtype),
#             axis=[rko, rki],
#         ),
#         name="C_pack",
#     )
#
#     C = te.compute(
#         (M, N),
#         lambda m, n: C_pack[m // MI, n // NI, m % MI, n % NI],
#         name="T_matmul_NT",
#     )
#
#     return C


# Compute function
def dense_gemm_compute(cfg, data, weight, bias=None, out_dtype=None, transpose_a=False, transpose_b=True):
    """
    Compute dense using GeMM.

    Parameters
    ----------
    cfg : Autotvm tuning space config file,
        empty in this case, but it's needed as an arg.

    data : tvm.te.Tensor
        2-D with shape [M, K] or [K, M].

    weight : tvm.te.Tensor
        2-D with shape [K, N] or [N, K].

    bias : Optional[tvm.te.Tensor]
        1-D with shape [N]


    out_dtype : Optional[str]
        Specifies the output data type.

    transpose_a : Optional[bool] = False
    Whether the data tensor is in transposed format.

    transpose_b : Optional[bool] = True
    Whether the weight tensor is in transposed format.

    Returns
    -------
    out : tvm.te.Tensor
        1-D with shape [out_dim]
    """

    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if bool(transpose_b):  # out_dim
        (N, _) = get_const_tuple(weight.shape)
    else:
        (_, N) = get_const_tuple(weight.shape)

    tile_M, tile_K = get_tiling_A(False, out_dtype)
    tile_N, _ = get_tiling_B_transformed(False, out_dtype, False)

    M_padded, pad_M = pad_dim_to_multiple(M, tile_M)
    K_padded, pad_K = pad_dim_to_multiple(K, tile_K)
    N_padded, pad_N = pad_dim_to_multiple(N, tile_N)
    m_pad_after = (pad_M, pad_K)
    n_pad_after = (pad_N, pad_K) if transpose_b else (pad_K, pad_N)

    if pad_M != 0 or pad_K != 0:
        data = nn.pad(data, pad_before=(0, 0), pad_after=m_pad_after, name="data_padded")

    k = te.reduce_axis((0, K_padded), name="k")

    if bool(transpose_b):
        weight = te.compute(
            (K_padded, N_padded), lambda x, y: weight[y, x], name="weight_transposed"
        )

    if pad_N != 0 or pad_K != 0:
        weight = nn.pad(weight, pad_before=(0, 0), pad_after=n_pad_after, name="weight_padded")

    C = te.compute(
        (M_padded, N_padded),
        lambda x, y: te.sum(
            data[x, k].astype(out_dtype) * weight[k, y].astype(out_dtype),
            axis=k,
        ).astype(out_dtype),
        name="C",
    )

    if bias is not None:
        C = te.compute(
            (M_padded, N_padded),
            lambda i, j: C[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
            name="dense_biased_output",
        )

    # We need to ensure that infer bound pass does not remove the padding
    # which is necessary for the tensorizations to work. So we need to
    # add a dummy reference to the padding area of the result
    zero = (
        tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
        - tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
    )

    out = te.compute(
        (M, N), lambda x, y: (C[x, y] + zero).astype(out_dtype), name="dense_gemm_output"
    )

    return out


def _dense_gemm_schedule(s, out):
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    out_type = A.dtype
    tile_M, tile_K = get_tiling_A(False, out_type)
    tile_N, _ = get_tiling_B_transformed(False, out_type, False)

    if C.op.name == "dense_biased_output":
        s[C].compute_inline()
        C = C.op.input_tensors[0]
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis

    k_outer, k_inner = s[C].split(k, factor=tile_K)
    x_outer, x_inner = s[C].split(x, factor=tile_M)
    y_outer, y_inner = s[C].split(y, factor=tile_N)
    y_inner_outer, y_inner_inner = s[C].split(y_inner, nparts=4)
    s[C].parallel(x_outer)
    s[C].reorder(
        x_outer,
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

    return s


def dense_gemm_schedule(cfg, outs):
    """Schedule the dense_gemm strategy"""
    s = te.create_schedule([x.op for x in outs])
    out = outs[0]
    x, y = out.op.axis
    _, inner = s[out].split(y, 4)
    s[out].parallel(x)
    s[out].vectorize(inner)

    def _callback(op):
        if "dense_gemm_output" in op.name:
            _dense_gemm_schedule(s, op.output(0))

    traverse_inline(s, out.op, _callback)
    return s
