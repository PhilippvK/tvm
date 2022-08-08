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
# pylint: disable=invalid-name, no-value-for-parameter
"""Defines gemm intrinsics for matrix multiplication with RISC-V V-Extension instructions."""

import random
import string

import tvm
from tvm import te
from . import common


##########################
# MxKxN MatMul Intrinsic #
##########################

# NOTE this is transposed matmul (A * B^T)
def intrin_gemm_MxKxN(M, K, N, in_dtype, out_dtype, stride_w=1):
    """Defines a RISC-V V-Extension accelerated transposed matmul."""
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO(weberlo, areusch): to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    if isinstance(M, tvm.tir.IntImm):
        M = M.value
    if isinstance(K, tvm.tir.IntImm):
        K = K.value
    if isinstance(N, tvm.tir.IntImm):
        N = N.value
    # TODO(weberlo, areusch): support more dtypes?
    assert in_dtype in ("int8", "int16")
    assert out_dtype == "int32"
    A = te.placeholder((M * stride_w - (stride_w - 1), K), name="a", dtype=in_dtype)
    B = te.placeholder((N, K), name="b", dtype=in_dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i * stride_w, k].astype(out_dtype) * B[j, k].astype(out_dtype), axis=k
        ),
        name="c",
    )
    A_buf = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="A", offset_factor=1, strides=[te.var("A_s"), 1]
    )
    B_buf = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="B", offset_factor=1, strides=[te.var("B_s"), 1]
    )
    C_buf = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="C", offset_factor=1, strides=[te.var("C_s"), 1]
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        gemm_func_prefix = "gemm" if in_dtype == "int8" else "gemm16"

        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{gemm_func_prefix}_{M}x{K}x{N}_update_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    cc.access_ptr("w"),
                    aa.strides[0] * stride_w,
                    bb.strides[0],
                    cc.strides[0],
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32", f"gemm_{M}x{K}x{N}_reset_{uniq_id}", cc.access_ptr("w"), cc.strides[0]
                )
            )
            return ib.get()

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{gemm_func_prefix}_{M}x{K}x{N}_body_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    cc.access_ptr("w"),
                    aa.strides[0] * stride_w,
                    bb.strides[0],
                    cc.strides[0],
                )
            )
            return ib.get()

        return _body(), _reduce_reset(), _reduce_update()

    intrin_decl = te.decl_tensor_intrin(C.op, intrin_func, binds={A: A_buf, B: B_buf, C: C_buf})
    return intrin_decl, uniq_id


def gemm_MxKxN_impl(M, K, N, uniq_id):
    """Emit C code for gemm impl."""
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    # aa_pad_size = M * K
    bb_pad_size = N * K
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = (
        common.common_includes
        + f"""


#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm_{M}x{K}x{N}_body_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K}; l++) {{
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }}
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm_{M}x{K}x{N}_body_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t bb_pad[{bb_pad_size}];
  int32_t retcode = 0;

  // if ( {M} < 16 || {N} < 16 ) {{
  if (0) {{
    retcode = gemm_{M}x{K}x{N}_body_loop_{uniq_id}(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      // int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      // int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      // int32_t sum = 0;
      // for (int l = 0; l < {K} / 4; l++) {{
      //   sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
      //   ++ aa_ptr; ++ bb_ptr;
      // }}
      size_t vl = vsetvl_e32m8({K});
      vint32m8_t res = vmv_v_x_i32m8(0, vl);
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];
      size_t remaining = {K};
      while (remaining > 0) {{
        vl = vsetvl_e32m8(remaining);
        vint16m4_t r0 = vsext_vf2_i16m4(vle8_v_i8m2(aa_ptr, vl), vl);
        vint16m4_t c0 = vsext_vf2_i16m4(vle8_v_i8m2(bb_ptr, vl), vl);
        res = vwmacc_vv_i32m8(res, r0, c0, vl);
        aa_ptr += vl;
        bb_ptr += vl;
        remaining -= vl;
      }}
      vl = vsetvl_e32m1(1);
      vint32m1_t red = vmv_v_x_i32m1(0, vl);
      vl = vsetvl_e32m8({N});
      red = vredsum_vs_i32m8_i32m1(red, res, red, vl);
      int32_t acc = vmv_x_s_i32m1_i32(red);
      cc[i*C_stride + j] = acc;
    }}
  }}

out:
  return retcode;
}}


#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm_{M}x{K}x{N}_update_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K}; l++) {{
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }}
      cc[i*C_stride + j] += sum;
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm_{M}x{K}x{N}_update_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int16_t bb_pad[{bb_pad_size}];
  int32_t retcode = 0;

  // if ( {M} < 16 || {N} < 16 ) {{
  if (0) {{
    retcode = gemm_{M}x{K}x{N}_update_loop_{uniq_id}(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      size_t vl = vsetvl_e32m8({K});
      vint32m8_t res = vmv_v_x_i32m8(0, vl);
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];
      size_t remaining = {K};
      while (remaining > 0) {{
        vl = vsetvl_e32m8(remaining);
        vint16m4_t r0 = vsext_vf2_i16m4(vle8_v_i8m2(aa_ptr, vl), vl);
        vint16m4_t c0 = vsext_vf2_i16m4(vle8_v_i8m2(bb_ptr, vl), vl);
        res = vwmacc_vv_i32m8(res, r0, c0, vl);
        aa_ptr += vl;
        bb_ptr += vl;
        remaining -= vl;
      }}
      vl = vsetvl_e32m1(1);
      vint32m1_t red = vmv_v_x_i32m1(0, vl);
      vl = vsetvl_e32m8({N});
      red = vredsum_vs_i32m8_i32m1(red, res, red, vl);
      int32_t acc = vmv_x_s_i32m1_i32(red);
      cc[i*C_stride + j] += acc;
    }}
  }}

out:
  return retcode;
}}


#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm16_{M}x{K}x{N}_body_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K}; l++) {{
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }}
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm16_{M}x{K}x{N}_body_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int32_t retcode = 0;

  // if ( {M} < 2 || {N} < 2 ) {{
  if (0) {{
    retcode = gemm16_{M}x{K}x{N}_body_loop_{uniq_id}(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }}

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){{
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      // int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      // int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      // int32_t sum = 0;
      // for (int l = 0; l < {K} / 2; l++) {{
      //   sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
      //   ++ aa_ptr; ++ bb_ptr;
      // }}
      size_t vl = vsetvl_e32m8({N});
      vint32m8_t res = vmv_v_x_i32m8(0, vl);
      int16_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int16_t *bb_ptr = (int32_t *) &bb[j*B_stride];
      size_t remaining = {N};
      while (remaining > 0) {{
        vl = vsetvl_e32m8(remaining);
        // vint32m8_t r0 = vsext_vf4_i32m8(vle8_v_i8m2(ip_r0, vl), vl);
        // vint32m8_t c0 = vsext_vf4_i32m8(vle8_v_i8m2(ip_c0, vl), vl);
        vint16m4_t r0 = vle16_v_i16m4(aa_ptr, vl);
        vint16m4_t c0 = vle16_v_i16m4(bb_ptr, vl);
        // vint32m8_t vwmacc_vv_i32m8 (vint32m8_t vd, vint16m4_t vs1, vint16m4_t vs2, size_t vl);
        res = vwmacc_vv_i32m8(res, r0, c0, vl);
        aa_ptr += vl;
        bb_ptr += vl;
        remaining -= vl;
      }}
      vl = vsetvl_e32m1(1);
      vint32m1_t red = vmv_v_x_i32m1(0, vl);
      vl = vsetvl_e32m8({N});
      red = vredsum_vs_i32m8_i32m1(red, res, red, vl);
      int32_t acc = vmv_x_s_i32m1_i32(red);
      cc[i*C_stride + j] = acc;
    }}
  }}

out:
  return retcode;
}}


#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm16_{M}x{K}x{N}_update_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      int32_t sum = 0;
      for (int l = 0; l < {K}; l++) {{
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }}
      cc[i*C_stride + j] += sum;
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm16_{M}x{K}x{N}_update_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int32_t retcode = 0;

  // if ( {M} < 2 || {N} < 2 ) {{
  if (0) {{
    retcode = gemm16_{M}x{K}x{N}_update_loop_{uniq_id}(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      // 4x unrolling this loop could lead to 50% less instructions
      // int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      // int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      // int32_t sum = 0;
      // for (int l = 0; l < {K} / 2; l++) {{
      //   sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
      //   ++ aa_ptr; ++ bb_ptr;
      // }}
      // cc[i*C_stride + j] += sum;
      size_t vl = vsetvl_e32m8({N});
      vint32m8_t res = vmv_v_x_i32m8(0, vl);
      int16_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int16_t *bb_ptr = (int32_t *) &bb[j*B_stride];
      size_t remaining = {N};
      while (remaining > 0) {{
        vl = vsetvl_e32m8(remaining);
        // vint32m8_t r0 = vsext_vf4_i32m8(vle8_v_i8m2(ip_r0, vl), vl);
        // vint32m8_t c0 = vsext_vf4_i32m8(vle8_v_i8m2(ip_c0, vl), vl);
        vint16m4_t r0 = vle16_v_i16m4(aa_ptr, vl);
        vint16m4_t c0 = vle16_v_i16m4(bb_ptr, vl);
        // vint32m8_t vwmacc_vv_i32m8 (vint32m8_t vd, vint16m4_t vs1, vint16m4_t vs2, size_t vl);
        res = vwmacc_vv_i32m8(res, r0, c0, vl);
        aa_ptr += vl;
        bb_ptr += vl;
        remaining -= vl;
      }}
      vl = vsetvl_e32m1(1);
      vint32m1_t red = vmv_v_x_i32m1(0, vl);
      vl = vsetvl_e32m8({N});
      red = vredsum_vs_i32m8_i32m1(red, res, red, vl);
      int32_t acc = vmv_x_s_i32m1_i32(red);
      cc[i*C_stride + j] += acc;
    }}
  }}

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__ ((always_inline)) int32_t gemm_{M}x{K}x{N}_reset_{uniq_id}(int32_t *cc, int C_stride) {{
  for (int i = 0; i < {M}; i++) {{
    for (int j = 0; j < {N}; j++) {{
      cc[i*C_stride + j] = 0;
    }}
  }}
  return 0;
}}

"""
    )
    return cc_code
