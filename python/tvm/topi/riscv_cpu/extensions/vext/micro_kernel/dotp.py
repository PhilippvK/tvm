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
"""Defines dotp intrinsics for matrix multiplication with RISC-V V-Extension instructions."""

import random
import string

import tvm
from tvm import te
from . import common


##########################
# N dot product Intrinsic #
##########################

# NOTE this is a dot product (a' * b)
def intrin_dotp_N(N, in_dtype, out_dtype, stride_w=1):
    """Defines a RISC-V V-Extension accelerated dot product."""
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO(weberlo, areusch): to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    if isinstance(N, tvm.tir.IntImm):
        N = N.value
    # TODO(weberlo, areusch): support more dtypes?
    assert in_dtype in ("int8", "int16")
    assert out_dtype == "int32"
    a = te.placeholder((N,), name="a", dtype=in_dtype)
    b = te.placeholder((N,), name="b", dtype=in_dtype)
    k = te.reduce_axis((0, N), name="k")
    c = te.compute(
        (1,),
        lambda i: te.sum(
            a[k].astype(out_dtype) * b[k].astype(out_dtype), axis=[k]
        ),
        name="c",
    )
    a_buf = tvm.tir.decl_buffer(
        # a.shape, a.dtype, name="a", offset_factor=1, strides=[te.var("a_s"), 1]
        a.shape, a.dtype, name="a", offset_factor=1
    )
    b_buf = tvm.tir.decl_buffer(
        # b.shape, b.dtype, name="b", offset_factor=1, strides=[te.var("b_s"), 1]
        b.shape, b.dtype, name="b", offset_factor=1
    )
    c_buf = tvm.tir.decl_buffer(
        # c.shape, c.dtype, name="c", offset_factor=1, strides=[te.var("c_s"), 1]
        c.shape, c.dtype, name="c", offset_factor=1
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        dotp_func_prefix = "dotp" if in_dtype == "int8" else "dotp16"

        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{dotp_func_prefix}_{N}_update_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    cc.access_ptr("w"),
                    # aa.strides[0] * stride_w,
                    # bb.strides[0],
                    # cc.strides[0],
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    # "int32", f"dotp_{N}_reset_{uniq_id}", cc.access_ptr("w"), cc.strides[0]
                    "int32", f"dotp_{N}_reset_{uniq_id}", cc.access_ptr("w"),
                )
            )
            return ib.get()

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{dotp_func_prefix}_{N}_body_{uniq_id}",
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    cc.access_ptr("w"),
                    # aa.strides[0] * stride_w,
                    # bb.strides[0],
                    # cc.strides[0],
                )
            )
            return ib.get()

        return _body(), _reduce_reset(), _reduce_update()

    intrin_decl = te.decl_tensor_intrin(c.op, intrin_func, binds={a: a_buf, b: b_buf, c: c_buf})
    return intrin_decl, uniq_id

def dotp_N_impl(N, uniq_id):
    """Emit C code for dotp impl."""
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    # bb_pad_size = N * K  # ???
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = (
        common.common_includes
        + f"""

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_body_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc) {{
    int32_t sum = 0;
  for (int i = 0; i < {N}; i++) {{
    sum += (int32_t) aa[i] * (int32_t) bb[i];
  }}
  cc[0] = sum;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_body_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc) {{
  // printf("dotp_{N}_body_{uniq_id}\\n");
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp_{N}_body_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  size_t vl = vsetvl_e32m8({N});
  vint32m8_t res = vmv_v_x_i32m8(0, vl);
  int8_t *aa_ptr = (int8_t *) &aa[0];
  int8_t *bb_ptr = (int8_t *) &bb[0];
  size_t remaining = {N};
  while (remaining > 0) {{
    vl = vsetvl_e32m8(remaining);
    // TODO: find out if this is faster with int8?
    vint16m4_t r0 = vsext_vf2_i16m4(vle8_v_i8m2(aa_ptr, vl), vl);
    vint16m4_t c0 = vsext_vf2_i16m4(vle8_v_i8m2(bb_ptr, vl), vl);
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
  cc[0] = acc;

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_update_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc) {{
  int32_t sum = 0;
  for (int i = 0; i < {N}; i++) {{
    sum += (int32_t) aa[i] * (int32_t) bb[i];
  }}
  cc[0] += sum;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_update_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc) {{
  // printf("dotp_{N}_update_{uniq_id}\\n");
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp_{N}_update_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  // printf("A\\n");
  size_t vl = vsetvl_e32m8({N});
  // printf("B\\n");
  vint32m8_t res = vmv_v_x_i32m8(0, vl);
  // printf("C\\n");
  int8_t *aa_ptr = (int8_t *) &aa[0];
  int8_t *bb_ptr = (int8_t *) &bb[0];
  size_t remaining = {N};
  // printf("D\\n");
  while (remaining > 0) {{
    // printf("E\\n");
    // printf("remaining=%u\\n", remaining);
    vl = vsetvl_e32m8(remaining);
    // printf("F\\n");
    // TODO: find out if this is faster with int8?
    // printf("G\\n");
    vint16m4_t r0 = vsext_vf2_i16m4(vle8_v_i8m2(aa_ptr, vl), vl);
    // printf("H\\n");
    vint16m4_t c0 = vsext_vf2_i16m4(vle8_v_i8m2(bb_ptr, vl), vl);
    // printf("I\\n");
    // vint32m8_t vwmacc_vv_i32m8 (vint32m8_t vd, vint16m4_t vs1, vint16m4_t vs2, size_t vl);
    res = vwmacc_vv_i32m8(res, r0, c0, vl);
    // printf("J\\n");
    aa_ptr += vl;
    bb_ptr += vl;
    remaining -= vl;
    // printf("K\\n");
  }}
  // printf("L\\n");
  vl = vsetvl_e32m1(1);
  // printf("M\\n");
  vint32m1_t red = vmv_v_x_i32m1(0, vl);
  // printf("N\\n");
  vl = vsetvl_e32m8({N});
  // printf("O\\n");
  red = vredsum_vs_i32m8_i32m1(red, res, red, vl);
  // printf("P\\n");
  int32_t acc = vmv_x_s_i32m1_i32(red);
  // printf("Q\\n");
  cc[0] += acc;
  // printf("R\\n");

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp16_{N}_body_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc) {{
  int32_t sum = 0;
  for (int i = 0; i < {N}; i++) {{
    sum += (int32_t) aa[i] * (int32_t) bb[i];
  }}
  cc[0] = sum;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp16_{N}_body_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {{
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp16_{N}_body_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){{
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }}

  size_t vl = vsetvl_e32m8({N});
  vint32m8_t res = vmv_v_x_i32m8(0, vl);
  int16_t *aa_ptr = (int16_t *) &aa[0];
  int16_t *bb_ptr = (int16_t *) &bb[0];
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
  cc[0] = acc;

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp16_{N}_update_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc) {{
  int32_t sum = 0;
  for (int i = 0; i < {N}; i++) {{
    sum += (int32_t) aa[i] * (int32_t) bb[i];
  }}
  cc[0] += sum;
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp16_{N}_update_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc) {{
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp16_{N}_update_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  size_t vl = vsetvl_e32m8({N});
  vint32m8_t res = vmv_v_x_i32m8(0, vl);
  int16_t *aa_ptr = (int16_t *) &aa[0];
  int16_t *bb_ptr = (int16_t *) &bb[0];
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
  cc[0] += acc;

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_reset_{uniq_id}(int32_t *cc) {{
  // for (int i = 0; i < {N}; i++) {{
  cc[0] = 0;
  // }}
  return 0;
}}

"""
    )
    return cc_code
