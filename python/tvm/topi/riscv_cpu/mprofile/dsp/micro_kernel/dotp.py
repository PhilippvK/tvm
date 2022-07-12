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
"""Defines dotp intrinsics for matrix multiplication with v7e-m DSP instructions."""

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
    """Defines a v7e-m DSP-accelerated dot product."""
    print(f"!intrin_dotp_{N}!")
    print("in_dtype =", in_dtype, "out_dtype =", out_dtype, "stride_w =", stride_w)
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
    print(f"!dotp_{N}_impl!")
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    # bb_pad_size = N * K  # ???
    bb_pad_size = N   # ???
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = (
        common.common_includes
        + f"""

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_body_rest_{uniq_id}(
    int N,
    int8_t *aa, int8_t *bb, int32_t *cc) {{
  int n_base = (N / 4) * 4;
  int8_t *a_ptr = &aa[n_base];
  int8_t *b_ptr = &bb[n_base];
  switch ( N % 4 ) {{
  case 1:
    cc[0] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    break;
  case 2:
    cc[0] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
            + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
    break;
  case 3:
    cc[0] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
            + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
            + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
    break;
  }}
  return 0;
}}

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
  // int16_t bb_pad[{bb_pad_size}];
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp_{N}_body_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  // for (int i = 0; i < {N} / 4; i++) {{
  //   read_and_pad(&bb[i*4], (int32_t*) &bb_pad[i*4], (int32_t*) &bb_pad[i*4 + 2]);
  // }}

  // int16_t aa_pad_line[{N}];
  // for (int i = 0; i < {N} / 4; i++) {{
  //   read_and_pad(&aa[i*4], (int32_t*) &aa_pad_line[i*4], (int32_t*) &aa_pad_line[i*4 + 2]);
  // }}

  // int32_t *aa_ptr = (int32_t *) aa_pad_line;
  // int32_t *bb_ptr = (int32_t *) &bb_pad[0];
  int32_t *aa_ptr = (int32_t *) &aa[0];
  int32_t *bb_ptr = (int32_t *) &bb[0];
  int32_t sum = 0;
  for (int i = 0; i < ({N} / 4); i++) {{
    // sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
    sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
    aa_ptr++ ; bb_ptr++;
  }}
  cc[0] = sum;

  if ( {N} % 4 != 0 )
    dotp_{N}_body_rest_{uniq_id}({N}, aa, bb, cc);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_update_rest_{uniq_id}(
    int N,
    int8_t *aa, int8_t *bb, int32_t *cc) {{
  int n_base = (N / 4) * 4;
  int8_t *a_ptr = &aa[n_base];
  int8_t *b_ptr = &bb[n_base];
  switch ( N % 4 ) {{
  case 1:
    cc[0] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    break;
  case 2:
    cc[0] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
             + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
    break;
  case 3:
    cc[0] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
             + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
             + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
    break;
  }}
  return 0;
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
    // printf("AAAA\\n");
  // int16_t bb_pad[{bb_pad_size}];  // ???
  // int16_t aa_pad_line[{N}];
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp_{N}_update_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}

  // for (int i = 0; i < {N} / 4; i++) {{
  //   read_and_pad(&bb[i*4], (int32_t*) &bb_pad[i*4], (int32_t*) &bb_pad[i*4 + 2]);
  //   read_and_pad(&aa[i*4], (int32_t*) &aa_pad_line[i*4], (int32_t*) &aa_pad_line[i*4 + 2]);
  // }}

  // int32_t *aa_ptr = (int32_t *) aa_pad_line;
  // int32_t *bb_ptr = (int32_t *) &bb_pad[0];
  int32_t *aa_ptr = (int32_t *) &aa[0];
  int32_t *bb_ptr = (int32_t *) &bb[0];
  int32_t sum = 0;
  for (int i = 0; i < ({N} / 4); i++) {{
    // sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
    sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
    ++aa_ptr; ++bb_ptr;
  }}
  cc[0] += sum;

  if ( {N} % 4 != 0 )
    dotp_{N}_update_rest_{uniq_id}({N}, aa, bb, cc);

  // printf("YYYY\\n");
out:
  // printf("Z\\n");
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
static inline __attribute__((always_inline)) int32_t dotp16_{N}_body_rest_{uniq_id}(
    int N,
    int16_t *aa, int16_t *bb, int32_t *cc) {{
  int n_base = (N / 2) * 2;
  int16_t *a_ptr = &aa[n_base];
  int16_t *b_ptr = &bb[n_base];
  cc[0] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
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

  int32_t *aa_ptr = (int32_t *) &aa[0];
  int32_t *bb_ptr = (int32_t *) &bb[0];
  int32_t sum = 0;
  for (int j = 0; j < {N} / 2; j++) {{
    // sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
    sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
    ++ aa_ptr; ++ bb_ptr;
  }}
  cc[0] = sum;

  if ( {N} % 2 != 0 )
    dotp16_{N}_body_rest_{uniq_id}({N}, aa, bb, cc);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp16_{N}_update_rest_{uniq_id}(
    int N,
    int16_t *aa, int16_t *bb, int32_t *cc) {{
  int n_base = (N / 2) * 2;
  int16_t *a_ptr = &aa[n_base];
  int16_t *b_ptr = &bb[n_base];
  cc[0] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
  return 0;
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
    // printf("AAAA\\n");
  int32_t retcode = 0;

  if (0) {{
    retcode = dotp16_{N}_update_loop_{uniq_id}(aa, bb, cc);
    goto out;
  }}
    // printf("BBBB\\n");

  int32_t sum = 0;
  int32_t *aa_ptr = (int32_t *) &aa[0];
  int32_t *bb_ptr = (int32_t *) &bb[0];
  // printf("CCCC\\n");
  for (int i = 0; i < {N} / 2; i++) {{
    // printf("DDDD\\n");
    // sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
    sum = __rv_smaqa(sum, *aa_ptr, *bb_ptr);
    // printf("EEEE\\n");
    ++aa_ptr; ++bb_ptr;
    // printf("FFFF\\n");
  }}
  // printf("GGGG\\n");
  cc[0] += sum;
  // printf("HHHH\\n");

  if ({N} % 2 != 0 ) {{
    // printf("HHHH\\n");
    dotp16_{N}_update_rest_{uniq_id}({N}, aa, bb, cc);
    // printf("IIII\\n");
  }}
  // printf("JJJJ\\n");

out:
  // printf("KKKK\\n");
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
static inline __attribute__((always_inline)) int32_t dotp_{N}_reset_{uniq_id}(int32_t *cc) {{
  for (int i = 0; i < {N}; i++) {{
    cc[i] = 0;
  }}
  return 0;
}}

"""
    )
    return cc_code
