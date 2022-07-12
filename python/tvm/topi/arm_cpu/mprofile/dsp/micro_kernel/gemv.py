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
"""Defines gemv intrinsics for matrix multiplication with v7e-m DSP instructions."""

import random
import string

import tvm
from tvm import te
from . import common


##########################
# MxN MatVecMul Intrinsic #
##########################

# NOTE this is transposed matvecmul (A * b)
def intrin_gemv_MxN(M, N, in_dtype, out_dtype, stride_w=1):
    """Defines a v7e-m DSP-accelerated matvecmul."""
    print(f"!intrin_gemv_{M}x{N}!")
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

    if isinstance(M, tvm.tir.IntImm):
        M = M.value
    if isinstance(N, tvm.tir.IntImm):
        N = N.value
    # TODO(weberlo, areusch): support more dtypes?
    assert in_dtype in ("int8", "int16")
    assert out_dtype == "int32"
    A = te.placeholder((M * stride_w - (stride_w - 1), N), name="a", dtype=in_dtype)
    b = te.placeholder((N,), name="b", dtype=in_dtype)
    k = te.reduce_axis((0, N), name="k")
    c = te.compute(
        (N,),
        lambda i: te.sum(
            A[i * stride_w, k].astype(out_dtype) * b[k].astype(out_dtype), axis=k
        ),
        name="c",
    )
    A_buf = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="A", offset_factor=1, strides=[te.var("A_s"), 1]
    )
    b_buf = tvm.tir.decl_buffer(
        b.shape, b.dtype, name="b", offset_factor=1, strides=[te.var("b_s"), 1]
    )
    c_buf = tvm.tir.decl_buffer(
        c.shape, c.dtype, name="c", offset_factor=1, strides=[te.var("c_s"), 1]
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        gemv_func_prefix = "gemv" if in_dtype == "int8" else "gemv16"

        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{gemv_func_prefix}_{M}x{N}_update_{uniq_id}",
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
                    "int32", f"gemv_{M}x{N}_reset_{uniq_id}", cc.access_ptr("w"), cc.strides[0]
                )
            )
            return ib.get()

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"{gemv_func_prefix}_{M}x{N}_body_{uniq_id}",
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

    intrin_decl = te.decl_tensor_intrin(c.op, intrin_func, binds={A: A_buf, b: b_buf, c: c_buf})
    return intrin_decl, uniq_id


def gemv_MxN_impl(M, N, uniq_id):
    """Emit C code for gemv impl."""
    print(f"!gemv_{M}x{N}_impl!")
    # TODO(weberlo, areusch): are there any SIMD tricks to zero out arrays quickly?
    bb_pad_size = N * M  # ?
    # code reference: CMSIS-NN paper (https://arxiv.org/abs/1801.06601)
    cc_code = (
        common.common_includes
        + f"""

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}_body_rest_{uniq_id}(
    int N,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  int n_base = (N / 4) * 4;
  switch ( N % 4 ) {{
  case 1:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }}
    break;
  case 2:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
              + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
    }}
    break;
  case 3:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
              + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
    }}
    break;
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}x{N}_body_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  for (int i = 0; i < {M}; i++) {{
    int32_t sum = 0;
    for (int j = 0; j < {N}; j++) {{
      sum += (int32_t) aa[i*A_stride + i] * (int32_t) bb[j];
      cc[i] = sum;
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}x{N}_body_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  int16_t bb_pad[{bb_pad_size}];
  int32_t retcode = 0;

  if (0) {{
    retcode = gemv_{M}x{N}_body_loop_{uniq_id}(aa, bb, cc, A_stride);
    goto out;
  }}

  for (int j = 0; j < {N} / 4; j++) {{
    read_and_pad(&bb[j*4], (int32_t*) &bb_pad[j*4], (int32_t*) &bb_pad[j*4 + 2]);
  }}

  for (int i = 0; i < {M}; i++) {{
    int16_t aa_pad_line[{N}];
    for (int j = 0; j < {N} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad_line[j*4], (int32_t*) &aa_pad_line[j*4 + 2]);
    }}

    int32_t *aa_ptr = (int32_t *) aa_pad_line;
    int32_t *bb_ptr = (int32_t *) &bb_pad[0];
    int32_t sum = 0;
    for (int j = 0; j < 2 * ({N} / 4); j++) {{
      sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
      ++ aa_ptr; ++ bb_ptr;
    }}
    cc[i] = sum;
  }}

  if ( {N} % 4 != 0 )
    gemv_{M}_body_rest_{uniq_id}({N}, aa, bb, cc, A_stride);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}_update_rest_{uniq_id}(
    int N,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  int n_base = (N / 4) * 4;
  switch ( N % 4 ) {{
  case 1:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }}
    break;
  case 2:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
    }}
    break;
  case 3:
    for (int i = 0; i < {M}; i++) {{
      int8_t *a_ptr = &aa[i * A_stride + n_base];
      int8_t *b_ptr = &bb[n_base];
      cc[i] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
    }}
    break;
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}x{N}_update_loop_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  for (int i = 0; i < {M}; i++) {{
    int32_t sum = 0;
    for (int j = 0; j < {N}; j++) {{
      sum += (int32_t) aa[i*A_stride + j] * (int32_t) bb[j];
    }}
    cc[i] += sum;
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{M}x{N}_update_{uniq_id}(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride) {{
  int16_t bb_pad[{bb_pad_size}];  // ???
  int32_t retcode = 0;

  if (0) {{
    retcode = gemv_{M}x{N}_update_loop_{uniq_id}(aa, bb, cc, A_stride);
    goto out;
  }}

  for (int i = 0; i < {N} / 4; i++) {{
    read_and_pad(&bb[i*4], (int32_t*) &bb_pad[i*4], (int32_t*) &bb_pad[i*4 + 2]);
  }}

  for (int i = 0; i < {M}; i++) {{
    int16_t aa_pad_line[{N}];
    for (int j = 0; j < {N} / 4; j++) {{
      read_and_pad(&aa[i*A_stride + j*4], (int32_t*) &aa_pad_line[j*4], (int32_t*) &aa_pad_line[j*4 + 2]);
    }}

    int32_t *aa_ptr = (int32_t *) aa_pad_line;
    int32_t *bb_ptr = (int32_t *) &bb_pad[0];
    int32_t sum = 0;
    for (int j = 0; j < 2 * ({N} / 4); j++) {{
      sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
      ++aa_ptr; ++bb_ptr;
    }}
    cc[i] += sum;
  }}

  if ( {N} % 4 != 0 )
    gemv_{M}_update_rest_{uniq_id}({N}, aa, bb, cc, A_stride);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}x{N}_body_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  for (int i = 0; i < {M}; i++) {{
    int32_t sum = 0;
    for (int j = 0; j < {N}; j++) {{
      sum += (int32_t) aa[i*A_stride + j] * (int32_t) bb[j];
    }}
    cc[i] = sum;
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}x{N}_body_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  int32_t retcode = 0;

  if (0) {{
    retcode = gemv16_{M}x{N}_body_loop_{uniq_id}(aa, bb, cc, A_stride);
    goto out;
  }}

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){{
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[0];
      int32_t sum = 0;
      for (int j = 0; j < {N} / 2; j++) {{
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }}
      cc[i] = sum;
    }}
  }}

  if ( {N} % 2 != 0 )
    gemv16_{M}_body_rest_{uniq_id}({N}, aa, bb, cc, A_stride);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}_body_rest_{uniq_id}(
    int N,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  int n_base = (N / 2) * 2;
  for (int i = 0; i < {M}; i++) {{
    int16_t *a_ptr = &aa[i * A_stride + n_base];
    int16_t *b_ptr = &bb[n_base];
    cc[i] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}_update_rest_{uniq_id}(
    int N,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  int n_base = (N / 2) * 2;
  for (int i = 0; i < {M}; i++) {{
    int16_t *a_ptr = &aa[i * A_stride + n_base];
    int16_t *b_ptr = &bb[n_base];
    cc[i] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }}
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}x{N}_update_loop_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  for (int i = 0; i < {M}; i++) {{
    int32_t sum = 0;
    for (int j = 0; j < {N}; j++) {{
      sum += (int32_t) aa[i*A_stride + j] * (int32_t) bb[j];
    }}
    cc[i] += sum;
  }}
  return 0;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv16_{M}x{N}_update_{uniq_id}(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride) {{
  int32_t retcode = 0;

  if (0) {{
    retcode = gemv16_{M}x{N}_update_loop_{uniq_id}(aa, bb, cc, A_stride);
    goto out;
  }}

  for (int i = 0; i < {M}; i++) {{
    int32_t sum = 0;
    int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
    int32_t *bb_ptr = (int32_t *) &bb[0];
    for (int j = 0; j < {N} / 2; j++) {{
      sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
      ++aa_ptr; ++bb_ptr;
    }}
    cc[i] += sum;
  }}

  if ( {N} % 2 != 0 )
    gemv16_{M}_update_rest_{uniq_id}({N}, aa, bb, cc, A_stride);

out:
  return retcode;
}}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemv_{N}_reset_{uniq_id}(int32_t *cc) {{
  for (int i = 0; i < {N}; i++) {{
      cc[i] = 0;
    }}
  }}
  return 0;
}}

"""
    )
    return cc_code
