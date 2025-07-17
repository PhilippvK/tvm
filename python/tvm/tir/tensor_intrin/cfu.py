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
# pylint: disable=invalid-name,missing-function-docstring
"""Dot product related intrinsics."""
from tvm.script import tir as T
from .. import TensorIntrin


def get_cfu_intrin(dtype_a, dtype_b, dtype_c, count):
    assert dtype_a == "int8"
    assert dtype_b == "int8"
    assert dtype_c == "int32"

    @T.prim_func
    def dp4a_desc(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((count,), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:count], B[0:count])
            T.writes(C[0])
            for i in range(0, count):
                with T.block("update"):
                    vi = T.axis.remap("R", [i])
                    C[0] = C[0] + T.cast(A[vi], dtype_c) * T.cast(B[vi], dtype_c)

    @T.prim_func
    def dp4a_impl(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((count,), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:count], B[0:count])
            T.writes(C[0])
            C[0] += T.call_pure_extern(
                f"cfu_kernel_{count}x",  # TODO: rename
                A.access_ptr("r", offset=0),
                B.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype=dtype_c,
            )

    return dp4a_desc, dp4a_impl


CFU_64X_INTRIN = "cfu_64x"
TensorIntrin.register(CFU_64X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 64))
CFU_56X_INTRIN = "cfu_56x"
TensorIntrin.register(CFU_56X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 56))
CFU_48X_INTRIN = "cfu_48x"
TensorIntrin.register(CFU_48X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 48))
CFU_40X_INTRIN = "cfu_40x"
TensorIntrin.register(CFU_40X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 40))
CFU_32X_INTRIN = "cfu_32x"
TensorIntrin.register(CFU_32X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 32))
CFU_24X_INTRIN = "cfu_24x"
TensorIntrin.register(CFU_24X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 24))
CFU_16X_INTRIN = "cfu_16x"
TensorIntrin.register(CFU_16X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 16))
CFU_8X_INTRIN = "cfu_8x"
TensorIntrin.register(CFU_8X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 8))
