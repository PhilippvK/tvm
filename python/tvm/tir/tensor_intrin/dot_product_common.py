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

# TODO: include types in names like this:
# def dot_product_4x4_i8i8i32_desc(

# phi_dotp4
@T.prim_func
def dp4a_desc(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[0] = C[0] + T.cast(A[vi], "int32") * T.cast(B[vi], "int32")

# phi_dotp4x4
@T.prim_func
def dp4a2_desc(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])
        for i in T.serial(0, 4):
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

# phi_conv23
@T.prim_func
def dp4a3_desc(
    A: T.Buffer((4, 4), "int8", offset_factor=1),
    B: T.Buffer((3, 3), "int8", offset_factor=1),
    C: T.Buffer((1,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4, 0:4], B[0:3, 0:3])
        T.writes(C[0])
        for y in T.serial(0, 2):
            for x in T.serial(0, 2):
                for ky in T.serial(0, 3):
                    for kx in T.serial(0, 3):
                        with T.block("update"):
                            # vy, vx, vky, vkx = T.axis.remap("SSRR", [y, x, ky, kx])
                            vy, vx, vky, vkx = T.axis.remap("RRRR", [y, x, ky, kx])
                            C[0] = C[0] + T.cast(A[vy + vky, vx + vkx], "int32") * T.cast(B[vy + vky, vx + vkx], "int32")

# phi_conv23x4 (4 channels at a time)
# how to ensure block memory layout?
# phi_gemm?+mac?

# phi_dotp4
@T.prim_func
def dp4a_impl(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])

        C[0] += T.call_pure_extern(
            # "__dp4a", A.vload([0], "int8x4"), B.vload([0], "int8x4"), T.int32(0), dtype="int32"
            "exp", A.vload([0], "int8x4"), B.vload([0], "int8x4"), T.int32(0), dtype="int32"
        )


# phi_dotp4x4
@T.prim_func
def dp4a2_impl(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_i8x4 = A.vload([0], "int8x4")
        A_i32 = T.reinterpret(A_i8x4, dtype="int32")
        vec_ai32 = T.broadcast(A_i32, 4)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x16")

        vec_b = B.vload([0, 0], dtype="int8x16")

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            # T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            "exp",
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            T.uint32(3),
            T.int32x4(0),
            vec_a,
            vec_b,
            dtype="int32x4",
        )

# phi_conv23
@T.prim_func
def dp4a3_impl(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_i8x4 = A.vload([0], "int8x4")
        A_i32 = T.reinterpret(A_i8x4, dtype="int32")
        vec_ai32 = T.broadcast(A_i32, 4)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x16")

        vec_b = B.vload([0, 0], dtype="int8x16")

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            # T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            "exp",
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            T.uint32(3),
            T.int32x4(0),
            vec_a,
            vec_b,
            dtype="int32x4",
        )


DP4A_INTRIN = "dp4a"
DP4A2_INTRIN = "dp4a2"
DP4A3_INTRIN = "dp4a3"

TensorIntrin.register(DP4A_INTRIN, dp4a_desc, dp4a_impl)
TensorIntrin.register(DP4A2_INTRIN, dp4a2_desc, dp4a2_impl)
TensorIntrin.register(DP4A3_INTRIN, dp4a3_desc, dp4a3_impl)
