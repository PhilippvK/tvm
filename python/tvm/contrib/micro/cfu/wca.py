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
"""Local builder for microTVM projects that compile on the local host"""

import traceback
from math import log2

import numpy as np

import tvm
from tvm.runtime import ndarray
# from tvm.tir.tensor_intrin.cfu import CFU_32X_INTRIN, CFU_24X_INTRIN, CFU_16X_INTRIN, CFU_8X_INTRIN
# from tvm.tir.tensor_intrin.cfu import CFU_40X_INTRIN, CFU_48X_INTRIN, CFU_56X_INTRIN, CFU_64X_INTRIN
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Schedule
from tvm.tir.schedule.analysis import has_block
from tvm.ir.supply import NameSupply
from tvm.tir import stmt_functor


def detect_clustered_weights(stmt):
    buffer_var = stmt.buffer_var
    name = buffer_var.name
    data = stmt.data.numpy()
    values, counts = np.unique(data, return_counts=True)
    values = list(values)
    counts = list(counts)

    # fill to next supported cluster count
    if len(values) in [1]:
        values += [0]
        counts += [0]
    elif len(values) in [3]:
        values += [0]
        counts += [0]
    elif len(values) in list(range(5, 16)):
        missing = 16 - len(values)
        values += [0] * missing
        counts += [0] * missing

    num_clusters = len(values)
    if num_clusters in [2, 4, 16]:  # TODO: 3, 5-15 also fine?
        if data.dtype == "int8":
            dtype_bits = 8
            cluster_bits = int(log2(num_clusters))
            pack_factor = dtype_bits / cluster_bits
            shape = data.shape
            extent = shape[-1]
            ok = extent % pack_factor == 0
            if not ok:
                return None
            packed_weights = [values.index(x) for x in data.flatten()]
            packed_weights = np.array(packed_weights, dtype="uint8")
            packed_weights = packed_weights.reshape(shape)
            packed_weights, factor = pack_bits(packed_weights, cluster_bits)
            if packed_weights is None:
                return None
            return packed_weights, values, name, factor
    return None


def detect_tensorize_block(stmt):
    tensorize_attr = stmt.annotations.get("meta_schedule.auto_tensorize")
    if tensorize_attr is None:
        return
    if not tensorize_attr.startswith("cfu_"):
        return
    block_name = stmt.name_hint
    tensorize_count = int(tensorize_attr.split("_", 1)[1][:-1])
    return tensorize_attr, tensorize_count, block_name


def _gen_cfu_kernel_code(num_clusters: int, cfu_mode: str, channel_count: int, kernel_name: str):
    # print("kernel_name", kernel_name)
    assert num_clusters in [2, 4, 16]
    assert cfu_mode in ["MODE_EMUL", "MODE_CFU"]
    assert channel_count in [8, 16, 24, 32, 40, 48, 56, 64]
    cfg = f"{num_clusters}_{channel_count}"
    ret = (
        """
#ifndef CFU_KERNEL_CODE_"""
        + cfg.upper()
        + """
#define CFU_KERNEL_CODE_"""
        + cfg.upper()
        + """
#include <stdint.h>

#ifndef MODE
#define MODE """
        + cfu_mode
        + """
#include "cfu_wca.h"
#undef MODE
#else
#include "cfu_wca.h"
#endif


static int32_t __attribute__((always_inline)) inline """
        + kernel_name
        + """(int8_t* data_ptr, int8_t* weights_ptr, int32_t* acc) {
    // COUNT="""
        + str(channel_count)
        + """, NUM_CLUSTERS="""
        + str(num_clusters)
        + """

    alu_rst();
"""
    )
    if num_clusters == 2:
        if channel_count == 64:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, code_word1);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 56:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1_lo = *((uint16_t*)(weights_ptr + 4));
    uint8_t code_word1_hi = *((uint8_t*)(weights_ptr + 6));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1 | ((uint32_t)code_word1_hi << 16));
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 48:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1 = *((uint16_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 40:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint8_t code_word1 = *((uint8_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 32:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b(code_word0, 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 24:
            ret += (
                """
    uint16_t code_word0_lo = *((uint16_t*)(weights_ptr));
    uint8_t code_word0_hi = *((uint8_t*)(weights_ptr + 2));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0 | ((uint32_t)code_word0_hi << 16), 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 16:
            ret += (
                """
    uint16_t code_word0 = *((uint16_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0, 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 8:
            ret += (
                """
    uint8_t code_word0 = *((uint8_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_2b((uint32_t)code_word0, 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
    elif num_clusters == 4:
        if channel_count == 32:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    // cfu_op0(CFU_FUNCT7_PUSH_WEIGHTS, code_word0, code_word1);
    push_weights_4b(code_word0, code_word1);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 24:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint16_t code_word1 = *((uint16_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b(code_word0, (uint32_t)code_word1);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 16:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b(code_word0, 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 8:
            ret += (
                """
    uint16_t code_word0 = *((uint16_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b((uint32_t)code_word0, 0);
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
    elif num_clusters == 16:
        if channel_count == 16:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_16b(code_word0, code_word1); // rename?
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
        elif channel_count == 8:
            ret += (
                """
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_16b(code_word0, 0); // rename?
    for (int i = 0; i < ("""
                + str(channel_count)
                + """ / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
"""
            )
    ret += (
        """
    return get_acc();
}
#endif  // CFU_KERNEL_CODE_"""
        + cfg.upper()
        + """
"""
    )
    return ret


@derived_object
class ImportCPostprocess(ms.postproc.PyPostproc):
    """A postproc that always fails."""

    def __init__(
        self,
        # num_clusters: int,
        mode: str,
        # channel_count: int,
        # f_initialize_with_tune_context: Callable = None,
        # f_apply: Callable = None,
        # f_clone: Callable = None,
        # f_as_string: Callable = None,
    ):
        # print("ImportCPostprocess.__init__")
        super().__init__(
            # self,
            # f_initialize_with_tune_context,
            # f_apply,
            # f_clone,
            # f_as_string,
        )
        # self.num_clusters = num_clusters
        self.mode = mode
        # self.channel_count = channel_count
        # print("ImportCPostprocess.__init__ done")

    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        pass

    def apply(self, sch: Schedule) -> bool:
        # print("apply", sch)
        # return False
        # has = has_block(sch, "block")
        has = has_block(sch, "root")
        # print("has", has)
        if has:
            has_tensorize = False
            is_legal = False
            try:
                block = sch.get_block("root")
                # sch.annotate(block, "foo", "bar")
                mod = sch.mod

                packed_weights_arr = None
                codebook_arr = None
                const_name = None
                pack_factor = None
                tensorize_func = None
                tensorize_block = None
                tensorize_count = None

                def _visit(stmt):
                    nonlocal has_tensorize, is_legal, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_block, tensorize_count
                    if isinstance(stmt, tvm.tir.Block):  # finding blocks to be tensorized?
                        res = detect_tensorize_block(stmt)
                        if res is not None:
                            assert not has_tensorize, "Can only tensorize once per block!"
                            has_tensorize = True
                            assert len(res) == 3
                            tensorize_func, tensorize_count, tensorize_block = res
                    elif isinstance(stmt, tvm.tir.AllocateConst):  # Finding constants for weight clustering
                        res = detect_clustered_weights(stmt)
                        if res is not None:
                            assert len(res) == 4
                            packed_weights_arr, codebook_arr, const_name, pack_factor = res
                            is_legal = True
                        else:
                            is_legal = False

                stmt_functor.ir_transform(mod["main"].body, _visit, None, ["tir.Block", "tir.AllocateConst"])
                # print("has_tensorize", has_tensorize)
                if has_tensorize:
                    # print("tensorize_func", tensorize_func)
                    # input(">>>")
                    if is_legal:
                        num_clusters = len(codebook_arr)
                        # code = _gen_cfu_kernel_code(num_clusters, self.mode, tensorize_count, tensorize_func)
                        func_name = f"cfu_kernel_{tensorize_count}x_{num_clusters}c"
                        # print("func_name", func_name)
                        code = _gen_cfu_kernel_code(num_clusters, self.mode, tensorize_count, func_name)
                        # print("code", code)
                        sch.annotate(block, "pragma_import_c", code)
                    else:
                        # print("illegal!")
                        block_ = sch.get_block(tensorize_block)
                        sch.unannotate(block_, "meta_schedule.auto_tensorize")
                        # input("#")
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                input("&&&")
                raise ex
        # print("sch", sch)
        # input(">")
        return True

    def clone(self) -> "ImportCPostprocess":
        # return ImportCPostprocess(self.num_clusters, self.mode, self.channel_count)
        return ImportCPostprocess(self.mode)

    def __str__(self) -> str:
        return "ImportCPostprocess"


def pack_bits(arr, n_bits: int):
    assert arr.dtype == np.uint8, "Input array must be of dtype uint8"
    max_val = 2**n_bits
    assert np.all(arr < max_val), f"All elements must be less than {max_val}"
    factor = 8 // n_bits
    if arr.shape[-1] % factor != 0:  # Innermost axis length must be divisible by factor
        return None, None

    # Reshape to group every 4 elements along the innermost axis
    shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
    grouped = arr.reshape(shape)

    # TODO: little or big endian?
    if n_bits == 1:
        # Pack each group of 8 uint1s into a uint8
        packed = (
            (grouped[..., 0] << 7)
            | (grouped[..., 1] << 6)
            | (grouped[..., 2] << 5)
            | (grouped[..., 3] << 4)(grouped[..., 4] << 3)
            | (grouped[..., 5] << 2)
            | (grouped[..., 6] << 1)
            | (grouped[..., 7])
        )
    elif n_bits == 2:
        # Pack each group of 4 uint2s into a uint8
        packed = (grouped[..., 0] << 6) | (grouped[..., 1] << 4) | (grouped[..., 2] << 2) | (grouped[..., 3])
    elif n_bits == 4:
        # Pack each group of 2 uint4s into a uint8
        packed = (grouped[..., 0] << 4) | (grouped[..., 1])
    packed = packed.astype(np.uint8)

    return packed, factor


# This seems to break the coudpickle dump required for the rpc-based builder....
from tvm._ffi.registry import register_func


@register_func("tvm.tir.transform.CompressWeights")
def CompressWeights():
    # print("CompressWeights")
    name_supply = NameSupply()

    def _transform(func, mod, ctx):
        # print("_transform")
        # nonlocal name_supply
        with ms.Profiler.timeit("CompressWeights/transform"):
            # print("_transform")
            has_call = False
            # has_tensorize = False
            packed_weights_arr = None
            codebook_arr = None
            const_name = None
            pack_factor = None
            # tensorize_func = None
            # tensorize_block = None
            tensorize_num_clusters = None

            def _visit(stmt):
                # print("_visit")
                # nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_num_clusters
                nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_num_clusters
                if isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
                    if stmt.op.name == "tir.call_pure_extern":
                        func_name = stmt.args[0]
                        if func_name.value.startswith("cfu_kernel"):
                            has_call = True
                elif isinstance(stmt, tvm.tir.AllocateConst):  # Finding constants for weight clustering
                    # print("alloc_const")
                    buffer_var = stmt.buffer_var
                    name = buffer_var.name
                    data = stmt.data.numpy()
                    values, counts = np.unique(data, return_counts=True)
                    values = list(values)
                    counts = list(counts)

                    # fill to next supported cluster count
                    if len(values) in [1]:
                        values += [0]
                        counts += [0]
                    elif len(values) in [3]:
                        values += [0]
                        counts += [0]
                    elif len(values) in list(range(5, 16)):
                        missing = 16 - len(values)
                        values += [0] * missing
                        counts += [0] * missing

                    num_clusters = len(values)
                    if num_clusters in [2, 4, 16]:
                        if data.dtype == "int8":
                            dtype_bits = 8
                            cluster_bits = int(log2(num_clusters))
                            pack_factor = dtype_bits / cluster_bits
                            shape = data.shape
                            extent = shape[-1]
                            ok = extent % pack_factor == 0
                            if not ok:
                                return None
                            packed_weights = [values.index(x) for x in data.flatten()]
                            packed_weights = np.array(packed_weights, dtype="uint8")
                            packed_weights = packed_weights.reshape(shape)
                            packed_weights, factor = pack_bits(packed_weights, cluster_bits)
                            packed_weights_arr = packed_weights
                            codebook_arr = values
                            const_name = name
                            pack_factor = factor
                            tensorize_num_clusters = num_clusters

            def _mutate(stmt):
                # print("_mutate")
                # nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_num_clusters
                if not has_call:
                    return stmt
                elif isinstance(stmt, tvm.tir.Call):  # rename function name to include cluster size
                    if stmt.op.name == "tir.call_pure_extern":
                        func_name = stmt.args[0]
                        if func_name.value.startswith("cfu_kernel"):
                            # print("func_name.value", func_name.value)
                            tensorize_count = int(func_name.value.split("x", 1)[0].split("_")[-1])
                            # print("tensorize_count", tensorize_count)
                            assert tensorize_count in [
                                8,
                                16,
                                24,
                                32,
                                40,
                                48,
                                56,
                                64,
                            ], f"tensorize_count={tensorize_count}"
                            new_func_name = f"cfu_kernel_{tensorize_count}x_{tensorize_num_clusters}c"
                            new_args = list(stmt.args)
                            new_args[0] = new_func_name
                            new_stmt = tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)
                            return new_stmt
                    elif stmt.op.name == "tir.tvm_access_ptr":
                        # print("PTR")
                        # print("stmt", stmt, dir(stmt))
                        # print("stmt.args[1]", stmt.args[1], dir(stmt.args[1]), type(stmt.args[1]))
                        # print("stmt.args[2]", stmt.args[2], dir(stmt.args[2]), type(stmt.args[2]))
                        const_name_ = stmt.args[1].name
                        # print("const_name_", const_name_)
                        if const_name_ == const_name:
                            # print("pack_factor", pack_factor)
                            offset_expr = stmt.args[2]
                            # print("offset_expr", offset_expr)
                            new_offset_expr = tvm.tir.indexdiv(offset_expr, pack_factor)
                            new_args = list(stmt.args)
                            new_args[2] = new_offset_expr
                            # print("new_args", new_args)
                            new_stmt = tvm.tir.Call(stmt.dtype, stmt.op, new_args, stmt.span)
                            # print("new_stmt", new_stmt)
                            return new_stmt
                elif isinstance(stmt, tvm.tir.AllocateConst):  # Replace constant for weight clustering
                    buffer_var = stmt.buffer_var
                    name = buffer_var.name
                    if name == const_name:
                        # TODO: change dtype?
                        new_extents = list(stmt.extents)
                        new_extents[-1] = new_extents[-1] // pack_factor
                        new_data = ndarray.array(packed_weights_arr)
                        codebook_name = name_supply.fresh_name("codebook_")
                        codebook_var = tvm.tir.Var(codebook_name, tvm.ir.PointerType(tvm.ir.PrimType("int8")))
                        codebook_buf = tvm.tir.decl_buffer(
                            shape=[len(codebook_arr)], dtype="int8", data=codebook_var  # Bind it to the actual var
                        )
                        set_codebook_stmt = tvm.tir.Evaluate(
                            tvm.tir.call_extern(
                                "void",
                                f"set_codebook_{tensorize_num_clusters}",
                                codebook_buf.access_ptr("r", offset=0),
                            )
                        )
                        new_body = tvm.tir.SeqStmt([set_codebook_stmt, stmt.body])
                        # annotations=None will lead to segfault in usmp pass
                        newer_body = tvm.tir.AllocateConst(
                            buffer_var=codebook_var,
                            dtype="int8",
                            extents=[len(codebook_arr)],
                            data_or_idx=ndarray.array(codebook_arr),
                            body=new_body,
                            annotations={},
                        )
                        ret = tvm.tir.AllocateConst(
                            buffer_var=stmt.buffer_var,
                            dtype=stmt.dtype,
                            extents=new_extents,
                            data_or_idx=new_data,
                            body=newer_body,
                            annotations=stmt.annotations,
                            span=stmt.span,
                        )
                        return ret
                return stmt

            # print("D")

            new_body = tvm.tir.stmt_functor.ir_transform(
                func.body,
                _visit,
                _mutate,
                # ["tir.Evaluate", "tir.Call", "tir.Block", "tir.AllocateConst"]
                ["tir.Call", "tir.AllocateConst"],
            )
            # print("E")
            # print("body", func.body)
            # print("new_body", new_body)
            # print("has_call", has_call)
            # input("@B")
            if has_call:
                # print("F")
                return func.with_body(new_body)
            # print("G")
            return func
            # return func

    return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="CompressWeights")
