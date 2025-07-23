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

import os
import traceback
import tempfile
from typing import Optional, Dict
import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.meta_schedule.builder import LocalBuilder
from tvm.driver.build_module import OperatorModule
from tvm import micro
from tvm.runtime import ndarray
from tvm.contrib.tar import tar
from tvm.relay.backend import Runtime
from tvm.driver import build as tvm_build
from tvm.tir.transform import RemoveWeightLayoutRewriteBlock
from tvm.tir.tensor_intrin.cfu import CFU_32X_INTRIN, CFU_24X_INTRIN, CFU_16X_INTRIN, CFU_8X_INTRIN
from tvm.tir.tensor_intrin.cfu import CFU_40X_INTRIN, CFU_48X_INTRIN, CFU_56X_INTRIN, CFU_64X_INTRIN


from math import log2
def CompressWeights():
    def _transform(func, mod, ctx):
        traceback.print_stack()
        print("CompressWeights")
        # print("func", func)
        # print("mod", mod)
        # print("ctx", ctx)
        # input("@A")
        # import time
        # time.sleep(2)
        has_call = False
        has_tensorize = False
        packed_weights_arr = None
        codebook_arr = None
        const_name = None
        pack_factor = None
        tensorize_func = None
        tensorize_block = None
        tensorize_num_clusters = None
        print("A")

        # def _visit(stmt):
        #     print("visit", stmt)
        #     nonlocal has_tensorize, is_legal, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_block
        #     if isinstance(stmt, tvm.tir.Block):  # finding blocks to be tensorized?
        #         tensorize_attr = stmt.annotations.get("meta_schedule.auto_tensorize")
        #         if tensorize_attr is None:
        #             return
        #         if not tensorize_attr.startswith("cfu_"):
        #             return
        #         # print("BLOCK", dir(stmt))
        #         # print("stmt.name_hint", stmt.name_hint, dir(stmt.name_hint))
        #         block_name = stmt.name_hint
        #         # print("block_name", block_name)
        #         tensorize_count = int(tensorize_attr.split("_", 1)[1][:-1])
        #         # print("tensorize_count", tensorize_count)
        #         assert not has_tensorize, "Can only tensorize once per block!"
        #         has_tensorize = True
        #         tensorize_func = tensorize_attr
        #         tensorize_block = block_name
        #         # print("stmt.annotations", stmt.annotations, dir(stmt.annotations))
        #         # print("stmt.annotations.items()", stmt.annotations.items())
        #         # print("stmt.annotations.keys()", stmt.annotations.keys())
        #         # print("A", stmt.annotations.get("meta_schedule.auto_tensorize"))
        #         # print("B", stmt.annotations.get("meta_schedule.auto_tensorize", None))
        #         # input("!!!")
        #     elif isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
        #         pass
        #         # print("CALL", dir(stmt))
        #         # print("stmt.op", stmt.op, dir(stmt.op))
        #         # print("stmt.op.name", stmt.op.name)
        #         # input("!!!")
        #     elif isinstance(stmt, tvm.tir.AllocateConst):  # Finding constants for weight clustering
        #         # print("alloc_const")
        #         # print("sch.mod", sch.mod)
        #         # print("mod.attrs", mod.attrs)
        #         # print("mod.functions", mod.functions)
        #         # print("stmt", stmt)
        #         # print("dir(stmt)", dir(stmt))
        #         # print("stmt.annotations", stmt.annotations)
        #         # print("stmt.body", stmt.body)
        #         # print("stmt.buffer_var", stmt.buffer_var)
        #         # print("dir(stmt.buffer_var)", dir(stmt.buffer_var))
        #         buffer_var = stmt.buffer_var
        #         name = buffer_var.name
        #         # print("buffer_var.name", name)
        #         # print("buffer_var.dtype", buffer_var.dtype)
        #         # print("stmt.data", stmt.data)
        #         #print("stmt.data.numpy()", stmt.data.numpy())
        #         # print("dir(stmt.data)", dir(stmt.data))
        #         data = stmt.data.numpy()
        #         # print("data", data.dtype)
        #         values, counts = np.unique(data, return_counts=True)
        #         num_clusters = len(values)
        #         if num_clusters in [2, 4, 16]:  # TODO: 3, 5-15 also fine?
        #             if data.dtype == "int8":
        #                 dtype_bits = 8
        #                 # print("values", values)
        #                 # print("counts", counts)
        #                 from math import log2
        #                 cluster_bits = int(log2(num_clusters))
        #                 # print("cluster_bits", cluster_bits)
        #                 pack_factor = dtype_bits / cluster_bits
        #                 # print("pack_factor", pack_factor)
        #                 shape = data.shape
        #                 # print("shape", shape)
        #                 extent = shape[-1]
        #                 # print("extent", extent)
        #                 ok = extent % pack_factor == 0
        #                 # print("ok?", ok)
        #                 packed_weights = [values.index(x) for x in data.flatten()]
        #                 # print("packed_weights", packed_weights)
        #                 packed_weights = np.array(packed_weights, dtype="uint8")
        #                 # print("packed_weights2", packed_weights)
        #                 packed_weights = packed_weights.reshape(shape)
        #                 # print("packed_weights3", packed_weights)
        #                 # packed_weights = packed_weights.astype("uint8")
        #                 # print("packed_weights4", packed_weights, packed_weights.shape)
        #                 def pack_bits(arr, n_bits: int):
        #                     assert arr.dtype == np.uint8, "Input array must be of dtype uint8"
        #                     max_val = 2**n_bits
        #                     assert np.all(arr < max_val), f"All elements must be less than {max_val}"
        #                     factor = 8 // n_bits
        #                     assert arr.shape[-1] % factor == 0, f"Innermost axis length must be divisible by {factor}"

        #                     # Reshape to group every 4 elements along the innermost axis
        #                     shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
        #                     grouped = arr.reshape(shape)

        #                     # TODO: little or big endian?
        #                     if n_bits == 1:
        #                         # Pack each group of 8 uint1s into a uint8
        #                         packed = (
        #                             (grouped[..., 0] << 7) |
        #                             (grouped[..., 1] << 6) |
        #                             (grouped[..., 2] << 5) |
        #                             (grouped[..., 3] << 4)
        #                             (grouped[..., 4] << 3) |
        #                             (grouped[..., 5] << 2) |
        #                             (grouped[..., 6] << 1) |
        #                             (grouped[..., 7])
        #                         )
        #                     elif n_bits == 2:
        #                         # Pack each group of 4 uint2s into a uint8
        #                         packed = (
        #                             (grouped[..., 0] << 6) |
        #                             (grouped[..., 1] << 4) |
        #                             (grouped[..., 2] << 2) |
        #                             (grouped[..., 3])
        #                         )
        #                     elif n_bits == 4:
        #                         # Pack each group of 2 uint4s into a uint8
        #                         packed = (
        #                             (grouped[..., 0] << 4) |
        #                             (grouped[..., 1])
        #                         )
        #                     packed = packed.astype(np.uint8)

        #                     return packed, factor
        #                 packed_weights, factor = pack_bits(packed_weights, cluster_bits)
        #                 # print("packed_weights5", packed_weights, packed_weights.shape)
        #                 packed_weights_arr = packed_weights
        #                 codebook_arr = values
        #                 const_name = name
        #                 pack_factor = factor
        #                 is_legal = True
        #                 # print("packed_weights5.shape", packed_weights.shape)
        #                 # print("stmt.dtype", stmt.dtype)
        #                 # print("stmt.extents", stmt.extents)
        #                 # print("stmt.span", stmt.span)
        #                 # annotations', 'body', 'buffer_var', 'data', 'dtype', 'extents', 'handle', 'irmod_storage_idx', 'legacy_repr', 'same_as', 'script', 'show', 'span
        #                 # input("€")
        print("B")

        def _visit(stmt):
            print("_visit")
            nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_num_clusters
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
                        packed_weights = [values.index(x) for x in data.flatten()]
                        packed_weights = np.array(packed_weights, dtype="uint8")
                        packed_weights = packed_weights.reshape(shape)
                        def pack_bits(arr, n_bits: int):
                            assert arr.dtype == np.uint8, "Input array must be of dtype uint8"
                            max_val = 2**n_bits
                            assert np.all(arr < max_val), f"All elements must be less than {max_val}"
                            factor = 8 // n_bits
                            assert arr.shape[-1] % factor == 0, f"Innermost axis length must be divisible by {factor}"

                            # Reshape to group every 4 elements along the innermost axis
                            shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
                            grouped = arr.reshape(shape)

                            # TODO: little or big endian?
                            if n_bits == 1:
                                # Pack each group of 8 uint1s into a uint8
                                packed = (
                                    (grouped[..., 0] << 7) |
                                    (grouped[..., 1] << 6) |
                                    (grouped[..., 2] << 5) |
                                    (grouped[..., 3] << 4)
                                    (grouped[..., 4] << 3) |
                                    (grouped[..., 5] << 2) |
                                    (grouped[..., 6] << 1) |
                                    (grouped[..., 7])
                                )
                            elif n_bits == 2:
                                # Pack each group of 4 uint2s into a uint8
                                packed = (
                                    (grouped[..., 0] << 6) |
                                    (grouped[..., 1] << 4) |
                                    (grouped[..., 2] << 2) |
                                    (grouped[..., 3])
                                )
                            elif n_bits == 4:
                                # Pack each group of 2 uint4s into a uint8
                                packed = (
                                    (grouped[..., 0] << 4) |
                                    (grouped[..., 1])
                                )
                            packed = packed.astype(np.uint8)

                            return packed, factor
                        packed_weights, factor = pack_bits(packed_weights, cluster_bits)
                        packed_weights_arr = packed_weights
                        codebook_arr = values
                        const_name = name
                        pack_factor = factor
                        tensorize_num_clusters = num_clusters

        # def _mutate(stmt):
        #     print("mutate", stmt)
        #     nonlocal has_tensorize, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func
        #     if not has_tensorize:
        #         return stmt
        #     if isinstance(stmt, tvm.tir.Block):  # finding blocks to be tensorized?
        #         block_name = stmt.name_hint
        #         # print("block_name", block_name)
        #         # if block_name == "root":
        #         #     # ann = stmt.annotations
        #         #     # print("stmt", dir(stmt))
        #         #     ann = {k: v for k, v in stmt.annotations.items()}
        #         #     # print("ann", ann, dir(ann))
        #         #     code = _gen_cfu_kernel_code(self.num_clusters, self.mode, self.channel_count)
        #         #     # sch.annotate(block, "pragma_import_c", code)
        #         #     ann["pragma_import_c"] = code
        #         #     # print("ann2", ann, dir(ann))
        #         #     # stmt.annotations = ann
        #         #     new_block = tvm.tir.Block()
        #         #     # print("stmt2", dir(stmt))
        #         #     # input("***")
        #         #     return stmt
        #     elif isinstance(stmt, tvm.tir.AllocateConst):  # Replace constant for weight clustering
        #         # print("alloc_const")
        #         buffer_var = stmt.buffer_var
        #         name = buffer_var.name
        #         if name == const_name:
        #             # TODO: change dtype?
        #             # buffer_var = tir.Var("v", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
        #             new_extents = list(stmt.extents)
        #             new_extents[-1] = new_extents[-1] // pack_factor
        #             new_data = ndarray.array(packed_weights_arr)
        #             codebook_var = tvm.tir.Var("codebook", tvm.ir.PointerType(tvm.ir.PrimType("int8")))
        #             # print("codebook_var", codebook_var, dir(codebook_var))
        #             # codebook_buf = tvm.tir.decl_buffer((len(codebook_arr),), "int8")
        #             codebook_buf = tvm.tir.decl_buffer(
        #                 shape=[len(codebook_arr)],
        #                 dtype="int8",
        #                 data=codebook_var  # Bind it to the actual var
        #             )
        #             # print("codebook_buf", codebook_buf)
        #             set_codebook_stmt = tvm.tir.Evaluate(tvm.tir.call_extern(
        #                 "void",
        #                 f"set_codebook_{self.num_clusters}",
        #                 codebook_buf.access_ptr("r", offset=0),
        #                 # codebook_var.access_ptr("r", offset=0),
        #             ))
        #             new_body = tvm.tir.SeqStmt([set_codebook_stmt, stmt.body])
        #             # print("new_body", new_body)
        #             newer_body = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype="int8", extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=new_body)
        #             # print("newer_body", newer_body)
        #             # new_body = ret = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype=tvm.tir.int8, extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=stmt.body)
        #                 # T.call_pure_extern(
        #             ret = tvm.tir.AllocateConst(buffer_var=stmt.buffer_var, dtype=stmt.dtype, extents=new_extents, data_or_idx=new_data, body=newer_body, annotations=stmt.annotations, span=stmt.span)
        #             # print("ret", ret)
        #             # input("€2")
        #             return ret
        #     return stmt
        print("C")

        def _mutate(stmt):
            print("_mutate")
            nonlocal has_call, packed_weights_arr, codebook_arr, const_name, pack_factor, tensorize_func, tensorize_num_clusters
            if not has_call:
                return stmt
            elif isinstance(stmt, tvm.tir.Call):  # finding call_extern after RewriteTensorize
                if stmt.op.name == "tir.tvm_access_ptr":
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
                    codebook_var = tvm.tir.Var("codebook", tvm.ir.PointerType(tvm.ir.PrimType("int8")))
                    codebook_buf = tvm.tir.decl_buffer(
                        shape=[len(codebook_arr)],
                        dtype="int8",
                        data=codebook_var  # Bind it to the actual var
                    )
                    set_codebook_stmt = tvm.tir.Evaluate(tvm.tir.call_extern(
                        "void",
                        f"set_codebook_{tensorize_num_clusters}",
                        codebook_buf.access_ptr("r", offset=0),
                    ))
                    new_body = tvm.tir.SeqStmt([set_codebook_stmt, stmt.body])
                    # annotations=None will lead to segfault in usmp pass
                    newer_body = tvm.tir.AllocateConst(buffer_var=codebook_var, dtype="int8", extents=[len(codebook_arr)], data_or_idx=ndarray.array(codebook_arr), body=new_body, annotations={})
                    ret = tvm.tir.AllocateConst(buffer_var=stmt.buffer_var, dtype=stmt.dtype, extents=new_extents, data_or_idx=new_data, body=newer_body, annotations=stmt.annotations, span=stmt.span)
                    return ret
            return stmt
        print("D")

        new_body = tvm.tir.stmt_functor.ir_transform(
            func.body,
            _visit,
            _mutate,
            ["tir.Evaluate", "tir.Call", "tir.Block", "tir.AllocateConst"]
        )
        print("E")
        print("body", func.body)
        print("new_body", new_body)
        print("has_call", has_call)
        # input("@B")
        if has_call:
            print("F")
            return func.with_body(new_body)
        print("G")
        return func
        # return func
    return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="CompressWeights")


def get_local_builder_micro():
    """Return micro-compatible Builder for meta schedule."""

    def _micro_build(
        mod: IRModule, target: Target, _params: Optional[Dict[str, NDArray]]
    ) -> OperatorModule:
        """Build function for micro targets.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be built.
        target : Target
            The target to be built.
        _params : Optional[Dict[str, NDArray]]
            The parameters to be used for the build. Must be None.

        Returns
        -------
        rt_mod : OperatorModule
            The built Module.
        """

        # Note: tvm_build assigns "global_symbol" to the name of generated C function
        # changing it is necessary for micro targets,
        # since the generated projects already include a main function.
        prim_func = mod["main"].with_attr("global_symbol", "default_function")
        mod = IRModule({"main": prim_func})
        runtime = Runtime("crt", {"system-lib": True})
        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        # try:
        #     mod = CompressWeights()(mod)
        # except Exception as ex:
        #     print(ex)
        #     print(traceback.format_exc())
        #     input("$$$")
        # cur_pass_ctx = tvm.transform.PassContext.current()
        # print("mod", mod)
        # # print("cur_pass_ctx", cur_pass_ctx)
        # # input(">>>")
        pass_config = {
            "tir.disable_vectorize": True,
            "tir.add_lower_pass": [(3, CompressWeights())],
        }
        with tvm.transform.PassContext(
            opt_level=3,
            config=pass_config,
            # disabled_pass=disabled_pass,
        ):
            try:
                rt_mod = tvm_build(mod, target=target, runtime=runtime)
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                input("$$$2")
        return rt_mod

    def _micro_export(mod: OperatorModule) -> str:
        """Export function for micro targets.

        Parameters
        ----------
        mod : OperatorModule
            The Module to be exported.

        Returns
        -------
        artifact_path : str
            The path to the exported Module.
        """
        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
        micro.export_model_library_format(mod, artifact_path)
        return artifact_path

    return LocalBuilder(f_build=_micro_build, f_export=_micro_export)
