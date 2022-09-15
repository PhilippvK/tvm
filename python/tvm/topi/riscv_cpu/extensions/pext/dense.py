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
"""Direct implementation of dense."""

from tvm import te
from tvm.topi.utils import traverse_inline, get_const_tuple
from tvm.autotvm.task.space import OtherOptionEntity

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)
#
# # Warning: broken, untested
# from .micro_kernel.gemv import (
#     intrin_gemv_MxN,
#     gemv_MxN_impl,
# )

from .micro_kernel.dotp import (
    intrin_dotp_N,
    dotp_N_impl,
)
from .... import tag


def dense_pext_compute(cfg, data, weight, bias=None, out_dtype=None):
    """Defines the P extension instructions of dense."""

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    cfg.define_split("tile_y", out_dim, policy="factors", num_outputs=2)  # TODO: 2 outputs?
    cfg.define_split("tile_x", batch, policy="factors", num_outputs=2)  # TODO: 2 coutputs
    cfg.define_split("tile_k", in_dim, policy="factors", num_outputs=2)
    cfg.define_knob("intrin_type", ["dotp", "gemm"])

    k = te.reduce_axis((0, in_dim), "k")
    C = te.compute(
        (batch, out_dim),
        lambda x, y: te.sum(
            data[x, k].astype(out_dtype) * weight[y, k].astype(out_dtype),
            axis=k,
        ),
        name="dense",
        tag="dense_pext",
    )
    if cfg.is_fallback:
        cfg.fallback_split("tile_y", [-1, batch])
        cfg.fallback_split("tile_x", [-1, out_dim])
        cfg.fallback_split("tile_k", [-1, 4])
        cfg["intrin_type"] = OtherOptionEntity("gemm")

    if bias is not None:
        C = te.compute((batch, out_dim), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


def dense_pext_schedule(cfg, outs):
    """Schedule function for RISC-V P-Extension instructions of dense."""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense" not in op.tag:
            return

        output = op.output(0)
        dense = op
        data = dense.input_tensors[0]

        M = cfg["tile_x"].size[-1]
        N = cfg["tile_y"].size[-1]
        K = cfg["tile_k"].size[-1]

        x, y = sched[dense].op.axis
        k = sched[dense].op.reduce_axis[0]

        x_o, x_i = cfg["tile_x"].apply(sched, dense, x)
        y_o, y_i = cfg["tile_y"].apply(sched, dense, y)
        k_o, k_i = cfg["tile_k"].apply(sched, dense, k)

        sched[dense].reorder(x_o, y_o, k_o, x_i, y_i, k_i)

        if cfg["intrin_type"].val == "gemm":
            gemm, uniq_id = intrin_gemm_MxKxN(M, K, N, data.dtype, output.dtype, stride_w=1)
            sched[output].tensorize(x_i, gemm)
            sched[output].pragma(x_o, "import_c", gemm_MxKxN_impl(M, K, N, uniq_id))
        elif cfg["intrin_type"].val == "dotp":
            dotp, uniq_id = intrin_dotp_N(K, data.dtype, output.dtype)
            sched[output].tensorize(k_i, dotp)
            sched[output].pragma(x_o, "import_c", dotp_N_impl(K, uniq_id))
        else:
            # Invalid
            pass

    traverse_inline(sched, outs[-1].op, _callback)
    return sched
