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
from tvm import autotvm
from tvm.topi.utils import traverse_inline
from ....utils import traverse_inline, get_const_tuple

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)

# Warning: broken, untested
from .micro_kernel.gemv import (
    intrin_gemv_MxN,
    gemv_MxN_impl,
)

from .micro_kernel.dotp import (
    intrin_dotp_N,
    dotp_N_impl,
)


def dense_pext_schedule(outs):
    """Schedule function for RISC-V P-Extension instructions of dense."""
    sched = te.create_schedule([x.op for x in outs])
    cfg = autotvm.get_config()

    def _callback(op):
        if "dense" not in op.tag:
            return

        output = op.output(0)
        s = sched
        data, weight = s[output].op.input_tensors
        batch, in_dim = get_const_tuple(data.shape)
        out_dim, _ = get_const_tuple(weight.shape)
        in_dim_factor = 4
        assert in_dim % in_dim_factor == 0, "Input dimension must divide {}".format(in_dim_factor)
        if in_dim % 16 == 0:
            in_dim_factor = 16

        # create tuning space
        cfg.define_split("tile_y", batch, num_outputs=4)
        cfg.define_split("tile_x", out_dim, num_outputs=4)
        cfg.define_split("tile_k", in_dim // in_dim_factor, num_outputs=2)
        cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
       
        # handle bias
        # if output.op not in s.outputs:
        #     s[output].compute_inline()
        #     output = s.outputs[0].output(0)

        n, x = s[output].op.axis
        kernel_scope, n = s[output].split(n, nparts=1)

        AA = data
        WW = weight
        CC = output
        ko = CC.op.reduce_axis[0]
        ko, ki = s[CC].split(ko, factor=4)
        # ko, kt = cfg["tile_k"].apply(s, CC, ko)

        dotp, uniq_id = intrin_dotp_N(4, data.dtype, output.dtype)
        sched[output].tensorize(ki, dotp)
        sched[output].pragma(ko, "import_c", dotp_N_impl(4, uniq_id))
        # =====
        # extract tensors
        # output = op.output(0)
        # dense = op
        # data_vec = dense.input_tensors[0]
        # M, K = data_vec.shape
        # N, N_ = dense.input_tensors[1].shape

        # # n, _ = sched[dense].op.axis
        # n, m = sched[dense].op.axis
        # # no, ni = sched[dense].split(n, nparts=1)
        # # mo, mi = sched[dense].split(m, nparts=1)
        # # sched[dense].reorder(n, m)
        # sched[dense].reorder(m, n)

        # print("M,K,N", M, K, N)
        # # mode = "gemm"
        # # mode = "gemv"
        # mode = "dotp"
        # # mode = "none"
        # if mode == "gemm":
        #     gemm, uniq_id = intrin_gemm_MxKxN(M, K, N, data_vec.dtype, output.dtype)
        #     sched[output].tensorize(ni, gemm)
        #     ## sched[output].pragma(no, "import_c", gemm_MxKxN_impl(M, K, N, uniq_id))
        # elif mode == "gemv":
        #     assert M == 1  # ???
        #     gemv, uniq_id = intrin_gemv_MxN(N, K, data_vec.dtype, output.dtype)
        #     sched[output].tensorize(m, gemv)
        #     # sched[output].pragma(no, "import_c", gemv_MxN_impl(N, K, uniq_id))
        # elif mode == "dotp":
        #     dotp, uniq_id = intrin_dotp_N(K, data_vec.dtype, output.dtype)
        #     sched[output].tensorize(m, dotp)
        #     # sched[output].pragma(no, "import_c", dotp_N_impl(K, uniq_id))
        # elif mode == "none":
        #     pass
        # else:
        #     raise RuntimeError(f"Invalid mode: {mode}")

    traverse_inline(sched, outs[-1].op, _callback)
    return sched
