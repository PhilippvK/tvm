
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
"""Estimate peak flops and bandwidth for x86 devices"""
import functools
import re
from typing import Dict, Optional, Tuple

import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend import Executor
from ... import build, get_global_func, nd, transform
from ...contrib import utils
from ...rpc.base import RPC_SESS_MASK
from ...rpc.client import RPCSession
from ...runtime import DataType, Device, num_threads
from ...script import tir as T
from ...target import Target, x86
from ...tir import PrimFunc
from . import registry


def _detect_vec_width_registers(
    target: Target, vec_width: Optional[int], num_vector_registers: Optional[int]
):
    """Get the vector width and number of vector registers for a target.

    Parameters
    ----------
    target : Target
        Target to detect vector width and registers for.
    vec_width : Optional[int]
        If None, try and detect vector width from target. Otherwise provided input is used.
    num_vector_registers : Optional[int]
        If None, try and number of vector registers from target. Otherwise provided input is used.

    Returns
    -------
    vec_width: int
        Width of a vector register on `target` in bytes.
    num_vector_registers: int
        Number of vector registers on `target`.
    """
    if vec_width is None:
        # Only implemented for x86 so far...
        if (
            str(target.kind) == "llvm"
            and target.device_name == ""
            and len(target.keys) == 1
            and target.keys[0] == "cpu"
        ):
            with target:
                vec_width = x86.get_simd_32bit_lanes() * 4  # in number of bytes
        else:
            raise RuntimeError(f"Cannot determine vector width for target {target}")
    if num_vector_registers is None:
        if target.device_name == "":  # indicates x86
            num_vector_registers = 16  # Assuming for all platforms, probably wrong on older ones
        else:
            raise RuntimeError(f"Cannot determine number of vector registers for target {target}")
    return vec_width, num_vector_registers


@functools.lru_cache(maxsize=None)
def estimate_peak_fma_vector_flops(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
    dtype: DataType,
    vec_width: Optional[int] = None,
    num_vector_registers: Optional[int] = None,
):
    """Estimate peak flops assuming vector fma instructions and no explicit
    intrinsics. See estimate_peak_fma_flops.
    """
    # target_str = "c"
    # target_str = "llvm"
    # target = tvm.target.Target(target_str)
    # vec_width = 4
    # num_vector_registers = 32

    @T.prim_func
    def peakflops_fma_tir(
        a: T.handle,
        vec_width: T.int32,
        iters: T.int32,
        num_vector_registers: T.int32,
        threads: T.int32,
    ) -> None:
        # pylint: disable=invalid-name, missing-function-docstring
        A = T.match_buffer(a, [threads, num_vector_registers, vec_width], dtype)
        for t in T.parallel(threads):
            for _j in range(iters):
                for l in T.unroll(num_vector_registers):
                    # We want to use as few registers as possible, so we perform
                    # all operations on the same element
                    for k in T.vectorized(vec_width):
                        A[t, l, k] = A[t, l, k] * A[t, l, k] + A[t, l, k]

    vec_width, num_vector_registers = _detect_vec_width_registers(
        target, vec_width, num_vector_registers
    )
    vec_width //= DataType(dtype).bits // 8
    iters = 1000000
    nthreads = num_threads()
    print("vec_width", vec_width)
    print("threads", threads)
    # nthreads = 1
    specialized = peakflops_fma_tir.specialize(
        {
            peakflops_fma_tir.params[1]: vec_width,
            peakflops_fma_tir.params[2]: iters,
            peakflops_fma_tir.params[3]: num_vector_registers,
            peakflops_fma_tir.params[4]: nthreads,
        }
    )
    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    # runtime = relay.backend.Runtime("cpp")
    # runtime = relay.backend.Runtime("crt")
    # executor = Executor("aot", {"link-params": True, "workspace-byte-alignment": alignment[0], "constant-byte-alignment": alignment[1], "unpacked-api": unpacked, "interface-api": api})
    # executor = Executor("aot", {"link-params": True, "workspace-byte-alignment": alignment[0], "constant-byte-alignment": alignment[1], "unpacked-api": unpacked, "interface-api": api})
    # executor = Executor("aot", {"link-params": True})
    # with transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    with transform.PassContext(opt_level=3, config={"tir.disable_vectorize": False}):
        # if not (specialized.attrs and "global_symbol" in specialized.attrs):
        #     specialized = specialized.with_attr("global_symbol", "main")
        #     # specialized = specialized.with_attr("global_symbol", "default_function")
        #     specialized = specialized.with_attr("tir.noalias", True)
        f = build(specialized, target=target, runtime=runtime)  # , executor=executor)  # params=params
        # mod = tvm.ir.IRModule({"main": specialized})
        # f = relay.build(mod, target=target, runtime=runtime) #, executor=executor)  # params=params

    options = {}
    work_dir = utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("crt")),
        f,
        # str("/tmp/project"),
        str(work_dir.path / "project"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        random_fill = session._rpc.get_function("tvm.contrib.random.random_fill")
        # random_fill = get_global_func("tvm.contrib.random.random_fill")
        assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"
        a = nd.empty((nthreads, num_vector_registers, vec_width), dtype=dtype, device=session.device)
        random_fill(a)
        # times = f.time_evaluator(f.entry_name, dev, repeat=100, number=1)(a)
        # times = f.time_evaluator("default_function", session.device, repeat=100, number=1)(a)
        f = session._rpc.get_function("runtime.SystemLib")()
        # times = f.time_evaluator(func.entry_name, session.device, repeat=100, number=1)(a)
        times = f.time_evaluator("default_function", session.device, repeat=100, number=1)(a)
        # print("ttiimmeess", times)
    work_dir.remove()
    #     aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    #     aot_executor.get_input(0).copyfrom(data_sample)
    #     result = aot_executor.module.time_evaluator("run", session.device, number=number, repeat=repeat)()
    #     print("result", result)
    #     # output = aot_executor.get_output(0).numpy()
    #     # print("output", output)

    # # upload to remote if running over rpc
    # if dev.device_type >= RPC_SESS_MASK:
    #     if remote is None:
    #         raise RuntimeError("A RPCSession must be provided when using a remote device.")
    #     temp = utils.tempdir()
    #     path = temp.relpath("peak_fma_flops.tar")
    #     f.export_library(path)
    #     remote.upload(path)
    #     f = remote.load_module("peak_fma_flops.tar")
    #     random_fill = remote.get_function("tvm.contrib.random.random_fill")
    # else:
    #     random_fill = get_global_func("tvm.contrib.random.random_fill")
    # assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"

    # a = nd.empty((nthreads, num_vector_registers, vec_width), dtype=dtype, device=dev)
    # random_fill(a)
    # times = f.time_evaluator(f.entry_name, dev, repeat=100, number=1)(a)
    flops = 2 * vec_width * num_vector_registers * nthreads * iters  # fma is two flops
    return flops / times.min

@T.prim_func
def peak_bandwidth_tir(a: T.handle, b: T.handle, threads: T.int32, vec_width: T.int32, N: T.int32) -> None:
    # pylint: disable=invalid-name, missing-function-docstring
    # N = T.int32()
    A = T.match_buffer(a, [threads, N, 4, vec_width], "float32")
    B = T.match_buffer(b, [threads, 4, vec_width], "float32")
    # Parallelism is necessary to hit all cores/nodes
    for i in T.parallel(threads):
        for k in T.serial(N):
            for l in T.unroll(4):
                # vectorized load is necessary to hit peak bandwidth
                for j in T.vectorized(vec_width):
                    # += is necessary to introduce a data dependency for all
                    # elements of A, preventing the backend from removing the
                    # `k` loop and setting `k` to the loop extent.
                    B[i, l, j] += A[i, k, l, j]


@functools.lru_cache(maxsize=None)
def estimate_peak_bandwidth_dram(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
    vec_width: Optional[int] = None,
) -> float:
    """Estimate peak bandwidth for DRAM. See estimate_peak_bandwidth."""
    threads = num_threads()
    vec_width, _ = _detect_vec_width_registers(target, vec_width, 1)
    print("vec_width", vec_width)
    # vec_width //= DataType(dtype).bits // 8
    # print("vec_width", vec_width)
    print("threads", threads)
    size = 10**8 // (4 * threads * vec_width)
    # size = 10**7 // (4 * threads * vec_width)
    # size = 10**6 // (4 * threads * vec_width)
    # size = 10**5 // (4 * threads * vec_width)
    specialized = peak_bandwidth_tir.specialize(
        {
            peak_bandwidth_tir.params[2]: threads,
            peak_bandwidth_tir.params[3]: vec_width,
            peak_bandwidth_tir.params[4]: size,
        }
    )
    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    with transform.PassContext(opt_level=3, config={"tir.disable_vectorize": False}):
        f = build(specialized, target=target, runtime=runtime)

    options = {
        # "workspace_size_bytes": 2 * 32 * 1024 * 1024,
        "workspace_size_bytes": 32 * 32 * 1024 * 1024,
    }
    work_dir = utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("crt")),
        f,
        # str("/tmp/project"),
        str(work_dir.path / "project"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        random_fill = session._rpc.get_function("tvm.contrib.random.random_fill")
        # random_fill = get_global_func("tvm.contrib.random.random_fill")
        assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"
        a = nd.empty((threads, size, 4, vec_width), dtype="float32", device=session.device)
        print("a.size()", a.numpy().size)
        random_fill(a)
        b = nd.empty((threads, 4, vec_width), dtype="float32", device=session.device)
        random_fill(b)
        # times = f.time_evaluator(f.entry_name, dev, repeat=100, number=1)(a)
        # times = f.time_evaluator("default_function", session.device, repeat=100, number=1)(a)
        f = session._rpc.get_function("runtime.SystemLib")()
        # times = f.time_evaluator(func.entry_name, session.device, repeat=100, number=1)(a)
        # times = f.time_evaluator("default_function", session.device, repeat=10, number=1)(a, b, threads)
        times = f.time_evaluator("default_function", session.device, repeat=10, number=1)(a, b)
        print("ttiimmeess", times)
        print("ttiimmeess.min", times.min)
        print("ttiimmeess.min", times.min)
        ans = a.numpy().size
    work_dir.remove()
    return ans * 4 / times.min  # 4 bytes per float32
