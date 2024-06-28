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

import ctypes
from typing import Tuple, Callable

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax, rpc, te, tir, topi
from tvm.contrib import utils, cc, popen_pool
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax.testing.vm import check_saved_func
from tvm.runtime import ShapeTuple

# EXEC_MODE = ["bytecode", "compiled"]


# @pytest.fixture(params=EXEC_MODE)
# def exec_mode(request):
#     return request.param

def get_relax_dense(dtype):
    matmul_input_size = 64
    matmul_hidden_size = 10
    matmul_output_size = 4

    matmul_weights_matrix = np.random.random((matmul_hidden_size, matmul_output_size)).astype(dtype)
    matmul_bias_matrix = np.random.random((matmul_output_size,)).astype(dtype)

    matmul_params = {"weights": tvm.nd.array(matmul_weights_matrix), "bias": tvm.nd.array(matmul_bias_matrix)}
    matmul_data = tvm.nd.array(np.random.rand(matmul_input_size, matmul_hidden_size).astype(dtype))

    builder = relax.BlockBuilder()

    with builder.function("main"):
        input = relax.Var("x", R.Tensor((matmul_input_size, matmul_hidden_size), dtype))
        weights = relax.Constant(tvm.nd.array(matmul_weights_matrix))
        bias = relax.Constant(tvm.nd.array(matmul_bias_matrix))
        output_matmul = relax.op.matmul(input, weights)
        output_bias = relax.op.add(output_matmul, bias)
        builder.emit_func_output(output_bias, params=[input])
    return builder.get(), matmul_data, matmul_params



def get_relax_conv2d(dtype):
    conv2d_input_n = 1
    conv2d_input_c = 16
    conv2d_input_h = 64
    conv2d_input_w = 64
    conv2d_kernel_h = 4
    conv2d_kernel_w = 4
    conv2d_kernel_ci = 16
    conv2d_kernel_co = 16
    conv2d_output_n = 1
    conv2d_output_c = 16
    conv2d_output_h = 61
    conv2d_output_w = 61

    conv2d_weights_matrix = np.random.random((conv2d_kernel_h, conv2d_kernel_w, conv2d_kernel_ci, conv2d_kernel_co)).astype(dtype)
    conv2d_bias_matrix = np.random.random((conv2d_output_w,)).astype(dtype)

    conv2d_params = {"weights": tvm.nd.array(conv2d_weights_matrix), "bias": tvm.nd.array(conv2d_bias_matrix)}
    conv2d_data = tvm.nd.array(np.random.rand(conv2d_input_n, conv2d_input_c, conv2d_input_h, conv2d_input_w).astype(dtype))

    builder = relax.BlockBuilder()

    with builder.function("main"):
        input = relax.Var("x", R.Tensor((conv2d_input_n, conv2d_input_c, conv2d_input_h, conv2d_input_w), dtype))
        weights = relax.Constant(tvm.nd.array(conv2d_weights_matrix))
        bias = relax.Constant(tvm.nd.array(conv2d_bias_matrix))
        output_conv2d = relax.op.nn.conv2d(input, weights, data_layout="NCHW", kernel_layout="HWIO")
        output_bias = relax.op.add(output_conv2d, bias)
        builder.emit_func_output(output_bias, params=[input])

    return builder.get(), conv2d_data, conv2d_params


def compare_aot_with_vm(mod, data, params):
    # config
    dev = tvm.cpu()
    target = tvm.target.Target("llvm", host="llvm")

    # build mod with vm and aot
    vm_ex = relax.build(mod, target, exec_mode="compiled")
    aot_ex = relax.build(mod, target, pipeline="micro2_build", exec_mode="crt", system_lib=True)  # TODO: system_lib yes/no?; Rename pipeline and exec mode

    # get aot result
    rt_mod = tvm.runtime.executor.AotModule(aot_ex["default"](dev))  # TODO: move relax build
    rt_mod.set_input("x", data)
    rt_mod.run()
    aot_out = rt_mod.get_output(0)

    # get vm result
    vm = relax.VirtualMachine(vm_ex, dev)
    vm_out = vm["main"](data)

    # compare aot with vm result
    tvm.testing.assert_allclose(vm_out.numpy(), aot_out.numpy(), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_relax_llvm_aot_dense(dtype):
    # get relax mod
    mod, data, params = get_relax_dense(dtype)

    # build, run and compare
    compare_aot_with_vm(mod, data, params)



@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_relax_llvm_aot_conv2d(dtype):
    # get relax mod
    mod, data, params = get_relax_conv2d(dtype)

    # build, run and compare
    compare_aot_with_vm(mod, data, params)



if __name__ == "__main__":
    tvm.testing.main()
