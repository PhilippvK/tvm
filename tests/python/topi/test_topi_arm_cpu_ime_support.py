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
"""IME capability checks and Relay fallback selection."""

import pytest
import tvm
from tvm import relay, te, topi
from tvm.relay.op.strategy.arm_cpu import conv2d_strategy_arm_cpu, schedule_dense_arm_cpu


@pytest.mark.parametrize(
    "dims,tiles,supported",
    [
        ((8, 8, 8), (8, 8, 8), True),
        ((1, 8, 8), (8, 8, 8), False),
        ((8, 8, 27), (8, 8, 8), False),
        ((8, 3, 8), (8, 8, 8), False),
        ((0, 8, 8), (8, 8, 8), False),
        ((4, 4, 16), (4, 4, 16), True),
        ((8, 8, 8), (8, 8, 0), False),
    ],
)
def test_ime_support(dims, tiles, supported):
    assert topi.arm_cpu.is_ime_shape_supported("int8", "int8", "int32", *dims, *tiles) == supported


def test_ime_symbolic_and_dtype():
    assert not topi.arm_cpu.is_ime_shape_supported("int8", "int8", "int32", te.var("m"), 8, 8)
    assert not topi.arm_cpu.is_ime_shape_supported("uint8", "int8", "int32", 8, 8, 8)
    assert not topi.arm_cpu.is_ime_shape_supported("int8", "int8", "int8", 8, 8, 8)


@pytest.mark.parametrize(
    "kind,data_shape,weight_shape,supported",
    [
        ("conv", (1, 4, 4, 3), (3, 3, 8, 3), False),
        ("conv", (1, 1, 1, 8), (1, 1, 8, 8), False),
        ("conv", (1, 4, 4, 8), (1, 1, 8, 8), True),
        ("dense", (1, 8), (8, 8), False),
        ("dense", (8, 27), (8, 27), False),
        ("dense", (8, 8), (8, 8), True),
    ],
)
def test_ime_strategy_fallback(kind, data_shape, weight_shape, supported):
    data = relay.var("data", shape=data_shape, dtype="int8")
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    if kind == "conv":
        call = relay.nn.conv2d(
            data,
            weight,
            data_layout="NHWC",
            kernel_layout="HWOI",
            padding=(1, 1) if weight_shape[0] == 3 else (0, 0),
            out_dtype="int32",
        )
        strategy_func = conv2d_strategy_arm_cpu
    else:
        call = relay.nn.dense(data, weight, out_dtype="int32")
        strategy_func = schedule_dense_arm_cpu
    call = relay.transform.InferType()(tvm.IRModule.from_expr(call))["main"].body
    inputs = [te.placeholder(shape, dtype="int8") for shape in (data_shape, weight_shape)]
    target = tvm.target.Target("llvm -keys=arm_cpu,cpu -libs=ime_gemm")
    with target:
        strategy = strategy_func(call.attrs, inputs, call.checked_type, target)
        impls = [impl for spec in strategy.specializations for impl in spec.implementations]
        assert any("ime_packed" in impl.name for impl in impls) == supported
        selected = max(impls, key=lambda impl: impl.plevel)
        if supported:
            assert "ime_packed" in selected.name
        else:
            assert selected.name.endswith(".generic")
        outputs = selected.compute(call.attrs, inputs, call.checked_type)
        assert tuple(outputs[0].shape) == tuple(call.checked_type.shape)
