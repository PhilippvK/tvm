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
"""Independent host tests for IME constant packing and MetaSchedule replay."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te, tir
from tvm.topi.arm_cpu.conv2d_gemm import conv2d_nhwc_hwoi_ime_packed_compute


def _nodes(mod, kind):
    found = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body, lambda node: found.append(node) if isinstance(node, kind) else None
    )
    return found


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize(
    "kernel,stride,padding,dilation,k_max,mi,ni",
    [
        ((1, 1), 1, 0, 1, 1024, 8, 8),
        ((3, 3), 1, 1, 1, 1024, 8, 8),
        ((3, 3), 1, 2, 2, 24, 4, 8),
        ((2, 3), 2, 1, 1, 16, 8, 4),
    ],
)
def test_conv2d_weight_packing(constant, kernel, stride, padding, dilation, k_max, mi, ni):
    data_shape = (2, 4, 8, 8)
    kh, kw = kernel
    weight_shape = (kh, kw, 16, 8)
    data = te.placeholder(data_shape, "int8", "data")
    weight = te.placeholder(weight_shape, "int8", "weight")
    output = conv2d_nhwc_hwoi_ime_packed_compute(
        None,
        data,
        weight,
        strides=(stride, stride),
        padding=padding,
        dilation=(dilation, dilation),
        K_MAX=k_max,
        MI=mi,
        NI=ni,
    )
    weights = ((np.arange(np.prod(weight_shape)).reshape(weight_shape) * 7 + 3) % 19 - 9).astype(
        "int8"
    )
    func = te.create_prim_func([data, weight, output])
    if constant:
        param = func.params[1]
        buffer = func.buffer_map[param]
        root = func.body.block
        body = tir.AllocateConst(
            buffer.data, "int8", buffer.shape, tvm.nd.array(weights), root.body
        )
        root = tir.Block(
            root.iter_vars,
            root.reads,
            root.writes,
            root.name_hint,
            body,
            root.init,
            root.alloc_buffers,
            root.match_buffers,
            root.annotations,
        )
        func = tir.PrimFunc(
            [func.params[0], func.params[2]],
            tir.BlockRealize([], True, root),
            buffer_map={p: b for p, b in func.buffer_map.items() if p != param},
            attrs=func.attrs,
        )
    before = tvm.IRModule({"main": func})
    after = tir.transform.FoldConstantWeightPacking()(before)
    blocks = {b.name_hint: b for b in _nodes(before, tir.Block)}
    assert blocks["B_pack"].annotations["tir.weight_packing"]
    assert "tir.weight_packing" not in blocks["A_pack"].annotations
    assert "layout_free_placeholders" not in blocks["C_pack"].annotations
    if constant:
        packed = _nodes(after, tir.AllocateConst)
        assert len(packed) == 1
        shape = tuple(int(x) for x in packed[0].extents)
        no_count, ko_count, kb_count, nt_count, _, _ = shape
        # HWOI -> output-channel rows, then split K and interleave microtiles.
        logical = weights.transpose(2, 0, 1, 3).reshape(16, -1)
        expected = logical.reshape(no_count, nt_count, 4, ko_count, kb_count, 8)
        expected = expected.transpose(0, 3, 4, 1, 2, 5)
        np.testing.assert_array_equal(packed[0].data.numpy(), expected)
        for mod in (after, tvm.lower(before)):
            assert not [s for s in _nodes(mod, tir.BufferStore) if s.buffer.name == "B_pack"]
        assert not [
            b
            for block in _nodes(after, tir.Block)
            for b in block.alloc_buffers
            if b.name == "B_pack"
        ]
        assert any(s.buffer.name == "A_pack" for s in _nodes(after, tir.BufferStore))
        tvm.ir.assert_structural_equal(after, tir.transform.FoldConstantWeightPacking()(after))
    else:
        tvm.ir.assert_structural_equal(before, after)

    inputs = ((np.arange(np.prod(data_shape)).reshape(data_shape) * 3) % 17 - 8).astype("int8")
    output_shape = tuple(int(x) for x in output.shape)
    expected = np.zeros(output_shape, dtype="int32")
    padded = np.pad(
        inputs.astype("int32"), ((0, 0), (padding, padding), (padding, padding), (0, 0))
    )
    for y in range(output_shape[1]):
        for x in range(output_shape[2]):
            for ky in range(kh):
                for kx in range(kw):
                    expected[:, y, x, :] += (
                        padded[:, y * stride + ky * dilation, x * stride + kx * dilation, :]
                        @ weights[ky, kx].astype("int32").T
                    )
    # Build the original module to exercise folding in the default lowering pipeline.
    result = tvm.nd.empty(output_shape, "int32")
    args = [tvm.nd.array(inputs)]
    if not constant:
        args.append(tvm.nd.array(weights))
    tvm.build(before, target="llvm")(*args, result)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_relay_parameter_binding(tmp_path):
    from tvm import relay
    from tvm.contrib import graph_executor

    from tvm import meta_schedule as ms
    from tvm.meta_schedule.testing.space_generation import generate_design_space
    from tvm.topi.testing import conv2d_nhwc_python

    weights = (np.arange(3 * 3 * 16 * 8).reshape(3, 3, 16, 8) % 19 - 9).astype("int8")
    data = relay.var("data", shape=(1, 4, 4, 8), dtype="int8")
    weight = relay.var("weight", shape=weights.shape, dtype="int8")
    mod = tvm.IRModule.from_expr(
        relay.Function(
            [data, weight],
            relay.nn.conv2d(
                data,
                weight,
                padding=(1, 1),
                channels=16,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWOI",
                out_dtype="int32",
            ),
        )
    )
    target = tvm.target.Target("llvm -keys=arm_cpu,cpu -libs=ime_gemm -num-cores=1")
    executor = relay.backend.Executor("graph", {"link-params": True})
    tasks = ms.relay_integration.extract_tasks(
        mod, target, params={"weight": weights}, executor=executor
    )
    assert len(tasks) == 1
    workload_mod = tasks[0].dispatched[0]
    assert _nodes(workload_mod, tir.AllocateConst)
    assert any(s.buffer.name == "B_pack" for s in _nodes(workload_mod, tir.BufferStore))
    spaces = generate_design_space(
        "llvm",
        workload_mod,
        target,
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTiling(structure="SSRSRS", max_innermost_factor=16),
        ],
    )
    assert spaces
    database = ms.database.JSONDatabase(work_dir=str(tmp_path))
    workload = database.commit_workload(workload_mod)
    database.commit_tuning_record(
        ms.database.TuningRecord(spaces[0].trace, workload, [1.0], target)
    )

    folded = []

    @tvm.instrument.pass_instrument
    class InspectPacking:
        def run_after_pass(self, ir_mod, info):
            if info.name == "tir.FoldConstantWeightPacking":
                for func in ir_mod.functions.values():
                    if isinstance(func, tir.PrimFunc):
                        single = tvm.IRModule({"main": func})
                        if _nodes(single, tir.AllocateConst):
                            folded.append(single)

    with database, tvm.transform.PassContext(
        opt_level=3,
        instruments=[InspectPacking()],
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": 4,
        },
    ):
        lib = relay.build(mod, target=target, params={"weight": weights}, executor=executor)
    assert folded
    assert not [s for s in _nodes(folded[-1], tir.BufferStore) if s.buffer.name == "B_pack"]
    assert tuple(_nodes(folded[-1], tir.AllocateConst)[0].data.shape) == (2, 1, 9, 2, 4, 8)
    runtime = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    inputs = ((np.arange(128).reshape(1, 4, 4, 8) * 3) % 17 - 8).astype("int8")
    runtime.set_input("data", inputs)
    runtime.run()
    np.testing.assert_array_equal(
        runtime.get_output(0).numpy(),
        conv2d_nhwc_python(
            inputs.astype("int32"), weights.transpose(0, 1, 3, 2).astype("int32"), 1, 1
        ),
    )


if __name__ == "__main__":
    tvm.testing.main()
