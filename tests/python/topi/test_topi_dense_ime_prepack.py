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
from tvm import te, tir, meta_schedule as ms
from tvm.script import tir as T
from tvm.meta_schedule.testing.space_generation import generate_design_space
from tvm.topi.arm_cpu.dense_gemm import dense_ime_packed_compute


def _workload(m, n, k, k_max=512, constant=True, mi=8, ni=8, shared=False):
    a = te.placeholder((m, k), "int8", "A")
    b = te.placeholder((n, k), "int8", "B")
    c = dense_ime_packed_compute(None, a, b, K_MAX=k_max, MI=mi, NI=ni)
    if shared:
        c = te.compute((m, n), lambda i, j: c[i, j] + b[j, 0].astype("int32"), name="shared")
    func = te.create_prim_func([a, b, c])
    weights = ((np.arange(n * k).reshape(n, k) * 7 + 3) % 19 - 9).astype("int8")
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
    return tvm.IRModule({"main": func}), weights


def _nodes(mod, kind):
    found = []
    tir.stmt_functor.post_order_visit(
        mod["main"].body, lambda node: found.append(node) if isinstance(node, kind) else None
    )
    return found


def _check_packed(before, after, weights):
    stores = [s for s in _nodes(before, tir.BufferStore) if s.buffer.name == "B_pack"]
    assert len(stores) == 1
    shape = tuple(int(x) for x in stores[0].buffer.shape)
    constants = _nodes(after, tir.AllocateConst)
    assert len(constants) == 1
    assert tuple(constants[0].data.shape) == shape
    # Independent reference: enumerate each microtile in the documented physical order.
    n, k = weights.shape
    if len(shape) == 6:
        ki = shape[2] * 8
    else:
        ki = k
    nt_count = shape[-3]
    ni_tile = nt_count * 4
    expected = np.array(
        [
            weights[no * ni_tile + nt * 4 + ni, ko * ki + kb * 8 + kk]
            for no in range(n // ni_tile)
            for ko in range(k // ki)
            for kb in range(ki // 8)
            for nt in range(nt_count)
            for ni in range(4)
            for kk in range(8)
        ],
        dtype="int8",
    ).reshape(shape)
    np.testing.assert_array_equal(constants[0].data.numpy(), expected)
    assert not [s for s in _nodes(after, tir.BufferStore) if s.buffer.name == "B_pack"]
    assert not [
        b for block in _nodes(after, tir.Block) for b in block.alloc_buffers if b.name == "B_pack"
    ]


def _run(mod, weights, m, constant=True):
    n, k = weights.shape
    inputs = ((np.arange(m * k).reshape(m, k) * 3) % 17 - 8).astype("int8")
    result = tvm.nd.empty((m, n), "int32")
    lib = tvm.build(mod, target="llvm")
    args = [tvm.nd.array(inputs)]
    if not constant:
        args.append(tvm.nd.array(weights))
    lib(*args, result)
    np.testing.assert_array_equal(
        result.numpy(), inputs.astype("int32") @ weights.astype("int32").T
    )


@pytest.mark.parametrize(
    "m,n,k,k_max",
    [
        (8, 8, 8, 512),
        (8, 16, 8, 512),
        (16, 8, 8, 512),
        (8, 8, 16, 512),
        (16, 24, 24, 512),
        (16, 24, 32, 16),
        (16, 16, 16, 8),
    ],
)
def test_constant_packing(m, n, k, k_max):
    before, weights = _workload(m, n, k, k_max)
    after = tir.transform.FoldConstantWeightPacking()(before)
    _check_packed(before, after, weights)
    tvm.ir.assert_structural_equal(after, tir.transform.FoldConstantWeightPacking()(after))
    _run(before, weights, m)  # Default build pipeline must fold, without a caller-supplied pass.
    _run(after, weights, m)
    lowered = tvm.lower(before)
    assert not [s for s in _nodes(lowered, tir.BufferStore) if s.buffer.name == "B_pack"]


@pytest.mark.parametrize("m,n,k", [(8, 16, 8), (16, 8, 8), (8, 8, 16), (16, 24, 32)])
def test_dynamic_weight_regression(m, n, k):
    before, weights = _workload(m, n, k, constant=False)
    tvm.ir.assert_structural_equal(before, tir.transform.FoldConstantWeightPacking()(before))
    _run(before, weights, m, constant=False)


def test_shared_constant_preserved():
    before, weights = _workload(16, 24, 16, shared=True)
    tvm.ir.assert_structural_equal(before, tir.transform.FoldConstantWeightPacking()(before))
    inputs = np.arange(16 * 16, dtype="int8").reshape(16, 16)
    output = tvm.nd.empty((16, 24), "int32")
    tvm.build(before, target="llvm")(tvm.nd.array(inputs), output)
    np.testing.assert_array_equal(
        output.numpy(), inputs.astype("int32") @ weights.astype("int32").T + weights[:, 0]
    )


def test_constant_pointer_escape_preserved():
    before, _ = _workload(16, 24, 16)

    def add_pointer_reader(node):
        if isinstance(node, tir.AllocateConst):
            body = tir.SeqStmt(
                [
                    node.body,
                    tir.Evaluate(tir.call_extern("int32", "read_weight", node.buffer_var)),
                ]
            )
            return tir.AllocateConst(node.buffer_var, node.dtype, node.extents, node.data, body)
        return None

    func = before["main"]
    body = tir.stmt_functor.ir_transform(func.body, None, add_pointer_reader, ["tir.AllocateConst"])
    before = tvm.IRModule({"main": func.with_body(body)})
    tvm.ir.assert_structural_equal(before, tir.transform.FoldConstantWeightPacking()(before))


@pytest.mark.parametrize("mi,ni", [(4, 4), (4, 8), (8, 4)])
def test_microtile_variants(mi, ni):
    before, weights = _workload(2 * mi, 3 * ni, 24, mi=mi, ni=ni)
    after = tir.transform.FoldConstantWeightPacking()(before)
    _check_packed(before, after, weights)
    _run(before, weights, 2 * mi)


# Same 8x16x8 microtile contract as the project's packed IME intrinsic.
# A portable implementation lets us execute tensorized candidates on the host.
@T.prim_func
def ime_desc(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (2, 2, 4, 8), "int8", offset_factor=1)
    B = T.match_buffer(b, (2, 2, 4, 8), "int8", offset_factor=1)
    C = T.match_buffer(c, (2, 2, 4, 4), "int32", offset_factor=1)
    with T.block("root"):
        T.reads(C[0:2, 0:2, 0:4, 0:4], A[0:2, 0:2, 0:4, 0:8], B[0:2, 0:2, 0:4, 0:8])
        T.writes(C[0:2, 0:2, 0:4, 0:4])
        for mt, nt, mi, ni, kb, kk in T.grid(2, 2, 4, 4, 2, 8):
            with T.block("update"):
                vm, vn, vi, vj, vk, vl = T.axis.remap("SSSSRR", [mt, nt, mi, ni, kb, kk])
                C[vm, vn, vi, vj] += T.Cast("int32", A[vk, vm, vi, vl]) * T.Cast(
                    "int32", B[vk, vn, vj, vl]
                )


@T.prim_func
def ime_extern(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (2, 2, 4, 8), "int8", offset_factor=1)
    B = T.match_buffer(b, (2, 2, 4, 8), "int8", offset_factor=1)
    C = T.match_buffer(c, (2, 2, 4, 4), "int32", offset_factor=1)
    with T.block("root"):
        T.reads(C[0:2, 0:2, 0:4, 0:4], A[0:2, 0:2, 0:4, 0:8], B[0:2, 0:2, 0:4, 0:8])
        T.writes(C[0:2, 0:2, 0:4, 0:4])
        T.evaluate(
            T.call_extern(
                "handle",
                "gemm_packed_8x16x8_s8s8s32_vl256_ime",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("rw"),
            )
        )


def test_meta_schedule_replay(tmp_path):
    name = "test_ime_prepack_8x16x8"
    tir.TensorIntrin.register(name, ime_desc, ime_desc, override=True)
    mod, weights = _workload(16, 24, 32, k_max=16)
    target = tvm.target.Target("llvm -num-cores=1")
    spaces = generate_design_space(
        "llvm",
        mod,
        target,
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                name, structure="SSRSRS", max_innermost_factor=16
            ),
        ],
    )
    assert spaces
    candidate = spaces[0]
    assert "meta_schedule.auto_tensorize" in candidate.mod.script()
    candidate.enter_postproc()
    assert ms.postproc.RewriteReductionBlock().apply(candidate)
    assert ms.postproc.RewriteTensorize().apply(candidate)
    database = ms.database.JSONDatabase(work_dir=str(tmp_path))
    workload = database.commit_workload(mod)
    database.commit_tuning_record(
        ms.database.TuningRecord(candidate.trace, workload, [1.0], target)
    )
    # Reload from disk and query the original workload; folding must not change the key.
    database = ms.database.JSONDatabase(work_dir=str(tmp_path))
    selected = ms.tir_integration.compile_tir(database, mod, target)
    assert selected is not None
    tvm.ir.assert_structural_equal(candidate.mod, selected.mod)
    packed = tir.transform.FoldConstantWeightPacking()(selected.mod)
    _check_packed(selected.mod, packed, weights)
    _run(selected.mod, weights, 16)
    # Replay exactly the same trace with the external IME implementation.
    # Build-only: no IME instructions or project experiment scripts execute here.
    tir.TensorIntrin.register(name, ime_desc, ime_extern, override=True)
    selected = ms.tir_integration.compile_tir(database, mod, target)
    packed = tir.transform.FoldConstantWeightPacking()(selected.mod)
    _check_packed(selected.mod, packed, weights)
    source = tvm.build(selected.mod, target="c").get_source()
    assert "gemm_packed_8x16x8_s8s8s32_vl256_ime(" in source
    riscv = tvm.build(
        selected.mod, target="llvm -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+a,+f,+d,+c,+v"
    )
    riscv.save(str(tmp_path / "ime.o"))


def test_relay_parameter_binding(tmp_path):
    from tvm import relay
    from tvm.contrib import graph_executor

    m, n, k = 16, 24, 16
    _, weights = _workload(m, n, k)
    data = relay.var("data", shape=(m, k), dtype="int8")
    weight = relay.var("weight", shape=(n, k), dtype="int8")
    mod = tvm.IRModule.from_expr(
        relay.Function([data, weight], relay.nn.dense(data, weight, out_dtype="int32"))
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
    _check_packed(workload_mod, folded[-1], weights)
    runtime = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    inputs = ((np.arange(m * k).reshape(m, k) * 3) % 17 - 8).astype("int8")
    runtime.set_input("data", inputs)
    runtime.run()
    np.testing.assert_array_equal(
        runtime.get_output(0).numpy(), inputs.astype("int32") @ weights.astype("int32").T
    )


if __name__ == "__main__":
    tvm.testing.main()
