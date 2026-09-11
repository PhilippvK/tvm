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
"""Parameterized structural priors for multi-level tiling."""
import pytest
import tvm
from tvm import meta_schedule as ms, te, tir
from tvm.meta_schedule.testing import te_workload


@pytest.mark.parametrize("prefix,fixed", [
    ({}, {}),
    ({"S0": [2, 16]}, {}),
    ({"S0": [2, 16], "S1": [2, 8]}, {"S1": [2, 16, 3, 1]}),
    ({"S0": [2, 8]}, {}),
])
def test_constraints_and_replay(prefix, fixed):
    mod = tvm.IRModule({"main": te.create_prim_func(te_workload.matmul(128, 128, 128))})
    rule = ms.schedule_rule.MultiLevelTiling(
        "SSRSRS", max_innermost_factor=128,
        tile_prefix_products=prefix, tile_fixed_factors=fixed,
    )
    restored = tvm.ir.load_json(tvm.ir.save_json(rule))
    tvm.ir.assert_structural_equal(rule.tile_prefix_products, restored.tile_prefix_products)
    tvm.ir.assert_structural_equal(rule.tile_fixed_factors, restored.tile_fixed_factors)
    for candidate in (rule, rule.clone()):
        for seed in range(1, 6):
            sch = tir.Schedule(mod, seed=seed)
            result, = candidate.apply(sch, sch.get_block("C"))
            loops = result.get_loops(result.get_block("C"))
            extents = [int(result.get(loop).extent) for loop in loops]
            for key, (_, product) in prefix.items():
                axis = int(key[1:])
                assert extents[axis] * extents[axis + 2] == product
            if fixed:
                assert extents[6] == 16
                assert extents[9] == 1
            if not prefix:
                samples = [i for i in result.trace.insts if i.kind.name == "SamplePerfectTile"]
                assert [int(i.attrs[0]) for i in samples] == [4, 4, 2]
            replay = tir.Schedule(mod)
            result.trace.apply_to_schedule(replay, remove_postproc=False)
            tvm.ir.assert_structural_equal(result.mod, replay.mod)


@pytest.mark.parametrize("prefix,fixed", [
    ({"M": [2, 16]}, {}),
    ({"S0": [2]}, {}),
    ({"S0": [1, 16]}, {}),
    ({"S0": [2, 0]}, {}),
    ({}, {"S1": [2, 16, 3, 1]}),
    ({"S1": [2, 8]}, {"S1": [2, 16, 2, 1]}),
    ({"S1": [2, 8]}, {"S1": [2, -16, 3, 1]}),
])
def test_invalid_configuration(prefix, fixed):
    with pytest.raises((ValueError, tvm.error.TVMError)):
        ms.schedule_rule.MultiLevelTiling(
            "SSRSRS", tile_prefix_products=prefix, tile_fixed_factors=fixed,
        )


@pytest.mark.parametrize("prefix,fixed", [
    ({"S0": [2, 3]}, {}),
    ({"S1": [2, 16]}, {"S1": [2, 16, 3, 1]}),
])
def test_incompatible_extent(prefix, fixed):
    sch = tir.Schedule(te.create_prim_func(te_workload.matmul(128, 128, 128)))
    rule = ms.schedule_rule.MultiLevelTiling(
        "SSRSRS", tile_prefix_products=prefix, tile_fixed_factors=fixed,
    )
    with pytest.raises((ValueError, tvm.error.TVMError)):
        rule.apply(sch, sch.get_block("C"))


def test_reduction_constraint():
    sch = tir.Schedule(te.create_prim_func(te_workload.matmul(128, 128, 128)), seed=1)
    rule = ms.schedule_rule.MultiLevelTiling(
        "SSRRSSRR", tile_prefix_products={"R0": [2, 8]},
        tile_fixed_factors={"R0": [3, 1, 2, 16]},
    )
    result, = rule.apply(sch, sch.get_block("C"))
    extents = [int(result.get(loop).extent) for loop in result.get_loops(result.get_block("C"))]
    assert extents[4] * extents[5] == 8
    assert extents[10:] == [16, 1]


def test_fixed_innermost_limit():
    sch = tir.Schedule(te.create_prim_func(te_workload.matmul(128, 128, 128)))
    rule = ms.schedule_rule.MultiLevelTiling(
        "SSRSRS", max_innermost_factor=4, tile_prefix_products={"S1": [2, 8]},
        tile_fixed_factors={"S1": [2, 1, 3, 16]},
    )
    with pytest.raises((ValueError, tvm.error.TVMError)):
        rule.apply(sch, sch.get_block("C"))
