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

""" Test rpc based launcher for hexagon """
import time
import pickle
import datetime
import argparse
from pathlib import Path
import tempfile
import hashlib
from types import MappingProxyType
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
import networkx as nx
# import pytest
import tvm.testing
import tvm.topi.testing
from tvm import meta_schedule as ms
from tvm import relay, te
from tvm.ir import structural_equal, structural_hash, transform
# from tvm.contrib.hexagon.meta_schedule import (
#     get_hexagon_local_builder,
#     get_hexagon_rpc_runner,
# )
from tvm.meta_schedule import postproc, schedule_rule
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput
from tvm.meta_schedule.runner import RunnerInput
from tvm.script import tir as T
from tvm.tir import FloatImm
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.tir import TensorIntrin

from rule_utils import generate_rules
from mod_utils import get_dense_relay_module, get_conv2d_relay_module
from estimate_utils import estimate_size


def get_dotp_intrin(dtype_a, dtype_b, dtype_c, count, rhs_lanes=1):
    global name_supply
    assert dtype_a == "int8"
    assert dtype_b == "int8"
    assert dtype_c == "int32"

    @T.prim_func
    def dotp_desc(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((rhs_lanes, count), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((rhs_lanes,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:rhs_lanes], A[0:count], B[0:count, 0:rhs_lanes])
            T.writes(C[0:rhs_lanes])
            for i in T.serial(0, rhs_lanes):
                for k in T.serial(0, count):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], dtype_c) * T.cast(B[vi, vk], dtype_c)

    @T.prim_func
    def dotp_impl(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((rhs_lanes, count), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((rhs_lanes,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:rhs_lanes], A[0:count], B[0:rhs_lanes, 0:count])
            T.writes(C[0:rhs_lanes])
            # C[0:rhs_lanes] = T.call_pure_extern(
            #     f"dotp_kernel_1x{count}x{rhs_lanes}",  # TODO: rename
            #     # f"dotp_kernel_{count}x_n{rhs_lanes}",  # TODO: rename
            #     # f"dotp_kernel_{count}x_m{lhs_lanes}",  # TODO: rename
            #     A.access_ptr("r", offset=0),
            #     B.access_ptr("r", offset=0),
            #     C.access_ptr("w", offset=0),
            #     dtype=f"{dtype_c}x{rhs_lanes}",
            # )
            T.evaluate(
                T.call_extern(
                    "handle",
                    f"dotp_kernel_1x{count}x{rhs_lanes}",  # TODO: rename
                    # f"dotp_kernel_{count}x_n{rhs_lanes}",  # TODO: rename
                    # f"dotp_kernel_{count}x_m{lhs_lanes}",  # TODO: rename
                    A.access_ptr("r", offset=0),
                    B.access_ptr("r", offset=0),
                    C.access_ptr("rw", offset=0),
                )
            )
            # C[T.ramp(T.int32(0), 1, rhs_lanes)] = ?

    # TODO: add types to name: dotp_i8i8i32_...
    return dotp_desc, dotp_impl


def get_gemm_intrin(dtype_a, dtype_b, dtype_c, M, K, N):
    global name_supply
    assert dtype_a == "int8"
    assert dtype_b == "int8"
    assert dtype_c == "int32"

    @T.prim_func
    def gemm_desc(
        A: T.Buffer((M, K,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((N, K), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((M, N), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:M, 0:N], A[0:M, 0:K], B[0:N, 0:K])
            T.writes(C[0:M, 0:N])
            for i in T.serial(0, M):
                for j in T.serial(0, N):
                    for k in T.serial(0, K):
                        with T.block("update"):
                            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                            C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], dtype_c) * T.cast(B[vj, vk], dtype_c)

    @T.prim_func
    def gemm_impl(
        A: T.Buffer((M, K), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((N, K), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((M, N), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:M, 0:N], A[0:M, 0:K], B[0:N, 0:K])
            T.writes(C[0:M, 0:N])
            # C[0:M, 0:N] = T.call_pure_extern(
            #     f"gemm_kernel_{N}x{K}x{M}",  # TODO: rename
            #     A.access_ptr("r", offset=0),
            #     B.access_ptr("r", offset=0),
            #     C.access_ptr("w", offset=0),
            #     dtype=f"{dtype_c}x{M}",
            # )
            T.evaluate(
                T.call_extern(
                    "handle",
                    f"gemm_kernel_{N}x{K}x{M}",  # TODO: rename
                    A.access_ptr("r", offset=0),
                    B.access_ptr("r", offset=0),
                    C.access_ptr("rw", offset=0),
                )
            )
            # C[T.ramp(T.int32(0), 1, rhs_lanes)] = ?

    # TODO: add types to name: gemm_i8i8i32_...
    return gemm_desc, gemm_impl


def gen_intrins(intrin):
    ret = []
    # TODO: handle dtypes!
    # print("intrin", intrin)
    if intrin.startswith("dotp_"):
        dims = intrin.split("_", 1)[1]
        # print("dims", dims)
        assert dims.count("x") == 1
        k, n = dims.split("x", 1)
        assert k == "K"
        num_lanes = int(n)
        # num_lanes = int(intrin.split("x", 1)[-1])
        # COUNTS = [32, 64]  # TODO: add others
        COUNTS = [64, 32, 16, 8]  # TODO: add others
        # TODO: add dtypes to name?
        for count in COUNTS:
            dotp_intrin_name = f"dotp_{count}x{num_lanes}"
            found = TensorIntrin.get(dotp_intrin_name, allow_missing=True)
            # print("found", found)
            if not found:
                # print("register")
                TensorIntrin.register(dotp_intrin_name, *get_dotp_intrin("int8", "int8", "int32", count, rhs_lanes=num_lanes))
            ret.append(dotp_intrin_name)
    elif intrin.startswith("gemm_"):
        dims = intrin.split("_", 1)[1]
        # print("dims", dims)
        assert dims.count("x") == 2
        m, k, n = dims.split("x", 2)
        m = int(m)
        assert k == "K"
        n = int(n)
        COUNTS = [64, 32, 16, 8]  # TODO: add others
        # TODO: add dtypes to name?
        for count in COUNTS:
            gemm_intrin_name = f"gemm_{m}x{count}x{n}"
            found = TensorIntrin.get(gemm_intrin_name, allow_missing=True)
            # print("found", found)
            if not found:
                # print("register")
                TensorIntrin.register(gemm_intrin_name, *get_gemm_intrin("int8", "int8", "int32", n, count, m))
            ret.append(gemm_intrin_name)
    else:
        raise NotImplementedError(f"Intrin: {intrin}")
    # print("ret", ret, len(ret))
    return ret







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-spaces", type=int, default=100000000)
    parser.add_argument("--base-out", default=None)
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--cont", default=None)
    parser.add_argument("--start-space-id", type=int, default=0)
    parser.add_argument("--start-mod-id", type=int, default=0)
    parser.add_argument("--strategy", default="evolutionary")

    # TODO: expose out dir
    args = parser.parse_args()
    if args.cont:
        assert args.out is None
        out_dir = Path(args.cont)
        assert out_dir.is_dir()
    elif args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True)
    else:
        out_dir_base = Path(args.base_out) if args.base_out is not None else Path("exp_out")
        dt = datetime.datetime.now()
        ts = dt.strftime("%Y%m%dT%H%M%S")
        print("ts", ts)
        out_dir = out_dir_base / ts
        out_dir.mkdir(parents=True)
    print("out_dir", out_dir)
    rules_dir = out_dir / "rules"
    rules_dir.mkdir(exist_ok=True)

    strategy_kind = args.strategy
    strategy_kwargs = dict(
        # population_size=512,
        # population_size=128,
        # population_size=1024,
        # population_size=1024 * 16 * 16,
        # population_size=1024 * 16,
        # population_size=1024 * 16 * 2,
        population_size=1024 * 16 // 2,
        # population_size=16,
        init_measured_ratio=0.2,
        # init_measured_ratio=0.9,
        init_min_unmeasured=50,
        # init_min_unmeasured=10,
        # max_fail_count=5,
        max_fail_count=1,
        # genetic_num_iters=4,
        genetic_num_iters=6,
        genetic_mutate_prob=0.85,
        # genetic_max_fail_count=10,
        genetic_max_fail_count=5,
        # eps_greedy=0.05,
        eps_greedy=0.25,
    )

    strategy_file = out_dir / "strategy.yaml"
    with open(strategy_file, "w") as f:
        yaml.dump({"kind": strategy_kind, **strategy_kwargs}, f)

    target = tvm.target.Target("llvm -num-cores=1")

    mods = []

    # OPS = ["dense", "conv2d_3x3"]
    # OPS = ["dense"]
    OPS = ["conv2d_3x3"]
    # OPS = ["conv2d_1x1"]
    # DTYPES = ["float32", "int32"]
    # DTYPES = ["int32"]
    DTYPES = ["int8"]
    # DIMS = [8, 16, 32, 64, 128]
    # DIMS = [8, 32, 128]
    # DIMS = [8]
    # DIMS = [16]
    DIMS = [32]
    LAYOUTS = [
        # ("NCHW", "OIHW"),
        ("NHWC", "OHWI"),
        # ("NHWC", "HWOI"),
        # ("NHWC", "HWIO"),
    ]

    for op in OPS:
        for dtype in DTYPES:
            if op == "dense":
                for dim in DIMS:
                    mod, params = get_dense_relay_module(dim, dtype)
                    mod_name = op
                    mods.append((mod, params, mod_name, None, None))
            elif op == "conv2d_3x3":
                for (data_layout, kernel_layout) in LAYOUTS:
                    print("data_layout", data_layout)
                    print("kernel_layout", kernel_layout)
                    for c in DIMS:
                        print("c", c)
                        mod, params = get_conv2d_relay_module(h=32, w=32, kw=3, kh=3, cin=c, cout=c, dtype=dtype, data_layout=data_layout, kernel_layout=kernel_layout)
                        mod_name = op
                        mods.append((mod, params, mod_name, data_layout, kernel_layout))
            elif op == "conv2d_1x1":
                for (data_layout, kernel_layout) in LAYOUTS:
                    print("data_layout", data_layout)
                    print("kernel_layout", kernel_layout)
                    for c in DIMS:
                        print("c", c)
                        mod, params = get_conv2d_relay_module(h=32, w=32, kw=1, kh=1, cin=c, cout=c, dtype=dtype, data_layout=data_layout, kernel_layout=kernel_layout)
                        mod_name = op
                        mods.append((mod, params, mod_name, data_layout, kernel_layout))
            else:
                raise NotImplementedError(f"Op: {op}")

    num_mods = len(mods)
    print("num_mods", num_mods)

    # sch_rules = "from-target"
    # structure = "SSRSRS"
    # structure = "RS"
    # structure = "RSRS"
    # structure = "SRSRS"
    all_rules = ["ApplyCustomRule", "InlineConstantScalars", "AutoInline", "AddRFactor", "MultiLevelTiling", "ParallelizeVectorizeUnroll", "RandomComputeLocation"]

    # INTRINS = ["none", "dotp", "dotp_1x2", "dotp_1x4", "dotp_2x1", "dotp_4x1"]  # TODO: gemm
    # INTRINS = ["none", "dotp"]
    # dotp_KxN
    # INTRINS = ["dotp_Kx1", "dotp_Kx2", "dotp_Kx4"]
    # INTRINS = ["dotp_Kx1"]
    # INTRINS = ["dotp_Kx2"]
    # INTRINS = ["dotp_Kx4"]
    # gemm_MxKxN
    # INTRINS = ["gemm_1xKx1"]
    # INTRINS = ["gemm_1xKx2"]
    # INTRINS = ["gemm_1xKx4"]
    # INTRINS = ["gemm_2xKx1"]
    # INTRINS = ["gemm_2xKx2"]
    # INTRINS = ["gemm_2xKx4"]
    # INTRINS = ["gemm_4xKx1"]
    # INTRINS = ["gemm_4xKx2"]
    # INTRINS = ["gemm_4xKx4"]
    # SWAP_MLT_RULES = [False, True]
    # SWAP_MLT_RULES = [False]
    # mixed
    # INTRINS = ["none", "dotp_Kx1", "dotp_Kx2", "dotp_Kx4", "gemm_2xKx2", "gemm_4xKx4"]
    INTRINS = ["none"]
    # INTRINS = ["gemm_4xKx4"]
    SWAP_MLT_RULES = [True]

    structures = []
    sch_rules_space = []
    SKIP_INTRIN_MLT = True
    MLT = [False, True]
    # MLT = [True]
    # MLT = [False]
    # MLT = [True]
    UNROLL = [False, True]
    # UNROLL = [True]
    # RFACTOR = [False, True]
    RFACTOR = [True]
    # MLT_STRUCTURE = ["S", "R", "RS", "SR", "RSRS"]
    # MLT_STRUCTURE = ["RS", "SR"]
    MLT_STRUCTURE = ["SSR", "SRS", "RSS"]
    # MLT_STRUCTURE = ["SSR"]
    # MLT_STRUCTURE = ["RSRS"]
    # MLT_STRUCTURE = ["S", "RSRS"]
    # MLT_MAX_INNERMOST_FACTOR = [64]
    # MLT_MAX_INNERMOST_FACTOR = [32, 64]
    MLT_MAX_INNERMOST_FACTOR = [64]
    RFACTOR_MAX_INNERMOST_FACTOR = [64]
    # RFACTOR_MAX_INNERMOST_FACTOR = [2, 4, 8, 16, 32, 64]
    # RFACTOR_MAX_INNERMOST_FACTOR = [2]
    # UNROLL_MAX_STEPS = [[0, 2, 4, 8, 16, 32, 64]]
    # UNROLL_MAX_STEPS = [[0, 2, 4, 8, 16, 32]]
    # UNROLL_MAX_STEPS = [[0, 2], [0, 2, 4], [0, 2, 4, 8], [0, 2, 4, 8, 16, 32], [0, 2, 4, 8, 16, 32, 64]]
    # UNROLL_MAX_STEPS = [[0, 2], [0, 2, 4], [0, 2, 4, 8], [0, 2, 4, 8, 16], [0, 2, 4, 8, 16, 32], [0, 2, 4, 8, 16, 32, 64]]
    # UNROLL_MAX_STEPS = [[0, 2, 4, 8, 16, 32, 64]]
    UNROLL_MAX_STEPS = [[0, 2, 4, 8], [0, 2, 4, 8, 16, 32, 64]]
    UNROLL_EXPLICIT = [False, True]
    # UNROLL_EXPLICIT = [True]
    RANDOM_COMPUTE_LOCATION = [False, True]
    # RANDOM_COMPUTE_LOCATION = [True]
    AUTO_INLINE = [False, True]
    # AUTO_INLINE = [True]
    INLINE_CONST = [False, True]
    # INLINE_CONST = [True]
    # TODO: add model with activation func
    # TODO: add model with bias
    # TODO: add model with requant?
    for intrin in INTRINS:
        # print("intrin", intrin)
        mlts_ = MLT if intrin == "none" or not SKIP_INTRIN_MLT else [False]
        for enable_mlt in mlts_:
            # print("  enable_mlt", enable_mlt)
            swap_mlt_rules_ = SWAP_MLT_RULES if intrin != "none" and enable_mlt else [False]
            for swap_mlt_rules in swap_mlt_rules_:
                # print("    swap_mlt_rules", swap_mlt_rules)
                for enable_unroll in UNROLL:
                    for enable_auto_inline in AUTO_INLINE:
                        for enable_random_compute_location in RANDOM_COMPUTE_LOCATION:
                            for enable_inline_const in INLINE_CONST:
                                for enable_rfactor in RFACTOR:
                                    rfactor_max_innermost_factors = RFACTOR_MAX_INNERMOST_FACTOR if enable_rfactor else [None]
                                    for rfactor_max_innermost_factor in rfactor_max_innermost_factors:
                                        mlt_structures = MLT_STRUCTURE if enable_mlt or intrin != "none" else [None]  # TODO: different structure for mlti?
                                        for mlt_structure in mlt_structures:
                                            assert mlt_structure is None or len(mlt_structure) > 1
                                            mlt_max_innermost_factors = MLT_MAX_INNERMOST_FACTOR if enable_mlt else [None]
                                            for mlt_max_innermost_factor in mlt_max_innermost_factors:
                                                unroll_max_steps_ = UNROLL_MAX_STEPS if enable_unroll else [None]
                                                for unroll_max_steps in unroll_max_steps_:
                                                    unroll_explicits = UNROLL_EXPLICIT if enable_unroll else [None]
                                                    for unroll_explicit in unroll_explicits:
                                                        enabled_rules = [rule for rule in all_rules]
                                                        if not enable_mlt:
                                                            enabled_rules.remove("MultiLevelTiling")
                                                        if not enable_unroll:
                                                            enabled_rules.remove("ParallelizeVectorizeUnroll")
                                                        if not enable_rfactor:
                                                            enabled_rules.remove("AddRFactor")
                                                        if not enable_auto_inline:
                                                            enabled_rules.remove("AutoInline")
                                                        if not enable_random_compute_location:
                                                            enabled_rules.remove("RandomComputeLocation")
                                                        if not enable_inline_const:
                                                            enabled_rules.remove("InlineConstantScalars")
                                                        rule_kwargs = dict(
                                                            rules=enabled_rules, rfactor_max_innermost_factor=rfactor_max_innermost_factor, mlt_structure=mlt_structure, mlt_max_innermost_factor=mlt_max_innermost_factor, unroll_max_steps=unroll_max_steps, unroll_explicit=unroll_explicit, intrin=intrin, swap_mlt_rules=swap_mlt_rules,
                                                        )
                                                        sch_rules = generate_rules(**rule_kwargs)
                                                        temp = (rule_kwargs, sch_rules)
                                                    sch_rules_space.append(temp)
    print("len(sch_rules_space)", len(sch_rules_space))
    input("!!!")
    max_spaces = args.max_spaces
    if max_spaces:
        sch_rules_space = sch_rules_space[:max_spaces]
    print("len(sch_rules_space)", len(sch_rules_space))
    for space_id, temp in enumerate(sch_rules_space):
        rule_kwargs, sch_rules = temp
        # print("space_id", space_id)
        # print("rule_kwargs", rule_kwargs)
        rules_file = rules_dir / f"{space_id}.yaml"
        with open(rules_file, "w") as f:
            yaml.dump(rule_kwargs, f)
    # input("!")
    mods_dir = out_dir / "mods"
    mods_dir.mkdir(exist_ok=True)
    for mod_id, (mod, params, mod_name, data_layout, kernel_layout) in enumerate(mods):
        if mod_id < args.start_mod_id:
            continue
        # summary_data = [{"space_id": None, "task_id": None, "task_name": None}]
        mod_dir = mods_dir / str(mod_id)
        mod_dir.mkdir(exist_ok=True)
        tasks_dir = mod_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        summary_file = mod_dir / "summary.csv"
        metrics_file = mod_dir / "metrics.csv"
        if args.cont:
            # TODO: restore rules from disk?
            summary_df = pd.read_csv(summary_file)
            summary_data = summary_df.to_dict(orient="records")
            # print("summary_data", summary_data, len(summary_data))
            metrics_df = pd.read_csv(metrics_file)
            metrics_data = metrics_df.to_dict(orient="records")
            # print("metrics_data", metrics_data, len(metrics_data))
        else:
            summary_df = pd.DataFrame([])
            summary_df.to_csv(summary_file, index=False)
            metrics_df = pd.DataFrame([])
            metrics_df.to_csv(metrics_file, index=False)
    for mod_id, (mod, params, mod_name, data_layout, kernel_layout) in enumerate(mods):
        if mod_id < args.start_mod_id:
            # TODO: auto
            continue
        if args.cont:
            # TODO: restore rules from disk?
            summary_df = pd.read_csv(summary_file)
            summary_data = summary_df.to_dict(orient="records")
            # print("summary_data", summary_data, len(summary_data))
            metrics_df = pd.read_csv(metrics_file)
            metrics_data = metrics_df.to_dict(orient="records")
            # print("metrics_data", metrics_data, len(metrics_data))
            space_sizes = {}
            task_shashs = defaultdict(list)
            task_space_shashs = defaultdict(dict)
            task_ids = sorted(map(lambda p: int(p.name), tasks_dir.glob("*")))
            # print("task_ids", task_ids)
            for task_id in task_ids:
                task_dir = tasks_dir / str(task_id)
                assert task_dir.is_dir()
                spaces_dir = task_dir / "space"
                assert spaces_dir.is_dir()
                space_ids = sorted(map(lambda p: int(p.name), spaces_dir.glob("*")))
                # assert len(sch_rules_space) == len(space_ids), f"{len(sch_rules_space)} vs. {len(space_ids)}"
                for space_id in space_ids:
                    rows = summary_df[(summary_df["task_id"] == task_id) & (summary_df["space_id"] == space_id)]
                    # print("rows", rows)
                    if len(rows) == 0:
                        continue
                    assert len(rows) == 1
                    row = rows.iloc[0]
                    space_size = row["search_space_size"]
                    space_sizes[(space_id, task_id)] = space_size
                    space_dir = spaces_dir / str(space_id)
                    if not space_dir.is_dir():
                        continue
                    assert space_dir.is_dir()
                    # assert shashs_txt.is_file()
                    shashs_txt = space_dir / "shashs.txt"
                    if shashs_txt.is_file():
                        with open(shashs_txt, "r") as f:
                            shashs = set(list(map(lambda x: int(x.strip()), f.readlines())))
                        # print("shashs", shashs, len(shashs))
                        task_space_shashs[task_id][space_id] = list(shashs)
                        # print("shashs[0]", list(shashs)[0], type(list(shashs)[0]))
                        task_shashs[task_id] = list(set(task_shashs[task_id]) | set(shashs))
            # print("space_sizes", space_sizes, len(space_sizes))
            # print("task_shashs", task_shashs, len(task_shashs))
            # print("task_space_shashs", task_space_shashs, len(task_space_shashs))
            # input("!!!")
        else:
            summary_data = []
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            metrics_data = []
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            space_sizes = {}
            task_shashs = defaultdict(list)
            task_space_shashs = defaultdict(dict)
        for space_id, temp in enumerate(sch_rules_space):
            if space_id < args.start_space_id:
                continue
            print("space_id", space_id)
            # print("summary_data[-1]['space_id']", summary_data[-1]["space_id"])
            # input("!!!")
            rule_kwargs, sch_rules = temp
            print("rule_kwargs", rule_kwargs)
            print("sch_rules", sch_rules)
            # TODO: move to helpers
            postprocs = "from-target"
            mutator_probs = "from-target"
            print("sch_rules", sch_rules, len(sch_rules))
            space = ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs=mutator_probs,
            )
            assert strategy_kind == "evolutionary"
            strategy = ms.search_strategy.EvolutionarySearch(
                **strategy_kwargs,
            )
            # task_scheduler = ms.task_scheduler.RoundRobin()
            task_scheduler = ms.task_scheduler.GradientBased()
            with tempfile.TemporaryDirectory() as work_dir, ms.Profiler() as profiler:
                opt_level = 3
                module_equality = "structural"
                pass_config = MappingProxyType({})
                disabled_pass = None
                instruments = None
                database = "json"
                builder = "local"  # TODO: fake
                runner = "local"  # TODO: fake
                cost_model = "xgb"
                measure_callbacks = "default"
                seed = None
                num_tuning_cores = "physical"
                tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
                    extracted_tasks=ms.relay_integration.extract_tasks(
                        mod,
                        target,
                        params,
                        opt_level=opt_level,
                        module_equality=module_equality,
                        pass_config=pass_config,
                        disabled_pass=disabled_pass,
                        instruments=instruments,
                    ),
                    work_dir=work_dir,
                    space=space,
                    strategy=strategy,
                    seed=seed,
                    num_tuning_cores=num_tuning_cores,
                )
                num_tasks = len(tasks)
                assert num_tasks > 0
                pass_config = dict(pass_config)
                for task_id in range(num_tasks):
                    if task_id >= len(metrics_data):
                        metrics_data.append({"task_id": task_id})
                        assert len(metrics_data) == (task_id + 1)
                    tasks_ = [tasks[task_id]]
                    task_weights_ = [task_weights[task_id]]
                    task_dir = tasks_dir / str(task_id)
                    task_dir.mkdir(exist_ok=True)
                    spaces_dir = task_dir / "space"
                    spaces_dir.mkdir(exist_ok=True)
                    space_dir = spaces_dir / str(space_id)
                    space_dir.mkdir(exist_ok=True)
                    annotation_hist_file = space_dir / "annotation_hist.csv"
                    annotation_val_hist_file = space_dir / "annotation_val_hist.csv"
                    inst_hist_file = space_dir / "inst_hist.csv"
                    space_shashs_file = space_dir / "shashs.txt"
                    with transform.PassContext(
                        opt_level=opt_level,
                        config=pass_config,
                        disabled_pass=disabled_pass,
                        instruments=instruments,
                    ):
                        t0 = time.time()
                        try:
                            # database = ms.relay_integration.tune_relay(
                            #     mod=mod,
                            #     params=params,
                            #     target=target,
                            #     max_trials_global=1000000,
                            #     num_trials_per_iter=100000,
                            #     # strategy="replay-trace",
                            #     strategy=strategy,
                            #     work_dir=work_dir,
                            #     space=space,
                            #     # builder=get_hexagon_local_builder(),
                            #     # runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
                            #     task_scheduler=task_scheduler,
                            # )
                            database = ms.tune_tasks(
                                tasks=tasks_,
                                task_weights=task_weights_,
                                work_dir=work_dir,
                                max_trials_global=1000000,
                                max_trials_per_task=1000000,
                                num_trials_per_iter=100000,
                                builder=builder,
                                runner=runner,
                                database=database,
                                cost_model=cost_model,
                                measure_callbacks=measure_callbacks,
                                task_scheduler=task_scheduler,
                                module_equality=module_equality,
                            )
                            assert len(task_scheduler.tasks_) == 1
                            from tvm.meta_schedule.database.analyze_traces import analyze_ms_db
                            annotation_hist, annotation_val_hist, inst_hist = analyze_ms_db(database)
                            annotation_hist_df = pd.DataFrame([annotation_hist])
                            annotation_val_hist_df = pd.DataFrame([annotation_val_hist])
                            inst_hist_df = pd.DataFrame([inst_hist])
                            annotation_hist_df.to_csv(annotation_hist_file, index=False)
                            annotation_val_hist_df.to_csv(annotation_val_hist_file, index=False)
                            inst_hist_df.to_csv(inst_hist_file, index=False)
                            # print("work_dir", work_dir)
                            # input("!!!")
                            recs = database.get_all_tuning_records()
                            # print("recs", len(recs))
                            status = "ok"
                            reason = None
                            # input("!")
                        except Exception as e:
                            recs = []
                            status = "failed"
                            reason = repr(e)
                            # raise e
                        # for rec in recs:
                        #     rec_hash
                        t1 = time.time()
                        sampling_sec = t1 - t0
                        # print("tasks[0]", tasks[0], dir(tasks[0]))
                        # print("task_scheduler", task_scheduler, dir(task_scheduler))
                        # print("task_scheduler.tasks_[0]", task_scheduler.tasks_[0], dir(task_scheduler.tasks_[0]))
                        # print("task_scheduler.tasks_[0].candidate_history", task_scheduler.tasks_[0].candidate_history, dir(task_scheduler.tasks_[0].candidate_history))
                        # print("num_tasks", num_tasks)
                        metrics_data[task_id]["num_spaces"] = len(sch_rules_space)
                        task_rec = task_scheduler.tasks_[task_id]
                        task_measure_candidates = task_rec.all_measure_candidates
                        # print("len(task_measure_candidates)", len(task_measure_candidates))
                        shashs = []
                        for c in task_measure_candidates:
                            # print("c", c, dir(c))
                            sch = c.sch
                            # print("sch", sch, dir(sch))
                            sch_mod = sch.mod
                            # print("sch_mod", sch_mod, dir(sch_mod))
                            shash = structural_hash(sch_mod)
                            # print("shash", shash, type(shash))
                            # input("!!!")
                            # print("shash", shash, dir(shash))
                            # exists = sch in task_schs[task_id]
                            # mod_exists = sch_mod in task_mods[task_id]
                            shash_exists = shash in task_shashs[task_id]
                            # print("exists", exists)
                            # print("mod_exists", mod_exists)
                            # print("shash_exists", shash_exists)
                            # if not exists:
                            #     task_schs[task_id].append(sch)
                            # if not mod_exists:
                            #     task_mods[task_id].append(sch_mod)
                            if not shash_exists:
                                task_shashs[task_id].append(shash)
                            # input("o")
                            shashs.append(shash)
                        with open(space_shashs_file, "w") as f:
                            shashs_content = "\n".join(map(str, shashs))
                            f.write(shashs_content)
                        task_space_shashs[task_id][space_id] = shashs
                        # num_unique_schs = len(task_schs[task_id])
                        # print("num_unique_schs", num_unique_schs)
                        # num_unique_mods = len(task_mods[task_id])
                        # print("num_unique_mods", num_unique_mods)
                        num_unique_shashs = len(task_shashs[task_id])
                        # print("num_unique_shash", num_unique_shashs)
                        # input("ooo")
                        task_context = task_rec.ctx
                        task_mod = task_context.mod
                        # task_workload = ms.database.Workload(task_mod)
                        # top_recs = database.get_top_k(task_workload, int(1e6))
                        # print("len(top_recs)", len(top_recs))
                        # assert len(top_recs) < 1e6
                        # for rec in top_recs:
                        #     trace = rec.trace
                        #     print("trace", trace, dir(trace))
                        #     workload = rec.workload
                        #     print("workload", workload, dir(workload))
                        #     mod = workload.mod
                        #     print("mode", mod, dir(mod))
                        #     # rec_hash = 
                        # print(f"Task: {task_id}")
                        history = task_scheduler.tasks_[task_id].candidate_history
                        # assert len(history) > len(top_recs)
                        # TODO: early stopping may lead to missing records for last batch...
                        # print("history", history)
                        if len(history) == 0:
                            # print("empty")
                            search_space_size = None
                            is_estimate = False
                        else:
                            task_trials = int(sum(history))
                            # print("task_trials", task_trials)

                            search_space_size, is_estimate = estimate_size(history)
                            # print("search_space_size", search_space_size)
                            space_sizes[(space_id, task_id)] = search_space_size
                        # print("?", task_scheduler.tasks_[task_id], dir(task_scheduler.tasks_[task_id]))
                        task_func = task_mod["main"]
                        task_args = ArgInfo.from_prim_func(task_func)
                        # print("task_args", task_args)
                        task_args_str = str(task_args)
                        task_args_str = task_args_str.replace("\"", "").replace(" ", "").replace("TensorInfo", "")
                        task_args_hash = hashlib.sha256(task_args_str.encode('utf-8')).hexdigest()
                        # print("task_args_hash", task_args_hash)
                        task_name = task_context.task_name
                        task_flop = task_rec.flop
                        task_weight = task_rec.task_weight
                        new_data = {"mod_id": mod_id, "mod_name": mod_name, "data_layout": data_layout, "kernel_layout": kernel_layout, "space_id": space_id, "task_id": task_id, "task_name": task_name, "task_args": task_args_str, "task_args_hash": task_args_hash, "search_space_size": search_space_size, "sampling_sec": sampling_sec, "is_estimate": is_estimate, "status": status, "reason": reason}
                        summary_data.append(new_data)
                        summary_df = pd.DataFrame(summary_data)
                        print(summary_df)
                        summary_df.to_csv(summary_file, index=False)
                        metrics_data[task_id]["num_evaluated_spaces"] = space_id + 1
                        # metrics_data[task_id]["num_tasks"] = len(list(task_shashs.keys()))
                        if search_space_size is not None:
                            if "num_total_candidates" not in metrics_data[task_id]:
                                metrics_data[task_id]["num_total_candidates"] = 0
                            metrics_data[task_id]["num_total_candidates"] += search_space_size
                        metrics_data[task_id]["num_unique_candidates"] = len(task_shashs[task_id])
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.to_csv(metrics_file, index=False)
                # input("!")
                # lib = ms.relay_integration.compile_relay(
                #     database=database,
                #     mod=mod,
                #     params=params,
                #     target=target,
                # )
            print(profiler.table())
        print(summary_df)
        print("space_sizes", space_sizes)
        sorted_by_size = sorted(space_sizes.items(), key=lambda x: x[1])
        metrics_data[0]["max_total_candidates"] = max(space_sizes.values())
        metrics_data[0]["finished"] = True
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        # TODO: fetch recs from DB
        # TODO: histogram of applied rules in sched trace?
        print("task_space_shashs", task_space_shashs)

        # Stores redundant spaces:
        #   redundant_spaces[task_id] = {redundant_space_id: representative_space_id}
        redundant_spaces = {}
        
        # Stores superset relations:
        #   supersets[task_id] = {
        #       larger_space_id: [smaller_space_ids...]
        #   }
        supersets = {}
        canonical_space = {}

        equivalent_spaces = {}
        
        for task_id in task_space_shashs.keys():
            print(f"\n=== Task {task_id} ===")
        
            task_space_sizes = {
                k[0]: v
                for k, v in space_sizes.items()
                if k[1] == task_id
            }
        
            print("task_space_sizes", task_space_sizes)
        
            # Sort by estimated size
            space_id_by_size = sorted(task_space_sizes.items(), key=lambda x: x[1])
        
            print("space_id_by_size", space_id_by_size)
        
            redundant_spaces[task_id] = {}
            supersets[task_id] = {}
            equivalent_spaces[task_id] = {}
        
            # Convert to sets once
            shash_sets = {
                sid: set(task_space_shashs[task_id][sid])
                for sid, _ in space_id_by_size
            }
        
            # Keep only non-redundant spaces
            filtered_spaces = []
        
            for i, (space_id, space_size) in enumerate(space_id_by_size):
                shashs = shash_sets[space_id]
        
                is_redundant = False
        
                # Compare against smaller spaces only
                for prev_space_id, prev_space_size in filtered_spaces:
                    prev_shashs = shash_sets[prev_space_id]
        
                    common_shashs = shashs & prev_shashs
        
                    print("\n--------------------------------")
                    print("space_id", space_id)
                    print("prev_space_id", prev_space_id)
                    print("space_size", space_size)
                    print("prev_space_size", prev_space_size)
                    print("len(shashs)", len(shashs))
                    print("len(prev_shashs)", len(prev_shashs))
                    print("common", len(common_shashs))
        
                    # -------------------------------------------------
                    # TODO #1:
                    # Detect redundant spaces
                    #
                    # Same candidate set as previous space
                    # -------------------------------------------------
                    if shashs == prev_shashs:
                        print(
                            f"[REDUNDANT] {space_id} identical to {prev_space_id}"
                        )

                        redundant_spaces[task_id][space_id] = prev_space_id
                        equivalent_spaces[task_id].setdefault(prev_space_id, [prev_space_id]).append(space_id)
        
                        is_redundant = True
                        break
        
                    # -------------------------------------------------
                    # TODO #2:
                    # Detect supersets
                    #
                    # Larger space fully contains smaller one
                    # -------------------------------------------------
                    if prev_shashs.issubset(shashs):
                        print(
                            f"[SUPERSET] {space_id} is superset of {prev_space_id}"
                        )
        
                        supersets[task_id].setdefault(space_id, []).append(
                            prev_space_id
                        )
        
                if not is_redundant:
                    filtered_spaces.append((space_id, space_size))
            filtered_space_ids = [space_id for space_id, _ in filtered_spaces]
            metrics_data[task_id]["num_unique_spaces"] = len(filtered_spaces)
            metrics_data[task_id]["num_redundant_spaces"] = len(sch_rules_space) - len(filtered_spaces)
            G = nx.DiGraph()
            for space_id, space_size in task_space_sizes.items():
                # print("===")
                # print("space_id", space_id)
                # print("space_size", space_size)
                num_candidates = len(shash_sets[space_id])
                # print("num_candidates", num_candidates)
                redundant = space_id not in filtered_space_ids
                # print("redundant", redundant)
                G.add_node(
                    space_id,
                    size=space_size,
                    num_candidates=num_candidates,
                    redundant=redundant,
                    label=f"{space_id}\nsize={space_size}\nnum_candidates={num_candidates}\nredundant={redundant}",
                )
            # input("!!!")
            for larger_space, smaller_spaces in supersets[task_id].items():
                shash_large = shash_sets[larger_space]
                for smaller_space in smaller_spaces:
                    shash_small = shash_sets[smaller_space]
                    intersection = len(shash_large & shash_small)

                    containment_ratio = (
                        # intersection / len(shash_small)
                        intersection / len(shash_large)
                    )

                    G.add_edge(
                        larger_space,
                        smaller_space,
                        containment_ratio=containment_ratio,
                        label=f"containment_ratio={containment_ratio}",
                    )
            G_reduced2 = nx.transitive_reduction(G)
            # print("G_reduced2.edges", G_reduced2.edges)
            G_reduced = G.copy()
            to_drop = []
            for edge in G_reduced.edges:
                # print("edge", edge)
                u, v = edge
                # print("u,v", u, v)
                if edge not in G_reduced2.edges:
                    to_drop.append(edge)
                    # print("drop")
            for edge in to_drop:
                u, v = edge
                G_reduced.remove_edge(u, v)
            redundant_nodes = [
                n
                for n, attrs in G_reduced.nodes(data=True)
                if attrs["redundant"]
            ]
            for n in redundant_nodes:
                G_reduced.remove_node(n)
            # print("G_reduced.edges", G_reduced.edges)
            # print("G", G)
            # print("G_reduced", G_reduced)
            # print("G_reduced2", G_reduced2)
            shashs_file = task_dir / "shashs.txt"
            with open(shashs_file, "w") as f:
                shashs_content = "\n".join(map(str, shashs))
                f.write(shashs_content)
            dot_file = task_dir / "graph.dot"
            reduced_dot_file = task_dir / "graph_reduced.dot"
            pkl_file = task_dir / "graph.pkl"
            reduced_pkl_file = task_dir / "graph_reduced.pkl"
            nx.drawing.nx_pydot.write_dot(
                G,
                dot_file,
            )
            nx.drawing.nx_pydot.write_dot(
                G_reduced,
                reduced_dot_file,
            )
            with open(pkl_file, "wb") as f:
                pickle.dump(G, f)
            with open(reduced_pkl_file, "wb") as f:
                pickle.dump(G_reduced, f)
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)

            print("\n==============================")
            print("redundant_spaces")
            print(redundant_spaces)

            print("\n==============================")
            print("equivalent_spaces")
            print(equivalent_spaces)

            print("\n==============================")
            print("supersets")
            print(supersets)
            # print("task_space_shashs", task_space_shashs)
            # for task_id in task_space_shashs.keys():
            #     task_space_sizes = {k[0]: v for k, v in space_sizes.items() if k[1] == task_id}
            #     print("task_space_sizes", task_space_sizes)
            #     space_id_by_size = sorted(task_space_sizes.items(), key=lambda x: x[1])
            #     print("space_id_by_size", space_id_by_size)

            #     for i, (space_id, space_size) in enumerate(space_id_by_size):
            #         for j, (space_id_, space_size_) in enumerate(space_id_by_size):
            #             if j >= i:
            #                 continue
            #             print("i,j", i, j)
            #             print("space_id,space_id_", space_id, space_id_)
            #             print("space_size,space_size_", space_size, space_size_)
            #             shashs = task_space_shashs[task_id][space_id]
            #             shashs_ = task_space_shashs[task_id][space_id_]
            #             common_shashs = set(shashs) & set(shashs_)
            #             print("len(shashs)", len(shashs))
            #             print("len(shashs_)", len(shashs_))
            #             print("common_shashs", common_shashs, len(common_shashs))
            #             # TODO: detect and filter all space_ids which are redundant (the have the same candidates as a previous one)
            #             # TODO: keep track of which spaces are a superset of a previous one


if __name__ == "__main__":
    main()
