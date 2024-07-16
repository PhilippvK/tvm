import sys
import argparse
import tempfile
from pathlib import Path
from types import MappingProxyType
from collections import defaultdict

import numpy as np
import pandas as pd

import tvm
from tvm import relay, tir
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.relay.backend import Executor
from tvm.tir.analysis import estimate_tir_flops


parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--workload", default="conv2d_relay", help="TODO")
parser.add_argument("--database", "--db", default=None, help="TODO")
parser.add_argument("--out", "-o", default=None, help="TODO")
parser.add_argument("--print-ir-mod", action="store_true", help="TODO")
parser.add_argument("--trivial-schedule", action="store_true", help="TODO")
parser.add_argument("--fallback-schedule", action="store_true", help="TODO")
parser.add_argument("--database-trials", "--trials", type=int, nargs="+", default=[], help="TODO")
parser.add_argument("--database-topk", "--topk", type=int, default=0, help="TODO")
parser.add_argument("--project-options", default=None, help="TODO")
parser.add_argument("--runtime", default="crt", help="TODO")
parser.add_argument("--executor", default="aot", help="TODO")
parser.add_argument("--alter-op", action="store_true", help="TODO")
args = parser.parse_args()

# TODO: move to workloads.py
# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off
@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def create_relay_module():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    np.random.seed(seed=1234)
    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info


if args.workload == "matmul_tir":
    ir_mod = Matmul
elif args.workload == "conv2d_relay":
    ir_mod, params, model_info = create_relay_module()
else:
    raise ValueError(f"Unsupported workload: {args.workload}")

if args.print_ir_mod:
    print("ir_mod", ir_mod)
    # TODO: fancy print?


schedules = []  # stores tuples (mode, record, trial_idx)
if args.trivial_schedule:
    schedules.append(("trivial", None, None))
if args.fallback_schedule:
    schedules.append(("fallback", None, None))
if args.database:
    db_path = Path(args.database)
    path_workload = db_path / "database_workload.json"
    path_tuning_record = db_path / "database_tuning_record.json"
    database = ms.database.JSONDatabase(path_workload=str(path_workload), path_tuning_record=str(path_tuning_record))
    records = database.get_all_tuning_records()
    workload2records = defaultdict(list)
    for i, record in enumerate(records):
        workload = record.workload
        workload2records[workload].append(records[i])
    print("workload2records", workload2records)
    main_workloads = [workload for workload, records in workload2records.items() if len(records) > 1]
    print("main_workloads", main_workloads)
    assert len(main_workloads) == 1
    main_workload = main_workloads[0]
    records_ = workload2records[main_workload]
    records_by_time = sorted(records_, key=lambda x: x.timestamp)
    print("records_", records_)
    if args.database_trials:
        for trial_idx in args.database_trials:
            if trial_idx >= len(records_by_time):
                break
            record = records_by_time[trial_idx]
            schedules.append(("meta_schedule", record, trial_idx))
    if args.database_topk:
        topk_records = database.get_top_k(main_workload, args.database_topk)
        print("topk_records", topk_records)
        for k in range(min(len(topk_records), args.database_topk)):
            record = topk_records[k]
            print("record", record)
            print("record.run_secs", record.run_secs)
            trial_idx = records_by_time.index(record)
            print("trial_idx", trial_idx)
            schedules.append(("meta_schedule", record, trial_idx))

print("schedules", schedules)
# input("1")


def _schedule_trivial(sch):
    return True
    # block = sch.get_block("matmul")
    # i, j, k = sch.get_loops(block=block)
    # i_tiles = [1, 1, 2, 512]
    # j_tiles = [1, 512, 1, 2]
    # k_tiles = [256, 4]
    # i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    # j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    # k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    # sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn


def _create_schedule(mod, sch_fn):
    sch = tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch

if args.runtime == "crt":
    runtime = relay.backend.Runtime("crt", {"system-lib": True})
else:
    raise ValueError(f"Unsupported runtime: {args.runtime}")
if args.executor == "aot":
    link_params = True
    executor = Executor("aot", {"link-params": link_params})
else:
    raise ValueError(f"Unsupported executor: {args.executor}")

# This line is necessary for link-params to take effect during
# task extraction and relay.build(...).
ir_mod = ir_mod.with_attr("executor", executor)
# print("mod2", mod)

target = tvm.target.Target("c")
print("target", target)

pass_config = {
    "tir.disable_vectorize": True
}
disabled_pass = []
if not args.alter_op:
    disabled_pass += ["AlterOpLayout"]
# #             # "relay.backend.use_meta_schedule_dispatch": 2,


def eval_single_record(mod, mode, record, target, pass_config, disabled_pass, runtime, executor, platform, options, dispatch=1, skip_io=True):
    if mode == "trivial":
        assert record is None
        flops_ = None
        db = ms.database.ScheduleFnDatabase(
            _schedule_dummy()
        )
    elif mode == "fallback":
        assert record is None
        flops_ = None
        db = None
    elif mode == "meta_schedule":
        assert record is not None
        flops_ = estimate_tir_flops(record.workload.mod)
        db = ms.database.MemoryDatabase()
        _ = db.commit_workload(record.workload.mod)
        db.commit_tuning_record(record)
        # TODO: skip db and just use mod from trace/workload?
    else:
        raise ValueError(f"Invalid mode: {mode}")
    ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        database=db,
        mod=mod,
        target=target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule": True,
                "relay.backend.tir_converter": "default",
                "relay.backend.use_meta_schedule_dispatch": dispatch,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        project = tvm.micro.generate_project(
            str(tvm.micro.get_microtvm_template_projects(platform)),
            ms_mod,
            str(work_dir / "project"),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            if not skip_io:
                data_sample = None  # TODO
                aot_executor.get_input(0).copyfrom(data_sample)
            result = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result", result)
            print("mean: ", result.mean)
            if not skip_io:
                output = aot_executor.get_output(0).numpy()
                print(output)
        return result.mean, flops_


if args.project_options is None:
    # use default
    toolchain = "gcc"
    platform = "/work/git/tvmtests/microtvm-etiss-template/template_project"
    project_options = {
        "verbose": True,
        "quiet": True,
        # "gcc_prefix": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/riscv_gcc",
        "gcc_prefix": "/tmp/riscv_tools_rv64imfd_lp64d_medany/gnu",
        # "gcc_name": "riscv32-unknown-elf",
        "gcc_name": "riscv64-unknown-elf",
        # "llvm_dir": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/llvm",
        "llvm_dir": "/tmp/seal5_llvm_corev/.seal5/build/release_assertions",
        # "toolchain": "gcc",
        # "toolchain": "llvm",
        "toolchain": toolchain,
        # "etiss_script": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/etiss/bin/run_helper.sh",
        "etiss_script": "/work/git/mlonmcu/mlonmcu/workspace_default/deps/install/etiss/bin/run_helper.sh",
        "etiss_args": "",
        # "arch": "rv32gc_zicsr_zifencei",
        "arch": "rv64imfd",
        # "abi": "ilp32d",
        "abi": "lp64d",
        # "cpu_arch": "RV32IMACFD",
        "cpu_arch": "RV64IMACFD",
        "cpu_freq": 100000000,
    }
else:
    raise NotImplementedError("load options from yaml")

df_data = []
for schedule in schedules:
    mode, record, trial_idx = schedule
    mean_s, flops = eval_single_record(ir_mod, mode, record, target, pass_config, disabled_pass, runtime, executor, platform, project_options)
    df_data_ = {"mode": mode, "record": trial_idx, "mean_s": mean_s, "flops": flops}
    df_data.append(df_data_)

df = pd.DataFrame(df_data)
unique_flops = df["flops"].dropna().unique()
print("unique_flops", unique_flops)
assert len(unique_flops) == 1
df["flops"].fillna(unique_flops[0], inplace=True)
df["flops_per_s"] = df["flops"] / df["mean_s"]
print("df")
print(df)
if args.out:
    df.to_csv(args.out, index=False)
