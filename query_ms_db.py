import sys
import argparse
import tempfile
from pathlib import Path
from types import MappingProxyType
from collections import defaultdict

import numpy as np

import tvm
from tvm import relay, tir
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.relay.backend import Executor
from tvm.tir.analysis import estimate_tir_flops


parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--workload", default="conv2d_relay", help="TODO")
parser.add_argument("--database", "--db", default=None, help="TODO")
parser.add_argument("--print-ir-mod", action="store_true", help="TODO")
parser.add_argument("--trivial-schedule", action="store_true", help="TODO")
parser.add_argument("--fallback-schedule", action="store_true", help="TODO")
parser.add_argument("--database-trials", "--trials", type=int, action="append", default=[], help="TODO")
parser.add_argument("--database-topk", "--topk", type=int, default=0, help="TODO")
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
    for trial_idx in args.database_trials:
        if trial_idx >= len(records_):
            break
        record = records_[trial_idx]
        schedules.append(("meta_schedule", record, trial_idx))
    if args.database_topk:
        topk_records = database.get_top_k(main_workload, args.database_topk)
        for k in range(min(len(topk_records), args.database_topk)):
            record = topk_records[k]
            trial_idx = records_.index(record)
            schedules.append(("meta_schedule", record, trial_idx))

print("schedules", schedules)
input("1")


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


def _create_schedule(mod, sch_fn):
    sch = tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch




assert len(sys.argv) == 2
db_path = Path(sys.argv[1])
path_workload = db_path / "database_workload.json"
path_tuning_record = db_path / "database_tuning_record.json"
# mod = Matmul
mod, params, model_info = create_relay_module()
# print("mod", mod)

link_params = True

if args.runtime == "crt":
    runtime = relay.backend.Runtime("crt", {"system-lib": True})
else:
    raise ValueError(f"Unsupported runtime: {args.runtime}")
if args.executor == "aot":
    executor = Executor("aot", {"link-params": link_params})
else:
    raise ValueError(f"Unsupported executor: {args.executor}")

# This line is necessary for link-params to take effect during
# task extraction and relay.build(...).
mod = mod.with_attr("executor", executor)
# print("mod2", mod)

# trace = _create_schedule(mod, _schedule_matmul).trace
database = ms.database.JSONDatabase(path_workload=str(path_workload), path_tuning_record=str(path_tuning_record))
print("database", database, dir(database))
# token = database.commit_workload(mod)
# workload = database.commit_workload(mod)
# ret = database.get_top_k(token, 2)
# ret = database.get_top_k(workload, 2)
# assert len(ret) == 2
# target = tvm.target.Target("llvm")
target = tvm.target.Target("c")
print("target", target)
record = database.query_tuning_record(mod=mod, target=target, workload_name="main")
print("record", record)
x = database.get_all_tuning_records()
print("x", x)
y = database.has_workload(mod)
print("y", y)
l = len(database)
print("l", l)
# print("x[0]", x[0], dir(x[0]))
# print("x[0].run_secs", x[0].run_secs)
# print("x[0].target", x[0].target)
# print("x[0].trace", x[0].trace)
# print("x[0].workload", x[0].workload, dir(x[0].workload))
# print("x[0].workload.mod", x[0].workload.mod, dir(x[0].workload.mod))
# print("x[1]", x[1], dir(x[1]))
# print("x[1].run_secs", x[1].run_secs)
# print("x[1].target", x[1].target)
# print("x[1].trace", x[1].trace)
# print("x[1].workload", x[1].workload, dir(x[1].workload))
# print("x[1].workload.mod", x[1].workload.mod, dir(x[1].workload.mod))
print("x[2]", x[2], dir(x[2]))
print("x[2].run_secs", x[2].run_secs)
print("x[2].target", x[2].target)
print("x[2].trace", x[2].trace)
print("x[2].workload", x[2].workload, dir(x[2].workload))
print("x[2].workload.mod", x[2].workload.mod, dir(x[2].workload.mod))
workload2 = x[2].workload
# record = database.query_tuning_record(mod=mod, target=target, workload_name="main")
# print("record", record)
x2 = database.get_all_tuning_records()
print("x2", x2)
# y2 = database.has_workload(workload2)
# print("y2", y2)
TOP = 3
topk = database.get_top_k(workload2, TOP)
print("topk", topk)
# workload = database.commit_workload(mod)
# as_json
# as_measure_candidate
# handle

alter_op = True
pass_config = {
    "tir.disable_vectorize": True
}
disabled_pass = []
if not alter_op:
    disabled_pass += ["AlterOpLayout"]
#
# # ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
# #     database=database,
# #     # database=None,
# #     mod=mod,
# #     target=target,
# #     params=params,
# #     pass_config=MappingProxyType(
# #         {
# #             **pass_config,
# #             "relay.backend.use_meta_schedule": True,
# #             "relay.backend.tir_converter": "default",
# #             # "relay.backend.use_meta_schedule_dispatch": 2,
# #             # "tir.disable_vectorize": True,
# #             # "tir.enable_debug": True,
# #         }
# #     ),
# #     disabled_pass=disabled_pass,
# #     executor=executor,
# #     runtime=runtime,
# # )
# # print("ms_mod", ms_mod)
# #
# # with database, tvm.transform.PassContext(
# #     opt_level=3,
# #     config={
# #         **pass_config,
# #         "relay.backend.use_meta_schedule": True,
# #         "relay.backend.use_meta_schedule_dispatch": 2,
# #     },
# # ):
# #     lib = relay.build(mod, target=target, params=params, runtime=runtime, executor=executor)


def eval_single_record(mod, record, target, pass_config, disabled_pass, runtime, executor, platform, options):
    # with tempfile.TemporaryDirectory() as tmpdir:
    # path_workload = osp.join(tmpdir, "workloads.json")
    # path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    # temp_db = ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)
    # mod_eq = "structural"
    # flops = estimate_tir_flops(mod)
    flops_ = estimate_tir_flops(record.workload.mod)
    # print("flops", flops, flops_)
    mem_db = ms.database.MemoryDatabase()
    _ = mem_db.commit_workload(record.workload.mod)
    mem_db.commit_tuning_record(record)
    # TODO: skip db and just use mod from trace/workload?
    ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        database=mem_db,
        mod=mod,
        target=target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule": True,
                "relay.backend.tir_converter": "default",
                # "relay.backend.use_meta_schedule_dispatch": 2,
                # "tir.disable_vectorize": True,
                # "tir.enable_debug": True,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )
    print("ms_mod", ms_mod)

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
            # aot_executor.get_input(0).copyfrom(data_sample)
            # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            result = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result", result)
            print("mean: ", result.mean)
            # output = aot_executor.get_output(0).numpy()
        flops_per_s = flops_ / result.mean
        return result.mean, flops_, flops_per_s


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

res = []
res2 = []
for tk in topk:
    res_, _, res2_ = eval_single_record(mod, tk, target, pass_config, disabled_pass, runtime, executor, platform, project_options)
    res.append(res_)
    res2.append(res2_ / 1e6)

print("topk!", [x.run_secs for x in topk])
print("res", res)
print("res2", res2)
