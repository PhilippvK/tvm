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

import os
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import pandas as pd
import numpy as np

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib import utils
from tvm import meta_schedule as ms
from tvm.driver import tvmc
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig
from tvm.meta_schedule.logging import get_logger
from tvm import transform
from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro


from tvm.contrib.micro.cfu.wca import CompressWeights, ImportCPostprocess, get_wca_tuning_config
from tvm.contrib.micro.cfu.model_utils import lookup_model_by_name
from tvm.contrib.micro.cfu.tuning_utils import _schedule_dummy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


logging.basicConfig(level=logging.ERROR)
get_logger("xgb_model").setLevel(logging.ERROR)

DIR = Path(__file__).parent.resolve()
BASE_DIR = Path(os.environ.get("BASE_DIR", DIR / "../../../../"))
print("BASE_DIR", BASE_DIR.resolve())
BASE_OUT_DIR = Path(os.environ.get("BASE_OUT_DIR", "/tmp/base"))
print("BASE_DIR_DIR", BASE_OUT_DIR.resolve())
ETISS_INSTALL_DIR = Path(os.environ.get("ETISS_INSTALL_DIR", BASE_DIR / "etiss/build/install/bin/run_helper.sh"))
LLVM_INSTALL_DIR = Path(os.environ.get("LLVM_INSTALL_DIR", BASE_DIR / "install/llvm"))
# SEAL5_LLVM_INSTALL_DIR = os.environ.get(str(BASE_DIR / "install/seal5_llvm"),
GNU_PREFIX = Path(os.environ.get("GNU_PREFIX", BASE_DIR / "install/rv32gc_ilp32d"))
GNU_NAME = os.environ.get("GNU_NAME", "riscv32-unknown-elf")
GNU_MULTILIB_PREFIX = Path(os.environ.get("GNU_MULTILIB_PREFIX", BASE_DIR / "install/multilib_old"))


def run_micro_tuning_with_meta_schedule(
    alter_op,
    toolchain,
    tvm_target,
    template,
    num_trials_per_iter,
    max_trials_per_task,
    max_trials_global,
    skip_tuning,
    skip_bench,
    enable_custom,
    enable_intrin,
    cfu_mode,
    module_equality,
    model,
    num_clusters,
    channel_count,
    transform_layout,
    options,
    ms_dispatch,
    ms_db=None,
    out_dir=None,
):
    print()
    platform = BASE_DIR / f"microtvm-{template}-template/template_project"
    # print("platform", platform)
    opt_level = 3
    pass_config = {
        "tir.disable_vectorize": True,
        "tir.add_lower_pass": [(3, CompressWeights())],
    }
    disabled_pass = []
    if not alter_op:
        disabled_pass += ["AlterOpLayout"]

    if out_dir is None:
        base_dir = BASE_OUT_DIR
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")

        def sanitize(x):
            if not isinstance(x, str):
                x = str(x)
            x = x.replace(" ", "").replace(",", "").replace("/", "").replace(";", "").replace("=", "-").replace("+", "")
            return x

        fields = [
            tvm_target,
            toolchain,
            alter_op,
            num_trials_per_iter,
            max_trials_per_task,
            max_trials_global,
            ts,
            opt_level,
            enable_custom,
            enable_intrin,
            cfu_mode,
            module_equality,
            model,
            num_clusters,
            channel_count,
            transform_layout,
            *[f"no{x}" for x in disabled_pass],
        ]
        label = "-".join([sanitize(x) for x in fields])
        work_dir_path = base_dir / label
    elif out_dir == "TEMP":
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    else:
        work_dir_path = Path(out_dir).resolve()
        work_dir_path.mkdir(exist_ok=True, parents=True)
        # assert work_dir_path.is_dir()
    print("work_dir_path", work_dir_path)

    mod, params, input_name, input_shape, input_dtype, data_sample = lookup_model_by_name(model, base_dir=BASE_DIR)

    if transform_layout:
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config=pass_config,
            disabled_pass=disabled_pass,
        ):
            desired_layouts = {"qnn.conv2d": ["NHWC", "HWOI"]}
            # desired_layouts = {"qnn.conv2d": ["NHWC", "OHWI"]}

            # Convert the layout of the graph where possible.
            seq = transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

    # print("model.mod", model.mod)
    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        if not skip_tuning:
            # print("a1")
            if enable_custom:
                sch_rules, postprocs, mutator_probs = get_wca_tuning_config(
                    enable_intrin, num_clusters, cfu_mode, channel_count
                )
                space = ms.space_generator.PostOrderApply(
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs=mutator_probs,
                )
                strategy = "evolutionary"
            else:
                space = "post-order-apply"
                strategy = "evolutionary"
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            micro_rpc_workers = num_trials_per_iter
            with get_rpc_runner_micro(
                platform=platform,
                options=options,
                # session_timeout_sec=120,
                session_timeout_sec=120*2,
                evaluator_config=evaluator_config,
                # serial_numbers=["server:micro", "server:micro"],
                # serial_numbers=["server:micro", "server:micro"] * 5,
                serial_numbers=["micro"] * micro_rpc_workers,
                tracker_host="127.0.0.1",
                tracker_port=9190,
                max_workers=micro_rpc_workers,
                rpc_timeout_sec=10,
            ) as runner:
                # print("runner", runner)
                # if True:
                if max_trials_global > 0:
                    if ms_db is not None:
                        db_path = Path(ms_db)
                        print(f"Using existing MS database for tuning: {db_path}")
                        assert db_path.is_dir(), f"Not found: {db_path}"
                        path_workload = db_path / "database_workload.json"
                        assert path_workload.is_file(), f"Not found: {path_workload}"
                        path_tuning_record = db_path / "database_tuning_record.json"
                        assert path_tuning_record.is_file(), f"Not found: {path_tuning_record}"
                        ms_db = ms.database.JSONDatabase(
                            path_workload=str(path_workload), path_tuning_record=str(path_tuning_record)
                        )
                    db: ms.Database = ms.relay_integration.tune_relay(
                        mod=mod,
                        params=params,
                        target=tvm_target,
                        builder=builder,
                        runner=runner,
                        strategy=strategy,
                        space=space,
                        # num_trials_per_iter=2,
                        num_trials_per_iter=num_trials_per_iter,
                        # max_trials_per_task=10,
                        max_trials_per_task=max_trials_per_task,
                        # max_trials_global=100,
                        max_trials_global=max_trials_global,
                        work_dir=str(work_dir_path),
                        module_equality=module_equality,
                        pass_config=MappingProxyType(
                            pass_config,
                            # {
                            #     "tir.disable_vectorize": True,
                            #     # "tir.enable_debug": True,
                            # }
                        ),
                        disabled_pass=disabled_pass,
                        # num_tuning_cores=1,
                    )
                else:
                    # db = ms.database.MemoryDatabase()
                    db = ms.database.ScheduleFnDatabase(_schedule_dummy())

            #  Build model using meta_schedule logs
            ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
                database=db,
                mod=mod,
                target=tvm_target,
                params=params,
                pass_config=MappingProxyType(
                    {
                        **pass_config,
                        "relay.backend.use_meta_schedule": True,
                        "relay.backend.tir_converter": "default",
                        "relay.backend.use_meta_schedule_dispatch": ms_dispatch,
                        # "tir.enable_debug": True,
                    }
                ),
                disabled_pass=disabled_pass,
                executor=executor,
                runtime=runtime,
            )
        elif ms_db is not None:
            db_path = Path(ms_db)
            print(f"Loading existing MS database from disk: {db_path}")
            assert db_path.is_dir(), f"Not found: {db_path}"
            path_workload = db_path / "database_workload.json"
            assert path_workload.is_file(), f"Not found: {path_workload}"
            path_tuning_record = db_path / "database_tuning_record.json"
            assert path_tuning_record.is_file(), f"Not found: {path_tuning_record}"
            db = ms.database.JSONDatabase(path_workload=str(path_workload), path_tuning_record=str(path_tuning_record))
            ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
                database=db,
                mod=mod,
                target=tvm_target,
                params=params,
                pass_config=MappingProxyType(
                    {
                        **pass_config,
                        "relay.backend.use_meta_schedule": True,
                        "relay.backend.tir_converter": "default",
                        "relay.backend.use_meta_schedule_dispatch": ms_dispatch,
                        # "tir.enable_debug": True,
                    }
                ),
                disabled_pass=disabled_pass,
                executor=executor,
                runtime=runtime,
            )
    print(profiler.table())
    # import time
    # print("sleeping")
    # time.sleep(10)
    non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        None,
        mod=mod,
        target=tvm_target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule_dispatch": ms_dispatch,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )

    force = True

    if not skip_bench:
        metrics = []

        if not skip_tuning or ms_db is not None:
            # TUNED
            # TODO: wrap in helper
            prj_dir = work_dir_path / "project"
            if prj_dir.is_dir() and force:
                print("Removing old project dir:", prj_dir)
                shutil.rmtree(prj_dir)
            project = tvm.micro.generate_project(
                # str(tvm.micro.get_microtvm_template_projects(platform)),
                str(platform),
                ms_mod,
                str(prj_dir),
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
                print("result.results", result.results)
                print("mean: ", result.mean)
                metrics.append({"mode": "tuned", "mean": result.mean})
                # output = aot_executor.get_output(0).numpy()

        # UNTUNED
        # TODO: wrap in helper
        prj_dir = work_dir_path / "project2"
        if prj_dir.is_dir() and force:
            print("Removing old project dir:", prj_dir)
            shutil.rmtree(prj_dir)
        project = tvm.micro.generate_project(
            # str(tvm.micro.get_microtvm_template_projects(platform)),
            str(platform),
            non_ms_mod,
            str(prj_dir),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            # aot_executor.get_input(0).copyfrom(data_sample)
            # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            result2 = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result2", result2)
            print("result2.results", result2.results)
            print("mean2:", result2.mean)
            metrics.append({"mode": "untuned", "mean": result2.mean})
            # output = aot_executor.get_output(0).numpy()
        if not skip_tuning or ms_db is not None:
            rel = result.mean / result2.mean
            print("rel:  ", rel)
            metrics.append({"mode": "REL", "mean": metrics[0]["mean"] / metrics[1]["mean"]})
        metrics_df = pd.DataFrame(metrics)
        metrics_file = work_dir_path / "metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        metrics_df.set_index("mode")
        print("Metrics:")
        print(metrics_df)


def parse_args():
    parser = argparse.ArgumentParser(description="MicroTVM + MetaSchedule CFU tuning runner")

    # === Core execution ===
    parser.add_argument("--model", type=str, default="new/pretrainedResnet_clustered_quant_remap")
    parser.add_argument("--tvm-target", type=str, default="c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1")
    parser.add_argument("--toolchain", type=str, choices=["gcc", "llvm"], default="gcc")
    parser.add_argument("--template", type=str, choices=["etiss", "cfu"], default="etiss")

    # === Tuning parameters ===
    parser.add_argument("--num-trials-per-iter", type=int, default=1)
    parser.add_argument("--max-trials-per-task", type=int, default=1)
    parser.add_argument("--max-trials-global", type=int, default=1_000_000)
    parser.add_argument("--skip-tuning", action="store_true", default=False)
    parser.add_argument("--skip-bench", action="store_true", default=False)
    parser.add_argument("--ms-db", default=None)
    parser.add_argument("--ms-dispatch", default=1, choices=[0, 1, 2, 4, 6], help="Metascheduler dispatch verbosity")
    # (dispatch & 2): controls whether to print TVMScript for missing TIR
    # (dispatch & 4): controls whether to raise fatal errors for missing TIR

    # === Feature flags ===
    parser.add_argument("--alter-op", action="store_true", default=True)
    parser.add_argument("--no-alter-op", dest="alter_op", action="store_false")
    parser.add_argument("--enable-custom", action="store_true", default=False)
    parser.add_argument("--enable-intrin", action="store_true", default=False)
    parser.add_argument("--transform-layout", action="store_true", default=True)
    parser.add_argument("--no-transform-layout", dest="transform_layout", action="store_false")

    # === CFU specific ===
    parser.add_argument("--cfu-mode", type=str, default=None, choices=[None, "MODE_EMUL", "MODE_CFU"])
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--channel-count", type=int, default=None)
    parser.add_argument("--cfu-verilog", type=str, default=None)

    # === Misc ===
    parser.add_argument("--module-equality", type=str, default="ignore-ndarray")
    parser.add_argument("-o", "--output", type=str, default=None)

    # === MicroTVM flags ===
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--arch", type=str, default="rv32im_zicsr_zifencei")
    parser.add_argument("--abi", type=str, default="ilp32")
    parser.add_argument("--etiss-arch", type=str, default="RV32IMACFDXCFU0")

    return parser.parse_args()


def main():
    args = parse_args()

    common_options = {
        "verbose": args.verbose,
        "quiet": not args.verbose,
        # "quiet": False,
    }
    if args.template == "etiss":
        options = {
            **common_options,
            "toolchain": args.toolchain,
            "etiss_script": str(ETISS_INSTALL_DIR / "bin" / "run_helper.sh"),
            "etiss_args": "",
            "arch": args.arch,
            "abi": args.abi,
            # "cpu_arch": "RV32IMACFD",
            "cpu_arch": args.etiss_arch,
            "cpu_freq": 100000000,
            "gcc_prefix": str(GNU_PREFIX),
            "gcc_name": GNU_NAME,
            "llvm_dir": str(LLVM_INSTALL_DIR),
        }
    elif args.template == "cfu":
        assert args.toolchain == "gcc"
        options = {
            **common_options,
            # "cfu_root": "?",
            # "verilator_install_dir": "?",
            "gcc_prefix": str(GNU_MULTILIB_PREFIX),
            "debug": True,
        }
        if args.cfu_verilog is not None:
            options["verilog_file"] = args.cfu_verilog
    else:
        raise ValueError(f"Unsupported template: {args.template}")

    run_micro_tuning_with_meta_schedule(
        alter_op=args.alter_op,
        toolchain=args.toolchain,
        tvm_target=args.tvm_target,
        template=args.template,
        num_trials_per_iter=args.num_trials_per_iter,
        max_trials_per_task=args.max_trials_per_task,
        max_trials_global=args.max_trials_global,
        skip_tuning=args.skip_tuning,
        skip_bench=args.skip_bench,
        ms_db=args.ms_db,
        enable_custom=args.enable_custom,
        enable_intrin=args.enable_intrin,
        cfu_mode=args.cfu_mode,
        module_equality=args.module_equality,
        model=args.model,
        num_clusters=args.num_clusters,
        channel_count=args.channel_count,
        transform_layout=args.transform_layout,
        ms_dispatch=args.ms_dispatch,
        options=options,
        out_dir=args.output,
    )


if __name__ == "__main__":
    main()
