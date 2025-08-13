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

import logging
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te, tir
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


logging.basicConfig(level=logging.ERROR)
get_logger("xgb_model").setLevel(logging.ERROR)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR / "../../../../"

MS_DISPATCH = 1  # silent?
# MS_DISPATCH = 2  # verbose
# MS_DISPATCH = ?  # error

# def _make_candidate(f_sch: Callable[[], tir.Schedule]) -> ms.MeasureCandidate:
#     return ms.MeasureCandidate(sch=f_sch(None), args_info=[])


def lookup_model_by_name(model):
    def _load_model(path):
        model = tvmc.load(
            str(path)
        )
        mod = model.mod
        params = model.params
        return mod, params

    if True:
        MODELS_DIR = BASE_DIR / "models"

        INPUT_SHAPE_LOOKUP = {
            "resnet": [1, 32, 32, 3],
        }
        DEFAULT_INPUT_SHAPE = [1, 32, 32, 3]
        INPUT_DTYPE_LOOKUP = {
            "resnet": "int8",
        }
        DEFAULT_INPUT_DTYPE = "int8"
        INPUT_NAME_LOOKUP = {
            "resnet": "input",
        }
        DEFAULT_INPUT_NAME = "input"

        model_file = model if ".tflite" in model else f"{model}.tflite"
        model_name = Path(model).stem
        model_path = MODELS_DIR / model_file
        assert model_path.is_file(), f"Model not found: {model_path}"
        mod, params = _load_model(model_path)

        model_name = Path(model).stem
        input_shape = INPUT_SHAPE_LOOKUP.get(model_name, DEFAULT_INPUT_SHAPE)
        input_dtype = INPUT_DTYPE_LOOKUP.get(model_name, DEFAULT_INPUT_DTYPE)
        input_name = INPUT_NAME_LOOKUP.get(model_name, DEFAULT_INPUT_NAME)
    data_sample = np.random.rand(*input_shape).astype(input_dtype)
    return mod, params, input_name, input_shape, input_dtype, data_sample


def get_tuning_config():
    def _get_sch_rules():
        structure = "SR"
        return [
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.InlineConstantScalars(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 2, 4, 8, 16, 32, 64],
                unroll_explicit=True,
                # unroll_explicit=False,
            ),
            ms.schedule_rule.RandomComputeLocation(),
        ]

    def _get_postprocs():
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
        ]

    def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
            # ms.mutator.Parallel(): 0.02,
        }

    sch_rules = _get_sch_rules()
    postprocs = _get_postprocs()
    mutator_probs = _get_mutator_probs()
    return sch_rules, postprocs, mutator_probs


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn


@pytest.mark.parametrize("alter_op", [
    # False,
    True
])
@pytest.mark.parametrize("toolchain", [
    "gcc",
    # "llvm",
])
@pytest.mark.parametrize("target", [
    "c -num-cores 1",
    # "c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc",
])
@pytest.mark.parametrize("num_trials_per_iter,max_trials_per_task,max_trials_global", [
    # (5, 100, 1000000),
    # (1, 1, 1000000),
    # (10, 10, 1000000),
    # (100, 100, 1000000),
    (50, 50, 1000000),
])
@pytest.mark.parametrize("module_equality", ["ignore-ndarray"])
@pytest.mark.parametrize("model", [
    "/work/git/mlonmcu/mlonmcu/workspace_default/models/resnet/resnet.tflite"
])
@pytest.mark.parametrize("transform_layout", [
    False,
    # True,
])
@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule(alter_op, toolchain, target, num_trials_per_iter, max_trials_per_task, max_trials_global, module_equality, model, transform_layout):
    print()
    platform = DIR / "../../../../microtvm-etiss-template/template_project"
    # print("platform", platform)
    options = {
        "verbose": True,
        "quiet": True,
        # "gcc_prefix": str(BASE_DIR / "install/rv32gc_ilp32d"),
        "gcc_prefix": "/work/git/mlonmcu/mlonmcu/workspace_default/deps/install/riscv_gcc_2024.09.03_gcc14",
        "gcc_name": "riscv32-unknown-elf",
        # "llvm_dir": str(BASE_DIR / "install/llvm"),
        "llvm_dir": "/work/git/mlonmcu/mlonmcu/workspace_default/deps/install/llvm",
        "toolchain": toolchain,
        "etiss_script": str(BASE_DIR / "etiss/build/install/bin/run_helper.sh"),
        # "etiss_script": "/work/git/mlonmcu/mlonmcu/workspace_default/deps/install/etiss/bin/run_helper.sh",
        "etiss_args": "",
        "arch": "rv32gc_zicsr_zifencei",
        "abi": "ilp32d",
        "cpu_arch": "RV32IMACFD",
        "cpu_freq": 100000000,
    }
    opt_level = 3
    pass_config = {
        "tir.disable_vectorize": True,
    }
    disabled_pass = []
    if not alter_op:
        disabled_pass += ["AlterOpLayout"]

    KEEP = True
    if KEEP:
        base_dir = Path("/tmp/base2")
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")

        def sanitize(x):
            if not isinstance(x, str):
                x = str(x)
            x = x.replace(" ", "").replace(",", "").replace("/", "").replace(";", "").replace("=", "-").replace("+", "")
            return x
        fields = [target, toolchain, alter_op, num_trials_per_iter, max_trials_per_task, max_trials_global, ts, opt_level, module_equality, model, transform_layout, *[f"no{x}" for x in disabled_pass]]
        label = "-".join([sanitize(x) for x in fields])
        work_dir_path = base_dir / label
    else:
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    print("work_dir_path", work_dir_path)
    mod, params, input_name, input_shape, input_dtype, data_sample = lookup_model_by_name(model)

    if transform_layout:
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config=pass_config,
            disabled_pass=disabled_pass,
        ):
            desired_layouts = {"qnn.conv2d": ["NCHW", "default"]}

            # Convert the layout of the graph where possible.
            seq = transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    SKIP_TUNING = False
    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        if not SKIP_TUNING:
            # print("a1")
            sch_rules, postprocs, mutator_probs = get_tuning_config()
            space = ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs=mutator_probs,
            )
            strategy = "evolutionary"
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            extractor = ms.feature_extractor.PerStoreFeature()
            # (feature,) = extractor.extract_from(
            #     _make_context(tvm.target.Target("llvm")),
            #     candidates=[_make_candidate(_create_schedule)],
            # )
            num_warmup_samples = 10
            cost_model = ms.cost_model.XGBModel(extractor=extractor, num_warmup_samples=num_warmup_samples)
            # micro_rpc_workers = num_trials_per_iter
            with get_rpc_runner_micro(
                platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
                # serial_numbers=["micro"] * micro_rpc_workers,
                # tracker_host="127.0.0.1",
                # tracker_port=9190,
                # max_workers=micro_rpc_workers,
                rpc_timeout_sec=10,

            ) as runner:
                # print("runner", runner)
                # if True:
                if max_trials_global > 0:
                    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
                        extracted_tasks=ms.relay_integration.extract_tasks(
                            mod,
                            target,
                            params,
                            opt_level=opt_level,
                            module_equality=module_equality,
                            pass_config=pass_config,
                            disabled_pass=disabled_pass,
                            # instruments=instruments,
                        ),
                        work_dir=str(work_dir_path),
                        space=space,
                        strategy=strategy,
                        # seed=seed,
                        num_tuning_cores=1,
                    )
                    tasks = tasks[0:2]
                    task_weights = task_weights[0:2]
                    pass_config = dict(pass_config)
                    with transform.PassContext(
                        opt_level=opt_level,
                        config=pass_config,
                        disabled_pass=disabled_pass,
                        # instruments=instruments,
                    ):
                        db: ms.Database = ms.tune.tune_tasks(
                            tasks=tasks,
                            task_weights=task_weights,
                            work_dir=str(work_dir_path),
                            max_trials_global=max_trials_global,
                            max_trials_per_task=max_trials_per_task,
                            num_trials_per_iter=num_trials_per_iter,
                            builder=builder,
                            runner=runner,
                            # database=database,
                            cost_model=cost_model,
                            # measure_callbacks=measure_callbacks,
                            # task_scheduler=task_scheduler,
                            module_equality=module_equality,
                        )
                else:
                    # db = ms.database.MemoryDatabase()
                    db = ms.database.ScheduleFnDatabase(
                        _schedule_dummy()
                    )

            #  Build model using meta_schedule logs
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
                        "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
                        # "tir.enable_debug": True,
                    }
                ),
                disabled_pass=disabled_pass,
                executor=executor,
                runtime=runtime,
            )
    # print("tasks[0]", tasks[0], dir(tasks[0]))
    # print("tasks[0]", tasks[0].mod)
    print(profiler.table())
    print("cost_model", cost_model, dir(cost_model))
    saved_model_path = work_dir_path / "cost_model.tar"
    # random_state = model.extractor.random_state
    cost_model.save(str(saved_model_path))
    cost_model.load(str(saved_model_path))
    cost_model.num_warmup_samples = 1  # Do not get random predictions
    # model.extractor.random_state = random_state
    # candidate = MeasureCandidate(Schedule(FullModule), [])
    dummy_preds = []
    record_preds = []
    for i in range(len(tasks)):
        tune_ctx = tasks[i]
        print("tune_ctx", tune_ctx, dir(tune_ctx))
        sched = tir.Schedule(tune_ctx.mod)
        print("sched", sched)
        # dummy_candidate = _make_candidate(sched)
        dummy_candidate = ms.MeasureCandidate(sch=sched, args_info=[])
        print("dummy_candidate", dummy_candidate)
        (dummy_feature,) = extractor.extract_from(
            tune_ctx,
            candidates=[dummy_candidate],
        )
        print("dummy_feature", dummy_feature, dir(dummy_feature))
        dummy_predictions = cost_model.predict(tune_ctx, [dummy_candidate])
        dummy_preds.append(dummy_predictions[0])
        print("dummy_predictions", dummy_predictions)
        workload = db.commit_workload(tasks[i].mod)
        records = db.get_top_k(workload, 3)
        print("records", records, len(records))
        if len(records) == 0:
            continue
        record = records[0]
        print("record", record, dir(record))
        db.commit_tuning_record(record)
        record_trace = record.trace
        print("record_trace", record_trace, dir(record_trace))
        record_sched = tir.Schedule(record.workload.mod)
        print("record_sched_init", record_sched, dir(record_sched))
        record_trace.apply_to_schedule(record_sched, remove_postproc=False)
        print("record_sched", record_sched, dir(record_sched))
        record_candidate = ms.MeasureCandidate(sch=record_sched, args_info=[])
        print("record_candidate", record_candidate)
        (record_feature,) = extractor.extract_from(
            tune_ctx,
            candidates=[record_candidate],
        )
        print("record_feature", record_feature, dir(record_feature))
        record_predictions = cost_model.predict(tune_ctx, [record_candidate])
        print("record_predictions", record_predictions)
        # assert len(record_predictions) == 1
        record_preds.append(record_predictions[0])
    print("dummy_preds", dummy_preds)
    sorted_dummy_idxs = list(np.argsort(dummy_preds))
    print("sorted_dummy_idxs", sorted_dummy_idxs)
    sorted_dummy_preds = [dummy_preds[i] for i in sorted_dummy_idxs]
    print("sorted_dummy_preds", sorted_dummy_preds)
    dummy_preds_sum = sum(dummy_preds)
    print("dummy_preds_sum", dummy_preds_sum)
    print("record_preds", record_preds)
    sorted_record_idxs = list(np.argsort(record_preds))
    print("sorted_record_idxs", sorted_record_idxs)
    sorted_record_preds = [record_preds[i] for i in sorted_record_idxs]
    print("sorted_record_preds", sorted_record_preds)
    record_preds_sum = sum(record_preds)
    print("record_preds_sum", record_preds_sum)
    # TODO: weighted sum!
    input("!!!")
    non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        None,
        mod=mod,
        target=target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )

    if not SKIP_TUNING:
        # TUNED
        # TODO: wrap in helper
        project = tvm.micro.generate_project(
            str(platform),
            ms_mod,
            str(work_dir_path / "project"),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            result = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result", result)
            print("mean: ", result.mean)

    # UNTUNED
    project = tvm.micro.generate_project(
        str(platform),
        non_ms_mod,
        str(work_dir_path / "project2"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        result2 = aot_executor.module.time_evaluator("run", session.device, number=1)()
        print("result2", result2)
        print("mean2:", result2.mean)
    if not SKIP_TUNING:
        rel = result.mean / result2.mean
        print("rel:  ", rel)


if __name__ == "__main__":
    tvm.testing.main()
