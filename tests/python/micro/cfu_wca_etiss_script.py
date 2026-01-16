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
from tvm.tir.tensor_intrin.cfu import CFU_32X_INTRIN, CFU_24X_INTRIN, CFU_16X_INTRIN, CFU_8X_INTRIN
from tvm.tir.tensor_intrin.cfu import CFU_40X_INTRIN, CFU_48X_INTRIN, CFU_56X_INTRIN, CFU_64X_INTRIN
from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro
from tvm.contrib.micro.cfu.wca import CompressWeights, ImportCPostprocess


logging.basicConfig(level=logging.ERROR)
get_logger("xgb_model").setLevel(logging.ERROR)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR / "../../../../"
# BASE_DIR = DIR / "../../../../../"
print("BASE_DIR", BASE_DIR.resolve())
BASE_OUT_DIR = Path(os.environ.get("BASE_OUT_DIR", "/tmp/base"))
print("BASE_DIR_DIR", BASE_OUT_DIR.resolve())
# input("!")

MS_DISPATCH = 1  # silent?
# MS_DISPATCH = 2  # verbose
# MS_DISPATCH = ?  # error


def lookup_model_by_name(model):
    def _load_model(path):
        model = tvmc.load(str(path))
        mod = model.mod
        params = model.params
        return mod, params

    if model == "default":
        # input("1")
        mod, params, model_info = create_relay_module()
        input_name = model_info["in_tensor"]
        input_shape = model_info["in_shape"]
        input_dtype = model_info["in_dtype"]
    else:
        MODELS_DIR = (BASE_DIR / "models").resolve()

        INPUT_SHAPE_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": [1, 32, 32, 3],
            "pretrainedResnet_clustered_quant_remap_packed": [1, 32, 32, 3],
        }
        DEFAULT_INPUT_SHAPE = [1, 32, 32, 3]
        INPUT_DTYPE_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": "int8",
            "pretrainedResnet_clustered_quant_remap_packed": "int8",
        }
        DEFAULT_INPUT_DTYPE = "int8"
        INPUT_NAME_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": "input",
            "pretrainedResnet_clustered_quant_remap_packed": "input",
        }
        DEFAULT_INPUT_NAME = "input"

        model_file = model if ".tflite" in model else f"{model}.tflite"
        model_name = Path(model).stem
        model_path = MODELS_DIR / model_file
        assert model_path.is_file(), f"Model not found: {model_path}"
        mod, params = _load_model(model_path)

        input_shape = INPUT_SHAPE_LOOKUP.get(model_name, DEFAULT_INPUT_SHAPE)
        input_dtype = INPUT_DTYPE_LOOKUP.get(model_name, DEFAULT_INPUT_DTYPE)
        input_name = INPUT_NAME_LOOKUP.get(model_name, DEFAULT_INPUT_NAME)
    data_sample = np.random.rand(*input_shape).astype(input_dtype)
    return mod, params, input_name, input_shape, input_dtype, data_sample


def get_tuning_config(
    enable_intrin: bool = False,
    num_clusters: Optional[int] = None,
    cfu_mode: Optional[str] = None,
    channel_count: Optional[int] = None,
):
    # print("get_tuning_config", enable_intrin, num_clusters, cfu_mode, channel_count)
    if num_clusters is not None:
        assert channel_count is not None
        from math import log2

        max_channels = 64 // int(log2(num_clusters))
        channel_count = min(max_channels, channel_count)
    # print("channel_count", channel_count)

    # def _get_sch_rules(intrin: Optional[str] = None, num_clusters: Optional[int] = None, channel_count: Optional[int] = None):
    def _get_sch_rules(
        intrin: Optional[str] = None, num_clusters: Optional[int] = None, channel_count: Optional[int] = None
    ):
        # print("_get_sch_rules", intrin, num_clusters, channel_count)
        # init_intrin = DP4A_S8S8S32_INIT_INTRIN
        # structure_lookup = {
        #     AMDGPU_SDOT4_INTRIN: "SSSRRSRS",
        #     VRMPY_i8i8i32_INTRIN: "SRSRS",
        #     DP4A_S8S8S32_INTRIN: "SR",
        #     # DP4A_S8S8S32_INIT_INTRIN: "SR",
        #     # ARM_DOT_4x4_i8_NEON_INTRIN: "SR",
        #     ARM_DOT_4x4_i8_NEON_INTRIN: "RS",
        # }
        if intrin is None:
            intrins = []
        elif intrin == "all":
            intrins = [
                CFU_64X_INTRIN,
                CFU_56X_INTRIN,
                CFU_48X_INTRIN,
                CFU_40X_INTRIN,
                CFU_32X_INTRIN,
                CFU_24X_INTRIN,
                CFU_16X_INTRIN,
                CFU_8X_INTRIN,
            ]
        elif intrin == "auto":
            assert channel_count is not None

            intrin_lookup = {
                # 32: DP4A_S8S8S32_INTRIN,
                64: CFU_64X_INTRIN,
                56: CFU_56X_INTRIN,
                48: CFU_48X_INTRIN,
                40: CFU_40X_INTRIN,
                32: CFU_32X_INTRIN,
                24: CFU_24X_INTRIN,
                16: CFU_16X_INTRIN,
                8: CFU_8X_INTRIN,
            }
            intrin = intrin_lookup.get(channel_count)
            assert intrin is not None, f"Could not determine intrin for channel_count: {channel_count}"
            intrins = [intrin]
        else:
            intrins = [intrin]

        structure = "SR"
        # print("intrin", intrin)
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
            # ms.schedule_rule.AddRFactor(max_jobs_per_core=1, max_innermost_factor=64),
            *(
                [
                    ms.schedule_rule.MultiLevelTilingWithIntrin(
                        intrin,
                        # structure=structure_lookup[intrin],
                        structure=structure,
                        # tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                        # max_innermost_factor=32,
                        # vector_load_lens=[1, 2, 3, 4],
                        # reuse_read=ms.schedule_rule.ReuseType(
                        #     req="must",
                        #     levels=[4],
                        #     scope="shared",
                        # ),
                        # reuse_write=ms.schedule_rule.ReuseType(
                        #     req="must",
                        #     levels=[3],
                        #     scope="local",
                        # ),
                    )
                    for intrin in intrins
                ]
            ),
            # *([ms.schedule_rule.MultiLevelTilingWithIntrin(
            #         init_intrin,
            #         structure=structure_lookup[init_intrin],
            #         # tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            #         # max_innermost_factor=32,
            #         # vector_load_lens=[1, 2, 3, 4],
            #         # reuse_read=ms.schedule_rule.ReuseType(
            #         #     req="must",
            #         #     levels=[4],
            #         #     scope="shared",
            #         # ),
            #         # reuse_write=ms.schedule_rule.ReuseType(
            #         #     req="must",
            #         #     levels=[3],
            #         #     scope="local",
            #         # ),
            #     )] if init_intrin is not None else []),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
                # structure="SSRSRS",
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

    # def _get_postprocs(num_clusters: Optional[int] = None, cfu_mode: Optional[str] = None, channel_count: Optional[int] = None):
    def _get_postprocs(cfu_mode: Optional[str] = None):
        # print("_get_postprocs", num_clusters, cfu_mode, channel_count)
        # print("_get_postprocs", cfu_mode)
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            # *([ImportCPostprocess(num_clusters, cfu_mode, channel_count)] if enable_intrin and num_clusters is not None else []),
            *([ImportCPostprocess(cfu_mode)] if enable_intrin else []),
            ms.postproc.RewriteTensorize(),
            # *([ImportC2Postprocess(num_clusters, cfu_mode, channel_count)] if enable_intrin and num_clusters is not None else []),
            # ms.postproc.RewriteTensorize(vectorize_init_loop=True),
        ]

    def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
            # ms.mutator.Parallel(): 0.02,
        }

    # default_intrin = DP4A_S8S8S32_INTRIN
    # default_intrin = "auto"
    default_intrin = "all" if channel_count is None else "auto"
    intrin = default_intrin if enable_intrin else None
    sch_rules = _get_sch_rules(intrin, num_clusters, channel_count)
    # sch_rules = _get_sch_rules(intrin)
    # postprocs = _get_postprocs(num_clusters, cfu_mode, channel_count)
    postprocs = _get_postprocs(cfu_mode)
    mutator_probs = _get_mutator_probs()
    # input(">>>")
    return sch_rules, postprocs, mutator_probs


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn


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
    weight_sample = np.random.rand(weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info


###

# def CompressWeights():
#     def _transform(func, mod, ctx):
#         print("CompressWeights")
#         print("func", func)
#         print("mod", mod)
#         print("ctx", ctx)
#         input("@A")
#         def stmt_post(stmt):
#             print("stmt_post", stmt)
#             return stmt
#
#         new_body = tvm.tir.stmt_functor.ir_transform(
#             func.body,
#             None,
#             stmt_post,
#             ["tir.Evaluate", "tir.Call"],
#         )
#         print("new_body", new_body)
#         input("@B")
#         return func.with_body(new_body)
#     return tvm.tir.transform.prim_func_pass(_transform, opt_level=0, name="CompressWeights")


def run_micro_tuning_with_meta_schedule(
    alter_op,
    toolchain,
    target,
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
    ms_db=None,
    out_dir=None,
):
    print()
    platform = BASE_DIR / "microtvm-etiss-template/template_project"
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
            target,
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

    mod, params, input_name, input_shape, input_dtype, data_sample = lookup_model_by_name(model)

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
                sch_rules, postprocs, mutator_probs = get_tuning_config(
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
                session_timeout_sec=120,
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
                        target=target,
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
    print(profiler.table())
    # import time
    # print("sleeping")
    # time.sleep(10)
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
    parser.add_argument("--target", type=str, default="c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1")
    parser.add_argument("--toolchain", type=str, default="gcc")

    # === Tuning parameters ===
    parser.add_argument("--num-trials-per-iter", type=int, default=1)
    parser.add_argument("--max-trials-per-task", type=int, default=1)
    parser.add_argument("--max-trials-global", type=int, default=1_000_000)
    parser.add_argument("--skip-tuning", action="store_true", default=False)
    parser.add_argument("--skip-bench", action="store_true", default=False)
    parser.add_argument("--ms-db", default=None)

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

    # === Misc ===
    parser.add_argument("--module-equality", type=str, default="ignore-ndarray")
    parser.add_argument("-o", "--output", type=str, default=None)

    # === MicroTVM flags ===
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.add_argument("--arch", type=str, default="rv32gc_zicsr_zifencei")
    parser.add_argument("--abi", type=str, default="ilp32d")
    parser.add_argument("--etiss-arch", type=str, default="RV32IMACFDXCFU0")

    return parser.parse_args()


def main():
    args = parse_args()

    options = {
        "verbose": args.verbose,
        "quiet": not args.verbose,
        # "quiet": False,
        "gcc_prefix": str(BASE_DIR / "install/rv32gc_ilp32d"),
        "gcc_name": "riscv32-unknown-elf",
        "llvm_dir": str(BASE_DIR / "install/seal5_llvm"),
        "toolchain": args.toolchain,
        "etiss_script": str(BASE_DIR / "etiss/build/install/bin/run_helper.sh"),
        "etiss_args": "",
        "arch": args.arch,
        "abi": args.abi,
        # "cpu_arch": "RV32IMACFD",
        "cpu_arch": args.etiss_arch,
        "cpu_freq": 100000000,
    }

    run_micro_tuning_with_meta_schedule(
        alter_op=args.alter_op,
        toolchain=args.toolchain,
        target=args.target,
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
        options=options,
        out_dir=args.output,
    )


if __name__ == "__main__":
    main()
