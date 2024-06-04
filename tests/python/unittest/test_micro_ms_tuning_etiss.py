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
import numpy as np
# import pytest
from types import MappingProxyType
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor, utils
from tvm import meta_schedule as ms

###
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig

###
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


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
###


@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule():
    # from tests.micro.zephyr.test_ms_tuning import create_relay_module
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    # platform = "crt"
    # platform = "etiss"
    platform = "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/src/microtvm_etiss/template_project"
    print("platform", platform)
    target = tvm.target.target.micro(model="host")
    # target = tvm.target.target.cpu("c")
    target = "llvm -num-cores 1 -mcpu generic-rv32 -mtriple=riscv32-unknown-elf -mabi ilp32d -mattr=+a,+c,+d,+f,+m -model=etiss-rv32gc"
    # target = "c"
    # options = {}
    options = {
        "verbose": False,
        "quiet": True,
        "gcc_prefix": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/riscv_gcc",
        "gcc_name": "riscv32-unknown-elf",
        "llvm_dir": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/llvm",
        "toolchain": "gcc",
        "etiss_script": "/var/tmp/ga87puy/microtvm/mlonmcu/workspace_default/deps/install/etiss/bin/run_helper.sh",
        "etiss_args": "",
        "arch": "rv32gc_zicsr_zifencei",
        "abi": "ilp32d",
        "cpu_arch": "RV32IMACFD",
        "cpu_freq": 100000000,
    }

    work_dir = utils.tempdir()
    print("work_dir", work_dir.path)
    # input("1")
    mod, params, model_info = create_relay_module()
    input_name = model_info["in_tensor"]
    input_shape = model_info["in_shape"]
    input_dtype = model_info["in_dtype"]
    data_sample = np.random.rand(*input_shape).astype(input_dtype)

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        with get_rpc_runner_micro(
            platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
        ) as runner:
            print("runner", runner)
            # if True:
            if False:
                db: ms.Database = ms.relay_integration.tune_relay(
                    mod=mod,
                    params=params,
                    target=target,
                    builder=builder,
                    runner=runner,
                    strategy="evolutionary",
                    # num_trials_per_iter=2,
                    num_trials_per_iter=1,
                    # max_trials_per_task=10,
                    max_trials_per_task=1,
                    # max_trials_global=100,
                    max_trials_global=3,
                    work_dir=str(work_dir.path),
                    module_equality="ignore-ndarray",
                    pass_config=MappingProxyType(
                        {
                            "tir.disable_vectorize": True,
                            "tir.enable_debug": True,
                        }
                    ),
                )
            else:
                tune_tasks = ms.relay_integration.extract_tasks(
                    mod,
                    params=params,
                    target=target,
                    # new!
                    module_equality="ignore-ndarray",
                    pass_config=MappingProxyType(
                        {
                            "tir.disable_vectorize": True,
                            "tir.enable_debug": True,
                        }
                    ),
                )
                intrin = VNNI_INTRIN
                postprocs = [
                    ms.postproc.DisallowDynamicLoop(),
                    ms.postproc.RewriteParallelVectorizeUnroll(),
                    ms.postproc.RewriteReductionBlock(),
                    ms.postproc.RewriteTensorize(vectorize_init_loop=True),
                ]
                sch_rules = [
                    ms.schedule_rule.ApplyCustomRule(),
                    ms.schedule_rule.AutoInline(
                        into_producer=False,
                        into_consumer=True,
                        inline_const_tensor=True,
                        disallow_if_then_else=True,
                        require_injective=True,
                        require_ordered=True,
                        disallow_op=["tir.exp"],
                    ),
                    ms.schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
                    ms.schedule_rule.MultiLevelTilingWithIntrin(
                        intrin,
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
                        max_jobs_per_core=16,
                        max_vectorize_extent=64,
                        unroll_max_steps=[0, 16, 64, 512],
                        unroll_explicit=True,
                    ),
                    ms.schedule_rule.RandomComputeLocation(),
                ]
                tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
                    extracted_tasks=tune_tasks,
                    work_dir=str(work_dir.path),
                    space=ms.space_generator.PostOrderApply(
                        sch_rules=sch_rules,
                        postprocs=postprocs,
                    ),
                )
                db: ms.Database = ms.tune.tune_tasks(
                    tasks=tasks,
                    task_weights=task_weights,
                    work_dir=str(work_dir.path),
                    max_trials_global=32,
                )

        import time
        time.sleep(60)
        #  Build model using meta_schedule logs
        ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
            database=db,
            mod=mod,
            target=target,
            params=params,
            pass_config=MappingProxyType(
                {
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.tir_converter": "default",
                    "tir.disable_vectorize": True,
                    # "tir.enable_debug": True,
                }
            ),
            executor=executor,
            runtime=runtime,
        )
    print(profiler.table())

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
        aot_executor.get_input(0).copyfrom(data_sample)
        result = aot_executor.module.time_evaluator("run", session.device, number=3)()
        print("result", result)
        output = aot_executor.get_output(0).numpy()

    # Build reference model (without tuning)
    dev = tvm.cpu()
    target = tvm.target.target.micro(model="host")
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
    ):
        ref_mod = relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
        )
    ref_mod.export_library(work_dir / "compiled_lib2.so")
    mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    graph_mod.set_input(input_name, data_sample)
    graph_mod.run()
    ref_output = graph_mod.get_output(0).numpy()

    assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    # work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
