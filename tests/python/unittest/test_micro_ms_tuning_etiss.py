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
from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
from types import MappingProxyType
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
# from tvm.contrib import graph_executor
from tvm.contrib import utils
from tvm import meta_schedule as ms

###
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig

###
# from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

import logging
logging.basicConfig(level=logging.ERROR)


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


@pytest.mark.parametrize("alter_op", [
    # False,
    True
])
@pytest.mark.parametrize("toolchain", [
    "gcc",
    # "llvm",
])
@pytest.mark.parametrize("target", [
    # "c",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc",
    "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc -global-isel=1 -global-isel-abort=2 -basic-block-sections=1",
])
@pytest.mark.parametrize("num_trials_per_iter,max_trials_per_task,max_trials_global", [
    # (0, 0, 0),
    # (1, 1, 3),
    # (5, 10, 3 * 10),
    # (5, 50, 3 * 50),
    # (5, 100, 3 * 100),
    # (5, 200, 3 * 200),
    # (5, 400, 3 * 400),
    # (5, 800, 3 * 800),
    (5, 1600, 3 * 800),
])
@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule(alter_op, toolchain, target, num_trials_per_iter, max_trials_per_task, max_trials_global):
    print()
    # from tests.micro.zephyr.test_ms_tuning import create_relay_module
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    # platform = "crt"
    # platform = "etiss"
    platform = "/work/git/tvmtests/microtvm-etiss-template/template_project"
    # print("platform", platform)
    # target = tvm.target.target.micro(model="host")
    # target = tvm.target.target.cpu("c")
    # target = "llvm -num-cores 1 -mcpu generic-rv32 -mtriple=riscv32-unknown-elf -mabi ilp32d -mattr=+a,+c,+d,+f,+m -model=etiss-rv32gc"
    # target = "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m -model=etiss-rv64gc"
    # target = "c"
    # options = {}
    options = {
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
    opt_level = 3
    pass_config = {
        "tir.disable_vectorize": True
    }
    disabled_pass = []
    if not alter_op:
        disabled_pass += ["AlterOpLayout"]

    KEEP = True
    if KEEP:
        base_dir = Path("/tmp/base")
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")
        def sanitize(x):
            if not isinstance(x, str):
                x = str(x)
            x = x.replace(" ", "").replace(",", "").replace("/", "").replace(";", "").replace("=", "-").replace("+", "")
            return x
        fields = [target, toolchain, alter_op, num_trials_per_iter, max_trials_per_task, max_trials_global, ts, opt_level, *sum(map(list, pass_config.items()), []), *[f"no{x}" for x in disabled_pass]]
        label = "-".join([sanitize(x) for x in fields])
        work_dir_path = base_dir / label
    else:
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    print("work_dir_path", work_dir_path)
    # input("1")
    mod, params, model_info = create_relay_module()
    input_name = model_info["in_tensor"]
    input_shape = model_info["in_shape"]
    input_dtype = model_info["in_dtype"]
    data_sample = np.random.rand(*input_shape).astype(input_dtype)
    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        # print("a1")
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        with get_rpc_runner_micro(
            platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
        ) as runner:
            # print("runner", runner)
            # if True:
            if max_trials_global > 0:
                db: ms.Database = ms.relay_integration.tune_relay(
                    mod=mod,
                    params=params,
                    target=target,
                    builder=builder,
                    runner=runner,
                    strategy="evolutionary",
                    # num_trials_per_iter=2,
                    num_trials_per_iter=num_trials_per_iter,
                    # max_trials_per_task=10,
                    max_trials_per_task=max_trials_per_task,
                    # max_trials_global=100,
                    max_trials_global=max_trials_global,
                    work_dir=str(work_dir_path),
                    module_equality="ignore-ndarray",
                    pass_config=MappingProxyType(
                        pass_config,
                        # {
                        #     "tir.disable_vectorize": True,
                        #     # "tir.enable_debug": True,
                        # }
                    ),
                    disabled_pass=disabled_pass,
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
                    "relay.backend.use_meta_schedule_dispatch": 2,
                    # "tir.disable_vectorize": True,
                    # "tir.enable_debug": True,
                }
            ),
            disabled_pass=disabled_pass,
            executor=executor,
            runtime=runtime,
        )
        non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
            None,
            mod=mod,
            target=target,
            params=params,
            pass_config=MappingProxyType(
                {
                    **pass_config,
                    "relay.backend.use_meta_schedule_dispatch": 2,
                }
            ),
            disabled_pass=disabled_pass,
            executor=executor,
            runtime=runtime,
        )
    print(profiler.table())
    import time
    print("sleeping")
    time.sleep(10)

    # TUNED
    # TODO: wrap in helper
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects(platform)),
        ms_mod,
        str(work_dir_path / "project"),
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

    # UNTUNED
    # TODO: wrap in helper
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects(platform)),
        non_ms_mod,
        str(work_dir_path / "project2"),
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
        print("mean2:", result2.mean)
        # import time
        # time.sleep(100)
        # output = aot_executor.get_output(0).numpy()
    rel = result.mean / result2.mean
    print("rel:  ", rel)

    # Build reference model (without tuning)
    # dev = tvm.cpu()
    # target = tvm.target.target.micro(model="host")
    # with tvm.transform.PassContext(
    #     opt_level=opt_level,
    #     config=pass_config,
    #     disabled_pass=disabled_pass,
    # ):
    #     ref_mod = relay.build(
    #         mod,
    #         target=target,
    #         params=params,
    #         runtime=runtime,
    #     )
    # ref_mod.export_library(work_dir / "compiled_lib2.so")
    # mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    # graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    # graph_mod.set_input(input_name, data_sample)
    # graph_mod.run()
    # ref_output = graph_mod.get_output(0).numpy()

    # assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    # if not KEEP:
    #     work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
