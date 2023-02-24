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
"""Test end-to-end network tuning with auto-scheduler"""
import pytest
import tempfile
import pathlib

import numpy as np

from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor, utils
from tvm.relay.backend import Runtime, Executor
import tvm.testing

from test_auto_scheduler_task_extraction import get_network


def tune_network(network, target, dev):
    # Extract tasks
    mod, params = get_network(network)
    target = tvm.target.Target(target)

    build_func = tvm.micro.autotvm_build_func
    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)
    template_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
    options = {"workspace_size_bytes": 16 * 1024 * 1024}

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=template_dir,
        project_options=options,
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}) as ctx:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        # Tuning
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(timeout=60, device=0, module_loader=module_loader)
        callbacks = [
            auto_scheduler.task_scheduler.PrintTableInfo(),
            auto_scheduler.task_scheduler.LogEstimatedLatency(("total_latency.tsv")),
        ]
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, callbacks=callbacks)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=10,
            num_measures_per_round=2,
            early_stopping=1,
            runner=measure_ctx.runner,
            builder = auto_scheduler.LocalBuilder(
                build_func=build_func,
                runtime=runtime,
            ),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            si_prefix="M",
            verbose=True,
        )
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}) as ctx:
            tuner.tune(tune_option, search_policy="sketch.random")
            assert tuner.best_score is not None and tuner.best_score < 1e9, "Tuning failed"
        del measure_ctx

        # Compile with the history best
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params, runtime=runtime, executor=executor)

        np.random.seed(0)
        if network == "mlp":
            input_name = "data"
            data = np.random.uniform(size=(1, 32))
        elif network == "winograd-test":
            input_name = "data"
            data = np.random.uniform(size=(1, 23, 40, 32))
        else:
            raise ValueError("Unknown network: " + network)

        work_dir = utils.tempdir()
        project = tvm.micro.generate_project(
            template_dir,
            lib,
            str(work_dir / "project"),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            aot_executor.get_input(0).copyfrom(data)
            result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            output = aot_executor.get_output(0).numpy()

        # Build reference model (without tuning + optimizations)
        dev = tvm.cpu()
        target = tvm.target.target.micro(model="host")
        with tvm.transform.PassContext(
            opt_level=0, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
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
        graph_mod.set_input(input_name, data)
        graph_mod.run()
        ref_output = graph_mod.get_output(0).numpy()

        assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
        work_dir.remove()


@pytest.mark.parametrize("network", ["mlp", "winograd-test"])
def test_tuning_micro(network):
    tune_network(network, "c", tvm.cpu())  # TODO: llvm?


if __name__ == "__main__":
    test_tuning_micro()
