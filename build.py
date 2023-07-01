import os

import numpy as np
import pathlib
import json
import shutil

import tvm
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

MODEL_PATH = "toycar.tflite"

tflite_model_buf = open(MODEL_PATH, "rb").read()

import tflite

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 640)
INPUT_NAME = "input_1"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)

RUNTIME = Runtime("crt", {"system-lib": True})

# TARGET = tvm.micro.testing.get_target("crt")
# TARGET = "llvm -mtriple=riscv32-unknown-elf -mabi=ilp32d -mcpu=generic-rv32 -mattr=+m,+c,+a,+f,+d,+v,+zvl128b -model=spike"
TARGET = "c"
# TARGET = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+c,+a,+f,+d,+v,+zvl128b"

# EXECUTOR = Executor("aot", {"link-params": True})
EXECUTOR = Executor("graph", {"link-params": False})

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
# with tvm.transform.PassContext(opt_level=3):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

template_project_path = pathlib.Path("/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/src/microtvm_spike/template_project/")
project_options = {
    "toolchain": "llvm",
    "llvm_dir": "/var/tmp/ga87puy/ll/llvm-project/install/",
    "gcc_prefix": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/riscv_gcc_vext/",
    "spike_exe": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/spike/spike",
    "spike_pk": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/spikepk/pk",
    "arch": "rv32gcv",
    # "arch": "rv32gc",
    "vlen": 128,
    "elen": 64,
}

# temp_dir = tvm.contrib.utils.tempdir()
cwd = pathlib.Path()
generated_project_dir = pathlib.Path("/var/tmp/ga87puy/tvm_sve/tvm2/project")
if generated_project_dir.is_dir():
    shutil.rmtree(generated_project_dir)
project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

project.build()
project.flash()

with tvm.micro.Session(project.transport()) as session:
    # aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    # aot_executor.run()
    debug_module = tvm.micro.create_local_debug_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )
    # debug_module = tvm.micro.create_local_graph_executor(
    #     module.get_graph_json(), session.get_system_lib(), session.device
    # )
    # debug_module.set_input(**module.get_params())
    print("########## Build with Autotuning ##########")
    debug_module.run()
    del debug_module
