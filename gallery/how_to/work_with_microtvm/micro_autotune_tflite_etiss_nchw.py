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

"""
.. _tutorial-micro-autotune:

6. Model Tuning with microTVM
=============================
**Authors**:
`Andrew Reusch <https://github.com/areusch>`_,
`Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial explains how to autotune a model using the C runtime.
"""
# import sys
# import logging
# logging.basicConfig(level="DEBUG", stream=sys.stdout)


######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_dependencies.rst
#


# You can skip the following section (installing Zephyr) if the following flag is False.
# Installing Zephyr takes ~20 min.
import sys
import os

assert len(sys.argv) == 4, "a arg telling the location of the model is needed."
model_path = sys.argv[1]
out_file = sys.argv[2]
xcorev = bool(int(sys.argv[3]))

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_zephyr.rst
#


######################################################################
# Import Python dependencies
# -------------------------------
#
import json
import numpy as np
import pathlib

import tvm
from tvm.relay.backend import Runtime
import tvm.micro.testing

####################
# Defining the model
####################
####################
# Defining the model
####################
from tflite.TensorType import TensorType as TType
from tvm import relay

class TensorInfo:
    def __init__(self, t):
        self.name = t.Name().decode()

        typeLookup = {
            TType.FLOAT32: (4, "float32"),
            TType.UINT8: (1, "uint8"),
            TType.INT8: (1, "int8")
        }
        self.tysz, self.ty = typeLookup[t.Type()]
        assert self.ty != ""

        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])
        self.shape = shape

        self.size = self.tysz
        for dimSz in self.shape:
            self.size *= dimSz


class ModelInfo:
    def __init__(self, model):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        self.inTensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            self.inTensors.append(TensorInfo(t))

        self.outTensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            self.outTensors.append(TensorInfo(t))


print("### TVMFlow.loadModel")

MODEL = model_path

import os
modelBuf = open(model_path, "rb").read()

import tflite
tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)

shapes = {}
types = {}

modelInfo = ModelInfo(tflModel)
for t in modelInfo.inTensors:
    print("Input", '"' + t.name + '"', t.ty, t.shape)
    shapes[t.name] = t.shape
    types[t.name] = t.ty

relay_mod, params = relay.frontend.from_tflite(tflModel, shape_dict=shapes, dtype_dict=types)


print("12")
desired_layouts = {
    "qnn.conv2d": ["NCHW", "default"],
    "nn.conv2d": ["NCHW", "default"],
    "nn.max_pool2d": ["NCHW", "default"],
    "nn.avg_pool2d": ["NCHW", "default"],
}

# Convert the layout of the graph where possible.
seq = tvm.transform.Sequential(
    [
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ConvertLayout(desired_layouts),
        relay.transform.FoldConstant(),
    ]
)

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    relay_mod = seq(relay_mod)
print("34")
#### #
#### # To begin with, define a model in Relay to be executed on-device. Then create an IRModule from relay model and
#### # fill parameters with random numbers.
#### #
####
#### data_shape = (1, 3, 10, 10)
#### weight_shape = (6, 3, 5, 5)
####
#### data = tvm.relay.var("data", tvm.relay.TensorType(data_shape, "float32"))
#### weight = tvm.relay.var("weight", tvm.relay.TensorType(weight_shape, "float32"))
####
#### y = tvm.relay.nn.conv2d(
####     data,
####     weight,
####     padding=(2, 2),
####     kernel_size=(5, 5),
####     kernel_layout="OIHW",
####     out_dtype="float32",
#### )
#### f = tvm.relay.Function([data, weight], y)
####
#### relay_mod = tvm.IRModule.from_expr(f)
#### relay_mod = tvm.relay.transform.InferType()(relay_mod)
####
#### weight_sample = np.random.rand(
####     weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
#### ).astype("float32")
#### params = {"weight": weight_sample}

#######################
# Defining the target
#######################
# Now we define the TVM target that describes the execution environment. This looks very similar
# to target definitions from other microTVM tutorials. Alongside this we pick the C Runtime to code
# generate our model against.
#
# When running on physical hardware, choose a target and a board that
# describe the hardware. There are multiple hardware targets that could be selected from
# PLATFORM list in this tutorial. You can chose the platform by passing --platform argument when running
# this tutorial.
#

RUNTIME = Runtime("crt", {"system-lib": True})
# TARGET = tvm.micro.testing.get_target("crt")
# TARGET = tvm.target.Target("c")
# TARGET = tvm.target.Target("llvm -device=riscv_cpu -mcpu=generic-rv32 -mtriple=riscv32-unknown-elf -mabi=ilp32d -mattr=+m,+a,+f,+d,+c -model etiss")
mattr = "+m,+a,+f,+d,+c"
if xcorev:
    mattr += ",+xcorevmac"
# TARGET = tvm.target.Target("llvm -device=riscv_cpu -mcpu=generic-rv32 -mtriple=riscv32-unknown-elf -mabi=ilp32d -mattr=+m,+a,+f,+d,+c,+xcorevmac -model etiss")
TARGET = tvm.target.Target(f"llvm -device=riscv_cpu -mcpu=generic-rv32 -mtriple=riscv32-unknown-elf -mabi=ilp32d -mattr={mattr} -model etiss")
# --target-llvm-device riscv_cpu --target-llvm-mcpu generic-rv32 --target-llvm-mtriple riscv32-unknown-elf --target-llvm-mabi ilp32d --target-llvm-mattr +m,+a,+f,+d,+c --target-llvm-model etiss

# Compiling for physical hardware
# --------------------------------------------------------------------------
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32L4R5ZI Nucleo target and board is chosen in the example below.
if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)


#########################
# Extracting tuning tasks
#########################
# Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
# a single implementation is defined; others don't make sense as tuning tasks. Using
# `extract_from_program`, you can produce a list of tunable tasks.
#
# Because task extraction involves running the compiler, we first configure the compiler's
# transformation passes; we'll apply the same configuration later on during autotuning.
#

pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0

######################
# Configuring microTVM
######################
# Before autotuning, we need to define a module loader and then pass that to
# a `tvm.autotvm.LocalBuilder`. Then we create a `tvm.autotvm.LocalRunner` and use
# both builder and runner to generates multiple measurements for auto tunner.
#
# In this tutorial, we have the option to use x86 host as an example or use different targets
# from Zephyr RTOS. If you choose pass `--platform=host` to this tutorial it will uses x86. You can
# choose other options by choosing from `PLATFORM` list.
#

module_loader = tvm.micro.AutoTvmModuleLoader(
    # template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("crt")),
    template_project_dir="/var/tmp/ga87puy/llvm-gen/mlonmcu/workspace/deps/src/tvm/apps/microtvm/etiss/template_project",
    project_options={
        "verbose": False,
        "etiss_script": "/var/tmp/ga87puy/llvm-gen/etiss/build/installed/bin/run_helper.sh",
    },
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=True,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=10000, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# Compiling for physical hardware
if use_physical_hw:
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")),
        project_options={
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
        },
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=RUNTIME,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=10000, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

##########################
# Run Autotuning
##########################
# Now we can run autotuning separately on each extracted task on microTVM device.
#

# autotune_log_file = pathlib.Path("microtvm_autotune.log.txt")
autotune_log_file = pathlib.Path(out_file)

if os.path.exists(autotune_log_file):
    os.remove(autotune_log_file)

# num_trials = 10
num_trials = 1000
for i, task in enumerate(tasks):
    prefix = "\n[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        early_stopping=num_trials//10,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file(str(autotune_log_file)),
            tvm.autotvm.callback.progress_bar(num_trials, prefix=prefix, si_prefix="M"),
        ],
        si_prefix="M",
    )

############################
# Timing the untuned program
############################
# For comparison, let's compile and run the graph without imposing any autotuning schedules. TVM
# will select a randomly-tuned implementation for each operator, which should not perform as well as
# the tuned operator.
#

with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    # str(tvm.micro.get_microtvm_template_projects("crt")),
    "/var/tmp/ga87puy/llvm-gen/mlonmcu/workspace/deps/src/tvm/apps/microtvm/etiss/template_project",
    lowered,
    temp_dir / "project",
    {
        "verbose": False,
        "etiss_script": "/var/tmp/ga87puy/llvm-gen/etiss/build/installed/bin/run_helper.sh",
    },
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered,
        temp_dir / "project",
        {
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module

##########################
# Timing the tuned program
##########################
# Once autotuning completes, you can time execution of the entire program using the Debug Runtime:

with tvm.autotvm.apply_history_best(str(autotune_log_file)):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    # str(tvm.micro.get_microtvm_template_projects("crt")),
    "/var/tmp/ga87puy/llvm-gen/mlonmcu/workspace/deps/src/tvm/apps/microtvm/etiss/template_project",
    lowered_tuned,
    temp_dir / "project",
    {
        "verbose": False,
        "etiss_script": "/var/tmp/ga87puy/llvm-gen/etiss/build/installed/bin/run_helper.sh",
    },
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered_tuned,
        temp_dir / "project",
        {
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered_tuned.get_params())
    print("########## Build with Autotuning ##########")
    debug_module.run()
    del debug_module
