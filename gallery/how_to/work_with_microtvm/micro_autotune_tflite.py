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

Autotuning with micro TVM
=========================
**Authors**:
`Andrew Reusch <https://github.com/areusch>`_,
`Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial explains how to autotune a model using the C runtime.
"""

import numpy as np
import subprocess
import pathlib

import tvm

####################
# Defining the model
####################
#
# To begin with, define a model in Relay to be executed on-device. Then create an IRModule from relay model and
# fill parameters with random numbers.
#


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

MODELS_DIR = "/work/git/prj/etiss_clint_uart/ml_on_mcu/data/"
MODEL = "aww"

import os
modelBuf = open(os.path.join(MODELS_DIR, MODEL, MODEL + ".tflite"), "rb").read()

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


#######################
# Defining the target #
#######################
# Now we define the TVM target that describes the execution environment. This looks very similar
# to target definitions from other microTVM tutorials.
#
# When running on physical hardware, choose a target and a board that
# describe the hardware. There are multiple hardware targets that could be selected from
# PLATFORM list in this tutorial. You can chose the platform by passing --platform argument when running
# this tutorial.
#
#TARGET = tvm.target.target.micro("host")
TARGET = tvm.target.Target("c --runtime=c -device=arm_cpu --system-lib")
# Compiling for physical hardware
# --------------------------------------------------------------------------
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32L4R5ZI Nucleo target and board is chosen in the example below.
#
#    TARGET = tvm.target.target.micro("stm32l4r5zi")
#    BOARD = "nucleo_l4r5zi"

#########################
# Extracting tuning tasks
#########################
# Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
# a single implementation is defined; others don't make sense as tuning tasks. Using
# `extract_from_program`, you can produce a list of tunable tasks.
#
# Because task extraction involves running the compiler, we first configure the compiler's
# transformation passes; we'll apply the same configuration later on during autotuning.

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

repo_root = pathlib.Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
)

module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=repo_root / "src" / "runtime" / "crt" / "host",
    project_options={"verbose": False},
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=True,
    build_func=tvm.micro.autotvm_build_func,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# Compiling for physical hardware
# --------------------------------------------------------------------------
#    module_loader = tvm.micro.AutoTvmModuleLoader(
#        template_project_dir=repo_root / "apps" / "microtvm" / "zephyr" / "template_project",
#        project_options={
#            "zephyr_board": BOARD,
#            "west_cmd": "west",
#            "verbose": False,
#            "project_type": "host_driven",
#        },
#    )
#    builder = tvm.autotvm.LocalBuilder(
#        n_parallel=1,
#        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
#        do_fork=False,
#        build_func=tvm.micro.autotvm_build_func,
#    )
#    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)
#
#    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

################
# Run Autotuning
################
# Now we can run autotuning separately on each extracted task.

#num_trials = 20
#num_trials = 20
num_trials = 200
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file("microtvm_autotune.log.txt"),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
    )

############################
# Timing the untuned program
############################
# For comparison, let's compile and run the graph without imposing any autotuning schedules. TVM
# will select a randomly-tuned implementation for each operator, which should not perform as well as
# the tuned operator.

with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, params=params)

temp_dir = tvm.contrib.utils.tempdir()

project = tvm.micro.generate_project(
    str(repo_root / "src" / "runtime" / "crt" / "host"),
    lowered,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
# --------------------------------------------------------------------------
#    project = tvm.micro.generate_project(
#        str(repo_root / "apps" / "microtvm" / "zephyr" / "template_project"),
#        lowered,
#        temp_dir / "project",
#        {
#            "zephyr_board": BOARD,
#            "west_cmd": "west",
#            "verbose": False,
#            "project_type": "host_driven",
#        },
#    )

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

with tvm.autotvm.apply_history_best("microtvm_autotune.log.txt"):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, params=params)

temp_dir = tvm.contrib.utils.tempdir()

project = tvm.micro.generate_project(
    str(repo_root / "src" / "runtime" / "crt" / "host"),
    lowered_tuned,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
# --------------------------------------------------------------------------
#    project = tvm.micro.generate_project(
#        str(repo_root / "apps" / "microtvm" / "zephyr" / "template_project"),
#        lowered_tuned,
#        temp_dir / "project",
#        {
#            "zephyr_board": BOARD,
#            "west_cmd": "west",
#            "verbose": False,
#            "project_type": "host_driven",
#        },
#    )

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
