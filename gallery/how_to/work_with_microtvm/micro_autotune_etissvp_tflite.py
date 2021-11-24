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

import logging
import sys
#logging.basicConfig(level="DEBUG", stream=sys.stdout)


project_options = {
    "project_type": "host_driven",
    "verbose": False,
    "debug": False,
    "transport": True,
    #"etiss_path": "/work/git/prj/etiss_clint_uart/ml_on_mcu/deps/install/etiss/etiss_default/",
    "etiss_path": "/work/git/prj/etiss_freertos/etiss-public-fork/build/installed",
    "riscv_path": "/usr/local/research/projects/SystemDesign/tools/riscv/current/",
    "etissvp_script": tvm.micro.get_microtvm_template_projects("etissvp") + "/scripts/run.sh",
    #"etissvp_script_args": "-i" + tvm.micro.get_microtvm_template_projects("etissvp") + "/scripts/memsegs.ini v",
    "etissvp_script_args": "v"
}

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

# Compiling for virtual hardware
# --------------------------------------------------------------------------
#TARGET = tvm.target.target.micro("host")
TARGET = tvm.target.Target("c --runtime=c -device=arm_cpu --system-lib")
#TARGET = tvm.target.target.riscv_cpu("bare_etiss_processor")
BOARD = "bare_etiss_processor"

#########################
# Extracting tuning tasks
#########################

pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0

######################
# Configuring microTVM
######################

# Compiling for virtual hardware
# --------------------------------------------------------------------------
module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("etissvp")),
    project_options=project_options,
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=False,
    build_func=tvm.micro.autotvm_build_func,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

################
# Run Autotuning
################

#num_trials = 100
num_trials = 10
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file("microtvm_autotune.log.txt"),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M", prefix=prefix),
        ],
        si_prefix="M",
    )

############################
# Timing the untuned program
############################

with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, params=params)

temp_dir = tvm.contrib.utils.tempdir()

# Compiling for virtual hardware
# --------------------------------------------------------------------------
project = tvm.micro.generate_project(
    str(repo_root / "apps" / "microtvm" / "etissvp" / "template_project"),
    lowered,
    temp_dir / "project",
    project_options,
)

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    #debug_module = tvm.micro.create_local_debug_executor(
    #    lowered.get_graph_json(), session.get_system_lib(), session.device
    #)
    debug_module = tvm.micro.create_local_graph_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module

##########################
# Timing the tuned program
##########################

with tvm.autotvm.apply_history_best("microtvm_autotune.log.txt"):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, params=params)

temp_dir = tvm.contrib.utils.tempdir()

# Compiling for virtual hardware
# --------------------------------------------------------------------------
project = tvm.micro.generate_project(
    str(repo_root / "apps" / "microtvm" / "etissvp" / "template_project"),
    lowered_tuned,
    temp_dir / "project",
    project_options,
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
