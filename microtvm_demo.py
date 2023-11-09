import os

import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.contrib import graph_executor
from tvm.micro import export_model_library_format
from tvm.contrib.download import download_testdata

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", nargs=1, choices=["sine_model", "toycar"], default=None, help="Which model to use (Choose from: sine_model, toycar)")
parser.add_argument("--profile", action="store_true", help="Enable per-layer profiling (needs host-driven graph executor)")
parser.add_argument("--benchmark", action="store_true", help="Enable full model benchmarking (?)")
parser.add_argument("--cmsisnn", action="store_true", help="Enable cmsisnn BYOC backend")
parser.add_argument("--executor", choices=["graph", "aot"], nargs=1, default=None, help="Choose executor (either graph or aot)")
parser.add_argument("--mode", choices=["host-driven", "standalone"], nargs=1, default=None, help="Choose executor (either standalone or host-driven)")
args = parser.parse_args()
MODEL = args.model[0]
assert MODEL is not None
MODE = args.mode[0]
assert MODE is not None
EXECUTOR = args.executor[0]
assert EXECUTOR is not None
PROFILE = args.profile
BENCHMARK = args.benchmark
assert not (PROFILE and BENCHMARK), "--profile and --benchmark can not be used together"
CMSISNN = args.cmsisnn

if MODEL == "sine_model":
    model_url = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/sine_model.tflite"
    model_file = "sine_model.tflite"
    input_tensor = "dense_4_input"
    input_shape = (1,)
    input_dtype = "float32"
elif MODEL == "toycar":
    model_url = "https://github.com/tum-ei-eda/mlonmcu-models/raw/main/toycar/toycar.tflite"
    model_file = "toycar.tflite"
    input_tensor = "input_1"
    input_shape = (1, 640)
    input_dtype = "int8"

model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

project_options = {
    "verbose": False,
}  # You can use options to provide platform-specific options through TVM.

RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
if EXECUTOR == "graph":
    EXECUTOR_ = tvm.relay.backend.Executor("graph", {"link-params": True})
elif EXECUTOR == "aot":
    EXECUTOR_ = tvm.relay.backend.Executor("aot", {"link-params": True})
elif EXECUTOR == "aot+usmp":
    # Conflicts with system-lib!
    raise NotImplementedError
else:
    assert False
# TARGET = tvm.micro.testing.get_target("crt")
TARGET = "c"
if CMSISNN:
    from tvm.relay.op.contrib import cmsisnn

    # config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
    # mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    mod = cmsisnn.partition_for_cmsisnn(mod, params)
    project_options["cmsis_path"] = str((pathlib.Path(__file__).parent / "CMSIS_5").resolve())


with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = relay.build(mod, target=TARGET, runtime=RUNTIME, executor=EXECUTOR_, params=params)

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "generated-project"
generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

generated_project.build()
generated_project.flash()

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    if EXECUTOR == "graph":
        if MODE == "host-driven":
            if PROFILE:
                rt_mod = tvm.micro.create_local_debug_executor(
                    module.get_graph_json(), session.get_system_lib(), session.device, dump_root="./prof"
                )
            else:
                rt_mod = tvm.micro.create_local_graph_executor(
                    module.get_graph_json(), session.get_system_lib(), session.device
                )
        elif MODE == "standalone":
            assert not PROFILE, "--profile needs host-driven graph executor"
            rt_mod = graph_executor.create(
                module.get_graph_json(), session.get_system_lib(), session.device
            )
        else:
            assert False
    elif EXECUTOR == "aot":
        assert not PROFILE, "--profile needs host-driven graph executor"
        assert MODE == "host-driven", "AoT executor only supports host-driven"
        if MODE == "host-driven":
            if BENCHMARK:
                pass # TODO: needs patch!
            rt_mod = tvm.micro.create_local_aot_executor(session)
        else:
            assert False

    rt_mod.set_input(**module.get_params())

    input_data = np.ones(shape=input_shape, dtype=input_dtype)
    rt_mod.set_input(input_tensor, input_data)
    rt_mod.run()  # will also do profiling if with debug_executor
    if BENCHMARK:
        times = rt_mod.benchmark(session.device, number=1, repeat=1, end_to_end=False)
        print(times)

    tvm_output = rt_mod.get_output(0).numpy()
    print("Output: " + str(tvm_output))
