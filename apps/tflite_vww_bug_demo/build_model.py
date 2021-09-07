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
"""Creates a simple TVM modules."""

import argparse
import os
from tvm import relay
import tvm
from tvm import te, runtime
import logging
import json
from tvm.contrib import cc as _cc
from tvm.contrib.download import download_testdata

RUNTIMES = {
    "c": "{name}_c.{ext}",
    #"c++": "{name}_cpp.{ext}",
}


def build_module(opts):
    # dshape = (1, 3, 224, 224)
    # from mxnet.gluon.model_zoo.vision import get_model

    # block = get_model("mobilenet0.25", pretrained=True)
    # shape_dict = {"data": dshape}
    # mod, params = relay.frontend.from_mxnet(block, shape_dict)
    #model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
    #model_url = "https://github.com/mlcommons/tiny/raw/master/v0.5/training/keyword_spotting/trained_models/kws_ref_model.tflite"
    model_url = "https://github.com/mlcommons/tiny/raw/master/v0.5/training/visual_wake_words/trained_models/vww_96_int8.tflite"
    #model_file = "kws_ref_model.tflite"
    model_file = "vww_96_int8.tflite"
    model_path = download_testdata(model_url, model_file, module="data")

    tflite_model_buf = open(model_path, "rb").read()

    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    from tflite.TensorType import TensorType as TType

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


    shapes = {}
    types = {}

    modelInfo = ModelInfo(tflite_model)
    for t in modelInfo.inTensors:
        print("Input", '"' + t.name + '"', t.ty, t.shape)
        shapes[t.name] = t.shape
        types[t.name] = t.ty


    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shapes, dtype_dict=types
    )

    for runtime_name, file_format_str in RUNTIMES.items():
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            graph, lib, params = relay.build(
                #mod, tvm.target.target.micro("host"), params=params
                mod, f"llvm --runtime=c --system-lib", params=params
            )

        build_dir = os.path.abspath(opts.out_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        ext = "tar" if runtime_name == "c" else "o"
        lib_file_name = os.path.join(build_dir, file_format_str.format(name="model", ext=ext))
        if runtime_name == "c":
            lib.export_library(lib_file_name)
        else:
            # NOTE: at present, export_libarary will always create _another_ shared object, and you
            # can't stably combine two shared objects together (in this case, init_array is not
            # populated correctly when you do that). So for now, must continue to use save() with the
            # C++ library.
            # TODO(areusch): Obliterate runtime.cc and replace with libtvm_runtime.so.
            lib.save(lib_file_name)
        with open(
            os.path.join(build_dir, file_format_str.format(name="graph", ext="json")), "w"
        ) as f_graph_json:
            f_graph_json.write(graph)
        with open(
            os.path.join(build_dir, file_format_str.format(name="params", ext="bin")), "wb"
        ) as f_params:
            f_params.write(runtime.save_param_dict(params))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default=".")
    #parser.add_argument("-t", "--test", action="store_true")
    opts = parser.parse_args()

    #if opts.test:
    #    build_test_module(opts)
    #else:
    build_module(opts)
    #build_inputs(opts)
