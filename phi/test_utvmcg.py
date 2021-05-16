import os
import sys
import logging
import shutil

import numpy as np

import tvm
import tvm.micro
from tvm import te
from tvm import relay
from tvm import ir
from tvm import autotvm
from tvm.contrib import graph_runtime

import tflite
from tflite.TensorType import TensorType as TType

path=sys.argv[1]

modelBuf = open(path, "rb").read()
import tflite
tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)
shapes = {}
types = {}

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

modelInfo = ModelInfo(tflModel)
for t in modelInfo.inTensors:
    print("Input", '"' + t.name + '"', t.ty, t.shape)
    shapes[t.name] = t.shape
    types[t.name] = t.ty

mod, params = relay.frontend.from_tflite(tflModel, shape_dict=shapes, dtype_dict=types)

cfg = { "tir.disable_vectorize": True }
target = tvm.target.target.micro("host")
opt_level = 3

with tvm.transform.PassContext(opt_level=opt_level, config=cfg):
    c_mod = relay.build(mod, target=target, params=params)
    graph = c_mod.get_graph_json()
    c_params = c_mod.get_params()

workspace = tvm.micro.Workspace(debug=True)

opts = tvm.micro.default_options(os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host"))
#compiler = compiler_ext.Compiler_Ext(target=target)
compiler = tvm.micro.DefaultCompiler(target=target)
micro_binary = tvm.micro.build_static_runtime(
    workspace,
    compiler,
    c_mod,
    opts,
    extra_libs=[tvm.micro.get_standalone_crt_lib("memory")]
)

print("MOD =", mod)
print("C_MOD =", c_mod)
