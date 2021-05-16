import os
import sys
import io
import struct
import numpy as np
import pathlib
import shutil
import subprocess
import tempfile
import tarfile
import pytest

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib import get_pattern_table
from tvm.contrib import utils
from tvm.relay.backend import compile_engine
from tvm.contrib import utils
from tvm.contrib import graph_executor
from tvm.micro import export_model_library_format
#from tvm.relay import testing
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator

class CcompilerAnnotator(ExprMutator):
    """
    This is used to create external functions for ccompiler.
    A simple annotator that creates the following program:
           |
      -- begin --
           |
          add
           |
        subtract
           |
        multiply
           |
       -- end --
           |
    """

    def __init__(self):
        super(CcompilerAnnotator, self).__init__()
        self.in_compiler = 0

    def visit_call(self, call):
        if call.op.name == "add":  # Annotate begin at args
            if self.in_compiler == 1:
                lhs = compiler_begin(super().visit(call.args[0]), "ccompiler")
                rhs = compiler_begin(super().visit(call.args[1]), "ccompiler")
                op = relay.add(lhs, rhs)
                self.in_compiler = 2
                return op
        elif call.op.name == "subtract":
            if self.in_compiler == 1:
                lhs = super().visit(call.args[0])
                rhs = super().visit(call.args[1])
                if isinstance(lhs, relay.expr.Var):
                    lhs = compiler_begin(lhs, "ccompiler")
                if isinstance(rhs, relay.expr.Var):
                    rhs = compiler_begin(rhs, "ccompiler")
                return relay.subtract(lhs, rhs)
        elif call.op.name == "multiply":  # Annotate end at output
            self.in_compiler = 1
            lhs = super().visit(call.args[0])
            rhs = super().visit(call.args[1])
            if isinstance(lhs, relay.expr.Var):
                lhs = compiler_begin(lhs, "ccompiler")
            if isinstance(rhs, relay.expr.Var):
                rhs = compiler_begin(rhs, "ccompiler")
            op = relay.multiply(lhs, rhs)
            if self.in_compiler == 2:
                op = compiler_end(op, "ccompiler")
            self.in_compiler = 0

x = relay.var("x", shape=(1, 1))
y = relay.var("y", shape=(1, 1))
#x = compiler_begin(x, "ccompiler")
#y = compiler_begin(y, "compiler")
z = relay.add(x, y)
#z = compiler_end(z, "ccompiler")
func = relay.Function([x, y], z)

mod = tvm.IRModule()
ann = CcompilerAnnotator()
mod["main"] = ann.visit(func)
#mod["main"] = func
mod = tvm.relay.transform.PartitionGraph()(mod)

x_in = np.ones((1, 1)).astype("float32")
y_in = np.random.uniform(size=(1, 1)).astype("float32")

#params = {"x": x_in}
params = {}
#inputs = {"x": x_in, "y": y_in}

#input_list = [x_in, y_in]

#mod=func

target = "c -runtime=c --link-params --executor=aot"

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    lib = tvm.relay.build(mod, target, target_host=target, params=params)

tmp_path = utils.tempdir()
tmp_dir = tmp_path.temp_dir

base_path = os.path.join(tmp_dir, "test")
build_path = os.path.join(base_path, "build")
os.makedirs(build_path, exist_ok=True)

tar_file = os.path.join(base_path, "test.tar")
export_model_library_format(lib, tar_file)
t = tarfile.open(tar_file)
t.extractall(base_path)

outDir = "codegen"
#os.makedirs(outDir, exist_ok=True)
shutil.copytree(os.path.join(base_path, "codegen"), outDir)
shutil.copy2(os.path.join(base_path, "relay.txt"), os.path.join(outDir, "relay.txt"))
shutil.copy2(os.path.join(base_path, "metadata.json"), os.path.join(outDir, "metadata.json"))

#print("base_path =", base_path)
#input("!")
sys.exit(-1)

#create_main("test.c", input_list, output_list, build_path)

file_dir = os.path.dirname(os.path.abspath(__file__))
makefile = os.path.join(file_dir, "aot_test.mk")
make_cmd = f"make -f {makefile} build_dir=" + build_path + f" TVM_ROOT={file_dir}/../../../.."

compile_log_path = os.path.join(build_path, "test_compile.log")
ret = subprocess_with_stdout_and_log(make_cmd, ".", compile_log_path, False)
assert ret == 0

# Verify that runs fine
run_log_path = os.path.join(build_path, "test_run.log")
ret = subprocess_with_stdout_and_log("./aot_test_runner", build_path, run_log_path, False)
assert ret == 0
