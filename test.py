import tvm
from tvm import relay
import numpy as np

data_type = "float32"
data_shape = (1, 3, 24, 24)
weight_shape = (10, 3, 3, 3)
strides = (2, 2)
padding = (1, 1, 1, 1)
layout = "NCHW"
output_shape = (1, 10, 12, 12)

data_exp = relay.var('data', shape=data_shape, dtype=data_type)
weight = tvm.nd.array((np.random.uniform(size=weight_shape)).astype(data_type))
weight_exp = weight_exp = relay.const(weight, dtype=data_type)
out = relay.nn.conv2d(data_exp, weight_exp, strides, padding, data_layout=layout)
mod = tvm.IRModule.from_expr(out)

from tvm.relay.op.contrib.csinn import partition_for_csinn
mod = partition_for_csinn(mod)

print("mod", mod)

# target = tvm.target.Target(
#     "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
# )
# target = tvm.target.Target("llvm")
target = tvm.target.Target("c")
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target=target)

lib_path = "lib_csinn2.so"
# cross_compile = 'riscv64-unknown-linux-gnu-g++'
# lib.export_library(lib_path, cc=cross_compile)
lib.export_library(lib_path)
