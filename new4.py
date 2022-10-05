import sys

import torch
import torchvision

import tvm
import tvm.relay as relay

from tvm.relay.op.contrib.register import get_pattern_table

model = torchvision.models.resnet18(pretrained=True)
model = model.eval()
input_name = "input0"
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
print(scripted_model)

shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


mod = tvm.relay.transform.InferType()(mod)
print("mod", mod)


def fold_optimize(mod, params=None):
    optimize = tvm.transform.Sequential( [
        tvm.relay.transform.InferType(),
        tvm.relay.transform.CanonicalizeOps(),
        tvm.relay.transform.SimplifyInference(),
        tvm.relay.transform.FoldScaleAxis(),
        tvm.relay.transform.FoldConstant(),
    ])
    if params:
        mod["main"] = tvm.relay.build_module.bind_params_by_name(mod["main"], params)

    mod = optimize(mod)
    return mod


with tvm.transform.PassContext(opt_level=3):
    mod_alt = fold_optimize(mod, params=params)

print("mod_alt", mod_alt)

assert len(sys.argv) == 3
backend = sys.argv[1]
merge = bool(int(sys.argv[2]))
pattern_table = get_pattern_table(backend)

sequence = [
    relay.transform.MergeComposite(pattern_table),
    relay.transform.AnnotateTarget(backend),
]

if merge:
    sequence.append(relay.transform.MergeCompilerRegions())

sequence.append(relay.transform.PartitionGraph())
sequential = tvm.transform.Sequential(sequence)

print("mod\n", mod)
mod_ = sequential(mod_alt)

print("mod_\n", mod_)

target = "c"
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    relay.build(mod_, target=target, params=params)
