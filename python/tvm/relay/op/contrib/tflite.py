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
"""TfLite codegen supported operators."""
import tvm.ir
#from tvm.contrib.target.tflite import _convert_map  # TODO: support tflite as a target
from ...expr import Constant
from .. import op as reg
from .. import strategy
from tvm.target import generic_func
from . import _make

from tvm import relay


def _register_tflite_op(op_name):
    """Register a function to check the given operator is supported by TfLite.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    """

    def _check_supported(expr):
        attrs, args = expr.attrs, expr.args
        #if op_name == "nn.conv2d":
        #    if not isinstance(args[1], Constant):
        #        return False
        #    if attrs["kernel_layout"] not in ["HWIO", "OIHW"]:
        #        return False
        # TODO: implement!
        # return True
        raise NotImplementedError

    tvm.ir.register_op_attr(op_name, "target.tflitecompiler", _check_supported)


# for op in _convert_map:
#     _register_tflite_op(op)

def tflite_extern(inputs, name="UNKNOWN", builtin=False, options=None, out_dtype=None, out_shape=None):
    """Addition with numpy-style broadcasting.
    Parameters
    ----------
    TODO
    Returns
    -------
    TODO
    Examples
    --------
    TODO
    """ # TODO: update docstring
    import numpy as np
    inputs.append(tvm.relay.Constant(tvm.nd.array(np.array(options, dtype="uint8"))))
    inputs = [ relay.annotation.compiler_begin(input_expr, "ccompiler") for input_expr in inputs]
    return _make.tflite_extern(tvm.relay.Tuple(inputs), name, builtin, options, out_dtype, out_shape)

from tvm import te, tir, ir, topi

def tflite_extern_topi(data, attrs, out_dtype):
    inputs = data[:-1]
    options = data[-1]
    print(options)
    print("inputs =",inputs)
    attrs_dict = { i.name: attrs[i.name] for i in attrs.list_field_info()}
    print("attrs =",attrs_dict)
    print("out_dtype =",out_dtype, "T:", out_dtype.dtype, "of", type(out_dtype.dtype), "S:", out_dtype.shape, "of", type(out_dtype.shape))
    input(">>>")

    output_dtype = out_dtype.dtype # TODO: get!
    output_shape = out_dtype.shape # TODO: get!
    #output_dtype = out_dtype # TODO: get!
    #output_shape = out_shape # TODO: get!

    num_outputs = 1 # TODO!

    outputs = []
    output_dtypes = []
    output_shapes = []
    for i in range(num_outputs):  # Normally there will not be num_outputs > 1 but just in case...
        outputs.append(te.placeholder(output_shape, name=f'outp{i}', dtype=output_dtype))
        output_dtypes.append(output_dtype)
        output_shapes.append(output_shape)

    op_name = attrs_dict['name']
    is_builtin = attrs_dict['is_builtin']
    rets = list(te.extern(output_shapes, data, lambda ins, outs: tir.call_packed("tvm.runtime.tflite_extern_wrapper", op_name, int(is_builtin), len(inputs), len(outputs), *ins[:-1], *outs, ins[-1]), name="C", dtype=output_dtypes))
    print("rets: ", rets)
    return rets[0]

@reg.register_compute("tflite_extern")
def compute_tflite_extern(attrs, inputs, out_type):
    args = [inputs, attrs, out_type]
    return [tflite_extern_topi(*args)]

@generic_func
def schedule_tflite_extern(attrs, outs, target):
    """Schedule for tflite_extern"""
    with target:
        return topi.generic.default.default_schedule(outs, False)

reg.register_schedule("tflite_extern", schedule_tflite_extern)
