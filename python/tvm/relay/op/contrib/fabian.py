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
# pylint: disable=invalid-name, unused-argument
"""Fabian supported operators."""
import tvm

from tvm import relay
from tvm._ffi import register_func
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant, is_tuple_get_item, is_tuple
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d


def partition_for_fabian(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to Fabian.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.MergeComposite(fabian_pattern_table()),
            transform.AnnotateTarget("fabian", False),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    result_mod = seq(mod)
    return result_mod

@register_pattern_table("fabian")
def fabian_pattern_table():
    """Get the Fabian pattern table."""

    def conv_pattern():
        """Create a convolution pattern."""
        pattern = is_op("nn.conv2d")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(is_tuple_get_item)
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = pattern.optional(is_op("clip"))
        pattern = pattern.optional(lambda x: is_op("add")(x, wildcard()))
        pattern = pattern.optional(lambda x: is_op("nn.max_pool2d")(x))
        pattern = pattern.optional(lambda x: is_op("nn.avg_pool2d")(x))
        return pattern

    def dense_pattern():
        """Create a dense pattern."""
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by Fabian."""
        call = extract
        if isinstance(call, tvm.relay.expr.TupleGetItem):
            call = call.tuple_value
        elif call.op.name == "nn.relu":
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value
        elif call.op.name == "clip":
            if call.attrs["a_min"] != 0.0 or call.attrs["a_max"] != 6.0:
                return False
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value

        while call.op.name != "nn.conv2d":
            call = call.args[0]
        attrs, args = call.attrs, call.args
        if attrs.data_layout != "NCHW":
            return False
        data_typ = args[0].checked_type
        kernel_typ = args[1].checked_type
        is_depthwise = is_depthwise_conv2d(
            data_typ.shape,
            attrs["data_layout"],
            kernel_typ.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        )
        if attrs.groups != 1 and not is_depthwise:
            return False
        return True

    return [
        ("fabian.conv2d", conv_pattern(), check_conv),
        ("fabian.dense", dense_pattern()),
    ]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.fabian")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


# _register_external_op_helper("clip")
_register_external_op_helper("nn.relu")
# _register_external_op_helper("nn.global_avg_pool2d")
# _register_external_op_helper("nn.global_max_pool2d")
# _register_external_op_helper("nn.avg_pool2d")
# _register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.softmax")
# _register_external_op_helper("reshape")
# _register_external_op_helper("add")
# _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")
# _register_external_op_helper("minimum")
# _register_external_op_helper("maximum")
_register_external_op_helper("nn.adaptive_avg_pool2d")
