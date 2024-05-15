# Licensed to thme Apache Software Foundation (ASF) under one
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
# pylint: disable=import-outside-toplevel, unused-argument

"""TFLite frontend of Relax."""
import math
from typing import Callable, Dict, List, Tuple, Union, Any, Optional

import numpy as np

import tvm
from tvm import relax, tir, topi
from ... import expr as _expr


def to_int_list(np_array):
    """Convert a np array to a python int list.

    Note: This function converts np.int32 to python's int.
    If we don't do this conversion, numpy's automatic upcast will make
    the shape / parameters be converted to int64 IntImm in relay and
    cause problems in relay/TOPI.
    """
    return [int(x) for x in np_array]


class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

    def __repr__(self):
        return f"TensorWrapper({self.tensor_idx}, {self.tensor}, {self.buffer}, {self.qnn_params})"


# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    tensor_name = subgraph.Tensors(tensor_idx).Name()
    if tensor_name is not None:
        tensor_name = tensor_name.decode("utf-8")
    else:
        tensor_name = "tvmgen_tensor_" + str(tensor_idx)
    return tensor_name


def _decode_type(n):
    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    return _tflite_m[n]


def _input_type(model):
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        inputs_count = subgraph.InputsLength()
        assert inputs_count >= 1
        for input_index in range(inputs_count):
            input_ = subgraph.Inputs(input_index)
            assert subgraph.TensorsLength() > input_
            tensor = subgraph.Tensors(input_)
            input_shape = tuple(tensor.ShapeAsNumpy())
            tensor_type = tensor.Type()
            input_name = get_tensor_name(subgraph, input_)
            shape_dict[input_name] = input_shape
            dtype_dict[input_name] = _decode_type(tensor_type)

    return shape_dict, dtype_dict


def build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret


def get_scalar_from_constant(expr):
    """Returns scalar value from Relay constant scalar."""
    print("get_scalar_from_constant", expr)
    assert (
        isinstance(expr, _expr.Constant) and not expr.data.shape
    ), "Expr is not a constant scalar."
    value = expr.data.numpy()
    print("value", value)
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(np.int8) or value.dtype == np.dtype(
        np.float32
    ), "value must be float32/int32"
    return value.item(0)


def get_tensor_from_constant(expr):
    """Returns tensor of values from Relay constant node."""
    assert isinstance(expr, _expr.Constant)
    value = expr.data.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(
        np.float32
    ), "value must be float32/int32"
    return value


class TFLiteImporter:
    """An importer from TFLite to Relax."""

    def __init__(self) -> None:

        # self._nodes: Dict[Union[str, mlir.ir.Operation], relax.Expr] = {}
        self._nodes: Dict[str, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.subgraph = None
        self.prefetched_nodes = {}
        self.create_convert_map()

    # @staticmethod
    # def _convert_data_type(input_type):
    #     """converts the data type from mlir to tvm."""
    #     from jaxlib import mlir

    #     if mlir.ir.ShapedType.isinstance(input_type):
    #         input_type = mlir.ir.ShapedType(input_type).element_type

    #     input_type = str(input_type)
    #     if input_type == "f16":
    #         return "float16"
    #     elif input_type in ["f32", "F32Type"]:
    #         return "float32"
    #     elif input_type in ["f64", "F64Type"]:
    #         return "float64"
    #     elif input_type == "i1":
    #         return "bool"
    #     elif input_type == "i8":
    #         return "int8"
    #     elif input_type == "i16":
    #         return "int16"
    #     elif input_type == "i32":
    #         return "int32"
    #     elif input_type == "i64":
    #         return "int64"
    #     elif input_type == "ui8":
    #         return "uint8"
    #     elif input_type == "ui16":
    #         return "uint16"
    #     elif input_type == "ui32":
    #         return "uint32"
    #     elif input_type == "ui64":
    #         return "uint64"
    #     else:
    #         raise NotImplementedError(f"input_type {input_type} is not handled yet")

    # def _attr2value(self, node) -> Union[Any, List[Any]]:
    #     from jaxlib import mlir
    #     import numpy as np

    #     if mlir.ir.IntegerAttr.isinstance(node):
    #         int_attr = mlir.ir.IntegerAttr(node)
    #         return int_attr.value
    #     if mlir.ir.FloatAttr.isinstance(node):
    #         float_attr = mlir.ir.FloatAttr(node)
    #         return float_attr.value
    #     if mlir.ir.DenseIntElementsAttr.isinstance(node):
    #         dense_attr = mlir.ir.DenseIntElementsAttr(node)
    #     elif mlir.ir.DenseFPElementsAttr.isinstance(node):
    #         dense_attr = mlir.ir.DenseFPElementsAttr(node)
    #     else:
    #         raise ValueError("Unsupported Attribute type: " + str(type(node)))
    #     ret = []
    #     for val in dense_attr:
    #         ret.append(val)
    #     shape = self.get_shape(node.type)
    #     dtype = self._convert_data_type(node.type)
    #     return np.asarray(ret, dtype).reshape(shape).tolist()

    # def retrieve_operands(self, node):
    #     return self._retrieve_operands(node.operands)

    # def _retrieve_operands(self, node):
    #     from jaxlib import mlir

    #     # the operand is one of the inputs of FuncOp
    #     if isinstance(node, mlir.ir.Operation):
    #         return self._nodes[node]
    #     if isinstance(node, tuple):
    #         return tuple(self._retrieve_operands(x) for x in node)
    #     if isinstance(node, (list, mlir.ir.OpOperandList)):
    #         return [self._retrieve_operands(x) for x in node]
    #     if isinstance(node, dict):
    #         return {self._retrieve_operands(k): self._retrieve_operands(v) for k, v in node.items()}
    #     if isinstance(node, mlir.ir.Value):
    #         if isinstance(node.owner, mlir.ir.Block):
    #             block_arg = mlir.ir.BlockArgument(node)
    #             return self._nodes["arg" + str(block_arg.arg_number)]
    #         return self._retrieve_operands(node.owner)
    #     return node

    # def get_shape(self, inpt_type) -> List[Any]:
    #     """Get the shape from Type like tensor<?x?xf32>"""
    #     from jaxlib import mlir

    #     shape_type = inpt_type
    #     if isinstance(shape_type, mlir.ir.Type):
    #         shape_type = mlir.ir.ShapedType(shape_type)
    #     ret = []
    #     for i in range(shape_type.rank):
    #         # get_dim_size
    #         if shape_type.is_dynamic_dim(i):
    #             n = tir.Var("n", "int64")
    #             ret.append(n)
    #         else:
    #             ret.append(shape_type.get_dim_size(i))

    #     return ret

    # @staticmethod
    # def _promote_binary_op_args(lhs, rhs):
    #     if not isinstance(lhs, relax.Expr) and not isinstance(rhs, relax.Expr):
    #         msg = "Both the lhs and the rhs are not expressions."
    #         raise AssertionError(msg)
    #     if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
    #         return lhs, rhs
    #     if isinstance(lhs, relax.Expr):
    #         assert isinstance(lhs.struct_info, relax.TensorStructInfo)
    #         return lhs, relax.const(rhs, lhs.struct_info.dtype)
    #     assert isinstance(rhs.struct_info, relax.TensorStructInfo)
    #     return relax.const(lhs, rhs.struct_info.dtype), rhs

    # def _call_binary_op(self, op, lhs, rhs):
    #     lhs, rhs = StableHLOImporter._promote_binary_op_args(lhs, rhs)
    #     return self.block_builder.emit(op(lhs, rhs))

    # def _add(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
    #         return self._call_binary_op(relax.op.add, lhs, rhs)
    #     return lhs + rhs
    def _convert_elemwise(self, relax_op, op, ignore_qnn_params=False, comparison_op=False):
        """Generic method to Convert TFLite elemwise"""
        try:
            from tflite.AddOptions import AddOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.DivOptions import DivOptions
            from tflite.MulOptions import MulOptions
            from tflite.SubOptions import SubOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_expr = self.get_tensor_expr(rhs_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # TFLite format demands equal scale and zero_point tuple parameters for some operations
        # to allow us to use non-quantized operation instead of quantized if ignore_qnn_params=True
        if ignore_qnn_params and not comparison_op:
            assert (
                lhs_tensor.qnn_params
                and self.has_same_qnn_params(lhs_tensor, output_tensor)
                and self.has_same_qnn_params(rhs_tensor, output_tensor)
            ), "All tensors should be quantized with the same (scale,zero-point) tuple parameters"

        # If quantized, extracts qnn params and call QNN add operator.
        if not ignore_qnn_params and lhs_tensor.qnn_params:
            assert rhs_tensor.qnn_params, "Both tensors should be quantized."
            assert output_tensor.qnn_params, "Output tensor should be quantized."
            # lhs_expr = relax.op.qdq.dequantize(lhs_expr, lhs_tensor.qnn_params["scale"], lhs_tensor.qnn_params["zero_point"])
            # rhs_expr = relax.op.qdq.dequantize(rhs_expr, rhs_tensor.qnn_params["scale"], rhs_tensor.qnn_params["zero_point"])
            lhs_expr = self.dequantize(lhs_expr, lhs_tensor)
            rhs_expr = self.dequantize(rhs_expr, rhs_tensor)
            out = relax_op(
                lhs_expr,
                rhs_expr,
                # lhs_scale=lhs_tensor.qnn_params["scale"],
                # lhs_zero_point=lhs_tensor.qnn_params["zero_point"],
                # rhs_scale=rhs_tensor.qnn_params["scale"],
                # rhs_zero_point=rhs_tensor.qnn_params["zero_point"],
                # output_scale=output_tensor.qnn_params["scale"],
                # output_zero_point=output_tensor.qnn_params["zero_point"],
            )
            # out = relax.op.qdq.quantize(out, output_tensor.qnn_params["scale"], output_tensor.qnn_params["zero_point"])
            out = self.quantize(out, output_tensor)
        else:
            out = relax_op(lhs_expr, rhs_expr)

        # Options (fused_activation_function)
        options = None
        if op.BuiltinOptionsType() == BuiltinOptions.AddOptions:
            options = AddOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.SubOptions:
            options = SubOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.MulOptions:
            options = MulOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.DivOptions:
            options = DivOptions()

        if options is not None:
            op_options = op.BuiltinOptions()
            options.Init(op_options.Bytes, op_options.Pos)
            fused_activation_fn = options.FusedActivationFunction()

            # Handle fused activations
            if not ignore_qnn_params and output_tensor.qnn_params:
                # TODO: avoid QDQ between op and act
                # scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
                # zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
                # output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
                # out = relax.op.qdq.dequantize(out, scale_val, zero_point_val)
                out = self.dequantize(out, output_tensor)
                # out = self.convert_qnn_fused_activation_function(
                #     expr=out,
                #     fused_activation_fn=fused_activation_fn,
                #     scale=scale_val,
                #     zero_point=zero_point_val,
                #     dtype=output_tensor_type_str,
                # )
                out = self.convert_fused_activation_function(out, fused_activation_fn)
                # out = relax.op.qdq.quantize(out, scale_val, zero_point_val)
                out = self.quantize(out, output_tensor)
            else:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
        return self.block_builder.emit(out)

    def convert_add(self, op):
        """Convert TFLite ADD"""
        # Check if the input tensor is quantized, call QNN op
        # if self.is_quantized(op):
        #     raise NotImplementedError("Quantized TFLite Ops are unsupported")
        #     return self._convert_elemwise(_qnn.op.add, op)
        return self._convert_elemwise(relax.op.add, op)

    def convert_sub(self, op):
        """Convert TFLite SUB"""
        # Check if the input tensor is quantized, call QNN op
        # if self.is_quantized(op):
        #     raise NotImplementedError("Quantized TFLite Ops are unsupported")
        #     return self._convert_elemwise(_qnn.op.subtract, op)
        return self._convert_elemwise(relax.op.subtract, op)


    # def _maximum(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.maximum(lhs, rhs))

    # def _minimum(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.minimum(lhs, rhs))

    # def _divide(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
    #         return self._call_binary_op(relax.op.divide, lhs, rhs)
    #     return lhs / rhs

    # def _multiply(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
    #         return self._call_binary_op(relax.op.multiply, lhs, rhs)
    #     return lhs * rhs
    def convert_mul(self, op):
        """Convert TFLite MUL"""
        # Check if the input tensor is quantized, call QNN op
        # if self.is_quantized(op):
        #     raise NotImplementedError("Quantized TFLite Ops are unsupported")
        #     return self._convert_elemwise(_qnn.op.mul, op)
        return self._convert_elemwise(relax.op.multiply, op)

    # def _subtract(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
    #         return self._call_binary_op(relax.op.subtract, lhs, rhs)
    #     return lhs - rhs

    # def _broadcast_in_dim(self, node: mlir.ir.Operation) -> relax.Expr:
    #     operands = self.retrieve_operands(node)
    #     data = operands[0]
    #     # broadcast_dims = self._attr2value(node.attributes["broadcast_dimensions"])
    #     shape = self.get_shape(node.result.type)
    #     # scalar
    #     if len(shape) == 0:
    #         return data
    #     return self.block_builder.emit(relax.op.broadcast_to(data, shape))

    # def _const(self, node: mlir.ir.Operation) -> relax.Expr:
    #     const_value = self._attr2value(node.attributes["value"])
    #     dtype = self._convert_data_type(node.result.type)
    #     return relax.const(const_value, dtype)

    # def _dot_general(self, node: mlir.ir.Operation) -> relax.Expr:
    #     lhs, rhs = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.matmul(lhs, rhs))

    # def _convolution(self, node) -> relax.Expr:
    #     from jaxlib import mlir

    #     x, weight = self.retrieve_operands(node)
    #     shaped_type = mlir.ir.ShapedType(node.result.type)
    #     out_dtype = self._convert_data_type(shaped_type.element_type)
    #     strides = self._attr2value(node.attributes["window_strides"])
    #     padding = self._attr2value(node.attributes["padding"])
    #     lhs_dilation = self._attr2value(node.attributes["lhs_dilation"])
    #     rhs_dilation = self._attr2value(node.attributes["rhs_dilation"])
    #     if len(lhs_dilation) > 0:
    #         lhs_dilation = lhs_dilation[0]
    #     if len(rhs_dilation) > 0:
    #         rhs_dilation = rhs_dilation[0]
    #     dilation = (lhs_dilation, rhs_dilation)
    #     groups = self._attr2value(node.attributes["batch_group_count"])
    #     conv2d = relax.op.nn.conv2d(
    #         x,
    #         weight,
    #         strides=strides,
    #         padding=padding[0],
    #         dilation=dilation,
    #         groups=groups,
    #         data_layout="NHWC",
    #         kernel_layout="HWIO",
    #         out_dtype=out_dtype,
    #     )

    #     return self.block_builder.emit(conv2d)
    def convert_fused_activation_function(self, in_expr, fused_activation_fn):
        """Convert TFLite fused activation function"""
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if fused_activation_fn == ActivationFunctionType.NONE:
            return in_expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            tmp = self.block_builder.emit_te(topi.maximum, in_expr, 0)
            return self.block_builder.emit_te(topi.minimum, tmp, 6)
        if fused_activation_fn == ActivationFunctionType.RELU:
            return relax.op.nn.relu(in_expr)
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            tmp = self.block_builder.emit_te(topi.maximum, in_expr, -1)
            return self.block_builder.emit_te(topi.minimum, tmp, 1)
        if fused_activation_fn == ActivationFunctionType.TANH:
            return relax.op.tanh(in_expr)
        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise tvm.error.OpNotImplemented(
            f"Fused activation {fused_activation_fn_str} is not supported yet."
        )

    def convert_conv(self, op, conv_type):
        """convolution implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        is_depthwise_conv = False
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "depthwise":
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise tvm.error.OpNotImplemented(
                f"Operator {conv_type} is not supported for frontend TFLite."
            )

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = to_int_list(self.get_tensor_shape(weight_tensor))
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, in_channels = to_int_list(
                self.get_tensor_shape(weight_tensor)
            )

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            # "kernel_size": [kernel_h, kernel_w],
            "strides": [stride_h, stride_w],
            "dilation": [dilation_h, dilation_w],
            "padding": [0, 0],
            "data_layout": "NHWC",
        }

        if is_depthwise_conv:
            # params["channels"] = int(in_channels)
            params["groups"] = int(input_c)
            # If number of input channels is 1, treat as normal
            # convolution.
            params["kernel_layout"] = "HWIO" if input_c == 1 else "HWOI"
        else:
            # params["channels"] = int(output_channels)
            params["kernel_layout"] = "HWIO"
            if input_c != in_channels:
                assert (
                    input_c % in_channels == 0
                ), "Input channels is not divisible of kernel in_channels."
                params["groups"] = int(input_c / in_channels)

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite converts float32 models to float16 models by introducing
        # a Dequantize op in every op that contains a float32 values.
        # (weights, biases, and constants etc. )
        # So conv op may have weight and bias as tensors instead of values.
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
            if is_depthwise_conv:
                weight_expr = relax.op.reshape(
                    weight_expr, (kernel_h, kernel_w, input_c, depth_multiplier)
                )
            else:
                weight_expr = _op.transpose(weight_expr, axes=(1, 2, 3, 0))
        else:
            if self.is_prefetched(weight_tensor.tensor_idx):
                weight_value = self.get_prefetched_node(weight_tensor.tensor_idx)
            else:
                weight_value = self.get_tensor_value(weight_tensor)

            # TFLite kernel layout:
            # convolution:
            # OC KH KW IC, we require KH KW IC OC (HWIO)
            # depthwise convolution:
            # 1 KH KW C(input_c * depth_multiplier), we require
            # KH KW IC M (depth_multiplier) (HWOI)
            print("is_dw", is_depthwise_conv)
            if is_depthwise_conv:
                weight_value = weight_value.reshape(kernel_h, kernel_w, input_c, depth_multiplier)
            else:
                weight_value = weight_value.transpose((1, 2, 3, 0))
            print("weight_value", weight_value)

            # weight_expr = self.exp_tab.new_const(
            #     weight_value, dtype=weight_tensor_type_str, source_name=weight_tensor.tensor.Name()
            # )
            weight_expr = relax.const(
                weight_value, dtype=weight_tensor_type_str
            )

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)

            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
            if do_pad:
                params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]

        else:
            raise tvm.error.OpAttributeUnImplemented(
                f"Padding format {padding} is not supported for operator Conv."
            )

        if input_tensor.qnn_params:
            if input_tensor.qnn_params is not None:  # if not float32
                in_expr = self.dequantize(in_expr, input_tensor)
            in_expr = self.block_builder.normalize(in_expr)
            if weight_tensor.qnn_params is not None:  # if not float32
                weight_expr = self.dequantize(weight_expr, weight_tensor, axis=-2 if is_depthwise_conv else -1)
            print("we", weight_expr)
            print("wt", weight_tensor)
            weight_expr = self.block_builder.normalize(weight_expr)
            # qnn_conv2d_params = dict(params)
            # qnn_conv2d_params["weight_zero_point"] = input_tensor.qnn_params["zero_point"]
            # qnn_conv2d_params["kernel_zero_point"] = weight_tensor.qnn_params["zero_point"]
            # qnn_conv2d_params["out_dtype"] = (
            #     "int64" if output_tensor_type_str == "int16" else "int32"
            # )
            # qnn_conv2d_params["input_scale"] = input_tensor.qnn_params["scale"]
            # qnn_conv2d_params["kernel_scale"] = weight_tensor.qnn_params["scale"]
            # out = _qnn.op.conv2d(in_expr, weight_expr, **qnn_conv2d_params)
            out = relax.op.nn.conv2d(in_expr, weight_expr, **params, out_dtype="float32")
            out = self.block_builder.normalize(out)
            out = self.quantize(out, output_tensor)  # TODO: use int32 here (also eleminates bias bug)
            out = self.block_builder.normalize(out)
        else:
            out = relax.op.nn.conv2d(in_expr, weight_expr, **params)
            out = self.block_builder.normalize(out)

        # if we have bias
        if len(input_tensors) == 3:
            print("has bias")
            bias_tensor = input_tensors[2]
            print("bias_tensor", bias_tensor)
            bias_tensor_type = bias_tensor.tensor.Type()
            print("bias_tensor_type", bias_tensor_type)
            # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            print("bias_tensor_type_str", bias_tensor_type_str)
            if self.has_expr(bias_tensor.tensor_idx):
                bias_expr = self.get_expr(bias_tensor.tensor_idx)
            else:
                # bias_expr = self.exp_tab.new_const(
                #     self.get_tensor_value(bias_tensor),
                #     dtype=bias_tensor_type_str,
                #     source_name=bias_tensor.tensor.Name(),
                # )
                bias_expr = relax.const(
                    self.get_tensor_value(bias_tensor),
                    dtype=bias_tensor_type_str,
                )
            # channel_axis = 3
            # out = _op.nn.bias_add(out, bias_expr, axis=channel_axis)
            bias_expr = relax.op.reshape(
                bias_expr,
                [1, 1, 1, -1]
            )
            # TODO: only cast if required (skip for fp32)
            out = relax.op.astype(out, bias_tensor_type_str)
            out = relax.op.add(out, bias_expr)
            out = self.block_builder.normalize(out)

        # Handle fused activation.
        if output_tensor.qnn_params:
            # Calculate the intermediate scale and zero point of the int32 output.
            data_scale = input_tensor.qnn_params["scale"]
            data_scale_val = get_scalar_from_constant(data_scale)

            weight_scale = weight_tensor.qnn_params["scale"]
            # If weight scale is scalar, it is per-tensor quantization
            if isinstance(weight_scale, float):
                weight_scale_val = get_scalar_from_constant(weight_scale)
            else:
                weight_scale_val = get_tensor_from_constant(weight_scale)

            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relax.const(new_input_scale_val, "float32")
            # new_input_zero_point = relax.const(0, "int32")
            new_input_zero_point = relax.const(0, "int8")

            # Finally requantize
            # TODO: combine into requantize op
            # TODO: add axis argument to self.quantize/dequantize
            out = relax.op.qdq.dequantize(out, new_input_scale, new_input_zero_point, axis=3)
            # out = _qnn.op.requantize(
            #     out,
            #     input_scale=new_input_scale,
            #     input_zero_point=new_input_zero_point,
            #     output_scale=output_tensor.qnn_params["scale"],
            #     output_zero_point=output_tensor.qnn_params["zero_point"],
            #     out_dtype=output_tensor_type_str,
            #     axis=3,
            # )

            # Call activation function
            output_scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            output_zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.block_builder.normalize(out)
            out = self.convert_fused_activation_function(out, fused_activation_fn)
            # out = self.convert_qnn_fused_activation_function(
            #     expr=out,
            #     fused_activation_fn=fused_activation_fn,
            #     scale=output_scale_val,
            #     zero_point=output_zero_point_val,
            #     dtype=output_tensor_type_str,
            # )
            out = relax.op.qdq.quantize(out, output_tensor.qnn_params["scale"], output_tensor.qnn_params["zero_point"], axis=3, out_dtype=output_tensor_type_str)
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)
        return self.block_builder.emit(out)

    def convert_conv2d(self, op):
        """Convert TFLite conv2d"""
        return self.convert_conv(op, "conv2d")

    def convert_depthwise_conv2d(self, op):
        """Convert TFLite depthwise conv2d"""
        return self.convert_conv(op, "depthwise")

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        params = {"axis": -1}  # -1 is channel
        in_expr = self.get_expr(input_tensor_idx)

        # TODO - Naive softmax int8 implementation leads to bad accuracy. Currently, we can
        # dequantize to FP32 and perform softmax on FP32. We can investigate an integer only softmax
        # implementation in future.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = relax.op.nn.softmax(in_expr, **params)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return self.block_builder.emit(out)

    # def _reshape(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     if isinstance(data, list):
    #         assert len(data) == 1
    #         data = data[0]
    #     new_shape = self.get_shape(node.result.type)
    #     return self.block_builder.emit(relax.op.reshape(data, new_shape))
    def convert_reshape(self, op):
        """Convert TFLite reshape"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReshapeOptions import ReshapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            if self.has_expr(shape_tensor.tensor_idx):
                target_expr = self.get_expr(shape_tensor.tensor_idx)
                target_value, success = try_infer_value(
                    target_expr,
                    parameters={k: _nd.array(np.array(v)) for k, v in self.exp_tab.params.items()},
                )
                if success:
                    # convert to flattened list
                    from itertools import chain

                    try:
                        target_shape = list(chain(*target_value))
                    except TypeError:
                        target_shape = list(chain(target_value))
                else:
                    target_shape = target_expr
            else:
                target_shape = self.get_tensor_value(shape_tensor)
                # convert to flattened list
                from itertools import chain

                try:
                    target_shape = list(chain(*target_shape))
                except TypeError:
                    target_shape = list(chain(target_shape))

        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = to_int_list(reshape_options.NewShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        # If the tensors are quantized, ensure that input/output qnn params are same.

        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        if input_tensor.qnn_params and input_tensor_type_str == "int8":
            # TFLite 2.x quantization spec requires qnn params to be same and dtype to be int8.
            # For TFLite 1.x, dtype can be uint8 and qnn params can be different
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(
                input_tensor, output_tensor
            ), "TFLite reshape requires input and output scale and zero points to be equal"

        out = relax.op.reshape(in_expr, target_shape)
        if input_tensor.qnn_params and input_tensor_type_str == "uint8":
            output_tensor = output_tensors[0]
            if not self.has_same_qnn_params(input_tensor, output_tensor):
                output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
                out = _qnn.op.requantize(
                    out,
                    input_scale=input_tensor.qnn_params["scale"],
                    input_zero_point=input_tensor.qnn_params["zero_point"],
                    output_scale=output_tensor.qnn_params["scale"],
                    output_zero_point=output_tensor.qnn_params["zero_point"],
                    out_dtype=output_tensor_type_str,
                )

        return self.block_builder.emit(out)

    def convert_quantize(self, op):
        """Convert TFLite Quantize"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        in_expr = self.get_tensor_expr(input_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())

        # The output must be quantized
        assert output_tensor.qnn_params

        # TFLite Quantize op can also act as Requantize op
        if input_tensor_type_str == "float32":
            out = self.quantize(in_expr, output_tensor)
        else:
            out = relax.op.qdq.requantize(  # TODO: implement
                in_expr,
                input_tensor.qnn_params["scale"],
                input_tensor.qnn_params["zero_point"],
                output_tensor.qnn_params["scale"],
                output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )
        return out

    def convert_dequantize(self, op):
        """Convert TFLite Dequantize"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        if input_tensor.tensor.Type() == TensorType.FLOAT16:
            dtype = self.get_tensor_type_str(input_tensor.tensor.Type())
            input_value = self.get_tensor_value(input_tensor)
            in_expr = self.exp_tab.new_const(
                input_value, dtype=dtype, source_name=input_tensor.tensor.Name()
            )
            out = relay.cast(in_expr, dtype="float32")
            return out

        in_expr = self.get_expr(input_tensor.tensor_idx)

        # The input must be quantized
        assert input_tensor.qnn_params
        # Dequantize the input.
        out = self.dequantize(in_expr, input_tensor)

        return out

    # def _reduce(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     dimensions = self._attr2value(node.attributes["dimensions"])
    #     if node.body is not None:
    #         reducer_op = node.body.blocks[0].operations[0].OPERATION_NAME
    #         assert reducer_op == "stablehlo.add", f"reducer {reducer_op} in reduce is not supported"
    #     return self.block_builder.emit(relax.op.sum(data[0], axis=dimensions))

    # def _reduce_window(self, node: mlir.ir.Operation) -> relax.Expr:
    #     operands = self.retrieve_operands(node)
    #     window_dimensions = self._attr2value(node.attributes["window_dimensions"])
    #     window_dilations = self._attr2value(node.attributes["window_dilations"])

    #     if node.body is not None:
    #         reducer_op = node.body.blocks[0].operations[0].OPERATION_NAME
    #         assert (
    #             reducer_op == "stablehlo.maximum"
    #         ), f"the reducer {reducer_op} in reduce_window is not supported"

    #     pool_size = []
    #     for i, window_dim in enumerate(window_dimensions):
    #         if window_dim == 0:
    #             pool_size.append(0)
    #         else:
    #             dilated_window_size = (window_dim - 1) * window_dilations[i] + 1
    #             pool_size.append(dilated_window_size)
    #     strides = self._attr2value(node.attributes["window_strides"])
    #     # padding = self._attr2value(node.attributes["padding"])

    #     # TODO (yongwww): Infer the layout automatically
    #     layout = "NHWC"

    #     ret = self.block_builder.emit(
    #         relax.op.nn.max_pool2d(
    #             operands[0],
    #             pool_size=pool_size[1:3],  # HW
    #             strides=strides[1:3],
    #             padding=[1, 1],
    #             dilation=window_dilations[1:3],
    #             layout=layout,
    #         )
    #     )
    #     return ret

    def convert_pool2d(self, op, pool_type):
        """pool2d implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Padding import Padding
            from tflite.Pool2DOptions import Pool2DOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        params = {
            "pool_size": (filter_h, filter_w),
            "strides": (stride_h, stride_w),
            "padding": [0, 0],
            "layout": "NHWC",
        }

        in_expr = self.get_expr(input_tensor_idx)

        _, input_h, input_w, _ = to_int_list(self.get_tensor_shape(input_tensor))

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise tvm.error.OpAttributeUnImplemented(
                f"Padding format {padding} for operator Pool2D is not supported."
            )

        if pool_type == "average":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), (
                    "TFLite avg_pool2dreshape requires input and output scale"
                    "and zero points to be equal"
                )
                out = relax.op.astype(in_expr, dtype="int32")
                out = relax.op.nn.avg_pool2d(out, **params)
                out = relax.op.astype(out, dtype=output_tensor_type_str)
            else:
                out = relax.op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(
                    input_tensor, output_tensor
                ), "qnn.op.max_pool2d requires input and output qnn params to be same"
            out = relax.op.nn.max_pool2d(in_expr, **params)
        elif pool_type == "l2":
            # L2_POOL_2D is equivalent to square_root(avg_pool(square(in_data)))
            # TFLite does not have support for quantised L2_POOL_2D op.
            assert (
                not input_tensor.qnn_params
            ), "As TFLite does not have support for quantized L2_POOL_2D, \
                Quantized input is not expected."
            exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
            square_exp = _op.power(in_expr, relax.const(2, exp_type))
            avg_pool_exp = relax.op.nn.avg_pool2d(square_exp, **params)
            out = _op.sqrt(avg_pool_exp)
        else:
            raise tvm.error.OpNotImplemented(
                f"Operator {pool_type} pool is not supported for frontend TFLite."
            )

        # Handle fused activations
        if output_tensor.qnn_params:
            # TODO: avoid QDQ between op and act
            # scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            # zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.dequantize(out, output_tensor)
            # out = self.convert_qnn_fused_activation_function(
            #     expr=out,
            #     fused_activation_fn=fused_activation_fn,
            #     scale=scale_val,
            #     zero_point=zero_point_val,
            #     dtype=output_tensor_type_str,
            # )
            out = self.convert_fused_activation_function(out, fused_activation_fn)
            out = self.quantize(out, output_tensor)
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)
        return self.block_builder.emit(out)

    def convert_average_pool2d(self, op):
        """Convert TFLite average pool2d"""
        return self.convert_pool2d(op, "average")

    # def _rsqrt(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.rsqrt(data[0]))

    # def _sin(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.sin(data[0]))

    # def _sinh(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.sinh(data[0]))

    # def _cos(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.cos(data[0]))

    # def _cosh(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.cosh(data[0]))

    # def _sqrt(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.sqrt(data[0]))

    # def _round(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.round(data[0]))

    # def _exp(self, node: mlir.ir.Operation) -> relax.Expr:
    #     data = self.retrieve_operands(node)
    #     return self.block_builder.emit(relax.op.exp(data[0]))

    # def _return(self, node: mlir.ir.Operation) -> relax.Expr:
    #     outputs = self.retrieve_operands(node)
    #     return self.block_builder.emit_output(self.nodes[outputs])
    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()

        op_c = self.model.OperatorCodes(op_code_list_idx)
        # In TFlite 2.4.x there was a change where the type of the field that contained
        # the builtin code changed from int8 to int32 in the flat buffer representation.
        # However, to retain support for old flat buffers that were created, they retained
        # the original 8 bit field, but named it "deprecated_builtin_code" in TFLite 2.4.
        # This means that the API function BuiltinCode() which originally returned the value
        # of the 8 bit field would now look for the value in the new int32 field in the
        # schema and DeprecatedBuiltinCode() will look at the old 8 bit field.
        # In TFLite 2.4, if the opcode value is less than 127, it can be in either field
        # (however, if it is only in the "builtin_code" field, the model is not backward
        # compatible), so similarly to TFLite 2.4 reader, we'll pick the higher value of the
        # two fields.
        # Remember however that this value came into existence only after Tensorflow
        # lite 2.4.x and hence encase it in a try -except block.
        # Phew !
        try:
            opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
        except AttributeError:
            # In versions before 2.4 the int8 field that holds the builtin code is accessed
            # by BuiltinCode() and DeprecatedBuiltinCode() doesn't exist
            opc = op_c.BuiltinCode()

        op_code_id = opc
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError(
                "TFLite operator with code "
                + str(op_code_id)
                + " is not supported by this version of the fbs schema."
            )
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(op_code_list_idx).CustomCode()

            if self.allow_custom_ops:
                return "CUSTOM"

            if custom_op_code_str == b"TFLite_Detection_PostProcess":
                return "DETECTION_POSTPROCESS"

            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            raise tvm.error.OpAttributeInvalid(
                                "TFLite per-axis quantization restricts all zero points to be"
                                + " 0, but a non-zero value is observed"
                            )
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            f"Quantized type {type(tflite_scale)} (scale) and  "
                            f"{type(tflite_zero_point)} (zero point) not supported"
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(f"Quantized type {type(tflite_scale)} not supported")

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = relax.const(scale, "float32")
                    # TODO: check if in int8 range
                    # qnn_params["zero_point"] = relax.const(zero_point, "int32")
                    print("zpp", zero_point)
                    qnn_params["zero_point"] = relax.const(zero_point, "int8")
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            from tflite.TensorType import TensorType

            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except ImportError:
            raise ImportError("The tflite package must be installed")
        except KeyError:
            raise NotImplementedError(
                f"Tensor type '{tensor_wrapper.tensor.Type()}' currently not supported"
            )

    # pylint: disable=no-else-return
    def get_tensor_value(self, tensor_wrapper, is_sparse=False):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(self.get_tensor_shape(tensor_wrapper))
        else:
            shape = []

        if is_sparse:
            return np.frombuffer(data, dtype=dtype)
        else:
            return np.frombuffer(data, dtype=dtype).reshape(shape)

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.INT16:
            return "int16"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(f"Tensor type {str(tensor_type)} is currently not supported")

    def flatten_to_nd(self, x, x_shape, nd=3):
        """Flatten input tensor to nd rank"""
        ndims = _infer_shape(x_shape)[0]
        if ndims == nd:
            return x
        newshape = _op.concatenate(
            [
                _expr.const([-1], dtype=_infer_type(x_shape).checked_type.dtype),
                _op.strided_slice(x_shape, [ndims - nd + 1], [ndims]),
            ],
            0,
        )
        out = _op.reshape(x, _fold_constant(newshape))
        return out

    def get_tensor_shape(self, tensor_wrapper):
        """Returns tensor shape. Infers shape if the shape is empty."""
        assert isinstance(tensor_wrapper, TensorWrapper), "Expecting TensorWrapper here"
        return (
            tensor_wrapper.tensor.ShapeAsNumpy()
            if tensor_wrapper.tensor.ShapeLength() > 0
            else _infer_shape(self.get_tensor_expr(tensor_wrapper))
        )

    def get_expr(self, input_tensor_idx):
        return self._nodes[get_tensor_name(self.subgraph, input_tensor_idx)]

    def has_expr(self, input_tensor_idx):
        return get_tensor_name(self.subgraph, input_tensor_idx) in self._nodes

    def is_prefetched(self, input_tensor_idx):
        return (
            self.prefetched_nodes.get(get_tensor_name(self.subgraph, input_tensor_idx)) is not None
        )

    def set_prefetched_node(self, input_tensor_idx, value):
        self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)] = value

    def get_prefetched_node(self, input_tensor_idx):
        return self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)]

    def get_tensor_expr(self, tensor, is_sparse=False):
        """Return the relax expr for tensor."""
        if self.has_expr(tensor.tensor_idx):
            expr = self.get_expr(tensor.tensor_idx)
        else:
            type_str = self.get_tensor_type_str(tensor.tensor.Type())
            expr = relax.const(
                self.get_tensor_value(tensor, is_sparse),
                dtype=type_str,
                # source_name=tensor.tensor.Name(),
            )
        return expr

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params["scale"]
        rhs_scale = rhs_tensor.qnn_params["scale"]
        lhs_zero_point = lhs_tensor.qnn_params["zero_point"]
        rhs_zero_point = rhs_tensor.qnn_params["zero_point"]
        # 0.1 + 0.2 != 0.3
        return np.allclose(
            lhs_scale.data.numpy(), rhs_scale.data.numpy(), rtol=1e-5, atol=1e-5
        ) and np.allclose(
            lhs_zero_point.data.numpy(), rhs_zero_point.data.numpy(), rtol=1e-5, atol=1e-5
        )

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

    def quantize(self, expr, tensor_to_quantize, axis=-1):
        """Helper function to quantize a tensor with Relay"""
        print("quantize", expr, tensor_to_quantize)
        print("scale", tensor_to_quantize.qnn_params["scale"], type(tensor_to_quantize.qnn_params["scale"]))
        print("zp", tensor_to_quantize.qnn_params["zero_point"], type(tensor_to_quantize.qnn_params["zero_point"]))
        tensor_type = tensor_to_quantize.tensor.Type()
        tensor_type_str = self.get_tensor_type_str(tensor_type)
        quantized = relax.op.qdq.quantize(
            expr,
            tensor_to_quantize.qnn_params["scale"],
            tensor_to_quantize.qnn_params["zero_point"],
            out_dtype=tensor_type_str,
            axis=axis,
        )
        return quantized

    def dequantize(self, expr, tensor, axis=-1):
        """Helper function to dequantize a tensor with Relay"""
        print("dequantize", expr, tensor)
        print("scale", tensor.qnn_params["scale"], type(tensor.qnn_params["scale"]))
        print("zp", tensor.qnn_params["zero_point"], type(tensor.qnn_params["zero_point"]))
        dequantized = relax.op.qdq.dequantize(
            expr,
            tensor.qnn_params["scale"],
            tensor.qnn_params["zero_point"],
            axis=axis,
        )
        return dequantized

    def create_convert_map(self):
        # TODO: Add more operators
        self.convert_map = {
            # "ABS": self.convert_abs,
            "ADD": self.convert_add,
            # "ADD_N": self.convert_add_n,
            # "ARG_MAX": self.convert_arg_max,
            # "ARG_MIN": self.convert_arg_min,
            "AVERAGE_POOL_2D": self.convert_average_pool2d,
            # "BATCH_TO_SPACE_ND": self.convert_batch_to_space_nd,
            # "BATCH_MATMUL": self.convert_batch_matmul,
            # "CAST": self.convert_cast,
            # "CEIL": self.convert_ceil,
            # "CONCATENATION": self.convert_concatenation,
            "CONV_2D": self.convert_conv2d,
            # "COS": self.convert_cos,
            # "DENSIFY": self.convert_densify,
            # "DEPTH_TO_SPACE": self.convert_depth_to_space,
            "DEPTHWISE_CONV_2D": self.convert_depthwise_conv2d,
            "DEQUANTIZE": self.convert_dequantize,
            # "DETECTION_POSTPROCESS": self.convert_detection_postprocess,
            # "DIV": self.convert_div,
            # "ELU": self.convert_elu,
            # "EQUAL": self.convert_equal,
            # "EXP": self.convert_exp,
            # "EXPAND_DIMS": self.convert_expand_dims,
            # "FAKE_QUANT": self.convert_fake_quant,
            # "FILL": self.convert_fill,
            # "FLOOR_DIV": self.convert_floor_div,
            # "FLOOR_MOD": self.convert_floor_mod,
            # "FLOOR": self.convert_floor,
            # "FULLY_CONNECTED": self.convert_fully_connected,
            # "GATHER": self.convert_gather,
            # "GATHER_ND": self.convert_gather_nd,
            # "GREATER_EQUAL": self.convert_greater_equal,
            # "GREATER": self.convert_greater,
            # "HARD_SWISH": self.convert_hard_swish,
            # "L2_NORMALIZATION": self.convert_l2_normalization,
            # "L2_POOL_2D": self.convert_l2_pool2d,
            # "LEAKY_RELU": self.convert_leaky_relu,
            # "LESS_EQUAL": self.convert_less_equal,
            # "LESS": self.convert_less,
            # "LOCAL_RESPONSE_NORMALIZATION": self.convert_lrn,
            # "LOG": self.convert_log,
            # "LOG_SOFTMAX": self.convert_log_softmax,
            # "LOGICAL_AND": self.convert_logical_and,
            # "LOGICAL_NOT": self.convert_logical_not,
            # "LOGICAL_OR": self.convert_logical_or,
            # "LOGISTIC": self.convert_logistic,
            # "MATRIX_DIAG": self.convert_matrix_diag,
            # "MATRIX_SET_DIAG": self.convert_matrix_set_diag,
            # "MAX_POOL_2D": self.convert_max_pool2d,
            # "MAXIMUM": self.convert_maximum,
            # "MEAN": self.convert_reduce_mean,
            # "MINIMUM": self.convert_minimum,
            # "MIRROR_PAD": self.convert_mirror_pad,
            "MUL": self.convert_mul,
            # "NEG": self.convert_neg,
            # "NOT_EQUAL": self.convert_not_equal,
            # "ONE_HOT": self.convert_one_hot,
            # "PACK": self.convert_pack,
            # "PAD": self.convert_pad,
            # "PADV2": self.convert_pad,
            # "POW": self.convert_pow,
            # "PRELU": self.convert_prelu,
            # "RANGE": self.convert_range,
            "QUANTIZE": self.convert_quantize,
            # "REDUCE_ANY": self.convert_reduce_any,
            # "REDUCE_MAX": self.convert_reduce_max,
            # "REDUCE_MIN": self.convert_reduce_min,
            # "REDUCE_PROD": self.convert_reduce_prod,
            # "RELU": self.convert_relu,
            # "RELU6": self.convert_relu6,
            # "RELU_N1_TO_1": self.convert_relu_n1_to_1,
            "RESHAPE": self.convert_reshape,
            # "RESIZE_BILINEAR": self.convert_resize_bilinear,
            # "RESIZE_NEAREST_NEIGHBOR": self.convert_resize_nearest_neighbor,
            # "ROUND": self.convert_round,
            # "RSQRT": self.convert_rsqrt,
            # "REVERSE_SEQUENCE": self.convert_reverse_sequence,
            # "REVERSE_V2": self.convert_reverse_v2,
            # "SELECT": self.convert_select,
            # "SHAPE": self.convert_shape,
            # "SIN": self.convert_sin,
            # "SLICE": self.convert_slice,
            "SOFTMAX": self.convert_softmax,
            # "SPACE_TO_BATCH_ND": self.convert_space_to_batch_nd,
            # "SPACE_TO_DEPTH": self.convert_space_to_depth,
            # "SPARSE_TO_DENSE": self.convert_sparse_to_dense,
            # "SPLIT": self.convert_split,
            # "SPLIT_V": self.convert_split_v,
            # "SQRT": self.convert_sqrt,
            # "SQUARE": self.convert_square,
            # "SQUARED_DIFFERENCE": self.convert_squared_difference,
            # "SQUEEZE": self.convert_squeeze,
            # "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": self.convert_sub,
            # "SUM": self.convert_reduce_sum,
            # "TAN": self.convert_tan,
            # "TANH": self.convert_tanh,
            # "TILE": self.convert_tile,
            # "TOPK_V2": self.convert_topk_v2,
            # "TRANSPOSE_CONV": self.convert_transpose_conv,
            # "TRANSPOSE": self.convert_transpose,
            # "UNPACK": self.convert_unpack,
            # "UNIDIRECTIONAL_SEQUENCE_LSTM": self.convert_unidirectional_sequence_lstm,
            # "WHERE": self.convert_select,
            # "ZEROS_LIKE": self.convert_zeros_like,
            # "NON_MAX_SUPPRESSION_V5": self.convert_nms_v5,
        }

    def check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()
        dynamic_range_ops_set = set()
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)
                continue

            # Trying to exclude "dynamic range quantization" optimized ops as not supported in TVM
            qnn_in_cnt = len(
                [_.qnn_params for _ in self.get_input_tensors(op)[0:1] if _.qnn_params is not None]
            )
            qnn_weight_cnt = len(
                [_.qnn_params for _ in self.get_input_tensors(op)[1:] if _.qnn_params is not None]
            )
            qnn_out_cnt = len(
                [_.qnn_params for _ in self.get_output_tensors(op) if _.qnn_params is not None]
            )

            if qnn_in_cnt == 0 and qnn_out_cnt == 0 and qnn_weight_cnt > 0:
                dynamic_range_ops_set.add(op_code_str)

        raise_msg = ""

        if unsupported_ops_set:
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise_msg += f"The following operators are not supported in frontend TFLite: {ops}\n"

        if dynamic_range_ops_set:
            ops = str(list(dynamic_range_ops_set)).strip("[,]")
            raise_msg += (
                f"The following operators are likely to have dynamic range quantization: {ops}. "
                f"If you are running an optimized graph, please turn off dynamic range "
                f"quantization or use full integer quantization"
            )

        if len(raise_msg) > 0:
            raise tvm.error.OpNotImplemented(raise_msg)

    def convert_op_to_relax(self):
        """Convert TFLite ops to relax ops"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)
            try:
                from tflite.Operator import Operator
            except ImportError:
                raise ImportError("The tflite package must be installed")

            assert isinstance(op, Operator)
            # ret = self.convert_map[op_code_str](op)
            ret = self.convert_map[op_code_str](op)
            self._nodes[op] = ret

            # In case the Op can be prefetched, the output can be optimized out
            if ret is None:
                continue

            output_names = ", ".join(
                [get_tensor_name(self.subgraph, tensor.tensor_idx) for tensor in output_tensors]
            )
            # ret = ret

            # print("self._nodes", self._nodes)
            if len(output_tensors) == 1:
                tensor_idx = output_tensors[0].tensor_idx
                self._nodes[get_tensor_name(self.subgraph, tensor_idx)] = ret
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    self._nodes[get_tensor_name(self.subgraph, output_tensor.tensor_idx)] = ret[idx]

    def from_tflite(
        self,
        model,
        shape_dict: Optional[Dict[str, List]] = None,
        dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
    ) -> tvm.IRModule:
        """Convert a TFLite Module to a Relax program.

        Parameters
        ----------
        model : ?
            The TFLite model to convert.

        shape_dict : dict of str to tuple, optional
            The input shape to the graph

        dtype_dict : str or dict of str to str, optional
            The input types to the graph


        Returns
        -------
        output : tvm.IRModule
            The result IRModule with entry function "main"
        """
        print("from_tflite2", model, shape_dict, dtype_dict)
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")
        # TODO: move to constructor
        self.model = model
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())

        # assert isinstance(model, mlir.ir.Module)

        # inputs of the function
        inputs = []
        _shape_dict, _dtype_dict = _input_type(model)
        if shape_dict is not None:
            _shape_dict.update(shape_dict)
        if dtype_dict is not None:
            _dtype_dict.update(dtype_dict)
        # for idx, arg in enumerate(block.arguments.types):

        # keep the same as tflite
        assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"
        self.subgraph = model.Subgraphs(0)

        # model inputs / outputs
        model_inputs = self.subgraph.InputsAsNumpy()
        model_outputs = self.subgraph.OutputsAsNumpy()
        for model_input in model_inputs:
            model_input_name = get_tensor_name(self.subgraph, model_input)
            shape = _shape_dict[model_input_name] if model_input_name in _shape_dict else None
            dtype = _dtype_dict[model_input_name] if model_input_name in _dtype_dict else "float32"
            var = relax.Var(model_input_name, relax.TensorStructInfo(shape, dtype))
            self._nodes[model_input_name] = var
            inputs.append(var)

        # TODO (yongwww): Handle mlir.ir.Module with multiple functions
        # Initialize the block builder with a function and a dataflow block.
        # Raise error if the input stablehlo op is impure
        func_name = "main"
        self.block_builder = relax.BlockBuilder()
        # self.check_unsupported_ops()

        with self.block_builder.function(name=func_name, params=inputs.copy()):
            output = None
            with self.block_builder.dataflow():
                self.convert_op_to_relax()
                outputs = [self._nodes[get_tensor_name(self.subgraph, i)] for i in model_outputs]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                output = self.block_builder.emit_output(outputs)
            #     pass
            #     output = relax.Var("x", relax.TensorStructInfo((16, 16), "float32"))
            #     output = self.block_builder.emit_output(output)
            #     # block = model.body.operations[0].regions[0].blocks[0]
            #     # for operation in block.operations:
            #     #     if isinstance(operation, (mlir.dialects.func.ReturnOp, stablehlo.ReturnOp)):
            #     #         operation = operation.operands[0].owner
            #     #         # TODO (yongwww): handle multiple outputs
            #     #         output = self.block_builder.emit_output(self._nodes[operation])
            #     #         break

            #     #     if isinstance(operation, mlir.ir.OpView):
            #     #         op_name = operation.operation.name
            #     #         assert op_name in self.convert_map, f"Unsupported operation {op_name}"
            #     #         self._nodes[operation] = self.convert_map[op_name](operation)
            #     #     else:
            #     #         raise ValueError(f"Unsupported op {operation}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        mod.show()
        return mod


def from_tflite(
    tflite_module,
    shape_dict: Optional[Dict[str, List]] = None,
    dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
) -> tvm.IRModule:
    """Convert a TFLite Module to a Relax program

    Parameters
    ----------
    tflite_module : TODO
        The TFLite model to convert.

    shape_dict : dict of str to tuple, optional
        The input shape to the graph

    dtype_dict : str or dict of str to str, optional
        The input types to the graph

    Returns
    -------
    output : tvm.IRModule
        The result IRModule with entry function "main"
    """

    return TFLiteImporter().from_tflite(tflite_module, shape_dict=shape_dict, dtype_dict=dtype_dict)
