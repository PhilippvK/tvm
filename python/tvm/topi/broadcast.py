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
"""Broadcast operators"""
from __future__ import absolute_import as _abs
from . import cpp as _cpp


def broadcast_to(data, shape):
    """Broadcast the src to the target shape

    We follows the numpy broadcasting rule.
    See also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data

    shape : list or tuple
        The target shape to be broadcasted.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return _cpp.broadcast_to(data, shape)


def add(lhs, rhs):
    """Addition with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.add(lhs, rhs)

from tvm import te, tir, ir

def add2_(inputs, attrs, out_dtype, test):
    A, B = inputs
    print("(A,B) =",A,",", B)
    print("inputs =",inputs)
    attrs_dict = { i.name: attrs[i.name] for i in attrs.list_field_info()}
    print("attrs =",attrs_dict)
    print("out_dtype =",out_dtype)
    print("test =",test)
    print(type(A.shape))
    input(">>>")
   
    dtype = A.dtype
    shape = A.shape
    C1 = te.placeholder(A.shape, name="c1", dtype=dtype)
    C2 = te.placeholder(A.shape, name="c2", dtype=dtype)
    outputs = [C1, C2]
    
    #A_buf = tir.decl_buffer(shape, dtype)
    #B_buf = tir.decl_buffer(shape, dtype)
    #C_buf = tir.decl_buffer(shape, dtype)


    #ret = te.extern(shape, [A, B], lambda ins, outs: tir.call_extern(dtype, "test_tflite_custom", ins[0].access_ptr("r"), ins[1].access_ptr("r"), outs[0].access_ptr("rw")), name="C")
    #ret = te.extern(shape, [A, B], lambda ins, outs: tir.call_extern(dtype, f"tflite_custom_{attrs_dict['name']}", ins[0].access_ptr("r"), ins[1].access_ptr("r"), outs[0].access_ptr("rw"), outs[0].access_ptr("rw"), str(shape), [dtype]), name="C")
    #ret = te.extern(shape, [A, B], lambda ins, outs: tir.call_extern(dtype, f"tflite_custom_{attrs_dict['name']}", ins[0].access_ptr("r"), ins[1].access_ptr("r"), outs[0].access_ptr("rw"), outs[0].access_ptr("rw"), str(shape), dtype), name="C")
    #rets = list(te.extern([shape, shape], [A, B], lambda ins, outs: tir.call_packed("tvm.runtime.tflite_custom_op", ins[0], ins[1], outs[0], dtype), name="C"))
    rets = list(te.extern([o.shape for o in outputs], inputs, lambda ins, outs: tir.call_packed("tvm.runtime.tflite_custom_op", len(ins), len(outs), *ins, *outs, 3, "\x01\x02\x03"), name="C"))
    #rets = list(te.extern([o.shape for o in outputs], inputs, lambda ins, outs: tir.call_extern("float32", "tvm.runtime.tflite_custom_op", len(ins), len(outs), *ins, *outs, 3, "\x01\x02\x03"), name="C"))

    #print("T: ", T)
    #print("T2: ", T2)
    print("rets: ", rets)
    print("ret1: ", rets[0])
    print("ret2: ", rets[1])
    #input(">>>")

    #return C
                        #aa.access_ptr("r"),
                    #bb.access_ptr("r"),
                    #cc.access_ptr("w"),
                    #aa.strides[0],
                    #bb.strides[0],
                    #cc.strides[0],
    #A = te.placeholder((1, 2), name="A")
    #B = te.placeholder((1, 2), name="B")
    #C = te.extern((1, 2), [lhs, rhs],
    #           lambda ins, outs: tir.call_packed(
    #              "tvm.contrib.cblas.matmul2",
    #                ins[0], ins[1], outs[0], 0, 0), name="C")

    #C = te.extern((1, 2), [lhs, rhs], lambda ins, outs: tir.call_extern("float32", "test_tflite_custom", 1, 2), name="C")
    #return _cpp.add(lhs, rhs)
    return rets[0]


def subtract(lhs, rhs):
    """Subtraction with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """Multiplication with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.multiply(lhs, rhs)


def divide(lhs, rhs):
    """Division with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.divide(lhs, rhs)


def floor_divide(lhs, rhs):
    """Floor division with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.floor_divide(lhs, rhs)


def mod(lhs, rhs):
    """Modulus with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.mod(lhs, rhs)


def floor_mod(lhs, rhs):
    """Floor modulus with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.floor_mod(lhs, rhs)


def maximum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.minimum(lhs, rhs)


def power(lhs, rhs):
    """Power with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.power(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.left_shift(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.right_shift(lhs, rhs)


def greater(lhs, rhs):
    """Compute (lhs>rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater(lhs, rhs)


def less(lhs, rhs):
    """Compute (lhs<rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less(lhs, rhs)


def equal(lhs, rhs):
    """Compute (lhs==rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Compute (lhs!=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.not_equal(lhs, rhs)


def greater_equal(lhs, rhs):
    """Compute (lhs>=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater_equal(lhs, rhs)


def less_equal(lhs, rhs):
    """Compute (lhs<=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less_equal(lhs, rhs)


def logical_and(lhs, rhs):
    """Compute element-wise logical and of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.logical_and(lhs, rhs)


def logical_or(lhs, rhs):
    """Compute element-wise logical or of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.logical_or(lhs, rhs)


def logical_xor(lhs, rhs):
    """Compute element-wise logical xor of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.logical_xor(lhs, rhs)


def bitwise_and(lhs, rhs):
    """Compute element-wise bitwise and of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.bitwise_and(lhs, rhs)


def bitwise_or(lhs, rhs):
    """Compute element-wise bitwise or of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.bitwise_or(lhs, rhs)


def bitwise_xor(lhs, rhs):
    """Compute element-wise bitwise xor of data.

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
          The left operand
    rhs : tvm.te.Tensor or Expr
          The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if both operands are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.bitwise_xor(lhs, rhs)


def logical_not(data):
    """Compute element-wise logical not of data.

    Parameters
    ----------
    data : tvm.te.Tensor or Expr

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if the operand are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.logical_not(data)


def bitwise_not(data):
    """Compute element-wise bitwise not of data.

    Parameters
    ----------
    data : tvm.te.Tensor or Expr

    Returns
    -------
    ret : tvm.te.Tensor or Expr
          Returns Expr if the operand are Expr.
          Otherwise returns Tensor.
    """
    return _cpp.bitwise_not(data)
