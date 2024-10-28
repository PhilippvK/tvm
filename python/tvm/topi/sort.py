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
# pylint: disable=too-many-arguments
"""Argsort operator"""
import tvm
from tvm import te
from .utils import get_const_tuple
from ..tir import ir_builder
from .math import cast


CUSTOM_QSORT = False


def sort(data, axis=-1, is_ascend=1):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)
    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.sort.sort", ins[0], outs[0], axis, is_ascend),
        dtype=data.dtype,
        in_buffers=[data_buf],
        out_buffers=out_buf,
        name="sort_cpu",
        tag="sort_cpu",
    )
    return out


def argsort_qsort_ir(ib, sorter, sorter_size, max_sorter_size, is_ascend, ret_type="indices"):
    sorted_indexes = ib.allocate("float32", (max_sorter_size,), name="sortedindexes_qsort", scope="global")
    i = ib.allocate("int32", (1,), name="i_qsort", scope="global")
    i[0] = cast(0, "int32")

    ib.emit(
        tvm.tir.call_extern(
            "",
            "qsort",
            sorter.asobject().data,
            sorter_size,
            8,
            tvm.tir.call_extern("", "ARGSORT_CMPFUNC"),
        )
    )

    if ret_type == "indices":
        with ib.while_loop(i[0] < sorter_size):
            sorted_indexes[i[0]] = cast(sorter[i[0], 1], "float32")
            i[0] = i[0] + 1

        return sorted_indexes
    elif ret_type == "both":
        return sorter
    else:
        return None


def argsort_te_old(ib, sorter, sorter_size, is_ascend):
    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")
    current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")
    sorted_indexes = ib.allocate("int32", (sorter_size,), name="sortedindexes", scope="local")

    current_max = ib.allocate("float32", (1,), name="current_max", scope="local")
    current_min = ib.allocate("float32", (1,), name="current_min", scope="local")

    if not is_ascend:
        # First, get the maximum value
        current_max[0] = sorter[0, 0]
        current_sortedindex[0] = cast(0, "int32")
        lo[0] = cast(1, "int32")
        hi[0] = cast(sorter_size, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(sorter[lo[0], 0] > current_max[0]):
                current_max[0] = sorter[lo[0], 0]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1
    else:
        # First, get the minimum value
        current_min[0] = cast(0x7FFFFFFF, "float32")
        current_sortedindex[0] = cast(0, "int32")
        lo[0] = cast(0, "int32")
        hi[0] = cast(sorter_size, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(sorter[lo[0], 0] < current_min[0]):
                current_min[0] = sorter[lo[0], 0]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1

    # Insert this into the first position of the output
    sorted_indexes[0] = current_sortedindex[0]

    # Now get all the other indexes
    lo[0] = cast(1, "int32")
    with ib.while_loop(lo[0] < hi[0]):
        lo_inner[0] = cast(0, "int32")
        current_max[0] = cast(0, "float")
        current_min[0] = cast(0x7FFFFFFF, "float")
        current_sortedindex[0] = cast(0, "int32")
        with ib.while_loop(lo_inner[0] < hi[0]):
            if not is_ascend:
                with ib.if_scope(sorter[lo_inner[0], 0] > current_max[0]):
                    with ib.if_scope(sorter[lo_inner[0], 0] < sorter[sorted_indexes[lo[0] - 1], 0]):
                        current_sortedindex[0] = lo_inner[0]
                        current_max[0] = sorter[lo_inner[0], 0]
            else:
                with ib.if_scope(sorter[lo_inner[0], 0] < current_min[0]):
                    with ib.if_scope(sorter[lo_inner[0], 0] > sorter[sorted_indexes[lo[0] - 1], 0]):
                        current_sortedindex[0] = lo_inner[0]
                        current_min[0] = sorter[lo_inner[0], 0]

            lo_inner[0] = lo_inner[0] + 1

        sorted_indexes[lo[0]] = current_sortedindex[0]
        lo[0] = lo[0] + 1

    return sorted_indexes


def argsort_nms_ir(data, valid_count=None, out=None, axis=-1, is_ascend=False):
    """
    Very naive, very ugly implementation of an argsort
    TODO: improve this!
    """
    # breakpoint()
    # TODO (FP): implement the use of valid_count!
    # TODO (FP): implement the use of is_ascend!

    ib = ir_builder.create()

    data_shape = data.shape
    data_dtype = data.dtype

    out = ib.buffer_ptr(out)
    data = ib.buffer_ptr(data)
    if valid_count:
        valid_count_buff = ib.buffer_ptr(valid_count)

    # lo = ib.allocate("int32", (1,), name="lo", scope="local")
    # hi = ib.allocate("int32", (1,), name="hi", scope="local")
    # lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")
    # axis_buff = ib.allocate("int32", (1,), name="axis_buff", scope="local")
    # axis_buff[0] = axis
    # axis_mul_before = ib.allocate("int32", (1,), name="axis_mul_before", scope="local")
    # axis_mul_after = ib.allocate("int32", (1,), name="axis_mul_after", scope="local")

    i = ib.allocate("int32", (1,), name="i_argsort", scope="global")
    j = ib.allocate("int32", (1,), name="j_argsort", scope="global")
    k = ib.allocate("int32", (1,), name="k_argsort", scope="global")
    current_sort_num = ib.allocate("int32", (1,), name="current_sort_num", scope="global")
    base_idx = ib.allocate("int32", (1,), name="base_idx", scope="local")
    sorter = ib.allocate("float32", (data_shape[axis], 2), name="sorter", scope="global")

    # current_max = ib.allocate(data_dtype, (1,), name="current_max", scope="local")
    # current_min = ib.allocate(data_dtype, (1,), name="current_min", scope="local")
    # current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")

    # hi[0] = cast(data_shape[axis], "int32")
    # current_sortedindex[0] = cast(0, "int32")

    # TODO (Improve this!)
    # axis_mul_before[0] = cast(1, "int32")
    # axis_mul_after[0] = cast(1, "int32")

    mul_bef = 1
    mul_aft = 1
    for i_dim in range(len(data_shape)):
        if i_dim < axis:
            mul_bef *= data_shape[i_dim]
        elif i_dim > axis:
            mul_aft *= data_shape[i_dim]

    axis_mul_before = ib.let("axis_mul_before", tvm.tir.const(int(mul_bef), "int32"))
    axis_mul_after = ib.let("axis_mul_after", tvm.tir.const(int(mul_aft), "int32"))

    i[0] = cast(0, "int32")

    with ib.while_loop(i[0] < axis_mul_before):
        j[0] = cast(0, "int32")
        with ib.while_loop(j[0] < axis_mul_after):
            # Get the number of valid bboxes!
            if valid_count:
                current_sort_num[0] = valid_count_buff[i[0] * axis_mul_after + j[0]]
            else:
                current_sort_num[0] = data_shape[axis]

            # Clean sorter
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                with ib.if_scope(k[0] < current_sort_num[0]):
                    sorter[k[0], 0] = cast(0, "float32")
                with ib.else_scope():
                    sorter[k[0], 0] = cast(-1, "float32")
                sorter[k[0], 1] = cast(k[0], "float32")
                k[0] += 1

            # Fill sorter
            base_idx = i[0] * data_shape[axis] * axis_mul_after + j[0]
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < current_sort_num[0]):
                sorter[k[0], 0] = data[base_idx + k[0] * axis_mul_after]
                sorter[k[0], 1] = cast(k[0], "float32")
                k[0] += 1

            # Actual sort
            sorted_indexes = argsort_qsort_ir(
                ib,
                sorter,
                current_sort_num[0],
                # data_shape[axis],
                data_shape[axis],
                is_ascend,
            )

            # Assign to output
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                with ib.if_scope(current_sort_num[0] > k[0]):
                    out[base_idx + k[0] * axis_mul_after] = cast(sorted_indexes[k[0]], "int32")
                with ib.else_scope():
                    out[base_idx + k[0] * axis_mul_after] = k[0]
                k[0] += 1

            j[0] += 1
        i[0] += 1

    return ib.get()


def argsort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    valid_count : tvm.te.Tensor, optional
        1-D tensor for valid number of boxes.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        axis = 0
        is_ascend = False
        out = argsort(data, axis=axis, is_ascend=is_ascend)
        np_data = np.random.uniform(dshape)
        s = topi.generic.schedule_argsort(out)
        f = tvm.build(s, [data, out], "llvm")
        dev = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, dev)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), dev)
        f(tvm_data, tvm_out)
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    target = tvm.target.Target.current()
    if valid_count is not None:
        valid_count_buf = tvm.tir.decl_buffer(valid_count.shape, valid_count.dtype, "valid_count_buf", data_alignment=4)
        out_buf = tvm.tir.decl_buffer(data.shape, "int32", "out_buf", data_alignment=8)
        # out = argsort_nms_te(data_buf, valid_count_buf, out_buf, axis, is_ascend)
        out = te.extern(
            data.shape,
            [data, valid_count],
            lambda ins, outs: (
                argsort_nms_ir(
                    data=ins[0],
                    valid_count=ins[1],
                    out=outs[0],
                    axis=axis,
                    is_ascend=is_ascend,
                    # ) if target.device_name == "gemmini" else tvm.tir.call_packed(
                )
                if CUSTOM_QSORT
                else tvm.tir.call_packed("tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend)
            ),
            dtype="int32",
            in_buffers=[data_buf, valid_count_buf],
            out_buffers=out_buf,
            name="argsort_nms_cpu",
            tag="argsort_nms_cpu",
        )
    else:
        out_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        # out = argsort_nms_te(data,valid_count,axis,is_ascend)
        out = te.extern(
            data.shape,
            [data],
            lambda ins, outs: (
                argsort_nms_ir(data=ins[0], out=outs[0], axis=axis, is_ascend=is_ascend)
                # if target.device_name == "gemmini"
                if CUSTOM_QSORT
                else tvm.tir.call_packed("tvm.contrib.sort.argsort", ins[0], outs[0], axis, is_ascend)
            ),
            dtype=dtype,
            in_buffers=[data_buf],
            out_buffers=out_buf,
            name="argsort_cpu",
            tag="argsort_cpu",
        )
    return out


def topk_ir(data, outputs, topk_to_ret, axis, ret_type, is_ascend):
    ib = ir_builder.create()

    data_shape = data.shape
    data_dtype = data.dtype

    if ret_type == "both":
        value_buf = ib.buffer_ptr(outputs[0])
        indices_buf = ib.buffer_ptr(outputs[1])
    elif ret_type == "values":
        value_buf = ib.buffer_ptr(outputs[0])
    elif ret_type == "indices":
        indices_buf = ib.buffer_ptr(outputs[0])

    data = ib.buffer_ptr(data)

    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")
    lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")
    axis_buff = ib.allocate("int32", (1,), name="axis_buff", scope="local")
    axis_buff[0] = axis
    axis_mul_before = ib.allocate("int32", (1,), name="axis_mul_before", scope="local")
    axis_mul_after = ib.allocate("int32", (1,), name="axis_mul_after", scope="local")

    i = ib.allocate("int32", (1,), name="i", scope="local")
    j = ib.allocate("int32", (1,), name="j", scope="local")
    k = ib.allocate("int32", (1,), name="k", scope="local")
    current_sort_num = ib.allocate("int32", (1,), name="current_sort_num", scope="local")
    base_idx = ib.allocate("int32", (1,), name="base_idx", scope="local")
    sorter = ib.allocate("float32", (data_shape[axis], 2), name="sorter", scope="local")

    # current_max = ib.allocate(data_dtype, (1,), name="current_max", scope="local")
    # current_min = ib.allocate(data_dtype, (1,), name="current_min", scope="local")
    # current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")

    # hi[0] = cast(data_shape[axis], "int32")
    # current_sortedindex[0] = cast(0, "int32")

    # TODO (Improve this!)
    # axis_mul_before[0] = cast(1, "int32")
    # axis_mul_after[0] = cast(1, "int32")

    mul_bef = 1
    mul_aft = 1
    for i_dim in range(len(data_shape)):
        if i_dim < axis:
            mul_bef *= data_shape[i_dim]
        elif i_dim > axis:
            mul_aft *= data_shape[i_dim]

    axis_mul_before[0] = mul_bef
    axis_mul_after[0] = mul_aft

    i[0] = cast(0, "int32")

    with ib.while_loop(i[0] < axis_mul_before[0]):
        j[0] = cast(0, "int32")
        with ib.while_loop(j[0] < axis_mul_after[0]):
            # Get the number of valid bboxes!
            current_sort_num[0] = data_shape[axis]

            # Clean sorter
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                with ib.if_scope(k[0] < current_sort_num[0]):
                    sorter[k[0], 0] = cast(0, "float32")
                with ib.else_scope():
                    sorter[k[0], 0] = cast(-1, "float32")
                sorter[k[0], 1] = cast(k[0], "float32")
                k[0] += 1

            # Fill sorter
            base_idx = i[0] * data_shape[axis] * axis_mul_after[0] + j[0]
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < current_sort_num[0]):
                sorter[k[0], 0] = data[base_idx + k[0] * axis_mul_after[0]]
                sorter[k[0], 1] = cast(k[0], "float32")
                k[0] += 1

            # Actual sort
            sorted_indexes = argsort_qsort_ir(
                ib, sorter, current_sort_num[0], data_shape[axis], is_ascend, ret_type=ret_type
            )

            # Assign to output
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                with ib.if_scope(topk_to_ret > k[0]):
                    if ret_type == "values":
                        value_buf[base_idx + k[0] * axis_mul_after[0]] = sorted_indexes[k[0]]
                    elif ret_type == "indices":
                        indices_buf[base_idx + k[0] * axis_mul_after[0]] = sorted_indexes[k[0]]
                    else:
                        # ret_type == "both"
                        value_buf[base_idx + k[0] * axis_mul_after[0]] = sorted_indexes[k[0], 0]
                        indices_buf[base_idx + k[0] * axis_mul_after[0]] = cast(sorted_indexes[k[0], 1], "int32")

                # with ib.else_scope():
                #    if ret_type == "values":
                #        value_buf[base_idx + k[0]*axis_mul_after[0]] = k[0]
                #    elif ret_type == "indices":
                #        indices_buf[base_idx + k[0]*axis_mul_after[0]] = k[0]
                #    else:
                #        # ret_type == "both"
                #        out[base_idx + k[0]*axis_mul_after[0]] = k[0]
                #        out[base_idx + k[0]*axis_mul_after[0]] = k[0]

                k[0] += 1

            j[0] += 1
        i[0] += 1

    return ib.get()


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_shape = list(get_const_tuple(data.shape))
    kvar = tvm.te.size_var("k")
    if not isinstance(k, int):
        out_shape[axis] = kvar
    elif k >= 1:
        out_shape[axis] = k
    out_bufs = []
    if ret_type in ["both", "values"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, data.dtype, "value_buf", data_alignment=8))
    if ret_type in ["both", "indices"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, dtype, "indices_buf", data_alignment=8))
    out_shapes = [out_shape] * len(out_bufs)

    kv = kvar if not isinstance(k, int) else k
    target = tvm.target.Target.current()
    out = te.extern(
        out_shapes,
        [data],
        lambda ins, outs: (
            topk_ir(ins[0], outs, kv, axis, ret_type, is_ascend)
            # if target.device_name == "gemmini"
            if CUSTOM_QSORT
            else tvm.tir.call_packed("tvm.contrib.sort.topk", ins[0], *outs, kv, axis, ret_type, is_ascend)
        ),
        in_buffers=[data_buf],
        out_buffers=out_bufs,
        name="topk_cpu",
        tag="topk_cpu",
    )
    return out
