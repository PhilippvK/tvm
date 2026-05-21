import numpy as np
import tvm
from tvm import relay, te
from tvm.topi.nn.utils import get_pad_tuple

def get_dense_relay_module(dim: int, dtype: str):
    print("get_dense_relay_module", dim, dtype)
    data_shape = (dim, dim)
    weight_shape = (dim, dim)
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    out_dtype = "int32" if dtype in ["int8"] else dtype
    out = relay.nn.dense(data, weight, out_dtype=out_dtype)
    mod = tvm.IRModule.from_expr(out)
    mod = relay.transform.InferType()(mod)
    mod = mod.with_attr("executor", relay.backend.Executor("graph", {"link-params": True}))
    weight_np = np.random.randn(*weight_shape).astype(dtype)
    data_np = np.random.randn(*data_shape).astype(dtype)
    params = {"weight": weight_np}
    return mod, params


def make_shape(layout: str, sizes: dict[str, int]) -> tuple[int, ...]:
    return tuple(sizes[c] for c in layout)

def get_conv2d_relay_module(h: int, w: int, kw: int, kh: int, cin: int, cout: int, dtype: str, data_layout: str, kernel_layout: str):
    print("get_conv2d_relay_module", h, w, kw, kh, cin, cout, dtype, data_layout, kernel_layout)
    sizes_data = {
        "N": 1,
        "C": cin,
        "H": h,
        "W": w,
    }
    sizes_weight = {
        "O": cout,
        "I": cin,
        "H": kh,
        "W": kw,
    }
    # data_shape = (1, cin, h, w)
    # weight_shape = (cout, cin, kh, kw)
    data_shape = make_shape(data_layout, sizes_data)
    weight_shape = make_shape(kernel_layout, sizes_weight)
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    out_dtype = "int32" if dtype in ["int8"] else dtype
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple("SAME", (kh, kw))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(pad_top, pad_left, pad_down, pad_right),
        channels=cout,
        kernel_size=(kw, kw),
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_dtype=out_dtype,
    )
    # f = relay.Function([data, weight], y)
    # mod = tvm.IRModule.from_expr(f)
    mod = tvm.IRModule.from_expr(y)
    mod = relay.transform.InferType()(mod)
    mod = mod.with_attr("executor", relay.backend.Executor("graph", {"link-params": True}))
    weight_np = np.random.randn(*weight_shape).astype(dtype)
    data_np = np.random.randn(*data_shape).astype(dtype)
    params = {"weight": weight_np}
    return mod, params
