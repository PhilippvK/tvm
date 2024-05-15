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

# pylint: disable=c-extension-no-member

import os
import functools
from typing import Union, Tuple, List
import pytest
import numpy as np
from packaging import version as package_version

from PIL import Image

import tvm
import tvm.testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.download import download_testdata
from tvm import relax, relay
from tvm.contrib import graph_executor
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.frontend.tflite import from_tflite

try:
    import tensorflow.compat.v1 as tf

    # tensorflow.python.framework.ops module itself is not part of
    # TensorFlow's public API: the precise contents of that module
    # may vary from one version to the next
    import tensorflow.compat.v1 as ops
except ImportError:
    import tensorflow as tf
    import tensorflow as ops

try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper


# TODO: share code with tflite relay frontend
#######################################################################
# Generic run functions for TVM & TFLite
# --------------------------------------
def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x
#######################################################################
# Get a real image for e2e testing
# --------------------------------
def get_real_image(im_height, im_width, quantized=True):
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8") if quantized else np.array(image).astype("float32")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def pre_processed_image(height, width):
    """Image preprocessed"""
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    with tf.name_scope("eval_image"):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to the specified height and width.
    image = tf.image.resize(image, [height, width], align_corners=False)
    image = tf.expand_dims(image, axis=0)
    return image

def run_tvm_graph(
    tflite_model_buf,
    input_data,
    input_node,
    num_output=1,
    target="llvm",
    out_names=None,
    mode="graph_executor",
    op_converter=relay.frontend.tflite.OperatorConverter,
):
    """Generic function to compile on relay and execute on tvm"""
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, node in enumerate(input_node):
        shape_dict[node] = input_data[i].shape
        dtype_dict[node] = input_data[i].dtype.name

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict, op_converter=op_converter
        )
    with tvm.testing.enable_span_filling():
        mod_with_span, _ = relay.frontend.from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict, op_converter=op_converter
        )
    assert tvm.ir.structural_equal(mod["main"], mod_with_span["main"])

    if mode in ["debug", "vm"]:
        inputs = []
        for param in mod["main"].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = relay.create_executor(mode, mod=mod, device=tvm.cpu(), target="llvm").evaluate()(
            *inputs
        )
        return vmobj_to_list(result)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)

        dev = tvm.device(target, 0)

        m = graph_executor.GraphModule(lib["default"](dev))
        # set inputs
        for i, node in enumerate(input_node):
            m.set_input(node, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(
            out_names
        ), f"out_names: {out_names} num_output: {num_output}"
        tvm_output_list = []
        for i in range(0, num_output):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list

def run_tflite_graph(tflite_model_buf, input_data):
    """Generic function to execute TFLite"""
    input_data = convert_to_list(input_data)

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_detail in enumerate(input_details):
        interpreter.resize_tensor_input(input_detail["index"], input_data[i].shape)
    interpreter.allocate_tensors()

    # set input
    assert len(input_data) == len(input_details)
    for i, input_detail in enumerate(input_details):
        interpreter.set_tensor(input_detail["index"], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = []
    for _, output_detail in enumerate(output_details):
        tflite_output.append(interpreter.get_tensor(output_detail["index"]))

    return tflite_output


# def generate_np_inputs(
#     input_shapes: Union[Tuple, List[Tuple]], dtype: str = "float32"
# ) -> Union[np.ndarray, List[np.ndarray]]:
#     """Generate numpy data as the inputs of model
#
#     Parameters
#     ----------
#     input_shapes: Union[Tuple, List[Tuple]]
#         shapes for inputs
#     dtype: str
#         the data type of inputs
#
#     Results
#     -------
#     out: List[np.ndarray]
#         numpy input data
#     """
#     if not isinstance(input_shapes[0], (list, tuple)):
#         return [np.random.uniform(size=input_shapes).astype(dtype)]
#     out = []
#     for input_shape in input_shapes:
#         out.append(np.random.uniform(size=input_shape).astype(dtype))
#     return out
#
#
# def np2jnp(inputs_np: Union[np.ndarray, List[np.ndarray]]):
#     """Convert data from numpy to jax.numpy
#
#     Parameters
#     ----------
#     inputs_np: Union[np.ndarray, List[np.ndarray]]
#         numpy input data
#
#     Results
#     -------
#     out: Union[jnp.ndarray, List[jnp.ndarray]]
#         jax numpy data
#     """
#     import jax.numpy as jnp
#
#     # Use jnp.asarray to avoid unnecessary memory copies
#     inputs_jnp = []
#     if isinstance(inputs_np, (tuple, list)):
#         for input_np in inputs_np:
#             inputs_jnp.append(jnp.asarray(input_np))
#         return inputs_jnp
#     return jnp.asarray(inputs_np)
#
#
# def check_correctness(
#     jax_jit_mod,
#     input_shapes: Union[Tuple, List[Tuple]],
#     dtype: str = "float32",
# ) -> None:
#     """Run a jax model and the translated TVM IRModule,
#        verify the inference accuracy.
#
#     Parameters
#     ----------
#     jax_jit_mod: jaxlib.xla_extension.CompiledFunction
#         The input jax jitted model
#     input_shapes: Union[Tuple, List[Tuple]]
#         shapes for inputs
#     dtype: str
#         the data type of inputs
#     """
#     # Generate numpy inputs
#     inputs_np = generate_np_inputs(input_shapes, dtype)
#     # Get the jax numpy data
#     inputs_jnp = np2jnp(inputs_np)
#
#     # lower the jitted function to StableHLO
#     lowered = jax_jit_mod.lower(*inputs_np)
#
#     # lowered.as_text(dialect="stablehlo") generates text format
#     # compiler_ir generates the related jaxlib.mlir.Module
#     stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
#
#     # Convert the StableHLO IR to Relax
#     ir_mod = from_stablehlo(stablehlo_module)
#
#     # Run the jax jitted model with the input jax numpy data
#     jax_output = jax_jit_mod(*inputs_jnp)
#
#     # TODO (yongwww): support multiple targets,
#     # "llvm" should be good for this check
#     target = tvm.target.Target("llvm", host="llvm")
#     # Compile and run
#     ex = relax.build(ir_mod, target)
#     vm = relax.VirtualMachine(ex, tvm.cpu())
#     vm.set_input("main", *inputs_np)
#     vm.invoke_stateful("main")
#     tvm_output = vm.get_outputs("main")
#
#     # Single ouput
#     if isinstance(tvm_output, tvm.nd.NDArray):
#         tvm.testing.assert_allclose(tvm_output.numpy(), jax_output, rtol=1e-5, atol=1e-5)
#         return
#
#     # Multiple ouputs
#     assert len(tvm_output) == len(jax_output), "numbers of outputs mismatch"
#     for tvm_out, jax_out in zip(tvm_output, jax_output):
#         tvm.testing.assert_allclose(tvm_out.numpy(), jax_out, rtol=1e-5, atol=1e-5)
#
#
def get_vm_res(
    ir_mod: tvm.IRModule, weights: Union[np.ndarray, List[np.ndarray]]
) -> Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]:
    """Compile and run an ir_module on Relax VM

    Parameters
    ----------
    ir_mod: tvm.IRModule
        input ir module

    weights: Union[np.ndarray, List[np.ndarray]]
         input weights

    Results
    -------
    out: Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]
        inference result
    """
    print("get_vm_res", weights)
    input("!")
    target = tvm.target.Target("llvm", host="llvm")
    # Compile and run
    ex = relax.build(ir_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *weights)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output
#
#
# @tvm.testing.requires_gpu
# def test_add_dynamic():
#     add_dyn = """
#     func.func @test(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
#       %1 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
#       func.return %1 : tensor<?x?xf32>
#     }
#     """
#
#     mod = from_stablehlo(add_dyn)
#
#     @I.ir_module
#     class Expected:
#         @R.function
#         def main(
#             arg0: R.Tensor(("n_0", "n_1"), dtype="float32"),
#             arg1: R.Tensor(("n_2", "n_3"), dtype="float32"),
#         ) -> R.Tensor(dtype="float32", ndim=2):
#             n_0 = T.int64()
#             n_1 = T.int64()
#             n_2 = T.int64()
#             n_3 = T.int64()
#             with R.dataflow():
#                 lv: R.Tensor(dtype="float32", ndim=2) = R.add(arg0, arg1)
#                 gv: R.Tensor(dtype="float32", ndim=2) = lv
#                 R.output(gv)
#             return gv
#
#     tvm.ir.assert_structural_equal(mod, Expected)
#
#
# @tvm.testing.requires_gpu
# def test_unary():
#     import jax
#
#     def _rsqrt(x):
#         return jax.lax.rsqrt(x)
#
#     def _sqrt(x):
#         return jax.lax.sqrt(x)
#
#     def _sin(x):
#         return jax.lax.sin(x)
#
#     def _sinh(x):
#         return jax.lax.sinh(x)
#
#     def _cos(x):
#         return jax.lax.cos(x)
#
#     def _cosh(x):
#         return jax.lax.cos(x)
#
#     def _exp(x):
#         return jax.lax.exp(x)
#
#     def _round(x):
#         return jax.lax.round(x)
#
#     input_shapes = (2, 3, 4)
#     for fn in [_rsqrt, _sqrt, _sin, _cos, _cosh, _exp, _round]:
#         check_correctness(jax.jit(fn), input_shapes)
#
#
# @tvm.testing.requires_gpu
# def test_binary():
#     import jax
#
#     def fn(x, y):
#         r1 = x + y
#         r2 = r1 * r1
#         r3 = r2 / r1
#         r = r2 - r3
#         return r
#
#     input_shape = (1, 2, 3)
#     input_shapes = (input_shape, input_shape)
#
#     # jit the function
#     jit_fn = jax.jit(fn)
#
#     # verify inference accuracy
#     check_correctness(jit_fn, input_shapes)
#
#
# @tvm.testing.requires_gpu
# def test_const():
#     import jax
#
#     def fn(x):
#         return x + 1
#
#     check_correctness(jax.jit(fn), (2,))
#
#
# @tvm.testing.requires_gpu
# def test_maximum():
#     import jax
#     import jax.numpy as jnp
#
#     def fn(x, y):
#         return jnp.maximum(x, y)
#
#     check_correctness(jax.jit(fn), ((2, 3), (2, 3)))
#
#
# @tvm.testing.requires_gpu
# def test_minimum():
#     import jax
#     import jax.numpy as jnp
#
#     def fn(x, y):
#         return jnp.minimum(x, y)
#
#     check_correctness(jax.jit(fn), ((2, 3), (2, 3)))
#
#
# @tvm.testing.requires_gpu
# def test_reduce():
#     import jax
#     import jax.numpy as jnp
#
#     def fn(x):
#         return jnp.mean(x, axis=(1, 2))
#
#     check_correctness(jax.jit(fn), (2, 3, 4, 5))
#
#
# @tvm.testing.requires_gpu
# def test_reduce_window():
#     import jax
#     from flax import linen as nn
#
#     def fn(x):
#         return nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
#
#     check_correctness(jax.jit(fn), (2, 3, 4))
#
#
# @tvm.testing.requires_gpu
# def test_dot_general():
#     import jax
#
#     def fn(x, y):
#         return jax.lax.dot_general(x, y, (([1], [0]), ([], [])))
#
#     input_shapes = ((1, 512), (512, 2))
#     check_correctness(jax.jit(fn), input_shapes)


def test_forward_mobilenet_v2():
    """Test the Mobilenet V2 TF Lite model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
        "mobilenet_v2_1.0_224.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    shape_dict = {"input": (1, 224, 224, 3)}
    dtype_dict = {"input": "float32"}
    input_info = [(shape_dict[name], dtype_dict[name]) for name in shape_dict]

    # convert in Relax
    ir_mod = from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict,
    )
    return
    # get ref output (tflite)
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    # compile and run (relay)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )
    # compile and run (relax)
    # tvm_output = get_vm_res(ir_mod, data)
    # verify accuracy
    # tvm.testing.assert_allclose(tvm_output.numpy(), expected  _output, rtol=1e-5, atol=1e-5)


@pytest.mark.skip("Relax QDQ does not support uint8")
def test_forward_qnn_mobilenet_v2_net():
    """Test the Quantized TFLite Mobilenet V2 model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/"
        "mobilenet_v2_1.0_224_quant.tgz",
        "mobilenet_v2_1.0_224_quant.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    shape_dict = {"input": (1, 224, 224, 3)}
    dtype_dict = {"input": "uint8"}
    input_info = [(shape_dict[name], dtype_dict[name]) for name in shape_dict]

    # convert in Relax
    ir_mod = from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict,
    )
    # tflite_output = run_tflite_graph(tflite_model_buf, data)
    # tflite_predictions = np.squeeze(tflite_output)
    # tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    # tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    # tvm_predictions = np.squeeze(tvm_output)
    # tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    # tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_tflite2_qnn_mobilenet_v2():
    """Test the Quantized TFLite version 2.1.0 Mobilenet V2 model."""
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        tflite_model_file = download_testdata(
            "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/"
            "mobilenet_v2_quantized.tflite",
            "mobilenet_v2_quantized.tflite",
        )
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()

        data = pre_processed_image(224, 224)

        try:
            import tflite.Model

            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite

            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except ImportError as exc:
            raise ImportError("The tflite package must be installed") from exc

        shape_dict = {"input": (1, 224, 224, 3)}
        dtype_dict = {"input": "float32"}
        input_info = [(shape_dict[name], dtype_dict[name]) for name in shape_dict]

        # convert in Relax
        ir_mod = from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict,
        )
        # return
        tvm_output = get_vm_res(ir_mod, np.array(data))

        # get ref output (tflite)
        # tvm_output = run_tvm_graph(tflite_model_buf, np.array(data), "input_1")
        tflite_output = run_tflite_graph(tflite_model_buf, data)
        tflite_predictions = np.squeeze(tflite_output)
        tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
        # compile and run (relay)
        tvm_output = run_tvm_graph(tflite_model_buf, np.array(data), "input_1")
        tvm_predictions = np.squeeze(tvm_output)
        tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
        # tvm.testing.assert_allclose(
        #     np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
        # )
        tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)
        # compile and run (relax)
        tvm_output = get_vm_res(ir_mod, np.array(data))
        # verify accuracy
        # tvm.testing.assert_allclose(tvm_output.numpy(), expected  _output, rtol=1e-5, atol=1e-5)



if __name__ == "__main__":
    tvm.testing.main()
