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

import os
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import pandas as pd
import numpy as np

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib import utils
from tvm import meta_schedule as ms
from tvm.driver import tvmc
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig
from tvm.meta_schedule.logging import get_logger
from tvm import transform
from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

from tvm.contrib.micro.cfu.wca import CompressWeights, ImportCPostprocess, get_wca_tuning_config


def create_relay_module():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    np.random.seed(seed=1234)
    weight_sample = np.random.rand(weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info


def lookup_model_by_name(model, base_dir=None):
    def _load_model(path):
        model = tvmc.load(str(path))
        mod = model.mod
        params = model.params
        return mod, params

    if model == "default":
        # input("1")
        mod, params, model_info = create_relay_module()
        input_name = model_info["in_tensor"]
        input_shape = model_info["in_shape"]
        input_dtype = model_info["in_dtype"]
    else:
        MODELS_DIR = (base_dir / "models").resolve()

        INPUT_SHAPE_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": [1, 32, 32, 3],
            "pretrainedResnet_clustered_quant_remap_packed": [1, 32, 32, 3],
        }
        DEFAULT_INPUT_SHAPE = [1, 32, 32, 3]
        INPUT_DTYPE_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": "int8",
            "pretrainedResnet_clustered_quant_remap_packed": "int8",
        }
        DEFAULT_INPUT_DTYPE = "int8"
        INPUT_NAME_LOOKUP = {
            "pretrainedResnet_clustered_quant_remap": "input",
            "pretrainedResnet_clustered_quant_remap_packed": "input",
        }
        DEFAULT_INPUT_NAME = "input"

        model_file = model if ".tflite" in model else f"{model}.tflite"
        model_name = Path(model).stem
        model_path = MODELS_DIR / model_file
        assert model_path.is_file(), f"Model not found: {model_path}"
        mod, params = _load_model(model_path)

        input_shape = INPUT_SHAPE_LOOKUP.get(model_name, DEFAULT_INPUT_SHAPE)
        input_dtype = INPUT_DTYPE_LOOKUP.get(model_name, DEFAULT_INPUT_DTYPE)
        input_name = INPUT_NAME_LOOKUP.get(model_name, DEFAULT_INPUT_NAME)
    data_sample = np.random.rand(*input_shape).astype(input_dtype)
    return mod, params, input_name, input_shape, input_dtype, data_sample
