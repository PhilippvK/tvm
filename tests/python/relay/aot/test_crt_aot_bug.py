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

from collections import OrderedDict
import sys

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay import testing, transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from aot_test_utils import (
    AOTTestModel,
    AOT_DEFAULT_RUNNER,
    generate_ref_data,
    convert_to_relay,
    compile_and_run,
    parametrize_aot_options,
)


@pytest.mark.parametrize("merge_compiler_regions", [False, True])
def test_byoc_microtvm_bug(merge_compiler_regions):
    """This is a simple test to reproduce a bug regarding multi-output  BYOC subgraphs and AOT"""
    use_unpacked_api = False
    interface_api = "packed"
    test_runner = AOT_DEFAULT_RUNNER

    # Inputs and Weights
    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))
    w2 = relay.var("w2", shape=(10, 10))

    # C compiler

    # z0 = x + w0
    x_ = compiler_begin(x, "ccompiler")
    w0_ = compiler_begin(w0, "ccompiler")
    z0_ = relay.add(x_, w0_)
    z0 = compiler_end(z0_, "ccompiler")

    # z1 = z0 + w1
    z0__ = compiler_begin(z0, "ccompiler")
    w1_ = compiler_begin(w1, "ccompiler")
    z1_ = relay.add(z0__, w1_)
    z1 = compiler_end(z1_, "ccompiler")

    # TVM Compiler

    # z2 = z0 + z1
    z2 = relay.add(z0, z1)

    f = relay.Function([x, w0, w1], z2)
    mod = tvm.IRModule()
    mod["main"] = f

    print("MOD BEFORE PARTITIONING:", mod)

    if merge_compiler_regions:
        print("MERGING COMPILER REGIONS...")
        mod = transform.MergeCompilerRegions()(mod)

    mod = transform.PartitionGraph("mod_name")(mod)
    mod = transform.InferType()(mod)

    print("PARTITIONED MOD:", mod)

    x_data = np.random.rand(10, 10).astype("float32")
    w_data = []
    for _ in range(2):
        w_data.append(np.random.rand(10, 10).astype("float32"))

    map_inputs = OrderedDict([("x", x_data)] + [("w{}".format(i), w_data[i]) for i in range(2)])
    output_list = generate_ref_data(mod, map_inputs)
    input_list = [map_inputs["x"]]
    input_list.extend([map_inputs["w{}".format(i)] for i in range(2)])
    compile_and_run(
        AOTTestModel(name="my_mod", module=mod, inputs=map_inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
