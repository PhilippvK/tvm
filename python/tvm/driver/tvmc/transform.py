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
# specific language
"""
TVMC Graph Transforms
"""

from tvm import relay, transform
from tvm.driver.tvmc import TVMCException


def convert_graph_layout(mod, desired_layout):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module to convert.
    desired_layout : str
        The layout to convert to. Either only data layour or data and kernel (colon-separated).

    Returns
    -------
    mod : tvm.IRModule
        The converted module.
    """
    if ":" in desired_layout:
        data_layout, kernel_layout = desired_layout.split(":")[:2]
    else:
        data_layout = desired_layout
        kernel_layout = "default"

    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": [data_layout, kernel_layout],
        "nn.conv2d_transpose": [data_layout, kernel_layout],
        "qnn.conv2d": [data_layout, kernel_layout],
    }

    # Convert the layout of the graph where possible.
    seq = transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )

    with transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise TVMCException(
                "Error converting layout to {0}: {1}".format(desired_layout, str(err))
            )
