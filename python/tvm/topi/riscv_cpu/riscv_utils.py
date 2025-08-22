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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""RISC-V target utility functions"""

import re
import tvm

def get_extensions_from_arch(arch):
    """Parse -march attribute to get all enabled ISA extensions."""
    arch = arch[4:]
    extensions = []
    for i, c in enumerate(arch):
        if c == "x":  # custom extensions
            # TODO: allow multiple?
            ext = arch[i:]
            extensions.append(ext)
        else:  # standard extension
            extensions.append(c)
    return extensions


def is_riscv_64():
    """Checks whether we are compiling for a 64 bit RISC-V target."""
    target = tvm.target.Target.current(allow_none=False)
    return "rv32" in target.attrs.get("march", "")

def is_pext_available():
    """Check if packed SIMD instructions are available."""
    target = tvm.target.Target.current(allow_none=False)
    arch = target.attrs.get("march", "")
    return "p" in get_extensions_from_arch(arch)

def is_vext_available():
    """Check if vector instructions are available."""
    target = tvm.target.Target.current(allow_none=False)
    arch = target.attrs.get("march", "")
    return "v" in get_extensions_from_arch(arch)
