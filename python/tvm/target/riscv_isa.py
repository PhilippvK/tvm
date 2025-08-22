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
"""Defines functions to analyze available opcodes in the RISC-V ISA."""

import tvm.target

# TODO: share with riscv_utils.py
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
    print()
    print()
    print("extensions", extensions)
    print()
    print()
    return extensions

class IsaAnalyzer(object):
    """Checks ISA support for given target"""

    def __init__(self, target):
        self.target = tvm.target.Target(target)

    @property
    def has_pext_support(self):
        return self.target.attrs.get("march", None) is not None and "p" in get_extensions_from_arch(self.target.attrs["march"])

    @property
    def has_vext_support(self):
        return self.target.attrs.get("march", None) is not None and "v" in get_extensions_from_arch(self.target.attrs["march"])

    @property
    def vlen(self):
        # Vector unit lenths in bytes (TODO: bits?)
        if self.has_vext_support:
            # TODO: lookup from -march if provided
            return None  # unknown
        else:
            return 0  # not available

