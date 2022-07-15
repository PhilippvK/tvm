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


class IsaAnalyzer(object):
    """Checks ISA support for given target"""

    def __init__(self, target):
        self.target = tvm.target.Target(target)

    @property
    def has_pext(self):
        return (self.target.mattr is not None and "+p" in self.target.mattr) or (
            self.target.march is not None and "p" in self.target.march[4:]
        )

    @property
    def has_vext(self):
        return (self.target.mattr is not None and "+v" in self.target.mattr) or (
            self.target.march is not None and "v" in self.target.march[4:]
        )
