#!/bin/bash
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

set -e
set -u
set -o pipefail

spike_dir="/opt/riscv/riscv-isa-sim"
spikepk_dir="/opt/riscv/spike-pk"

mkdir -p /opt/riscv

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

# Ubuntu 18.04 dependencies
apt-get update

# apt-get install -y \
#     bsdmainutils \
#     build-essential \
#     cpp \
#     git \
#     linux-headers-generic \
#     make \
#     python-dev \
#     python3 \
#     ssh \
#     wget \
#     xxd

apt-get install -y \
    build-essential \
    git \
    make

# Install the RISCV GCC toolchain with vector support
mkdir -p /opt/riscv/gcc-riscv64-unknown-elf/
gcc_riscv_url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2?revision=ca0cbf9c-9de2-491c-ac48-898b5bbc0443&la=en&hash=68760A8AE66026BCF99F05AC017A6A50C6FD832A'  # TODO
curl --retry 64 -sSL ${gcc_riscv_url} | tar -C /opt/riscv/riscv64-unknown-elf --strip-components=1 -jx
export PATH="/opt/riscv/riscv64-unknown-elf/bin:${PATH}"

# Clone spike repo
mkdir -p "$spike_dir"
cd "$tmpdir"
curl -sL https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.15_24.tgz | tar -xz
./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --no-interactive -d "$fvp_dir"
rm -rf FVP_Corstone_SSE-300.sh license_terms

# Clone riscv proxy kernel
mkdir -p "$spikepk_dir"
cd "$tmpdir"
curl -sL https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.15_24.tgz | tar -xz
./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --no-interactive -d "$fvp_dir"
rm -rf FVP_Corstone_SSE-300.sh license_terms
