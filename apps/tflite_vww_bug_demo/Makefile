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

# TODO

# Setup build environment
TVM_ROOT=$(shell cd ../..; pwd)
CRT_ROOT ?= ../../build/standalone_crt
ifeq ($(shell ls -lhd $(CRT_ROOT)),)
$(error "CRT not found. Ensure you have built the standalone_crt target and try again")
endif

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O2 -fPIC
PKG_CXXFLAGS = ${PKG_COMPILE_OPTS} -std=c++14 \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-Icrt_config \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-Icrt_config \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -pthread -lm

build_dir := build

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

CRT_SRCS = $(shell find $(CRT_ROOT))

MODEL_OBJ = $(build_dir)/model_c/devc.o $(build_dir)/model_c/lib0.o $(build_dir)/model_c/lib1.o

demo_static: $(build_dir)/demo_static
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/demo_static

$(build_dir)/crt/libcommon.a: $(CRT_SRCS)
	$(QUIET)cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath $(build_dir))/crt CRT_CONFIG=$(abspath crt_config/crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" common

$(build_dir)/crt/libgraph_executor.a: $(CRT_SRCS)
	$(QUIET)cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath $(build_dir))/crt CRT_CONFIG=$(abspath crt_config/crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" graph_executor

$(build_dir)/crt/libmemory.a: $(CRT_SRCS)
	$(QUIET)cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath $(build_dir))/crt CRT_CONFIG=$(abspath crt_config/crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" memory

$(build_dir)/demo_static: demo_static.c ${build_dir}/bundle_static.o $(MODEL_OBJ) ${build_dir}/crt/libmemory.a ${build_dir}/crt/libgraph_executor.a ${build_dir}/crt/libcommon.a ${build_dir}/graph_c.json.c ${build_dir}/params_c.bin.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS)

# Serialize our graph.json file.
$(build_dir)/graph_cpp.json.c: $(build_dir)/graph_cpp.json
	$(QUIET)xxd -i $^  > $@

$(build_dir)/graph_c.json.c: $(build_dir)/graph_c.json
	$(QUIET)xxd -i $^  > $@

# Serialize our params.bin file.
$(build_dir)/params_c.bin.c: $(build_dir)/params_c.bin
	$(QUIET)xxd -i $^  > $@

$(build_dir)/params_cpp.bin.c: $(build_dir)/params_cpp.bin
	$(QUIET)xxd -i $^  > $@

$(MODEL_OBJ) $(build_dir)/graph_c.json $(build_dir)/model_cpp.o $(build_dir)/graph_cpp.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	$(QUIET)python3 $< -o $(build_dir)
	$(QUIET)mkdir -p build/model_c
	$(QUIET)tar -C build/model_c -xvf build/model_c.tar

# Build our bundle against the serialized bundle.c API, the runtime.cc API, and
# the serialized graph.json and params.bin
$(build_dir)/bundle.so: bundle.cc runtime.cc $(build_dir)/model_cpp.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)g++ -shared $(PKG_CXXFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/bundle_c.so: bundle.c $(MODEL_OBJ) ${build_dir}/crt/libmemory.a ${build_dir}/crt/libgraph_executor.a ${build_dir}/crt/libcommon.a
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/bundle_static.o: bundle_static.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -c $(PKG_CFLAGS) -o $@  $^

clean:
	$(QUIET)rm -rf $(build_dir)/bundle.so $(build_dir)/bundle_c.so $(build_dir)/crt

cleanall:
	$(QUIET)rm -rf $(build_dir)

# Don't define implicit rules; they tend to match on logical target names that aren't targets (i.e. bundle_static)
.SUFFIXES:

.DEFAULT: demo_static
