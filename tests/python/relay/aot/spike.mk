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

# Makefile to build and run AOT tests against the reference system

# Setup build environment
build_dir := build
TVM_ROOT=$(shell cd ../../../../..; pwd)
CRT_ROOT ?= ${TVM_ROOT}/build/standalone_crt
ifeq ($(shell ls -lhd $(CRT_ROOT)),)
$(error "CRT not found. Ensure you have built the standalone_crt target and try again")
endif

SPIKE ?= /opt/riscv/spike/bin/spike
# rv32 only?
PK ?= /opt/riscv/spike/bin/pk

ARCH ?= rv32gc
ABI ?= ilp32d
ISA ?= ${ARCH}
TRIPLE ?= riscv64-unknown-elf

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O2 -Wno-incompatible-pointer-types -Wno-format -march=${ARCH} -mabi=${ABI}
CMAKE = cmake
CC = ${TRIPLE}-gcc
AR = ${TRIPLE}-ar
RANLIB = ${TRIPLE}-ranlib
CC_OPTS = CC=$(CC) AR=$(AR) RANLIB=$(RANLIB)
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	${CFLAGS} \
	-I$(build_dir)/../include \
	-I$(CODEGEN_ROOT)/host/include \
	-isystem$(STANDALONE_CRT_DIR)/include

PKG_LDFLAGS = -lm

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

CRT_SRCS = $(shell find $(CRT_ROOT))
C_CODEGEN_SRCS= $(wildcard $(CODEGEN_ROOT)/host/src/*.c)
CC_CODEGEN_SRCS= $(wildcard $(CODEGEN_ROOT)/most/src/*.cc)
c_lib_objs =$(c_source_libs:.c=.o)
cc_lib_objs =$(cc_source_libs:.cc=.o)
C_CODEGEN_OBJS = $(subst .c,.o,$(C_CODEGEN_SRCS))
CC_CODEGEN_OBJS = $(subst .cc,.o,$(CC_CODEGEN_SRCS))

aot_test_runner: $(build_dir)/aot_test_runner

$(build_dir)/stack_allocator.o: $(TVM_ROOT)/src/runtime/crt/memory/stack_allocator.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

$(build_dir)/crt_backend_api.o: $(TVM_ROOT)/src/runtime/crt/common/crt_backend_api.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

$(build_dir)/libcodegen.a: $(C_CODEGEN_SRCS) $(CC_CODEGEN_SRCS)
	echo C_CODEGEN_SRCS=$(C_CODEGEN_SRCS)
	echo CC_CODEGEN_SRCS=$(CC_CODEGEN_SRCS)
	$(QUIET)cd $(abspath $(CODEGEN_ROOT)/host/src) && $(CC) -c $(PKG_CFLAGS) $(C_CODEGEN_SRCS) $(CC_CODEGEN_SRCS)
	$(QUIET)$(AR) -cr $(abspath $(build_dir)/libcodegen.a) $(C_CODEGEN_OBJS) $(CC_CODEGEN_OBJS)
	$(QUIET)$(RANLIB) $(abspath $(build_dir)/libcodegen.a)


$(build_dir)/aot_test_runner: $(build_dir)/test.c $(build_dir)/crt_backend_api.o $(build_dir)/stack_allocator.o $(build_dir)/libcodegen.a $(ETHOSU_DRIVER_LIBS) $(ETHOSU_RUNTIME)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(PKG_CFLAGS) $(ETHOSU_INCLUDE) -o $@ -Wl,--whole-archive $^ -Wl,--no-whole-archive $(PKG_LDFLAGS)

clean:
	$(QUIET)rm -rf $(build_dir)/crt

cleanall:
	$(QUIET)rm -rf $(build_dir)

run: $(build_dir)/aot_test_runner
	$(SPIKE) --isa=$(ISA) $(PK) $(build_dir)/aot_test_runner

.SUFFIXES:

.DEFAULT: aot_test_runner

.PHONY: run
