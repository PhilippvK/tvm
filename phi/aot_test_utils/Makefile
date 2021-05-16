TVM_ROOT ?= /home/ga87puy/src/tvm_aot/
BUILD_DIR ?= build
SRCS_DIR ?= srcs
CODEGEN_DIR ?= codegen
AOT_ROOT ?= $(TVM_ROOT)/src/runtime/crt/aot
EXECUTABLE = aot_test

ENABLE_TVM_PLATFORM_ABORT_BACKTRACE = 0
DMLC_CORE=$(TVM_ROOT)/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g
CC = gcc
AR = ar
RANLIB = ranlib
CC_OPTS = CC=$(CC) AR=$(AR) RANLIB=$(RANLIB)

PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I$(TVM_ROOT)/src/runtime/crt/include \
	-I$(TVM_ROOT)/src/runtime/crt/host \
	-I$(TVM_ROOT)/include \
	-I$(DMLC_CORE)/include \
	-I$(TVM_ROOT)/3rdparty/dlpack/include \
	-I$(AOT_ROOT) \
	-I$(BUILD_DIR) \
	-I$(SRCS_DIR)

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

CRT_SRCS = $(shell find $(CRT_ROOT))

${EXECUTABLE}: $(BUILD_DIR)/${EXECUTABLE}

source_libs= $(wildcard $(CODEGEN_DIR)/host/src/lib*.c)
lib_objs =$(source_libs:.c=.o)

$(BUILD_DIR)/${EXECUTABLE}: $(SRCS_DIR)/main.c  $(BUILD_DIR)/aot_executor.o  $(source_libs) $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS) $(BACKTRACE_LDFLAGS) $(BACKTRACE_CFLAGS) -lm

$(BUILD_DIR)/%.o: $(CODEGEN_DIR)/host/src/%.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(BUILD_DIR)/aot_executor.o: $(TVM_ROOT)/src/runtime/crt/aot_executor/aot_executor.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(BUILD_DIR)/stack_allocator.o: $(TVM_ROOT)/src/runtime/crt/memory/stack_allocator.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(BUILD_DIR)/crt_backend_api.o: $(TVM_ROOT)/src/runtime/crt/common/crt_backend_api.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

.PHONY: run
run: ${EXECUTABLE}
	${QUIET}${BUILD_DIR}/${EXECUTABLE}

clean:
	$(QUIET)rm -rf $(BUILD_DIR)/crt
cleanall:
	$(QUIET)rm -rf $(BUILD_DIR)
cleancodegen:
	$(QUIET)rm -rf $(CODEGEN_DIR)

