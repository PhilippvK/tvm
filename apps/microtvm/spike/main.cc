/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file main.cc
 * \brief main entry point for host subprocess-based CRT
 */
#include <inttypes.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "crt_config.h"
#include "riscv_util.h"

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
#include <tvm/runtime/crt/graph_executor_module.h>
#endif

#include <tvm/runtime/crt/aot_executor_module.h>

#ifndef SPIKE_CPU_FREQ_HZ
// Default: 100MHz
#define SPIKE_CPU_FREQ_HZ (100000000)
#endif  // SPIKE_CPU_FREQ_HZ

// #define DBG

#ifdef DBG
FILE *fp;
#define dbginit() fp = fopen("/tmp/test.txt", "w+");
#define dbgprintf(...) fprintf(fp, __VA_ARGS__); fflush(fp);
#define dbgend() fclose(fp);
#else
#define dbginit()
#define dbgprintf(...)
#define dbgend()
#endif  // DBG


extern "C" {

ssize_t MicroTVMWriteFunc(void* context, const uint8_t* data, size_t num_bytes) {
  ssize_t to_return = write(STDOUT_FILENO, data, num_bytes);
  fflush(stdout);
  return to_return;
}

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  dbgprintf("TVMPlatformAbort: %d\n", error_code);
  exit(1);
}

MemoryManagerInterface* memory_manager;

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return memory_manager->Allocate(memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return memory_manager->Free(memory_manager, ptr, dev);
}

uint64_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  dbgprintf("TVMPlatformTimerStart\n");
  if (g_microtvm_timer_running) {
    dbgprintf("timer already running\n");
    return kTvmErrorPlatformTimerBadState;
  }
  g_microtvm_start_time = rdcycle64();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  dbgprintf("TVMPlatformTimerStop\n");
  if (!g_microtvm_timer_running) {
    dbgprintf("timer not running\n");
    return kTvmErrorPlatformTimerBadState;
  }
  uint64_t microtvm_stop_time = rdcycle64();
  *elapsed_time_seconds = (microtvm_stop_time - g_microtvm_start_time) / (float)(SPIKE_CPU_FREQ_HZ);
  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

static_assert(RAND_MAX >= (1 << 8), "RAND_MAX is smaller than acceptable");
unsigned int random_seed = 0;
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  if (random_seed == 0) {
    random_seed = (unsigned int)time(NULL);
  }
  for (size_t i = 0; i < num_bytes; ++i) {
    int random = rand();
    buffer[i] = (uint8_t)random;
  }

  return kTvmErrorNoError;
}
}

uint8_t memory[2048 * 1024];

static char** g_argv = NULL;

int main(int argc, char** argv) {
  dbginit();
  dbgprintf("main\n");
  srand(random_seed);
  g_argv = argv;
  int status =
      PageMemoryManagerCreate(&memory_manager, memory, sizeof(memory), 8 /* page_size_log2 */);
  if (status != 0) {
    dbgprintf("error initiailizing memory manager\n");
    dbgend();
    return 2;
  }
  dbgprintf("b\n");

  microtvm_rpc_server_t rpc_server = MicroTVMRpcServerInit(&MicroTVMWriteFunc, nullptr);

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
  CHECK_EQ(TVMGraphExecutorModule_Register(), kTvmErrorNoError,
           "failed to register GraphExecutor TVMModule");
#endif

  setbuf(stdin, NULL);
  setbuf(stdout, NULL);

  for (;;) {
    uint8_t c;
    int ret_code = read(STDIN_FILENO, &c, 1);
    if (ret_code < 0) {
      dbgprintf("microTVM runtime: read failed");
      dbgend();
      return 2;
    } else if (ret_code == 0) {
      dbgprintf("microTVM runtime: 0-length read, exiting!\n");
      dbgend();
      return 2;
    }
    uint8_t* cursor = &c;
    size_t bytes_to_process = 1;
    while (bytes_to_process > 0) {
      tvm_crt_error_t err = MicroTVMRpcServerLoop(rpc_server, &cursor, &bytes_to_process);
      if (err == kTvmErrorPlatformShutdown) {
        break;
      } else if (err != kTvmErrorNoError) {
        dbgprintf("microTVM runtime: MicroTVMRpcServerLoop error: %08x", err);
        dbgend();
        return 2;
      }
    }
  }
  dbgend();
  return 0;
}
