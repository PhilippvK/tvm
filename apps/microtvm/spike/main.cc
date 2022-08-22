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
// #include <time.h>
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

// using namespace std::chrono;

extern "C" {

ssize_t MicroTVMWriteFunc(void* context, const uint8_t* data, size_t num_bytes) {
  ssize_t to_return = write(STDOUT_FILENO, data, num_bytes);
  fflush(stdout);
  // fsync(STDOUT_FILENO);
  return to_return;
}

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  // std::cerr << "TVMPlatformAbort: " << error_code << std::endl;
  dbgprintf("TVMPlatformAbort: %d\n", error_code);
  // throw "Aborted";
  exit(1);
}

MemoryManagerInterface* memory_manager;

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return memory_manager->Allocate(memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return memory_manager->Free(memory_manager, ptr, dev);
}

// steady_clock::time_point g_microtvm_start_time;
// double g_microtvm_start_time;
uint64_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of clock cycles executed by the processor.
 */
static inline uint64_t rdcycle64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t cycles;
    uint32_t cyclesh1;
    uint32_t cyclesh2;

    /* Reads are not atomic. So ensure, that we are never reading inconsistent
     * values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh1));
        __asm__ volatile("rdcycle %0" : "=r"(cycles));
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh2));
    } while (cyclesh1 != cyclesh2);

    return (((uint64_t)cyclesh1) << 32) | cycles;
#else
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#endif
#else
    return 0;
#endif
}

tvm_crt_error_t TVMPlatformTimerStart() {
  dbgprintf("TVMPlatformTimerStart\n");
  if (g_microtvm_timer_running) {
    // std::cerr << "timer already running" << std::endl;
    dbgprintf("timer already running\n");
    return kTvmErrorPlatformTimerBadState;
  }
  // g_microtvm_start_time = std::chrono::steady_clock::now();
  // struct timeval tv;
  // gettimeofday(&tv, NULL);
  // g_microtvm_start_time = tv.tv_sec + 1e-6f * tv.tv_usec;
  g_microtvm_start_time = rdcycle64();
  // dbgprintf("g_microtvm_start_time=%f\n", g_microtvm_start_time);
  dbgprintf("g_microtvm_start_time=%llu\n", g_microtvm_start_time);
  // TVMLogf("g_microtvm_start_time=%llu\n", g_microtvm_start_time);
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  dbgprintf("TVMPlatformTimerStop\n");
  if (!g_microtvm_timer_running) {
    dbgprintf("timer not running\n");
    return kTvmErrorPlatformTimerBadState;
  }
  // auto microtvm_stop_time = std::chrono::steady_clock::now();
  // struct timeval tv;
  // gettimeofday(&tv, NULL);
  // float microtvm_stop_time = tv.tv_sec + 1e-6f * tv.tv_usec;
  uint64_t microtvm_stop_time = rdcycle64();
  // dbgprintf("microtvm_stop_time=%f\n", microtvm_stop_time);
  dbgprintf("microtvm_stop_time=%llu\n", microtvm_stop_time);
  // std::chrono::microseconds time_span = std::chrono::duration_cast<std::chrono::microseconds>(
  //     microtvm_stop_time - g_microtvm_start_time);
  // *elapsed_time_seconds = static_cast<double>(time_span.count()) / 1e6;
  dbgprintf("elapsed_time_seconds=%f\n", *elapsed_time_seconds);
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
    // int random = rand_r(&random_seed);
    int random = rand();
    buffer[i] = (uint8_t)random;
  }

  return kTvmErrorNoError;
}
}

uint8_t memory[2048 * 1024];

static char** g_argv = NULL;

// int testonly_reset_server(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
//                           int* out_ret_tcode, void* resource_handle) {
//   execvp(g_argv[0], g_argv);
//   perror("microTVM runtime: error restarting");
//   return -1;
// }

int main(int argc, char** argv) {
  dbginit();
  dbgprintf("main\n");
  srand(random_seed);
  // dbgprintf("a\n");
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
  // dbgprintf("c\n");

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
  CHECK_EQ(TVMGraphExecutorModule_Register(), kTvmErrorNoError,
           "failed to register GraphExecutor TVMModule");
#endif

  // int error = TVMFuncRegisterGlobal("tvm.testing.reset_server",
  //                                   (TVMFunctionHandle)&testonly_reset_server, 0);
  // if (error) {
  //   fprintf(
  //       stderr,
  //       "microTVM runtime: internal error (error#: %x) registering global packedfunc; exiting\n",
  //       error);
  //   return 2;
  // }

  // dbgprintf("d\n");
  setbuf(stdin, NULL);
  setbuf(stdout, NULL);
  // dbgprintf("e\n");

  for (;;) {
    // dbgprintf("f\n");
    uint8_t c;
    int ret_code = read(STDIN_FILENO, &c, 1);
    // dbgprintf("ret_code=%d\n", ret_code)
    // dbgprintf("c=%x\n", c)
    // dbgprintf("g\n");
    if (ret_code < 0) {
      // dbgprintf("h\n");
      dbgprintf("microTVM runtime: read failed");
      dbgend();
      return 2;
    } else if (ret_code == 0) {
      // dbgprintf("i\n");
      dbgprintf("microTVM runtime: 0-length read, exiting!\n");
      dbgend();
      return 2;
    }
    // dbgprintf("j\n");
    uint8_t* cursor = &c;
    size_t bytes_to_process = 1;
    // dbgprintf("k\n");
    // dbgprintf("bytes_to_process=%u\n", bytes_to_process);
    while (bytes_to_process > 0) {
      // dbgprintf("l\n");
      tvm_crt_error_t err = MicroTVMRpcServerLoop(rpc_server, &cursor, &bytes_to_process);
      // dbgprintf("m\n");
      // dbgprintf("err=%d\n", err);
      if (err == kTvmErrorPlatformShutdown) {
        // dbgprintf("n\n");
        break;
      } else if (err != kTvmErrorNoError) {
        // dbgprintf("m\n");
        dbgprintf("microTVM runtime: MicroTVMRpcServerLoop error: %08x", err);
        dbgend();
        return 2;
      }
      // dbgprintf("o\n");
    }
    // dbgprintf("p\n");
  }
  // dbgprintf("q\n");
  dbgend();
  return 0;
}
