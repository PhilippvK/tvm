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
 * \brief Implementation of TVMPlatform functions in tvm/runtime/crt/platform.h
 */

#include <dlpack/dlpack.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/logging.h>
#include <unistd.h>

// #include <chrono>
#include <iostream>

#ifndef SPIKE_CPU_FREQ_HZ
// Default: 100MHz
#define SPIKE_CPU_FREQ_HZ (100000000)
#endif  // SPIKE_CPU_FREQ_HZ


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


// using namespace std::chrono;

extern "C" {

uint8_t memory[TVM_WORKSPACE_SIZE_BYTES];
MemoryManagerInterface* memory_manager;

// steady_clock::time_point g_microtvm_start_time;
uint64_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

// Called when an internal error occurs and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error_code) {
  TVMLogf("ABORT\n");
  // std::cerr << "TVMPlatformAbort: " << error_code << std::endl;
  throw "Aborted";
}

// Called by the microTVM RPC server to implement TVMLogf.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsprintf(out_buf, fmt, args);
}

// Allocate memory for use by TVM.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  // TVMLogf("Alloc: %u\n", num_bytes);
  return memory_manager->Allocate(memory_manager, num_bytes, dev, out_ptr);
}

// Free memory used by TVM.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  // TVMLogf("Free\n");
  return memory_manager->Free(memory_manager, ptr, dev);
}

// Start a device timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  // TVMLogf("Start\n");
  if (g_microtvm_timer_running) {
    std::cerr << "timer already running" << std::endl;
    return kTvmErrorPlatformTimerBadState;
  }
  // g_microtvm_start_time = std::chrono::steady_clock::now();
  g_microtvm_start_time = rdcycle64();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

// Stop the running device timer and get the elapsed time (in microseconds).
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  // TVMLogf("Stop\n");
  if (!g_microtvm_timer_running) {
    std::cerr << "timer not running" << std::endl;
    return kTvmErrorPlatformTimerBadState;
  }
  // auto microtvm_stop_time = std::chrono::steady_clock::now();
  // std::chrono::microseconds time_span = std::chrono::duration_cast<std::chrono::microseconds>(
  //     microtvm_stop_time - g_microtvm_start_time);
  // *elapsed_time_seconds = static_cast<double>(time_span.count()) / 1e6;
  // *elapsed_time_seconds = 0.042;
  uint64_t microtvm_stop_time = rdcycle64();
  *elapsed_time_seconds = (microtvm_stop_time - g_microtvm_start_time) / (float)(SPIKE_CPU_FREQ_HZ);
  // TVMLogf("delta: %f\n", *elapsed_time_seconds);
  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

// Platform-specific before measurement call.
tvm_crt_error_t TVMPlatformBeforeMeasurement() { return kTvmErrorNoError; }

// Platform-specific after measurement call.
tvm_crt_error_t TVMPlatformAfterMeasurement() { return kTvmErrorNoError; }

static_assert(RAND_MAX >= (1 << 8), "RAND_MAX is smaller than acceptable");
unsigned int random_seed = 0;
// Fill a buffer with random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  // TVMLogf("TVMPlatformGenerateRandom\n");
  // if (random_seed == 0) {
  //   // random_seed = (unsigned int)time(NULL);
  //   random_seed = 42;
  // }
  // for (size_t i = 0; i < num_bytes; ++i) {
  //   int random = rand_r(&random_seed);
  //   buffer[i] = (uint8_t)random;
  // }
  return kTvmErrorNoError;
}

// Initialize TVM inference.
tvm_crt_error_t TVMPlatformInitialize() {
  int status =
      PageMemoryManagerCreate(&memory_manager, memory, sizeof(memory), 8 /* page_size_log2 */);
  if (status != 0) {
    fprintf(stderr, "error initiailizing memory manager\n");
    return kTvmErrorPlatformMemoryManagerInitialized;
  }
  return kTvmErrorNoError;
}

}  // extern C
