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

//#include <assert.h>
//#include <float.h>
//#include <kernel.h>
//#include <power/reboot.h>
//#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <unistd.h>
#include <inttypes.h>
//#include <zephyr.h>

//#include "input_data.h"
//#include "output_data.h"
#include "tvmgen_default.h"
//#include "zephyr_uart.h"

#define WORKSPACE_SIZE (270 * 1024)

uint32_t micros() { return 0; }

static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

//size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
//                                va_list args) {
//  return vsnprint(out_buf, out_buf_size_bytes, fmt, args);
//}

void TVMLogf(const char* msg, ...) {
  char buffer[256];
  int size;
  va_list args;
  va_start(args, msg);
  size = vsprintf(buffer, msg, args);
  va_end(args);
  //TVMPlatformWriteSerial(buffer, (uint32_t)size);
  printf(buffer);
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: %08x\n", error);
  //sys_reboot(SYS_REBOOT_COLD);
  // TODO
  for (;;)
    ;
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

//void timer_expiry_function(struct k_timer* timer_id) { return; }

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
//struct k_timer g_microtvm_timer;
uint32_t g_utvm_start_time_micros;
int g_utvm_timer_running = 0;

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

  g_utvm_timer_running = 1;
  g_utvm_start_time_micros = micros();
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

  //uint32_t stop_time = k_cycle_get_32();
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time_micros;
  *elapsed_time_seconds = ((double)g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  tvm_crt_error_t err = kTvmErrorNoError;
  void* ptr = 0;
  DLDevice dev = {device_type, device_id};
  //assert(nbytes > 0);
  err = TVMPlatformMemoryAllocate(nbytes, dev, &ptr);
  CHECK_EQ(err, kTvmErrorNoError,
           "TVMBackendAllocWorkspace(%d, %d, %" PRIu64 ", %d, %d) -> %" PRId32, device_type,
           device_id, nbytes, dtype_code_hint, dtype_bits_hint, err);
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  tvm_crt_error_t err = kTvmErrorNoError;
  DLDevice dev = {device_type, device_id};
  err = TVMPlatformMemoryFree(ptr, dev);
  return err;
}

//static uint8_t main_rx_buf[128];
//static uint8_t cmd_buf[128];
//static size_t g_cmd_buf_ind;

void main(void) {
  TVMLogf("ETISSVP AOT Runtime\n");

  struct tvmgen_default_inputs inputs = {
      //.input_1 = input_data,
      .input_1 = (void*)0,
  };
  struct tvmgen_default_outputs outputs = {
      //.output = output_data,
      .output = (void*)0,
  };

  StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

  //double elapsed_time = 0;
  //TVMPlatformTimerStart();
  int ret_val = tvmgen_default_run(&inputs, &outputs);
  //TVMPlatformTimerStop(&elapsed_time);

  if (ret_val != 0) {
    TVMLogf("Error: %d\n", ret_val);
    TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
  }

  //size_t max_ind = -1;
  //float max_val = -FLT_MAX;
  //for (size_t i = 0; i < output_data_len; i++) {
  //  if (output_data[i] >= max_val) {
  //    max_ind = i;
  //    max_val = output_data[i];
  //  }
  //}
  //TVMLogf("#result:%d\n", max_ind);
  //TVMLogf("#result:%d:%d\n", max_ind, (uint32_t)(elapsed_time * 1000));
}
