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

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This is a sample Zephyr-based application that contains the logic
 * needed to control a microTVM-based model via the UART. This is only
 * intended to be a demonstration, since typically you will want to incorporate
 * this logic into your own application.
 */

//#include <drivers/gpio.h>
//#include <drivers/uart.h>
//#include <fatal.h>
//#include <kernel.h>
//#include <power/reboot.h>
//#include <random/rand32.h>
#include <stdio.h>
#include <stdbool.h>
//#include <sys/printk.h>
//#include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <unistd.h>
//#include <zephyr.h>

#include "crt_config.h"
#include "ringbuffer.h"

static const struct device* tvm_uart;

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;
static size_t g_num_bytes_in_rx_buffer = 0;


// Circular buffers for transmit and receive
#define BUFLEN (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)

//static uint8_t _tx_buffer[sizeof(RingBuffer) + BUFLEN] __attribute__ ((aligned(4)));
static uint8_t _rx_buffer[sizeof(RingBuffer) + BUFLEN] __attribute__ ((aligned(4)));

//static RingBuffer *const tx_buffer = (RingBuffer *) &_tx_buffer;
static RingBuffer *const rx_buffer = (RingBuffer *) &_rx_buffer;

void UART0_IRQHandler() {
    // ...
    
    // If there is received data, read it into the receive buffer.  If the
    // buffer is full, disable the receive interrupt.
    if (!buf_isfull(rx_buffer)) {
        buf_put_byte(rx_buffer, UART0_D);
        //if(buf_isfull(rx_buffer))
        //    UART0_C2 &= ~UART_C2_RIE_MASK;
    }

    // ...
}

unsigned long micros() {
  return 0;
}

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  g_num_bytes_requested += size;

  for (size_t i = 0; i < size; i++) {
    //uart_poll_out(tvm_uart, data[i]);
    printf("uart_poll_out: %d (%c)", data[i], data[i]);
    g_num_bytes_written++;
  }

  return size;
}

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  //sys_reboot(SYS_REBOOT_COLD);
  // TODO
  for (;;)
    ;
}

// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  uint32_t random;  // one unit of random data.

  // Fill parts of `buffer` which are as large as `random`.
  size_t num_full_blocks = num_bytes / sizeof(random);
  for (int i = 0; i < num_full_blocks; ++i) {
    //random = sys_rand32_get();
    // TODO
    memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  }

  // Fill any leftover tail which is smaller than `random`.
  size_t num_tail_bytes = num_bytes % sizeof(random);
  if (num_tail_bytes > 0) {
    //random = sys_rand32_get();
    // TODO
    memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  }
  return kTvmErrorNoError;
}

// Heap for use by TVMPlatformMemoryAllocate.
//K_HEAP_DEFINE(tvm_heap, 216 * 1024);
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  if (num_bytes == 0) {
    num_bytes = sizeof(int);
  }
  *out_ptr = malloc(num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  free(ptr);
  return kTvmErrorNoError;
}

// Called by TVM to allocate memory.
/*tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  //*out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  // TODO
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  //k_heap_free(&tvm_heap, ptr);
  // TODO
  return kTvmErrorNoError;
}*/

//#define MILLIS_TIL_EXPIRY 200
//#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
//K_TIMER_DEFINE(g_microtvm_timer, /* expiry func */ NULL, /* stop func */ NULL);

unsigned long g_utvm_start_time_micros;
int g_utvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 1;
  g_utvm_start_time_micros = micros();
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time_micros;
  *elapsed_time_seconds = ((double)g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

// Ring buffer used to store data read from the UART on rx interrupt.
// This ring buffer size is only required for testing with QEMU and not for physical hardware.
//#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)
//RING_BUF_ITEM_DECLARE_SIZE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);

// UART interrupt callback.
/*void uart_irq_cb(const struct device* dev, void* user_data) {
  uart_irq_update(dev);
  if (uart_irq_is_pending(dev)) {
    struct ring_buf* rbuf = (struct ring_buf*)user_data;
    if (uart_irq_rx_ready(dev) != 0) {
      uint8_t* data;
      uint32_t size;
      size = ring_buf_put_claim(rbuf, &data, RING_BUF_SIZE_BYTES);
      int rx_size = uart_fifo_read(dev, data, size);
      // Write it into the ring buffer.
      g_num_bytes_in_rx_buffer += rx_size;

      if (g_num_bytes_in_rx_buffer > RING_BUF_SIZE_BYTES) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef3);
      }

      if (rx_size < 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef1);
      }

      int err = ring_buf_put_finish(rbuf, rx_size);
      if (err != 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef2);
      }
      // CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
      // bytes_written);
    }
  }
}*/

// Used to initialize the UART receiver.
/*void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
}*/

// The main function of this application.
//extern void __stdout_hook_install(int (*hook)(int));
void main(void) {

  // Claim console device.
  //tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  //uart_rx_init(&uart_rx_rbuf, tvm_uart);

  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM ETISSVP runtime - running");

  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (true) {
    uint8_t* data;
    //unsigned int key = irq_lock();
    //uint32_t bytes_read = ring_buf_get_claim(&uart_rx_rbuf, &data, RING_BUF_SIZE_BYTES);
    /*if (bytes_read > 0) {
      g_num_bytes_in_rx_buffer -= bytes_read;
      size_t bytes_remaining = bytes_read;
      while (bytes_remaining > 0) {
        // Pass the received bytes to the RPC server.
        tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &data, &bytes_remaining);
        if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
          TVMPlatformAbort(err);
        }
        if (g_num_bytes_written != 0 || g_num_bytes_requested != 0) {
          if (g_num_bytes_written != g_num_bytes_requested) {
            TVMPlatformAbort((tvm_crt_error_t)0xbeef5);
          }
          g_num_bytes_written = 0;
          g_num_bytes_requested = 0;
        }
      }
      int err = ring_buf_get_finish(&uart_rx_rbuf, bytes_read);
      if (err != 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef6);
      }
    }*/
    //irq_unlock(key);
    break;
  }
  TVMLogf("microTVM ETISSVP runtime - done");
  for (;;) {}

}
