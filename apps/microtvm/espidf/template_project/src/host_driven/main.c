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
 * This is a sample ESP-IDF-based application that contains the logic
 * needed to control a microTVM-based model via the UART. This is only
 * intended to be a demonstration, since typically you will want to incorporate
 * this logic into your own application.
 */
#include <driver/gpio.h>
#include <driver/uart.h>
#include <stdio.h>
#include <string.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "sdkconfig.h"
#include "crt_config.h"

static const char* TAG = "microtvm";

#if CONFIG_IDF_TARGET_ESP32C3

#define CONFIG_LED_PIN_RED ((gpio_num_t)3)
#define CONFIG_LED_PIN_GREEN ((gpio_num_t)4)
#define CONFIG_LED_PIN_BLUE ((gpio_num_t)5)

#endif

/**
 * This example shows how to use the UART driver to handle special UART events.
 *
 * It also reads data from UART0 directly, and echoes it to console.
 *
 * - Port: UART0
 * - Receive (Rx) buffer: on
 * - Transmit (Tx) buffer: off
 * - Flow control: off
 * - Event queue: on
 * - Pin assignment: TxD (default), RxD (default)
 */

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;
static size_t g_num_bytes_in_rx_buffer = 0;

#define EX_UART_NUM UART_NUM_0

#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)
static RingbufHandle_t buf_handle;

#define BUF_SIZE (1024)
static QueueHandle_t uart0_queue;

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
#ifdef CONFIG_LED_PIN_RED
  gpio_set_level(CONFIG_LED_PIN_RED, 1);
#endif
  g_num_bytes_requested += size;

  uart_write_bytes(EX_UART_NUM, data, size);
  g_num_bytes_written += size;

#ifdef CONFIG_LED_PIN_RED
  gpio_set_level(CONFIG_LED_PIN_RED, 0);
#endif

  return size;
}

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                const char* fmt, va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot
// continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: CALL\n");
#ifdef CONFIG_LED_PIN_RED
  gpio_set_level(CONFIG_LED_PIN_RED, 1);
#endif
#ifdef CONFIG_LED_PIN_GREEN
  gpio_set_level(CONFIG_LED_PIN_GREEN, 1);
#endif
#ifdef CONFIG_LED_PIN_BLUE
  gpio_set_level(CONFIG_LED_PIN_BLUE, 1);
#endif
  for (;;) vTaskDelay(1000);
}

// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  // TVMLogf("TVMPlatformGenerateRandom: CALL(%u)\n", num_bytes);
  // uint32_t random;  // one unit of random data.

  // // Fill parts of `buffer` which are as large as `random`.
  // size_t num_full_blocks = num_bytes / sizeof(random);
  // for (int i = 0; i < num_full_blocks; ++i) {
  //   // random = sys_rand32_get();
  //   random = 0;  // TODO(@PhilippvK)
  //   memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  // }
  // memset(buffer, 42, num_bytes);

  // // Fill any leftover tail which is smaller than `random`.
  // size_t num_tail_bytes = num_bytes % sizeof(random);
  // if (num_tail_bytes > 0) {
  //   // random = sys_rand32_get();
  //   random = 0;  // TODO(@PhilippvK)
  //   memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  // }
  // TVMLogf("TVMPlatformGenerateRandom: RET\n");
  return kTvmErrorNoError;
}

// #define CRT_MEMORY_NUM_PAGES 216
#define CRT_MEMORY_NUM_PAGES 108
#define CRT_MEMORY_PAGE_SIZE_LOG2 10

// Heap for use by TVMPlatformMemoryAllocate.
static uint8_t
    tvm_heap[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];
static MemoryManagerInterface* g_memory_manager;

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                          void** out_ptr) {
  return g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return g_memory_manager->Free(g_memory_manager, ptr, dev);
}

// #define MILLIS_TIL_EXPIRY 200
// #define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
// K_TIMER_DEFINE(g_microtvm_timer, /* expiry func */ NULL, /* stop func */
// NULL);
// TODO

int64_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

#ifdef CONFIG_LED_PIN
  // gpio_set_level(CONFIG_LED_PIN, 1);
#endif
  g_microtvm_start_time = esp_timer_get_time();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

  int64_t stop_time = esp_timer_get_time();
#ifdef CONFIG_LED_PIN
  // gpio_set_level(CONFIG_LED_PIN, 0);
#endif

  // compute how long the work took
  int64_t us_spent = stop_time - g_microtvm_start_time;

  // we do not expect a rollover because of using a 64 bit data type

  *elapsed_time_seconds = us_spent / 1e6;

  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

static void uart_event_task(void* pvParameters) {
  uart_event_t event;
  size_t buffered_size;
  uint8_t* data;
  for (;;) {
    // Waiting for UART event.
    if (xQueueReceive(uart0_queue, (void*)&event,
                      (portTickType)portMAX_DELAY)) {
      switch (event.type) {
        // Event of UART receving data
        /*We'd better handler data event fast, there would be much more data
        events than other types of events. If we take too much time on data
        event, the queue might be full.*/
        case UART_DATA: {
#ifdef CONFIG_LED_PIN_BLUE
          gpio_set_level(CONFIG_LED_PIN_BLUE, 1);
#endif
          UBaseType_t res = xRingbufferSendAcquire(
              buf_handle, (void**)&data, event.size, pdMS_TO_TICKS(10000));
          if (res != pdTRUE) {
            TVMLogf("Failed to acquire memory for data\n");
            break;
          }
          uart_read_bytes(EX_UART_NUM, data, event.size, portMAX_DELAY);
          // Write it into the ring buffer.
          g_num_bytes_in_rx_buffer += event.size;

          if (g_num_bytes_in_rx_buffer > RING_BUF_SIZE_BYTES) {
            TVMPlatformAbort((tvm_crt_error_t)0xbeef3);
            TVMLogf("Buffer full\n");
          }

          res = xRingbufferSendComplete(buf_handle, (void*)data);
          if (res != pdTRUE) {
            TVMLogf("Failed to send item\n");
          }
          break;
        }
        // Event of HW FIFO overflow detected
        case UART_FIFO_OVF:
          TVMLogf("hw fifo overflow");
          // If fifo overflow happened, you should consider adding flow control
          // for your application. The ISR has already reset the rx FIFO, As an
          // example, we directly flush the rx buffer here in order to read more
          // data.
          uart_flush_input(EX_UART_NUM);
          xQueueReset(uart0_queue);
          break;
        // Event of UART ring buffer full
        case UART_BUFFER_FULL:
          TVMLogf("ring buffer full");
          // If buffer full happened, you should consider encreasing your buffer
          // size As an example, we directly flush the rx buffer here in order
          // to read more data.
          uart_flush_input(EX_UART_NUM);
          xQueueReset(uart0_queue);
          break;
        // Event of UART RX break detected
        case UART_BREAK:
          TVMLogf("uart rx break");
          break;
        // Event of UART parity check error
        case UART_PARITY_ERR:
          TVMLogf("uart parity error");
          break;
        // Event of UART frame error
        case UART_FRAME_ERR:
          TVMLogf("uart frame error");
          break;
        // UART_PATTERN_DET
        case UART_PATTERN_DET:
          break;
        // Others
        default:
          TVMLogf("uart event type: %d", event.type);
          break;
      }
#ifdef CONFIG_LED_PIN_BLUE
      gpio_set_level(CONFIG_LED_PIN_BLUE, 0);
#endif
    }
  }
  vTaskDelete(NULL);
}

void app_main(void) {
  esp_log_level_set(TAG, ESP_LOG_INFO);

  // initialize gpios
#ifdef CONFIG_LED_PIN_RED
  gpio_reset_pin(CONFIG_LED_PIN_RED);
  gpio_set_direction(CONFIG_LED_PIN_RED, GPIO_MODE_OUTPUT);
  gpio_set_level(CONFIG_LED_PIN_RED, 0);
#endif  // CONFIG_LED_PIN_RED
#ifdef CONFIG_LED_PIN_GREEN
  gpio_reset_pin(CONFIG_LED_PIN_GREEN);
  gpio_set_direction(CONFIG_LED_PIN_GREEN, GPIO_MODE_OUTPUT);
  gpio_set_level(CONFIG_LED_PIN_GREEN, 0);
#endif  // CONFIG_LED_PIN_GREEN
#ifdef CONFIG_LED_PIN_BLUE
  gpio_reset_pin(CONFIG_LED_PIN_BLUE);
  gpio_set_direction(CONFIG_LED_PIN_BLUE, GPIO_MODE_OUTPUT);
  gpio_set_level(CONFIG_LED_PIN_BLUE, 0);
#endif  // CONFIG_LED_PIN_BLUE
  // Setup ring buffer
  // buf_handle = xRingbufferCreate(RING_BUF_SIZE_BYTES, RINGBUF_TYPE_BYTEBUF);
  buf_handle = xRingbufferCreate(RING_BUF_SIZE_BYTES, RINGBUF_TYPE_NOSPLIT);
  if (buf_handle == NULL) {
    printf("Failed to create ring buffer\n");
  }

  // Configure parameters of an UART driver, communication pins and install the
  // driver
  uart_config_t uart_config = {
      .baud_rate = 115200,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .source_clk = UART_SCLK_APB,
  };
  // Install UART driver, and get the queue.
  uart_driver_install(EX_UART_NUM, BUF_SIZE * 2, BUF_SIZE * 2, 200,
                      &uart0_queue, 0);
  uart_param_config(EX_UART_NUM, &uart_config);

  // Set UART log level
  esp_log_level_set(TAG, ESP_LOG_INFO);
  // Set UART pins (using UART0 default pins ie no changes.)
  uart_set_pin(EX_UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
               UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);

  // Create a task to handler UART event from ISR
  xTaskCreate(uart_event_task, "uart_event_task", 2048, NULL, 12, NULL);

  // setup memory manager
  tvm_crt_error_t ret = PageMemoryManagerCreate(
      &g_memory_manager, tvm_heap, sizeof(tvm_heap), CRT_MEMORY_PAGE_SIZE_LOG2);

  if (ret != kTvmErrorNoError) {
    TVMPlatformAbort(ret);
  }

  // Initialize microTVM RPC server, which will receive commands from the UART
  // and execute them.
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM ESPIDF runtime - running\n");

  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (true) {
    uint8_t* data;
    uint8_t* data2;
    // unsigned int key = irq_lock(); // ??
    size_t bytes_read = 0;
    data2 =
        (uint8_t*)xRingbufferReceive(buf_handle, &bytes_read, pdMS_TO_TICKS(0));
    data = data2;

    if (bytes_read > 0) {
#ifdef CONFIG_LED_PIN_RED
      gpio_set_level(CONFIG_LED_PIN_GREEN, 1);
#endif
      g_num_bytes_in_rx_buffer -= bytes_read;
      size_t bytes_remaining = bytes_read;
      while (bytes_remaining > 0) {
        // Pass the received bytes to the RPC server.
        tvm_crt_error_t err =
            MicroTVMRpcServerLoop(server, &data, &bytes_remaining);
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
      vRingbufferReturnItem(buf_handle, (void*)data2);
    }
#ifdef CONFIG_LED_PIN_RED
    gpio_set_level(CONFIG_LED_PIN_GREEN, 0);
#endif
    // irq_unlock(key);  // ??
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }

  for (int i = 10; i >= 0; i--) {
    printf("Restarting in %d seconds...\n", i);
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
  printf("Restarting now.\n");
  fflush(stdout);
  esp_restart();
}
