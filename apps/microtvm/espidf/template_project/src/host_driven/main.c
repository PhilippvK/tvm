/* Hello World Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_spi_flash.h"

#include <driver/gpio.h>
#include <driver/uart.h>
// #include <random/rand32.h>
// #include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <unistd.h>

#include "crt_config.h"

// static const struct device* tvm_uart;
static const char *TAG = "microtvm";

#ifdef CONFIG_LED_PIN
// #define LED0_NODE DT_ALIAS(led0)
// #define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
// #define LED0_PIN DT_GPIO_PIN(LED0_NODE, gpios)
// #define LED0_FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)
// static const struct device* led0_pin;
#endif  // CONFIG_LED

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;
static size_t g_num_bytes_in_rx_buffer = 0;

// void TVMLogf(const char* msg, ...) {
//     va_list args;
//     va_start(args, msg);
//     printf(msg, args);
//     va_end(args);
// }

#define EX_UART_NUM UART_NUM_0

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
#ifdef CONFIG_LED_PIN
  // gpio_pin_set(led0_pin, LED0_PIN, 1);
  gpio_set_level(CONFIG_LED_PIN, 1);
#endif
  g_num_bytes_requested += size;

  uart_write_bytes(EX_UART_NUM, data, size);
  g_num_bytes_written += size;

#ifdef CONFIG_LED_PIN
  gpio_set_level(CONFIG_LED_PIN, 0);
#endif

  return size;
}

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  // return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}


// Called by TVM when an internal invariant is violated, and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  // sys_reboot(SYS_REBOOT_COLD);
  esp_restart();
#ifdef CONFIG_LED_PIN
  gpio_set_level(CONFIG_LED_PIN, 1);
#endif
  for (;;)
    ;
}


// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  printf("TVMPlatformGenerateRandom\n");
  uint32_t random;  // one unit of random data.

  // Fill parts of `buffer` which are as large as `random`.
  size_t num_full_blocks = num_bytes / sizeof(random);
  for (int i = 0; i < num_full_blocks; ++i) {
    // random = sys_rand32_get();
    random = 0;  // TODO(@PhilippvK)
    memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  }

  // Fill any leftover tail which is smaller than `random`.
  size_t num_tail_bytes = num_bytes % sizeof(random);
  if (num_tail_bytes > 0) {
    // random = sys_rand32_get();
    random = 0;  // TODO(@PhilippvK)
    memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  }
  return kTvmErrorNoError;
}


#define CRT_MEMORY_NUM_PAGES 216
#define CRT_MEMORY_PAGE_SIZE_LOG2 10


// Heap for use by TVMPlatformMemoryAllocate.
// K_HEAP_DEFINE(tvm_heap, 216 * 1024);
static uint8_t tvm_heap[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];
static MemoryManagerInterface* g_memory_manager;


tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  // *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  return g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return g_memory_manager->Free(g_memory_manager, ptr, dev);
}


// #define MILLIS_TIL_EXPIRY 200
// #define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
// K_TIMER_DEFINE(g_microtvm_timer, /* expiry func */ NULL, /* stop func */ NULL);
// TODO

uint32_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

#ifdef CONFIG_LED_PIN
  gpio_set_level(CONFIG_LED_PIN, 1);
#endif
  // k_timer_start(&g_microtvm_timer, TIME_TIL_EXPIRY, TIME_TIL_EXPIRY);
  // g_microtvm_start_time = k_cycle_get_32();
  esp_cpu_ccount_t ccount = esp_cpu_get_ccount();
  g_microtvm_start_time = ccount;
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

  // uint32_t stop_time = k_cycle_get_32();
  esp_cpu_ccount_t ccount = esp_cpu_get_ccount();
  uint32_t stop_time = ccount;
#ifdef CONFIG_LED_PIN
  gpio_set_level(CONFIG_LED_PIN, 0);
#endif

  // compute how long the work took
  uint32_t cycles_spent = stop_time - g_microtvm_start_time;
  if (stop_time < g_microtvm_start_time) {
    // we rolled over *at least* once, so correct the rollover it was *only*
    // once, because we might still use this result
    cycles_spent = ~((uint32_t)0) - (g_microtvm_start_time - stop_time);
  }

  // uint32_t ns_spent = (uint32_t)k_cyc_to_ns_floor64(cycles_spent);
  // uint32_t ns_spent = cycles_spent /100 * 625;  // (1000000000/160000000)
  // TODO: get frequency?
  double hw_clock_res_us = cycles_spent / 6.25;

  // need to grab time remaining *before* stopping. when stopped, this function
  // always returns 0.
  // int32_t time_remaining_ms = k_timer_remaining_get(&g_microtvm_timer);
  // k_timer_stop(&g_microtvm_timer);
  // // check *after* stopping to prevent extra expiries on the happy path
  // if (time_remaining_ms < 0) {
  //   TVMLogf("negative time remaining");
  //   return kTvmErrorSystemErrorMask | 3;
  // }
  // uint32_t num_expiries = k_timer_status_get(&g_microtvm_timer);
  // uint32_t timer_res_ms = ((num_expiries * MILLIS_TIL_EXPIRY) + time_remaining_ms);
  // double approx_num_cycles =
  //     (double)k_ticks_to_cyc_floor32(1) * (double)k_ms_to_ticks_ceil32(timer_res_ms);
  // // if we approach the limits of the HW clock datatype (uint32_t), use the
  // // coarse-grained timer result instead
  // if (approx_num_cycles > (0.5 * (~((uint32_t)0)))) {
  //   *elapsed_time_seconds = timer_res_ms / 1000.0;
  // } else {
  //   *elapsed_time_seconds = hw_clock_res_us / 1e6;
  // }
  *elapsed_time_seconds = hw_clock_res_us / 1e6;  // TODO: overflow possible!

  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

// Ring buffer used to store data read from the UART on rx interrupt.
// This ring buffer size is only required for testing with QEMU and not for physical hardware.
#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)
// RING_BUF_ITEM_DECLARE_SIZE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);
// RING_BUF_ITEM_DECLARE_SIZE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);
static RingbufHandle_t buf_handle;

// UART interrupt callback.
// void uart_irq_cb(const struct device* dev, void* user_data) {

#define BUF_SIZE (1024)
#define RD_BUF_SIZE (BUF_SIZE)
static QueueHandle_t uart0_queue;

static void uart_event_task(void *pvParameters)
{
    uart_event_t event;
    // size_t buffered_size;
    uint8_t* data;
    for(;;) {
        //Waiting for UART event.
        if(xQueueReceive(uart0_queue, (void * )&event, (portTickType)portMAX_DELAY)) {
            // bzero(dtmp, RD_BUF_SIZE);
            ESP_LOGI(TAG, "uart[%d] event:", EX_UART_NUM);
            switch(event.type) {
                //Event of UART receving data
                /*We'd better handle data event fast, there would be much more data events than
                other types of events. If we take too much time on data event, the queue might
                be full.*/
                case UART_DATA:
                    ESP_LOGI(TAG, "[UART DATA]: %d", event.size);
                    ////
                    // struct ring_buf* rbuf = (struct ring_buf*)user_data;
                    // if (uart_irq_rx_ready(dev) != 0) {
                    // size = ring_buf_put_claim(rbuf, &data, RING_BUF_SIZE_BYTES);
                    UBaseType_t res =  xRingbufferSendAcquire(buf_handle, (void**)&data, event.size, pdMS_TO_TICKS(1000));
                    uart_read_bytes(EX_UART_NUM, data, event.size, portMAX_DELAY);
                    if (res != pdTRUE) {
                      ESP_LOGI(TAG, "Failed to acquire memory for data\n");
                      // TVMPlatformAbort((tvm_crt_error_t)0xbeef4);
                      break;
                    }
                    // int rx_size = uart_fifo_read(dev, data, event.size);
                    // Write it into the ring buffer.
                    g_num_bytes_in_rx_buffer += event.size;

                    if (g_num_bytes_in_rx_buffer > RING_BUF_SIZE_BYTES) {
                      TVMPlatformAbort((tvm_crt_error_t)0xbeef3);
                    }

                    // int err = ring_buf_put_finish(rbuf, rx_size);
                    res = xRingbufferSendComplete(buf_handle, (void**)&data);
                    if (res != pdTRUE) {
                      ESP_LOGI(TAG, "Failed to send item\n");
                      TVMPlatformAbort((tvm_crt_error_t)0xbeef4);
                    }
                    // CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
                    // bytes_written);
                    // }
                    ////
                    // ESP_LOGI(TAG, "[DATA EVT]:");
                    // uart_write_bytes(EX_UART_NUM, (const char*) dtmp, event.size);
                    break;
                //Event of HW FIFO overflow detected
                case UART_FIFO_OVF:
                    ESP_LOGI(TAG, "hw fifo overflow");
                    // If fifo overflow happened, you should consider adding flow control for your application.
                    // The ISR has already reset the rx FIFO,
                    // As an example, we directly flush the rx buffer here in order to read more data.
                    uart_flush_input(EX_UART_NUM);
                    xQueueReset(uart0_queue);
                    break;
                //Event of UART ring buffer full
                case UART_BUFFER_FULL:
                    ESP_LOGI(TAG, "ring buffer full");
                    // If buffer full happened, you should consider encreasing your buffer size
                    // As an example, we directly flush the rx buffer here in order to read more data.
                    uart_flush_input(EX_UART_NUM);
                    xQueueReset(uart0_queue);
                    break;
                //Event of UART RX break detected
                case UART_BREAK:
                    ESP_LOGI(TAG, "uart rx break");
                    break;
                //Event of UART parity check error
                case UART_PARITY_ERR:
                    ESP_LOGI(TAG, "uart parity error");
                    break;
                //Event of UART frame error
                case UART_FRAME_ERR:
                    ESP_LOGI(TAG, "uart frame error");
                    break;
                //UART_PATTERN_DET
                case UART_PATTERN_DET:
                    ESP_LOGI(TAG, "uart pattern detect");
                    // uart_get_buffered_data_len(EX_UART_NUM, &buffered_size);
                    // int pos = uart_pattern_pop_pos(EX_UART_NUM);
                    // ESP_LOGI(TAG, "[UART PATTERN DETECTED] pos: %d, buffered size: %d", pos, buffered_size);
                    // if (pos == -1) {
                    //     // There used to be a UART_PATTERN_DET event, but the pattern position queue is full so that it can not
                    //     // record the position. We should set a larger queue size.
                    //     // As an example, we directly flush the rx buffer here.
                    //     uart_flush_input(EX_UART_NUM);
                    // } else {
                    //     uart_read_bytes(EX_UART_NUM, dtmp, pos, 100 / portTICK_PERIOD_MS);
                    //     uint8_t pat[PATTERN_CHR_NUM + 1];
                    //     memset(pat, 0, sizeof(pat));
                    //     uart_read_bytes(EX_UART_NUM, pat, PATTERN_CHR_NUM, 100 / portTICK_PERIOD_MS);
                    //     ESP_LOGI(TAG, "read data: %s", dtmp);
                    //     ESP_LOGI(TAG, "read pat : %s", pat);
                    // }
                    break;
                //Others
                default:
                    ESP_LOGI(TAG, "uart event type: %d", event.type);
                    break;
            }
        }
    }
    // free(dtmp);
    // dtmp = NULL;
    vTaskDelete(NULL);
}


// Used to initialize the UART receiver.
// void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
void uart_rx_init() {
  // uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);  // ?
  // uart_irq_rx_enable(dev); // ?

  // configure uart
  /* Configure parameters of an UART driver,
     * communication pins and install the driver */
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    //Install UART driver, and get the queue.
    uart_driver_install(EX_UART_NUM, BUF_SIZE * 2, BUF_SIZE * 2, 20, &uart0_queue, 0);
    uart_param_config(EX_UART_NUM, &uart_config);

    //Set UART log level
    esp_log_level_set(TAG, ESP_LOG_INFO);
    //Set UART pins (using UART0 default pins ie no changes.)
    uart_set_pin(EX_UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    xTaskCreate(uart_event_task, "uart_event_task", 2048, NULL, 12, NULL);
}


void app_main(void) {
  printf("Hello world!\n");

  esp_log_level_set(TAG, ESP_LOG_DEBUG);

  // setup memory manager
  tvm_crt_error_t ret = PageMemoryManagerCreate(&g_memory_manager, tvm_heap, sizeof(tvm_heap), CRT_MEMORY_PAGE_SIZE_LOG2);

  if (ret != kTvmErrorNoError) {
    TVMLogf("%s: %d: error: %s\\n", __FILE__, __LINE__, TVMGetLastError());
    TVMPlatformAbort(ret);
  }


  // /* Print chip information */
  // esp_chip_info_t chip_info;
  // esp_chip_info(&chip_info);
  // printf("This is %s chip with %d CPU core(s), WiFi%s%s, ",
  //         CONFIG_IDF_TARGET,
  //         chip_info.cores,
  //         (chip_info.features & CHIP_FEATURE_BT) ? "/BT" : "",
  //         (chip_info.features & CHIP_FEATURE_BLE) ? "/BLE" : "");

  // printf("silicon revision %d, ", chip_info.revision);

  // printf("%dMB %s flash\n", spi_flash_get_chip_size() / (1024 * 1024),
  //         (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");

  // printf("Minimum free heap size: %d bytes\n", esp_get_minimum_free_heap_size());

#ifdef CONFIG_LED_PIN
  gpio_reset_pin(CONFIG_LED_PIN);
  gpio_set_direction(CONFIG_LED_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(CONFIG_LED_PIN, 1);
#endif

  // Claim console device.
  // tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  uart_rx_init();


  // Setup ring buffer
  buf_handle = xRingbufferCreate(RING_BUF_SIZE_BYTES, RINGBUF_TYPE_NOSPLIT);
  if (buf_handle == NULL) {
      printf("Failed to create ring buffer\n");
  }

  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM Zephyr runtime - running");
#ifdef CONFIG_LED
  gpio_set_level(CONFIG_LED_PIN, 0);
#endif

  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (true) {
    uint8_t* data;
    // unsigned int key = irq_lock(); // ??
    // uint32_t bytes_read = ring_buf_get_claim(&uart_rx_rbuf, &data, RING_BUF_SIZE_BYTES);
    size_t bytes_read = 0;
    data = (uint8_t*)xRingbufferReceiveUpTo(buf_handle, &bytes_read,
                                            pdMS_TO_TICKS(0), RING_BUF_SIZE_BYTES);

    if (bytes_read > 0) {
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
      // int err = ring_buf_get_finish(&uart_rx_rbuf, bytes_read);
      vRingbufferReturnItem(buf_handle, (void*)data);
    }
    // irq_unlock(key);  // ??
  }

  for (int i = 10; i >= 0; i--) {
    printf("Restarting in %d seconds...\n", i);
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
  printf("Restarting now.\n");
  fflush(stdout);
  esp_restart();
}
