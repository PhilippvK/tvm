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
#include "uart.h"
#include "int.h"

#include "crt_config.h"
#include "ringbuffer.h"


//#define IRQ_DEBUG

/* For getting core ID. */
static inline int get_core_id()
{
  return 0;
}

// get number of cores
static inline int get_core_num()
{
  return 0;
}

//#ifdef __riscv__
/**
 * @brief Write to CSR.
 * @param CSR register to write.
 * @param Value to write to CSR register.
 * @return void
 *
 * Function to handle CSR writes.
 *
 */
#define csrw(csr, value)  asm volatile ("csrw\t\t" #csr ", %0" : /* no output */ : "r" (value));

/**
 * @brief Read from CSR.
 * @param void
 * @return 32-bit unsigned int
 *
 * Function to handle CSR reads.
 *
 */
#define csrr(csr, value)  asm volatile ("csrr\t\t%0, " #csr "": "=r" (value));

/**
 * @brief Request to put the core to sleep.
 * @param void
 *
 * Set the core to sleep state and wait for events/interrupt to wake up.
 *
 */
/*static inline void sleep(void) {
#ifdef SCR
    SCR = 0x01;
#endif
  asm volatile ("nop;nop;wfi");
}*/

/* Loops/exits simulation */
void exit(int i);

/* end of computation */
void eoc(int i);

// sleep some cycles
void sleep_busy(volatile int);

//#include "cust_print.h"
#define MSTATUS_UIE 0x00000001
#define MSTATUS_SIE 0x00000002
#define MSTATUS_HIE 0x00000004
#define MSTATUS_MIE 0x00000008
#define MSTATUS_UPIE 0x00000010
#define MSTATUS_SPIE 0x00000020
#define MSTATUS_HPIE 0x00000040
#define MSTATUS_MPIE 0x00000080
#define MSTATUS_SPP 0x00000100
#define MSTATUS_HPP 0x00000600
#define MSTATUS_MPP 0x00001800
#define MSTATUS_FS 0x00006000
#define MSTATUS_XS 0x00018000
#define MSTATUS_MPRV 0x00020000
#define MSTATUS_SUM 0x00040000
#define MSTATUS_MXR 0x00080000
#define MSTATUS_TVM 0x00100000
#define MSTATUS_TW 0x00200000
#define MSTATUS_TSR 0x00400000
#define MSTATUS32_SD 0x80000000
#define MSTATUS_UXL 0x0000000300000000
#define MSTATUS_SXL 0x0000000C00000000
#define MSTATUS64_SD 0x8000000000000000

#define SSTATUS_UIE 0x00000001
#define SSTATUS_SIE 0x00000002
#define SSTATUS_UPIE 0x00000010
#define SSTATUS_SPIE 0x00000020
#define SSTATUS_SPP 0x00000100
#define SSTATUS_FS 0x00006000
#define SSTATUS_XS 0x00018000
#define SSTATUS_SUM 0x00040000
#define SSTATUS_MXR 0x00080000
#define SSTATUS32_SD 0x80000000
#define SSTATUS_UXL 0x0000000300000000
#define SSTATUS64_SD 0x8000000000000000

#define MIP_SSIP (1 << IRQ_S_SOFT)
#define MIP_HSIP (1 << IRQ_H_SOFT)
#define MIP_MSIP (1 << IRQ_M_SOFT)
#define MIP_STIP (1 << IRQ_S_TIMER)
#define MIP_HTIP (1 << IRQ_H_TIMER)
#define MIP_MTIP (1 << IRQ_M_TIMER)
#define MIP_SEIP (1 << IRQ_S_EXT)
#define MIP_HEIP (1 << IRQ_H_EXT)
#define MIP_MEIP (1 << IRQ_M_EXT)

#define SIP_SSIP MIP_SSIP
#define SIP_STIP MIP_STIP

#define PRV_U 0
#define PRV_S 1
#define PRV_H 2
#define PRV_M 3

#define SATP32_MODE 0x80000000
#define SATP32_ASID 0x7FC00000
#define SATP32_PPN 0x003FFFFF
#define SATP64_MODE 0xF000000000000000
#define SATP64_ASID 0x0FFFF00000000000
#define SATP64_PPN 0x00000FFFFFFFFFFF

#define SATP_MODE_OFF 0
#define SATP_MODE_SV32 1
#define SATP_MODE_SV39 8
#define SATP_MODE_SV48 9
#define SATP_MODE_SV57 10
#define SATP_MODE_SV64 11

#define PMP_R 0x01
#define PMP_W 0x02
#define PMP_X 0x04
#define PMP_A 0x18
#define PMP_L 0x80
#define PMP_SHIFT 2

#define PMP_TOR 0x08
#define PMP_NA4 0x10
#define PMP_NAPOT 0x18

#define IRQ_S_SOFT 1
#define IRQ_H_SOFT 2
#define IRQ_M_SOFT 3
#define IRQ_S_TIMER 5
#define IRQ_H_TIMER 6
#define IRQ_M_TIMER 7
#define IRQ_S_EXT 9
#define IRQ_H_EXT 10
#define IRQ_M_EXT 11
#define IRQ_COP 12
#define IRQ_HOST 13

#define DEFAULT_RSTVEC 0x00001000
#define CLINT_BASE 0x02000000
#define CLINT_SIZE 0x000c0000
#define EXT_IO_BASE 0x40000000
#define DRAM_BASE 0x80000000

// page table entry (PTE) fields
#define PTE_V 0x001    // Valid
#define PTE_R 0x002    // Read
#define PTE_W 0x004    // Write
#define PTE_X 0x008    // Execute
#define PTE_U 0x010    // User
#define PTE_G 0x020    // Global
#define PTE_A 0x040    // Accessed
#define PTE_D 0x080    // Dirty
#define PTE_SOFT 0x300 // Reserved for Software

#define PTE_PPN_SHIFT 10

#define PTE_TABLE(PTE) (((PTE) & (PTE_V | PTE_R | PTE_W | PTE_X)) == PTE_V)

#define CSR_SSTATUS 0x100
#define CSR_SIE 0x104
#define CSR_STVEC 0x105
#define CSR_SCOUNTEREN 0x106
#define CSR_SSCRATCH 0x140
#define CSR_SEPC 0x141
#define CSR_SCAUSE 0x142
#define CSR_STVAL 0x143
#define CSR_SIP 0x144
#define CSR_SATP 0x180
#define CSR_MSTATUS 0x300
#define CSR_MISA 0x301
#define CSR_MEDELEG 0x302
#define CSR_MIDELEG 0x303
#define CSR_MIE 0x304
#define CSR_MTVEC 0x305
#define CSR_MCOUNTEREN 0x306
#define CSR_MSCRATCH 0x340
#define CSR_MEPC 0x341
#define CSR_MCAUSE 0x342
#define CSR_MTVAL 0x343
#define CSR_MIP 0x344

#define CAUSE_MISALIGNED_FETCH 0x0
#define CAUSE_FETCH_ACCESS 0x1
#define CAUSE_ILLEGAL_INSTRUCTION 0x2
#define CAUSE_BREAKPOINT 0x3
#define CAUSE_MISALIGNED_LOAD 0x4
#define CAUSE_LOAD_ACCESS 0x5
#define CAUSE_MISALIGNED_STORE 0x6
#define CAUSE_STORE_ACCESS 0x7
#define CAUSE_USER_ECALL 0x8
#define CAUSE_SUPERVISOR_ECALL 0x9
#define CAUSE_HYPERVISOR_ECALL 0xa
#define CAUSE_MACHINE_ECALL 0xb
#define CAUSE_FETCH_PAGE_FAULT 0xc
#define CAUSE_LOAD_PAGE_FAULT 0xd
#define CAUSE_STORE_PAGE_FAULT 0xf

//#include "uart_drv.h"
//#include "aes_ha_drv.h"

//#define ETISSVP_LOGGER_ADDR 0xf0000000
//#define ETISSVP_LOGGER ((volatile char*)ETISSVP_LOGGER_ADDR)

//#include "edaduino.h"
#define CLINT_BASE_ADDR               (0x02000000) //<< CLINT

/** Registers and pointers */
//#define REGP(x) ((volatile unsigned int*)(x))
//#define REG(x) (*((volatile unsigned int*)(x)))
//#define REGP_8(x) (((volatile uint8_t*)(x)))

//#include "utils.h"
//#include "stdint.h"

#define CLINT_MTIMECMPLO_OFFSET    0x4000
#define CLINT_MTIMECMPHI_OFFSET    (0x4000 + 4)
#define CLINT_MTIMELO_OFFSET       0xBFF8
#define CLINT_MTIMEHI_OFFSET       (0xBFF8 + 4)
#define CLINT_TICKS_OFFSET         (0x4000 + 8)

#define CLINT_MTIMECMPLO        (CLINT_BASE_ADDR + CLINT_MTIMECMPLO_OFFSET)
#define CLINT_MTIMECMPHI        (CLINT_BASE_ADDR + CLINT_MTIMECMPHI_OFFSET)

#define CLINT_MTIMELO           (CLINT_BASE_ADDR + CLINT_MTIMELO_OFFSET)
#define CLINT_MTIMEHI           (CLINT_BASE_ADDR + CLINT_MTIMEHI_OFFSET)

#define CLINT_TICKS           (CLINT_BASE_ADDR + CLINT_TICKS_OFFSET)

#define CLINT_TIMER_PERIOD_NS 30518

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

volatile uint32_t ticks = 0;

#define TICKS_FREQ 10000  // 1 Tick: 0.01ms

void ISR_MTIP (void) {
    //printf("CLINT\n");
    const uint64_t cmp_increment = 40;  // TODO
    uint32_t * mtimelo = (uint32_t*)(CLINT_MTIMELO);
    uint32_t * mtimehi = (uint32_t*)(CLINT_MTIMEHI);
    uint32_t * mtimecmplo = (uint32_t*)(CLINT_MTIMECMPLO);
    uint32_t * mtimecmphi = (uint32_t*)(CLINT_MTIMECMPHI);

    uint64_t cmp_new = ((uint64_t)*mtimehi << 32 | *mtimelo) + cmp_increment;

    *mtimecmplo = cmp_new & 0xffffffff;
    *mtimecmphi = cmp_new >> 32;

    ticks++;
}

volatile char lock = 0;

void ISR_MEIP() {
    /*if(lock) {
        printf("LOCKED\n");
    } else {
    //lock = 1;
    //printf("UART\n");
    char c = *(volatile int*)UART_REG_RBR;
    if (!buf_isfull(rx_buffer)) {
        buf_put_byte(rx_buffer, c);
        size_t l = buf_len(rx_buffer);
    } else {
      printf("ERR: buf full!\n");
      TVMPlatformAbort((tvm_crt_error_t)0xbeef7);
    }
    //lock = 0;
    }*/

}

unsigned long micros() {
  uint32_t * mtimelo = (uint32_t*)(CLINT_MTIMELO);
  uint32_t * mtimehi = (uint32_t*)(CLINT_MTIMEHI);
  //static long fake_ticks = 0;
  //fake_ticks += 100;
  //return fake_ticks * 1000000/TICKS_FREQ;
  //return ticks * 1000000/TICKS_FREQ;
  //uint32_t my_ticks = ((uint64_t)*((volatile int*)CLINT_TICKS)) * 1000000/TICKS_FREQ;
  // TODO: Does this need to be atomic?
  uint64_t micros = ((uint64_t)*mtimehi << 32 | *mtimelo) * CLINT_TIMER_PERIOD_NS / 1000;
  return micros;
}

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  g_num_bytes_requested += size;

  for (size_t i = 0; i < size; i++) {
    //uart_poll_out(tvm_uart, data[i]);
    //printf("uart_sendchar: %d (%c)\n", data[i], data[i]);
    uart_sendchar(data[i]);
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
  //TVMLogf("TVMError: 0x%x", error);
  printf("TVMError: 0x%x", error);
  //exit(1);
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

void clint_cfg_timecompare(uint64_t compare){
    uint32_t * mtimecmplo = (uint32_t*)(CLINT_MTIMECMPLO);
    uint32_t * mtimecmphi = (uint32_t*)(CLINT_MTIMECMPHI);

    *mtimecmplo = (uint32_t)compare;
    *mtimecmphi = (uint32_t)(compare >> 32);
}

void clint_cfg_timecompare_us(const uint32_t period_us){
    uint64_t compare = ((uint64_t)period_us*1000)/CLINT_TIMER_PERIOD_NS;
    clint_cfg_timecompare(compare);
}

void main(void) {

  // Claim console device.
  //tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  //uart_rx_init(&uart_rx_rbuf, tvm_uart);
  buf_reset(rx_buffer, BUFLEN);
  uint32_t systemtimer_us = 100;
  clint_cfg_timecompare_us(systemtimer_us);
  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  //TVMLogf("microTVM ETISSVP runtime - running");

  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (true) {
    
    /*//printf("LOOP\n");
    int x = 0, y = 1;
    //while (x < 100000) {
    while (x < 10000) {
      y++;
      x++;
    }
    if (x == y){
      x = 1;
      y = 0;
    }*/
    static uint8_t data[BUFLEN];
    //while(lock) {printf("LOCKED\n");}
    //lock = 1;
    //int_disable(); // TODO: reenable this?
    //size_t bytes_remaining = buf_len(rx_buffer);
    size_t bytes_remaining = 0;
    //while((*((volatile int*)UART_REG_LSR) & 0x1) != 0x1) {
    //  //printf(";\n");
    //}
    while((*((volatile int*)UART_REG_LSR) & 0x1) == 0x1) {
      char c = *(volatile int*)UART_REG_RBR;
      //printf("UART - %u [%02x]\n", c, c);
      data[bytes_remaining] = c;
      bytes_remaining++;
      //printf("LOOP\n");
      /*int x = 0, y = 1;
      while (x < 10) {
        y++;
        x++;
      }
      if (x == y){
        x = 1;
        y = 0;
      }*/
    }

    if (bytes_remaining > 0) {
      
      //printf("bytes_remaining=%ld\n", bytes_remaining);
      //for (size_t i = 0; i < bytes_remaining; i++) {
      //  data[i] = buf_get_byte(rx_buffer);
      //}
      //int_enable();
      //lock = 0;
      //printf("Micros: %lu\n", micros());
      // TODO: disable interrupts
      uint8_t* arr_ptr = data;
      while (bytes_remaining > 0) {
        printf("bytes_remaining_=%ld\n", bytes_remaining);
        // Pass the received bytes to the RPC server.
        tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &arr_ptr, &bytes_remaining);
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
      printf("done\n");
    } else {
      //int_enable();
    }
    //lock = 0;
    
    
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
    //break;
  }
  //TVMLogf("microTVM ETISSVP runtime - done");
  //for (;;) {}

}
