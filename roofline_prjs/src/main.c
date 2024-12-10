#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "tvm_wrapper.h"

#if defined(__riscv) || defined(__riscv__)
static inline uint64_t cycles()
{
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
}
#else
static inline uint64_t cycles(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}
#endif

#ifndef NUMBER
#define NUMBER 1
#endif  // NUMBER
#ifndef REPEAT
#define REPEAT 1
#endif  // NUMBER

int main() {
  int res = 0;
  TVMWrap_Init();
  for (size_t r = 0; r < REPEAT; r++) {
    clock_t tic = clock();
    uint64_t t0 = cycles();
    for (size_t n = 0; n < NUMBER; n++) {
      res |= TVMWrap_Run();
    }
    clock_t toc = clock();
    uint64_t t1 = cycles();
    clock_t diff = toc - tic;
    uint64_t td = t1 - t0;
    // printf("Time: %f s Cycles: %lu\n", (double)diff / CLOCKS_PER_SEC, td);
    printf("Time: %f s Cycles: %lu\n", (double)diff / CLOCKS_PER_SEC, td);
  }
  return 0;
}
