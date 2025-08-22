// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -link-params=0 -march=armv7e-m -mcpu=cortex-m7 -model=stm32f746xx
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// #include <arm_math.h>
// #include <arm_nnsupportfunctions.h>
#include "riscv_vector.h"

#include <tvm/runtime/crt/error_codes.h>

#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE                   __attribute__((always_inline)) static inline
#endif


#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_reset_NRBPFSMA(
    int8_t *res,
    int N) {
  memset(res, (int8_t)-128, N * sizeof(*res));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_loop_NRBPFSMA(
    int8_t *arg,
    int8_t *res,
    int N) {
  printf("&arg=%p &res=%p N=%d\n", arg, res, N);
  // for ( int i = 0; i < N; ++ i )
  //   if ( arg[i] > res[i] )
  //     res[i] = arg[i];
  ////
  size_t vl = 0;
  size_t cnt;
  for (cnt = N; cnt > 0; cnt -= vl, arg += vl, res += vl) {
    printf("cnt=%u\n", cnt);
    vl = vsetvl_e8m8(cnt);

    /* Load the vector groups from memory. */
    vint8m8_t op_1 = vle8_v_i8m8(arg, vl);
    vint8m8_t op_2 = vle8_v_i8m8(res, vl);

    /* Apply max operation on whole vector group at once. */
    vint8m8_t max = vmax_vv_i8m8(op_1, op_2, vl);

    /* Store result back to memory. */
    vse8_v_i8m8(res, max, vl);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_NRBPFSMA(
    int8_t *arg,
    int8_t *res,
    int N) {
  int32_t *parg32, *pres32;
  int una_arg = (int32_t)arg & 0x3, una_res = (int32_t)res & 0x3;
  int32_t retcode = 0;

  // if ( N < 4 || ((una_arg || una_res) && una_arg != una_res) ) {
  if (1) {
    retcode = max8_loop_NRBPFSMA(arg, res, N);
    goto out;
  }
  // if ( una_arg ) {
  //   int n = (4 - una_arg);
  //   if ( n > N || (N - n) < 4 )
  //     n = N;
  //   retcode = max8_loop_NRBPFSMA(arg, res, n);
  //   N -= n;
  //   if ( N == 0 )
  //     goto out;
  //   arg += n; res += n;
  // }

  // parg32 = (int32_t *)arg;
  // pres32 = (int32_t *)res;

  // for ( int i = 0; i < N / 4; ++ i ) {
  //   int32_t arg32 = *parg32 ++;
  //   int32_t res32 = *pres32;
  //   __SSUB8(arg32, res32);
  //   res32 = __SEL(arg32, res32);
  //   *pres32 ++ = res32;
  // }

  // if ( N & 0x3 ) {
  //   retcode = max8_loop_NRBPFSMA((int8_t *)parg32, (int8_t *)pres32, N & 0x3);
  //   goto out;
  // }

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_layout_transform(int8_t* placeholder, int8_t* T_layout_trans) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1960; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      T_layout_trans[((ax0_ax1_fused_ax2_fused * 8) + ax3_inner)] = placeholder[((ax3_inner * 1960) + ax0_ax1_fused_ax2_fused)];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_layout_transform_1(int8_t* placeholder, int8_t* T_layout_trans) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 192; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        if (((ax3_outer * 4) + (ax3_inner >> 2)) < 5) {
          T_layout_trans[(((ax0_ax1_fused_ax2_fused * 20) + (ax3_outer * 16)) + ax3_inner)] = placeholder[(((((ax0_ax1_fused_ax2_fused % 24) * 160) + (ax3_outer * 128)) + (ax3_inner * 8)) + (ax0_ax1_fused_ax2_fused / 24))];
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d(int8_t* placeholder, int8_t* tensor) {
  for (int32_t ax1 = 0; ax1 < 24; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
      max8_reset_NRBPFSMA((&(tensor[((ax1 * 160) + (ax2 * 8))])), 8);
      for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
          max8_NRBPFSMA((&(placeholder[((((ax1 * 640) + (rv0 * 320)) + (ax2 * 16)) + (rv1 * 8))])), (&(tensor[((ax1 * 160) + (ax2 * 8))])), 8);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_buffer_var, int8_t* output_buffer_var) {
  void* sid_3 = TVMBackendAllocWorkspace(1, 0, (uint64_t)15680, 0, 8);
  if (sid_3 == NULL) {
    return -1;
  }
  void* sid_2 = TVMBackendAllocWorkspace(1, 0, (uint64_t)3840, 0, 8);
  if (sid_2 == NULL) {
    return -1;
  }
  if (tvmgen_default_fused_layout_transform(input_buffer_var, sid_3) != 0 ) return -1;
  if (tvmgen_default_fused_nn_max_pool2d(sid_3, sid_2) != 0 ) return -1;
  if (tvmgen_default_fused_layout_transform_1(sid_2, output_buffer_var) != 0 ) return -1;
  if (TVMBackendFreeWorkspace(1, 0, sid_2) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, sid_3) != 0) {
    return -1;
  }
  return 0;
}

