// tvm target: c -keys=riscv_cpu,cpu -device=riscv_cpu -link-params=0 -march=rv32gcp
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>


#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <rvp_intrinsic.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_reset_AYFSYTCM(
    int8_t *res,
    int N) {
  memset(res, (int8_t)-128, N * sizeof(*res));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_loop_AYFSYTCM(
    int8_t *arg,
    int8_t *res,
    int N) {
  for ( int i = 0; i < N; ++ i )
    if ( arg[i] > res[i] )
      res[i] = arg[i];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t max8_AYFSYTCM(
    int8_t *arg,
    int8_t *res,
    int N) {
  int32_t *parg32, *pres32;
  int una_arg = (int32_t)arg & 0x3, una_res = (int32_t)res & 0x3;
  int32_t retcode = 0;

  if ( N < 4 || ((una_arg || una_res) && una_arg != una_res) ) {
    retcode = max8_loop_AYFSYTCM(arg, res, N);
    goto out;
  }
  if ( una_arg ) {
    int n = (4 - una_arg);
    if ( n > N || (N - n) < 4 )
      n = N;
    retcode = max8_loop_AYFSYTCM(arg, res, n);
    N -= n;
    if ( N == 0 )
      goto out;
    arg += n; res += n;
  }

  parg32 = (int32_t *)arg;
  pres32 = (int32_t *)res;

  for ( int i = 0; i < N / 4; ++ i ) {
    int32_t arg32 = *parg32 ++;
    int32_t res32 = *pres32;
    res32 = __rv_smax8(arg32, res32);
    *pres32 ++ = res32;
  }

  if ( N & 0x3 ) {
    retcode = max8_loop_AYFSYTCM((int8_t *)parg32, (int8_t *)pres32, N & 0x3);
    goto out;
  }

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_max_pool2d(int8_t* placeholder, int8_t* tensor) {
  for (int32_t ax1 = 0; ax1 < 24; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 20; ++ax2) {
      max8_reset_AYFSYTCM((&(tensor[((ax1 * 160) + (ax2 * 8))])), 8);
      for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
        for (int32_t rv1 = 0; rv1 < 2; ++rv1) {
          max8_AYFSYTCM((&(placeholder[((((ax1 * 640) + (rv0 * 320)) + (ax2 * 16)) + (rv1 * 8))])), (&(tensor[((ax1 * 160) + (ax2 * 8))])), 8);
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
  if (tvmgen_default_fused_nn_max_pool2d(input_buffer_var, output_buffer_var) != 0 ) return -1;
  return 0;
}

