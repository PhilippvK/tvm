// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -link-params=0 -march=armv7e-m -mcpu=cortex-m7 -model=stm32f746xx
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>


#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_math.h>
#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>



#ifdef __cplusplus
extern "C"
#endif // __cplusplus
__STATIC_FORCEINLINE int32_t sum16_reset_SPAICCSW(
    int16_t *res) {
  *res = (int16_t)0;
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t sum16_2_SPAICCSW(
    int16_t *arr,
    int16_t *res16,
    long arr_offset,
    int reset) {
  int n;
  int32_t *p32;
  int32_t res = reset ? 0 : *res16;

  if ( arr_offset % 4 != 0 ) {
    res += *arr;
    p32 = (int32_t *)(&arr[1]);
    n = 2 - 1;
  } else {
    p32 = (int32_t *)arr;
    n = 2;
  }

  for ( int i = 0; i < n / 2; ++ i ) {
    res = __SMLAD(*p32, 0x00010001, res);
    ++ p32;
  }

  if ( n % 2 != 0 )
    res += *(int16_t *)p32;

  *res16 = res;

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_avg_pool2d(int16_t* placeholder, int32_t* tensor) {
  void* tensor1 = TVMBackendAllocWorkspace(1, 0, (uint64_t)7680, 0, 16);
  if (tensor1 == NULL) {
    return -1;
  }
  for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 24; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 20; ++ax3) {
        sum16_reset_SPAICCSW((&(((int16_t*)tensor1)[(((ax1 * 480) + (ax2 * 20)) + ax3)])));
        for (int32_t rv0 = 0; rv0 < 2; ++rv0) {
          int32_t cse_var_1 = ((((ax1 * 1960) + (ax2 * 80)) + (rv0 * 40)) + (ax3 * 2));
          sum16_2_SPAICCSW((&(placeholder[cse_var_1])), (&(((int16_t*)tensor1)[(((ax1 * 480) + (ax2 * 20)) + ax3)])), cse_var_1, 0);
        }
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 8; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 24; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 20; ++ax31) {
        int32_t cse_var_2 = (((ax11 * 480) + (ax21 * 20)) + ax31);
        tensor[cse_var_2] = (((int32_t)((int16_t*)tensor1)[cse_var_2]) / 4);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, 0, tensor1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int16_t* input_buffer_var, int16_t* output_buffer_var) {
  if (tvmgen_default_fused_nn_avg_pool2d(input_buffer_var, output_buffer_var) != 0 ) return -1;
  return 0;
}

