// tvm target: c -keys=cpu -link-params=0
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(int8_t* placeholder, int8_t* T_add) {
  for (int32_t ax0 = 0; ax0 < 16; ++ax0) {
    for (int32_t ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
        if (((ax1_outer * 16) + ax1_inner) < 29) {
          int32_t cse_var_1 = (((ax0 * 29) + (ax1_outer * 16)) + ax1_inner);
          T_add[cse_var_1] = (placeholder[cse_var_1] + placeholder[cse_var_1]);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* data_buffer_var, int8_t* output_buffer_var) {
  if (tvmgen_default_fused_add(data_buffer_var, output_buffer_var) != 0 ) return -1;
  return 0;
}

