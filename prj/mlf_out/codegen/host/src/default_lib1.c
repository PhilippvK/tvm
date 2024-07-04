// tvm target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) fused_constant_1[4] = {
    0x1.042c28p-1, 0x1.30629ap-3, 0x1.12ca6cp-1, 0x1.587ee8p-4
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) fused_constant[40] = {
    0x1.d56ac4p-1, 0x1.10aa4ap-3, 0x1.890156p-1, 0x1.b1bd5p-1, 0x1.216f78p-6, 0x1.fbfcd4p-1, 0x1.9ac71ep-1, 0x1.743dfp-1, 
    0x1.81a1ep-1, 0x1.e23384p-1, 0x1.0006ep-6, 0x1.cb2452p-1, 0x1.5a0868p-4, 0x1.711d2cp-1, 0x1.7f34fap-1, 0x1.6a0136p-5, 
    0x1.9ff01p-2, 0x1.561606p-2, 0x1.83e082p-1, 0x1.2c3f28p-1, 0x1.928d04p-3, 0x1.a219b8p-1, 0x1.398062p-1, 0x1.68e73ep-1, 
    0x1.814bf8p-1, 0x1.326e94p-6, 0x1.2327e6p-1, 0x1.01063ap-2, 0x1.df8576p-3, 0x1.82fe86p-2, 0x1.2fa32ep-2, 0x1.6e0f7ep-3, 
    0x1.bd8238p-4, 0x1.bc034cp-3, 0x1.5422f6p-1, 0x1.59fd86p-1, 0x1.fe8bdap-2, 0x1.45890cp-2, 0x1.b0cbd8p-3, 0x1.4cd4f6p-2
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(float* p0, float* T_add);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_matmul(float* p0, float* T_matmul_NN);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* x_buffer_var, float* output_buffer_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(float* p0, float* T_add) {
  for (int32_t ax0 = 0; ax0 < 64; ++ax0) {
    for (int32_t ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      int32_t cse_var_1 = ((ax0 * 4) + ax1_inner);
      T_add[cse_var_1] = (p0[cse_var_1] + ((float*)fused_constant_1)[ax1_inner]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_matmul(float* p0, float* T_matmul_NN) {
  for (int32_t i0 = 0; i0 < 64; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      T_matmul_NN[((i0 * 4) + i1)] = 0.000000e+00f;
      for (int32_t k = 0; k < 10; ++k) {
        int32_t cse_var_1 = ((i0 * 4) + i1);
        T_matmul_NN[cse_var_1] = (T_matmul_NN[cse_var_1] + (p0[((i0 * 10) + k)] * ((float*)fused_constant)[((k * 4) + i1)]));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* x_buffer_var, float* output_buffer_var) {
  void* sid_1 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1024, 0, 8);
  if (sid_1 == NULL) {
    return -1;
  }
  if (tvmgen_default_fused_nn_matmul(x_buffer_var, sid_1) != 0 ) return -1;
  if (tvmgen_default_fused_add(sid_1, output_buffer_var) != 0 ) return -1;
  if (TVMBackendFreeWorkspace(1, 0, sid_1) != 0) {
    return -1;
  }
  return 0;
}

