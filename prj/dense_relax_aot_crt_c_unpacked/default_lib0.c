// tvm target: c -keys=cpu
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_0[40] = {
    0x1.1506f8p-1, 0x1.cb46bep-1, 0x1.b16bfep-3, 0x1.75b54ep-2, 0x1.ab0e98p-3, 0x1.09f1a8p-2, 0x1.a649f6p-2, 0x1.762bd2p-3,
    0x1.e2eb4p-5, 0x1.c794e8p-1, 0x1.33f248p-1, 0x1.105b82p-2, 0x1.01bedcp-4, 0x1.4081b2p-1, 0x1.813f64p-1, 0x1.003006p-1,
    0x1.7c3c5ep-3, 0x1.2f515ap-3, 0x1.b03d7ep-1, 0x1.487cb8p-2, 0x1.fa7c7p-1, 0x1.2578cp-1, 0x1.4a3b34p-1, 0x1.266d38p-1,
    0x1.d4bfb4p-1, 0x1.bfbda6p-4, 0x1.3129bep-3, 0x1.e6de7cp-1, 0x1.44b3b4p-3, 0x1.7e02b2p-1, 0x1.2551dap-1, 0x1.ecbb1cp-7,
    0x1.ffc8f4p-1, 0x1.2e4ea8p-1, 0x1.9784acp-1, 0x1.10e5d2p-1, 0x1.ac7936p-1, 0x1.5c9ab6p-1, 0x1.e69878p-1, 0x1.9727fcp-1
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_3[4] = {
    0x1.1ce288p-3, 0x1.c8659ep-5, 0x1.c8002ep-2, 0x1.559acep-2
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_1[40] = {
    0x1.1506f8p-1, 0x1.cb46bep-1, 0x1.b16bfep-3, 0x1.75b54ep-2, 0x1.ab0e98p-3, 0x1.09f1a8p-2, 0x1.a649f6p-2, 0x1.762bd2p-3,
    0x1.e2eb4p-5, 0x1.c794e8p-1, 0x1.33f248p-1, 0x1.105b82p-2, 0x1.01bedcp-4, 0x1.4081b2p-1, 0x1.813f64p-1, 0x1.003006p-1,
    0x1.7c3c5ep-3, 0x1.2f515ap-3, 0x1.b03d7ep-1, 0x1.487cb8p-2, 0x1.fa7c7p-1, 0x1.2578cp-1, 0x1.4a3b34p-1, 0x1.266d38p-1,
    0x1.d4bfb4p-1, 0x1.bfbda6p-4, 0x1.3129bep-3, 0x1.e6de7cp-1, 0x1.44b3b4p-3, 0x1.7e02b2p-1, 0x1.2551dap-1, 0x1.ecbb1cp-7,
    0x1.ffc8f4p-1, 0x1.2e4ea8p-1, 0x1.9784acp-1, 0x1.10e5d2p-1, 0x1.ac7936p-1, 0x1.5c9ab6p-1, 0x1.e69878p-1, 0x1.9727fcp-1
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_2[4] = {
    0x1.1ce288p-3, 0x1.c8659ep-5, 0x1.c8002ep-2, 0x1.559acep-2
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t add(float* A, float* B, float* T_add);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t matmul(float* A, float* B, float* matmul);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* x_buffer_var, float* output_buffer_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t add(float* A, float* B, float* T_add) {
  for (int32_t ax0 = 0; ax0 < 64; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
      int32_t cse_var_1 = ((ax0 * 4) + ax1);
      T_add[cse_var_1] = (A[cse_var_1] + B[ax1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t matmul(float* A, float* B, float* matmul) {
  for (int32_t i0 = 0; i0 < 64; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      for (int32_t k = 0; k < 10; ++k) {
        int32_t cse_var_1 = ((i0 * 4) + i1);
        if (k == 0) {
          matmul[cse_var_1] = 0.000000e+00f;
        }
        matmul[cse_var_1] = (matmul[cse_var_1] + (A[((i0 * 10) + k)] * B[((k * 4) + i1)]));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* x_buffer_var, float* output_buffer_var) {
  void* sid_2 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1024, 0, 8);
  if (sid_2 == NULL) {
    return -1;
  }
    uint64_t v_ = (uint64_t)0;
  matmul(x_buffer_var, constant_1, sid_2);
    uint64_t v__1 = (uint64_t)0;
  add(sid_2, constant_3, output_buffer_var);
  if (TVMBackendFreeWorkspace(1, 0, sid_2) != 0) {
    return -1;
  }
  return 0;
}

// // CodegenC: NOTE: Auto-generated entry function
// #ifdef __cplusplus
// extern "C"
// #endif
// TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
//   return tvmgen_default___tvm_main__(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
// }
