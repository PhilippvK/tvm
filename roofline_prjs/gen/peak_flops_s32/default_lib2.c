#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t a_code = arg_type_ids[0];
  void* a = (((TVMValue*)args)[0].v_handle);
  void* A = (((DLTensor*)a)[0].data);
  void* default_function_a_shape = (((DLTensor*)a)[0].shape);
  void* default_function_a_strides = (((DLTensor*)a)[0].strides);
  int32_t dev_id = (((DLTensor*)a)[0].device.device_id);
  if (!(default_function_a_strides == NULL)) {
  }
  for (int32_t t = 0; t < 8; ++t) {
    for (int32_t _j = 0; _j < 1000; ++_j) {
      int32_t cse_var_32 = (t * 32);
      int32_t cse_var_31 = (cse_var_32 + 9);
      int32_t cse_var_30 = (cse_var_32 + 8);
      int32_t cse_var_29 = (cse_var_32 + 7);
      int32_t cse_var_28 = (cse_var_32 + 6);
      int32_t cse_var_27 = (cse_var_32 + 5);
      int32_t cse_var_26 = (cse_var_32 + 4);
      int32_t cse_var_25 = (cse_var_32 + 31);
      int32_t cse_var_24 = (cse_var_32 + 30);
      int32_t cse_var_23 = (cse_var_32 + 3);
      int32_t cse_var_22 = (cse_var_32 + 29);
      int32_t cse_var_21 = (cse_var_32 + 28);
      int32_t cse_var_20 = (cse_var_32 + 27);
      int32_t cse_var_19 = (cse_var_32 + 26);
      int32_t cse_var_18 = (cse_var_32 + 25);
      int32_t cse_var_17 = (cse_var_32 + 24);
      int32_t cse_var_16 = (cse_var_32 + 23);
      int32_t cse_var_15 = (cse_var_32 + 22);
      int32_t cse_var_14 = (cse_var_32 + 21);
      int32_t cse_var_13 = (cse_var_32 + 20);
      int32_t cse_var_12 = (cse_var_32 + 2);
      int32_t cse_var_11 = (cse_var_32 + 19);
      int32_t cse_var_10 = (cse_var_32 + 18);
      int32_t cse_var_9 = (cse_var_32 + 17);
      int32_t cse_var_8 = (cse_var_32 + 16);
      int32_t cse_var_7 = (cse_var_32 + 15);
      int32_t cse_var_6 = (cse_var_32 + 14);
      int32_t cse_var_5 = (cse_var_32 + 13);
      int32_t cse_var_4 = (cse_var_32 + 12);
      int32_t cse_var_3 = (cse_var_32 + 11);
      int32_t cse_var_2 = (cse_var_32 + 10);
      int32_t cse_var_1 = (cse_var_32 + 1);
      ((int32_t*)A)[cse_var_32] = (((int32_t*)A)[cse_var_32] * (((int32_t*)A)[cse_var_32] + 1));
      ((int32_t*)A)[cse_var_1] = (((int32_t*)A)[cse_var_1] * (((int32_t*)A)[cse_var_1] + 1));
      ((int32_t*)A)[cse_var_12] = (((int32_t*)A)[cse_var_12] * (((int32_t*)A)[cse_var_12] + 1));
      ((int32_t*)A)[cse_var_23] = (((int32_t*)A)[cse_var_23] * (((int32_t*)A)[cse_var_23] + 1));
      ((int32_t*)A)[cse_var_26] = (((int32_t*)A)[cse_var_26] * (((int32_t*)A)[cse_var_26] + 1));
      ((int32_t*)A)[cse_var_27] = (((int32_t*)A)[cse_var_27] * (((int32_t*)A)[cse_var_27] + 1));
      ((int32_t*)A)[cse_var_28] = (((int32_t*)A)[cse_var_28] * (((int32_t*)A)[cse_var_28] + 1));
      ((int32_t*)A)[cse_var_29] = (((int32_t*)A)[cse_var_29] * (((int32_t*)A)[cse_var_29] + 1));
      ((int32_t*)A)[cse_var_30] = (((int32_t*)A)[cse_var_30] * (((int32_t*)A)[cse_var_30] + 1));
      ((int32_t*)A)[cse_var_31] = (((int32_t*)A)[cse_var_31] * (((int32_t*)A)[cse_var_31] + 1));
      ((int32_t*)A)[cse_var_2] = (((int32_t*)A)[cse_var_2] * (((int32_t*)A)[cse_var_2] + 1));
      ((int32_t*)A)[cse_var_3] = (((int32_t*)A)[cse_var_3] * (((int32_t*)A)[cse_var_3] + 1));
      ((int32_t*)A)[cse_var_4] = (((int32_t*)A)[cse_var_4] * (((int32_t*)A)[cse_var_4] + 1));
      ((int32_t*)A)[cse_var_5] = (((int32_t*)A)[cse_var_5] * (((int32_t*)A)[cse_var_5] + 1));
      ((int32_t*)A)[cse_var_6] = (((int32_t*)A)[cse_var_6] * (((int32_t*)A)[cse_var_6] + 1));
      ((int32_t*)A)[cse_var_7] = (((int32_t*)A)[cse_var_7] * (((int32_t*)A)[cse_var_7] + 1));
      ((int32_t*)A)[cse_var_8] = (((int32_t*)A)[cse_var_8] * (((int32_t*)A)[cse_var_8] + 1));
      ((int32_t*)A)[cse_var_9] = (((int32_t*)A)[cse_var_9] * (((int32_t*)A)[cse_var_9] + 1));
      ((int32_t*)A)[cse_var_10] = (((int32_t*)A)[cse_var_10] * (((int32_t*)A)[cse_var_10] + 1));
      ((int32_t*)A)[cse_var_11] = (((int32_t*)A)[cse_var_11] * (((int32_t*)A)[cse_var_11] + 1));
      ((int32_t*)A)[cse_var_13] = (((int32_t*)A)[cse_var_13] * (((int32_t*)A)[cse_var_13] + 1));
      ((int32_t*)A)[cse_var_14] = (((int32_t*)A)[cse_var_14] * (((int32_t*)A)[cse_var_14] + 1));
      ((int32_t*)A)[cse_var_15] = (((int32_t*)A)[cse_var_15] * (((int32_t*)A)[cse_var_15] + 1));
      ((int32_t*)A)[cse_var_16] = (((int32_t*)A)[cse_var_16] * (((int32_t*)A)[cse_var_16] + 1));
      ((int32_t*)A)[cse_var_17] = (((int32_t*)A)[cse_var_17] * (((int32_t*)A)[cse_var_17] + 1));
      ((int32_t*)A)[cse_var_18] = (((int32_t*)A)[cse_var_18] * (((int32_t*)A)[cse_var_18] + 1));
      ((int32_t*)A)[cse_var_19] = (((int32_t*)A)[cse_var_19] * (((int32_t*)A)[cse_var_19] + 1));
      ((int32_t*)A)[cse_var_20] = (((int32_t*)A)[cse_var_20] * (((int32_t*)A)[cse_var_20] + 1));
      ((int32_t*)A)[cse_var_21] = (((int32_t*)A)[cse_var_21] * (((int32_t*)A)[cse_var_21] + 1));
      ((int32_t*)A)[cse_var_22] = (((int32_t*)A)[cse_var_22] * (((int32_t*)A)[cse_var_22] + 1));
      ((int32_t*)A)[cse_var_24] = (((int32_t*)A)[cse_var_24] * (((int32_t*)A)[cse_var_24] + 1));
      ((int32_t*)A)[cse_var_25] = (((int32_t*)A)[cse_var_25] * (((int32_t*)A)[cse_var_25] + 1));
    }
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return default_function(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
