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
  int32_t b_code = arg_type_ids[1];
  void* a = (((TVMValue*)args)[0].v_handle);
  void* b = (((TVMValue*)args)[1].v_handle);
  void* A = (((DLTensor*)a)[0].data);
  void* default_function_a_shape = (((DLTensor*)a)[0].shape);
  void* default_function_a_strides = (((DLTensor*)a)[0].strides);
  int32_t dev_id = (((DLTensor*)a)[0].device.device_id);
  void* B = (((DLTensor*)b)[0].data);
  void* default_function_b_shape = (((DLTensor*)b)[0].shape);
  void* default_function_b_strides = (((DLTensor*)b)[0].strides);
  if (!(default_function_a_strides == NULL)) {
  }
  if (!(default_function_b_strides == NULL)) {
  }
  for (int32_t i = 0; i < 8; ++i) {
    for (int32_t k = 0; k < 7812; ++k) {
      for (int32_t j = 0; j < 4; ++j) {
        int32_t cse_var_1 = ((i * 16) + j);
        ((int32_t*)B)[cse_var_1] = (((int32_t*)B)[cse_var_1] + ((int32_t*)A)[(((i * 124992) + (k * 16)) + j)]);
      }
      for (int32_t j_1 = 0; j_1 < 4; ++j_1) {
        int32_t cse_var_2 = (((i * 16) + j_1) + 4);
        ((int32_t*)B)[cse_var_2] = (((int32_t*)B)[cse_var_2] + ((int32_t*)A)[((((i * 124992) + (k * 16)) + j_1) + 4)]);
      }
      for (int32_t j_2 = 0; j_2 < 4; ++j_2) {
        int32_t cse_var_3 = (((i * 16) + j_2) + 8);
        ((int32_t*)B)[cse_var_3] = (((int32_t*)B)[cse_var_3] + ((int32_t*)A)[((((i * 124992) + (k * 16)) + j_2) + 8)]);
      }
      for (int32_t j_3 = 0; j_3 < 4; ++j_3) {
        int32_t cse_var_4 = (((i * 16) + j_3) + 12);
        ((int32_t*)B)[cse_var_4] = (((int32_t*)B)[cse_var_4] + ((int32_t*)A)[((((i * 124992) + (k * 16)) + j_3) + 12)]);
      }
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
