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
      for (int32_t k = 0; k < 4; ++k) {
        int32_t cse_var_1 = ((t * 128) + k);
        ((int8_t*)A)[cse_var_1] = ((((int8_t*)A)[cse_var_1] * ((int8_t*)A)[cse_var_1]) + ((int8_t*)A)[cse_var_1]);
      }
      for (int32_t k_1 = 0; k_1 < 4; ++k_1) {
        int32_t cse_var_2 = (((t * 128) + k_1) + 4);
        ((int8_t*)A)[cse_var_2] = ((((int8_t*)A)[cse_var_2] * ((int8_t*)A)[cse_var_2]) + ((int8_t*)A)[cse_var_2]);
      }
      for (int32_t k_2 = 0; k_2 < 4; ++k_2) {
        int32_t cse_var_3 = (((t * 128) + k_2) + 8);
        ((int8_t*)A)[cse_var_3] = ((((int8_t*)A)[cse_var_3] * ((int8_t*)A)[cse_var_3]) + ((int8_t*)A)[cse_var_3]);
      }
      for (int32_t k_3 = 0; k_3 < 4; ++k_3) {
        int32_t cse_var_4 = (((t * 128) + k_3) + 12);
        ((int8_t*)A)[cse_var_4] = ((((int8_t*)A)[cse_var_4] * ((int8_t*)A)[cse_var_4]) + ((int8_t*)A)[cse_var_4]);
      }
      for (int32_t k_4 = 0; k_4 < 4; ++k_4) {
        int32_t cse_var_5 = (((t * 128) + k_4) + 16);
        ((int8_t*)A)[cse_var_5] = ((((int8_t*)A)[cse_var_5] * ((int8_t*)A)[cse_var_5]) + ((int8_t*)A)[cse_var_5]);
      }
      for (int32_t k_5 = 0; k_5 < 4; ++k_5) {
        int32_t cse_var_6 = (((t * 128) + k_5) + 20);
        ((int8_t*)A)[cse_var_6] = ((((int8_t*)A)[cse_var_6] * ((int8_t*)A)[cse_var_6]) + ((int8_t*)A)[cse_var_6]);
      }
      for (int32_t k_6 = 0; k_6 < 4; ++k_6) {
        int32_t cse_var_7 = (((t * 128) + k_6) + 24);
        ((int8_t*)A)[cse_var_7] = ((((int8_t*)A)[cse_var_7] * ((int8_t*)A)[cse_var_7]) + ((int8_t*)A)[cse_var_7]);
      }
      for (int32_t k_7 = 0; k_7 < 4; ++k_7) {
        int32_t cse_var_8 = (((t * 128) + k_7) + 28);
        ((int8_t*)A)[cse_var_8] = ((((int8_t*)A)[cse_var_8] * ((int8_t*)A)[cse_var_8]) + ((int8_t*)A)[cse_var_8]);
      }
      for (int32_t k_8 = 0; k_8 < 4; ++k_8) {
        int32_t cse_var_9 = (((t * 128) + k_8) + 32);
        ((int8_t*)A)[cse_var_9] = ((((int8_t*)A)[cse_var_9] * ((int8_t*)A)[cse_var_9]) + ((int8_t*)A)[cse_var_9]);
      }
      for (int32_t k_9 = 0; k_9 < 4; ++k_9) {
        int32_t cse_var_10 = (((t * 128) + k_9) + 36);
        ((int8_t*)A)[cse_var_10] = ((((int8_t*)A)[cse_var_10] * ((int8_t*)A)[cse_var_10]) + ((int8_t*)A)[cse_var_10]);
      }
      for (int32_t k_10 = 0; k_10 < 4; ++k_10) {
        int32_t cse_var_11 = (((t * 128) + k_10) + 40);
        ((int8_t*)A)[cse_var_11] = ((((int8_t*)A)[cse_var_11] * ((int8_t*)A)[cse_var_11]) + ((int8_t*)A)[cse_var_11]);
      }
      for (int32_t k_11 = 0; k_11 < 4; ++k_11) {
        int32_t cse_var_12 = (((t * 128) + k_11) + 44);
        ((int8_t*)A)[cse_var_12] = ((((int8_t*)A)[cse_var_12] * ((int8_t*)A)[cse_var_12]) + ((int8_t*)A)[cse_var_12]);
      }
      for (int32_t k_12 = 0; k_12 < 4; ++k_12) {
        int32_t cse_var_13 = (((t * 128) + k_12) + 48);
        ((int8_t*)A)[cse_var_13] = ((((int8_t*)A)[cse_var_13] * ((int8_t*)A)[cse_var_13]) + ((int8_t*)A)[cse_var_13]);
      }
      for (int32_t k_13 = 0; k_13 < 4; ++k_13) {
        int32_t cse_var_14 = (((t * 128) + k_13) + 52);
        ((int8_t*)A)[cse_var_14] = ((((int8_t*)A)[cse_var_14] * ((int8_t*)A)[cse_var_14]) + ((int8_t*)A)[cse_var_14]);
      }
      for (int32_t k_14 = 0; k_14 < 4; ++k_14) {
        int32_t cse_var_15 = (((t * 128) + k_14) + 56);
        ((int8_t*)A)[cse_var_15] = ((((int8_t*)A)[cse_var_15] * ((int8_t*)A)[cse_var_15]) + ((int8_t*)A)[cse_var_15]);
      }
      for (int32_t k_15 = 0; k_15 < 4; ++k_15) {
        int32_t cse_var_16 = (((t * 128) + k_15) + 60);
        ((int8_t*)A)[cse_var_16] = ((((int8_t*)A)[cse_var_16] * ((int8_t*)A)[cse_var_16]) + ((int8_t*)A)[cse_var_16]);
      }
      for (int32_t k_16 = 0; k_16 < 4; ++k_16) {
        int32_t cse_var_17 = (((t * 128) + k_16) + 64);
        ((int8_t*)A)[cse_var_17] = ((((int8_t*)A)[cse_var_17] * ((int8_t*)A)[cse_var_17]) + ((int8_t*)A)[cse_var_17]);
      }
      for (int32_t k_17 = 0; k_17 < 4; ++k_17) {
        int32_t cse_var_18 = (((t * 128) + k_17) + 68);
        ((int8_t*)A)[cse_var_18] = ((((int8_t*)A)[cse_var_18] * ((int8_t*)A)[cse_var_18]) + ((int8_t*)A)[cse_var_18]);
      }
      for (int32_t k_18 = 0; k_18 < 4; ++k_18) {
        int32_t cse_var_19 = (((t * 128) + k_18) + 72);
        ((int8_t*)A)[cse_var_19] = ((((int8_t*)A)[cse_var_19] * ((int8_t*)A)[cse_var_19]) + ((int8_t*)A)[cse_var_19]);
      }
      for (int32_t k_19 = 0; k_19 < 4; ++k_19) {
        int32_t cse_var_20 = (((t * 128) + k_19) + 76);
        ((int8_t*)A)[cse_var_20] = ((((int8_t*)A)[cse_var_20] * ((int8_t*)A)[cse_var_20]) + ((int8_t*)A)[cse_var_20]);
      }
      for (int32_t k_20 = 0; k_20 < 4; ++k_20) {
        int32_t cse_var_21 = (((t * 128) + k_20) + 80);
        ((int8_t*)A)[cse_var_21] = ((((int8_t*)A)[cse_var_21] * ((int8_t*)A)[cse_var_21]) + ((int8_t*)A)[cse_var_21]);
      }
      for (int32_t k_21 = 0; k_21 < 4; ++k_21) {
        int32_t cse_var_22 = (((t * 128) + k_21) + 84);
        ((int8_t*)A)[cse_var_22] = ((((int8_t*)A)[cse_var_22] * ((int8_t*)A)[cse_var_22]) + ((int8_t*)A)[cse_var_22]);
      }
      for (int32_t k_22 = 0; k_22 < 4; ++k_22) {
        int32_t cse_var_23 = (((t * 128) + k_22) + 88);
        ((int8_t*)A)[cse_var_23] = ((((int8_t*)A)[cse_var_23] * ((int8_t*)A)[cse_var_23]) + ((int8_t*)A)[cse_var_23]);
      }
      for (int32_t k_23 = 0; k_23 < 4; ++k_23) {
        int32_t cse_var_24 = (((t * 128) + k_23) + 92);
        ((int8_t*)A)[cse_var_24] = ((((int8_t*)A)[cse_var_24] * ((int8_t*)A)[cse_var_24]) + ((int8_t*)A)[cse_var_24]);
      }
      for (int32_t k_24 = 0; k_24 < 4; ++k_24) {
        int32_t cse_var_25 = (((t * 128) + k_24) + 96);
        ((int8_t*)A)[cse_var_25] = ((((int8_t*)A)[cse_var_25] * ((int8_t*)A)[cse_var_25]) + ((int8_t*)A)[cse_var_25]);
      }
      for (int32_t k_25 = 0; k_25 < 4; ++k_25) {
        int32_t cse_var_26 = (((t * 128) + k_25) + 100);
        ((int8_t*)A)[cse_var_26] = ((((int8_t*)A)[cse_var_26] * ((int8_t*)A)[cse_var_26]) + ((int8_t*)A)[cse_var_26]);
      }
      for (int32_t k_26 = 0; k_26 < 4; ++k_26) {
        int32_t cse_var_27 = (((t * 128) + k_26) + 104);
        ((int8_t*)A)[cse_var_27] = ((((int8_t*)A)[cse_var_27] * ((int8_t*)A)[cse_var_27]) + ((int8_t*)A)[cse_var_27]);
      }
      for (int32_t k_27 = 0; k_27 < 4; ++k_27) {
        int32_t cse_var_28 = (((t * 128) + k_27) + 108);
        ((int8_t*)A)[cse_var_28] = ((((int8_t*)A)[cse_var_28] * ((int8_t*)A)[cse_var_28]) + ((int8_t*)A)[cse_var_28]);
      }
      for (int32_t k_28 = 0; k_28 < 4; ++k_28) {
        int32_t cse_var_29 = (((t * 128) + k_28) + 112);
        ((int8_t*)A)[cse_var_29] = ((((int8_t*)A)[cse_var_29] * ((int8_t*)A)[cse_var_29]) + ((int8_t*)A)[cse_var_29]);
      }
      for (int32_t k_29 = 0; k_29 < 4; ++k_29) {
        int32_t cse_var_30 = (((t * 128) + k_29) + 116);
        ((int8_t*)A)[cse_var_30] = ((((int8_t*)A)[cse_var_30] * ((int8_t*)A)[cse_var_30]) + ((int8_t*)A)[cse_var_30]);
      }
      for (int32_t k_30 = 0; k_30 < 4; ++k_30) {
        int32_t cse_var_31 = (((t * 128) + k_30) + 120);
        ((int8_t*)A)[cse_var_31] = ((((int8_t*)A)[cse_var_31] * ((int8_t*)A)[cse_var_31]) + ((int8_t*)A)[cse_var_31]);
      }
      for (int32_t k_31 = 0; k_31 < 4; ++k_31) {
        int32_t cse_var_32 = (((t * 128) + k_31) + 124);
        ((int8_t*)A)[cse_var_32] = ((((int8_t*)A)[cse_var_32] * ((int8_t*)A)[cse_var_32]) + ((int8_t*)A)[cse_var_32]);
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
