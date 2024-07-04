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
    0x1.ff54aap-2, 0x1.2e91b2p-1, 0x1.4a4ea2p-5, 0x1.1c1cd6p-1
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) fused_constant[40] = {
    0x1.8af106p-1, 0x1.efa568p-1, 0x1.5fce94p-3, 0x1.9fe4fp-2, 0x1.2e053ap-1, 0x1.11b868p-2, 0x1.f0bd94p-3, 0x1.e67d3ap-8,
    0x1.92cfeep-1, 0x1.014aap-1, 0x1.870b04p-3, 0x1.cdabeap-1, 0x1.2de662p-1, 0x1.e542ep-3, 0x1.62b514p-1, 0x1.9003dep-2,
    0x1.0e0754p-1, 0x1.e22f6cp-1, 0x1.e5254ap-1, 0x1.40f262p-1, 0x1.d91ec2p-4, 0x1.02e2fcp-2, 0x1.4ed4aep-2, 0x1.22b67cp-1,
    0x1.70882p-2, 0x1.90bfb8p-1, 0x1.c5ff32p-2, 0x1.486dacp-1, 0x1.0aa278p-2, 0x1.e4323cp-1, 0x1.3ab11p-1, 0x1.e3a59ep-4,
    0x1.f98134p-1, 0x1.a38decp-1, 0x1.df5606p-1, 0x1.5037c8p-1, 0x1.91f1c6p-1, 0x1.672e44p-2, 0x1.d9ddf8p-1, 0x1.4cba4ep-2
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_matmul(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t p0_code = arg_type_ids[0];
  int32_t T_add_code = arg_type_ids[1];
  void* p0 = (((TVMValue*)args)[0].v_handle);
  void* T_add = (((TVMValue*)args)[1].v_handle);
  void* tvmgen_default_fused_add_p0_shape = (((DLTensor*)p0)[0].shape);
  void* tvmgen_default_fused_add_p0_strides = (((DLTensor*)p0)[0].strides);
  int32_t dev_id = (((DLTensor*)p0)[0].device.device_id);
  void* p0_1 = (((DLTensor*)p0)[0].data);
  void* tvmgen_default_fused_add_T_add_shape = (((DLTensor*)T_add)[0].shape);
  void* tvmgen_default_fused_add_T_add_strides = (((DLTensor*)T_add)[0].strides);
  void* T_add_1 = (((DLTensor*)T_add)[0].data);
  if (!(tvmgen_default_fused_add_p0_strides == NULL)) {
  }
  if (!(tvmgen_default_fused_add_T_add_strides == NULL)) {
  }
  for (int32_t ax0 = 0; ax0 < 64; ++ax0) {
    for (int32_t ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      int32_t cse_var_1 = ((ax0 * 4) + ax1_inner);
      ((float*)T_add_1)[cse_var_1] = (((float*)p0_1)[cse_var_1] + ((float*)fused_constant_1)[ax1_inner]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_matmul(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t p0_code = arg_type_ids[0];
  int32_t T_matmul_NN_code = arg_type_ids[1];
  void* p0 = (((TVMValue*)args)[0].v_handle);
  void* T_matmul_NN = (((TVMValue*)args)[1].v_handle);
  void* tvmgen_default_fused_nn_matmul_p0_shape = (((DLTensor*)p0)[0].shape);
  void* tvmgen_default_fused_nn_matmul_p0_strides = (((DLTensor*)p0)[0].strides);
  int32_t dev_id = (((DLTensor*)p0)[0].device.device_id);
  void* p0_1 = (((DLTensor*)p0)[0].data);
  void* tvmgen_default_fused_nn_matmul_T_matmul_NN_shape = (((DLTensor*)T_matmul_NN)[0].shape);
  void* tvmgen_default_fused_nn_matmul_T_matmul_NN_strides = (((DLTensor*)T_matmul_NN)[0].strides);
  void* T_matmul_NN_1 = (((DLTensor*)T_matmul_NN)[0].data);
  if (!(tvmgen_default_fused_nn_matmul_p0_strides == NULL)) {
  }
  if (!(tvmgen_default_fused_nn_matmul_T_matmul_NN_strides == NULL)) {
  }
  for (int32_t i0 = 0; i0 < 64; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      ((float*)T_matmul_NN_1)[((i0 * 4) + i1)] = 0.000000e+00f;
      for (int32_t k = 0; k < 10; ++k) {
        int32_t cse_var_1 = ((i0 * 4) + i1);
        ((float*)T_matmul_NN_1)[cse_var_1] = (((float*)T_matmul_NN_1)[cse_var_1] + (((float*)p0_1)[((i0 * 10) + k)] * ((float*)fused_constant)[((k * 4) + i1)]));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  TVMValue stack[2];
  void* stack_tcode = stack;
  TVMValue stack_1[3];
  void* stack_value = stack_1;
  TVMValue stack_2[12];
  void* stack_array = stack_2;
  TVMValue stack_3[4];
  void* stack_shape = stack_3;
  int32_t x_code = arg_type_ids[0];
  int32_t output_code = arg_type_ids[1];
  void* x = (((TVMValue*)args)[0].v_handle);
  void* output = (((TVMValue*)args)[1].v_handle);
  void* tvmgen_default___tvm_main___x_shape = (((DLTensor*)x)[0].shape);
  void* tvmgen_default___tvm_main___x_strides = (((DLTensor*)x)[0].strides);
  int32_t dev_id = (((DLTensor*)x)[0].device.device_id);
  void* x_buffer_var = (((DLTensor*)x)[0].data);
  void* tvmgen_default___tvm_main___output_shape = (((DLTensor*)output)[0].shape);
  void* tvmgen_default___tvm_main___output_strides = (((DLTensor*)output)[0].strides);
  void* output_buffer_var = (((DLTensor*)output)[0].data);
  if (!(tvmgen_default___tvm_main___x_strides == NULL)) {
  }
  if (!(tvmgen_default___tvm_main___output_strides == NULL)) {
  }
  void* sid_1 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 0, 8);
  if (sid_1 == NULL) {
    return -1;
  }
  ((int64_t*)stack_shape)[0] = (int64_t)64;
  ((int64_t*)stack_shape)[1] = (int64_t)10;
  (((DLTensor*)stack_array)[0].data) = x_buffer_var;
  (((DLTensor*)stack_array)[0].shape) = (&(((int64_t*)stack_shape)[0]));
    uint64_t v_ = (uint64_t)0;
  (((DLTensor*)stack_array)[0].strides) = (int64_t*)(*(void* *)(&(v_)));
  (((DLTensor*)stack_array)[0].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[0].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[0].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[0].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[0].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[0].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[0].device.device_type) = (DLDeviceType)1;
  ((int64_t*)stack_shape)[2] = (int64_t)64;
  ((int64_t*)stack_shape)[3] = (int64_t)4;
  (((DLTensor*)stack_array)[1].data) = sid_1;
  (((DLTensor*)stack_array)[1].shape) = (&(((int64_t*)stack_shape)[2]));
    uint64_t v__1 = (uint64_t)0;
  (((DLTensor*)stack_array)[1].strides) = (int64_t*)(*(void* *)(&(v__1)));
  (((DLTensor*)stack_array)[1].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[1].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[1].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[1].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[1].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[1].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[1].device.device_type) = (DLDeviceType)1;
  (((TVMValue*)stack_value)[0].v_handle) = (((DLTensor*)stack_array) + 0);
  ((int32_t*)stack_tcode)[0] = 7;
  (((TVMValue*)stack_value)[1].v_handle) = (((DLTensor*)stack_array) + 1);
  ((int32_t*)stack_tcode)[1] = 7;
  TVMValue ret_val;
  int ret_type_code;
  if (tvmgen_default_fused_nn_matmul( (TVMValue*) stack_value , (int*) stack_tcode, 1, &ret_val, &ret_type_code, NULL) != 0){
    return -1;
  }
  ((int64_t*)stack_shape)[0] = (int64_t)64;
  ((int64_t*)stack_shape)[1] = (int64_t)4;
  (((DLTensor*)stack_array)[0].data) = sid_1;
  (((DLTensor*)stack_array)[0].shape) = (&(((int64_t*)stack_shape)[0]));
    uint64_t v__2 = (uint64_t)0;
  (((DLTensor*)stack_array)[0].strides) = (int64_t*)(*(void* *)(&(v__2)));
  (((DLTensor*)stack_array)[0].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[0].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[0].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[0].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[0].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[0].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[0].device.device_type) = (DLDeviceType)1;
  ((int64_t*)stack_shape)[2] = (int64_t)64;
  ((int64_t*)stack_shape)[3] = (int64_t)4;
  (((DLTensor*)stack_array)[1].data) = output_buffer_var;
  (((DLTensor*)stack_array)[1].shape) = (&(((int64_t*)stack_shape)[2]));
    uint64_t v__3 = (uint64_t)0;
  (((DLTensor*)stack_array)[1].strides) = (int64_t*)(*(void* *)(&(v__3)));
  (((DLTensor*)stack_array)[1].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[1].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[1].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[1].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[1].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[1].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[1].device.device_type) = (DLDeviceType)1;
  (((TVMValue*)stack_value)[0].v_handle) = (((DLTensor*)stack_array) + 0);
  ((int32_t*)stack_tcode)[0] = 7;
  (((TVMValue*)stack_value)[1].v_handle) = (((DLTensor*)stack_array) + 1);
  ((int32_t*)stack_tcode)[1] = 7;
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (tvmgen_default_fused_add( (TVMValue*) stack_value , (int*) stack_tcode, 1, &ret_val_1, &ret_type_code_1, NULL) != 0){
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, sid_1) != 0) {
    return -1;
  }
  return 0;
}
