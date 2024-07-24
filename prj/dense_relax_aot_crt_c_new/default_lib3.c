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
    0x1.b85074p-1, 0x1.c521dap-1, 0x1.a5779ap-1, 0x1.47ea38p-1, 0x1.0fdbb4p-4, 0x1.98c026p-1, 0x1.f8b602p-5, 0x1.4b484ep-1, 
    0x1.45e91cp-2, 0x1.010ea8p-1, 0x1.5b8cccp-1, 0x1.058fe2p-1, 0x1.c5a0e8p-4, 0x1.d116c4p-1, 0x1.ed109ep-1, 0x1.07b08ap-1, 
    0x1.13e868p-2, 0x1.5a291ap-4, 0x1.217f3cp-2, 0x1.aab97p-1, 0x1.aa0d72p-2, 0x1.37b6c4p-1, 0x1.b07e36p-1, 0x1.ee33f2p-2, 
    0x1.cbd02cp-2, 0x1.6d4bap-2, 0x1.6e7406p-2, 0x1.710cdep-1, 0x1.45137cp-3, 0x1.378694p-5, 0x1.58d6d4p-1, 0x1.7e5d5p-4, 
    0x1.90d674p-2, 0x1.82e1acp-1, 0x1.0da9ap-2, 0x1.347eaep-1, 0x1.3588a2p-2, 0x1.1f5e26p-1, 0x1.8c5bdep-1, 0x1.ace78ap-1
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_1[40] = {
    0x1.b85074p-1, 0x1.c521dap-1, 0x1.a5779ap-1, 0x1.47ea38p-1, 0x1.0fdbb4p-4, 0x1.98c026p-1, 0x1.f8b602p-5, 0x1.4b484ep-1, 
    0x1.45e91cp-2, 0x1.010ea8p-1, 0x1.5b8cccp-1, 0x1.058fe2p-1, 0x1.c5a0e8p-4, 0x1.d116c4p-1, 0x1.ed109ep-1, 0x1.07b08ap-1, 
    0x1.13e868p-2, 0x1.5a291ap-4, 0x1.217f3cp-2, 0x1.aab97p-1, 0x1.aa0d72p-2, 0x1.37b6c4p-1, 0x1.b07e36p-1, 0x1.ee33f2p-2, 
    0x1.cbd02cp-2, 0x1.6d4bap-2, 0x1.6e7406p-2, 0x1.710cdep-1, 0x1.45137cp-3, 0x1.378694p-5, 0x1.58d6d4p-1, 0x1.7e5d5p-4, 
    0x1.90d674p-2, 0x1.82e1acp-1, 0x1.0da9ap-2, 0x1.347eaep-1, 0x1.3588a2p-2, 0x1.1f5e26p-1, 0x1.8c5bdep-1, 0x1.ace78ap-1
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_2[4] = {
    0x1.c545ccp-2, 0x1.f91a3ap-1, 0x1.4a96bcp-3, 0x1.c57538p-2
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const float __attribute__((section(".rodata.tvm"), aligned(16))) constant_3[4] = {
    0x1.c545ccp-2, 0x1.f91a3ap-1, 0x1.4a96bcp-3, 0x1.c57538p-2
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t matmul(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t var_A_code = arg_type_ids[0];
  int32_t var_B_code = arg_type_ids[1];
  int32_t var_T_add_code = arg_type_ids[2];
  void* var_A = (((TVMValue*)args)[0].v_handle);
  void* var_B = (((TVMValue*)args)[1].v_handle);
  void* var_T_add = (((TVMValue*)args)[2].v_handle);
  void* add_var_A_shape = (((DLTensor*)var_A)[0].shape);
  void* add_var_A_strides = (((DLTensor*)var_A)[0].strides);
  int32_t dev_id = (((DLTensor*)var_A)[0].device.device_id);
  void* A = (((DLTensor*)var_A)[0].data);
  void* add_var_B_shape = (((DLTensor*)var_B)[0].shape);
  void* add_var_B_strides = (((DLTensor*)var_B)[0].strides);
  void* B = (((DLTensor*)var_B)[0].data);
  void* add_var_T_add_shape = (((DLTensor*)var_T_add)[0].shape);
  void* add_var_T_add_strides = (((DLTensor*)var_T_add)[0].strides);
  void* T_add = (((DLTensor*)var_T_add)[0].data);
  if (!(add_var_A_strides == NULL)) {
  }
  if (!(add_var_B_strides == NULL)) {
  }
  if (!(add_var_T_add_strides == NULL)) {
  }
  for (int32_t ax0 = 0; ax0 < 64; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
      int32_t cse_var_1 = ((ax0 * 4) + ax1);
      ((float*)T_add)[cse_var_1] = (((float*)A)[cse_var_1] + ((float*)B)[ax1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t matmul(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t var_A_code = arg_type_ids[0];
  int32_t var_B_code = arg_type_ids[1];
  int32_t var_matmul_code = arg_type_ids[2];
  void* var_A = (((TVMValue*)args)[0].v_handle);
  void* var_B = (((TVMValue*)args)[1].v_handle);
  void* var_matmul = (((TVMValue*)args)[2].v_handle);
  void* matmul_var_A_shape = (((DLTensor*)var_A)[0].shape);
  void* matmul_var_A_strides = (((DLTensor*)var_A)[0].strides);
  int32_t dev_id = (((DLTensor*)var_A)[0].device.device_id);
  void* A = (((DLTensor*)var_A)[0].data);
  void* matmul_var_B_shape = (((DLTensor*)var_B)[0].shape);
  void* matmul_var_B_strides = (((DLTensor*)var_B)[0].strides);
  void* B = (((DLTensor*)var_B)[0].data);
  void* matmul_var_matmul_shape = (((DLTensor*)var_matmul)[0].shape);
  void* matmul_var_matmul_strides = (((DLTensor*)var_matmul)[0].strides);
  void* matmul = (((DLTensor*)var_matmul)[0].data);
  if (!(matmul_var_A_strides == NULL)) {
  }
  if (!(matmul_var_B_strides == NULL)) {
  }
  if (!(matmul_var_matmul_strides == NULL)) {
  }
  for (int32_t i0 = 0; i0 < 64; ++i0) {
    for (int32_t i1 = 0; i1 < 4; ++i1) {
      for (int32_t k = 0; k < 10; ++k) {
        int32_t cse_var_1 = ((i0 * 4) + i1);
        if (k == 0) {
          ((float*)matmul)[cse_var_1] = 0.000000e+00f;
        }
        ((float*)matmul)[cse_var_1] = (((float*)matmul)[cse_var_1] + (((float*)A)[((i0 * 10) + k)] * ((float*)B)[((k * 4) + i1)]));
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
  TVMValue stack_1[4];
  void* stack_value = stack_1;
  TVMValue stack_2[18];
  void* stack_array = stack_2;
  TVMValue stack_3[6];
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
  void* sid_2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 0, 8);
  if (sid_2 == NULL) {
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
  ((int64_t*)stack_shape)[2] = (int64_t)10;
  ((int64_t*)stack_shape)[3] = (int64_t)4;
  (((DLTensor*)stack_array)[1].data) = constant_1;
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
  ((int64_t*)stack_shape)[4] = (int64_t)64;
  ((int64_t*)stack_shape)[5] = (int64_t)4;
  (((DLTensor*)stack_array)[2].data) = sid_2;
  (((DLTensor*)stack_array)[2].shape) = (&(((int64_t*)stack_shape)[4]));
    uint64_t v__2 = (uint64_t)0;
  (((DLTensor*)stack_array)[2].strides) = (int64_t*)(*(void* *)(&(v__2)));
  (((DLTensor*)stack_array)[2].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[2].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[2].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[2].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[2].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[2].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[2].device.device_type) = (DLDeviceType)1;
  (((TVMValue*)stack_value)[0].v_handle) = (((DLTensor*)stack_array) + 0);
  ((int32_t*)stack_tcode)[0] = 7;
  (((TVMValue*)stack_value)[1].v_handle) = (((DLTensor*)stack_array) + 1);
  ((int32_t*)stack_tcode)[1] = 7;
  (((TVMValue*)stack_value)[2].v_handle) = (((DLTensor*)stack_array) + 2);
  ((int32_t*)stack_tcode)[2] = 7;
  TVMValue ret_val;
  int ret_type_code;
  if (matmul( (TVMValue*) stack_value , (int*) stack_tcode, 2, &ret_val, &ret_type_code, NULL) != 0){
    return -1;
  }
  ((int64_t*)stack_shape)[0] = (int64_t)64;
  ((int64_t*)stack_shape)[1] = (int64_t)4;
  (((DLTensor*)stack_array)[0].data) = sid_2;
  (((DLTensor*)stack_array)[0].shape) = (&(((int64_t*)stack_shape)[0]));
    uint64_t v__3 = (uint64_t)0;
  (((DLTensor*)stack_array)[0].strides) = (int64_t*)(*(void* *)(&(v__3)));
  (((DLTensor*)stack_array)[0].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[0].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[0].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[0].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[0].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[0].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[0].device.device_type) = (DLDeviceType)1;
  ((int64_t*)stack_shape)[2] = (int64_t)4;
  (((DLTensor*)stack_array)[1].data) = constant_3;
  (((DLTensor*)stack_array)[1].shape) = (&(((int64_t*)stack_shape)[2]));
    uint64_t v__4 = (uint64_t)0;
  (((DLTensor*)stack_array)[1].strides) = (int64_t*)(*(void* *)(&(v__4)));
  (((DLTensor*)stack_array)[1].ndim) = (uint32_t)1;
  (((DLTensor*)stack_array)[1].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[1].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[1].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[1].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[1].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[1].device.device_type) = (DLDeviceType)1;
  ((int64_t*)stack_shape)[3] = (int64_t)64;
  ((int64_t*)stack_shape)[4] = (int64_t)4;
  (((DLTensor*)stack_array)[2].data) = output_buffer_var;
  (((DLTensor*)stack_array)[2].shape) = (&(((int64_t*)stack_shape)[3]));
    uint64_t v__5 = (uint64_t)0;
  (((DLTensor*)stack_array)[2].strides) = (int64_t*)(*(void* *)(&(v__5)));
  (((DLTensor*)stack_array)[2].ndim) = (uint32_t)2;
  (((DLTensor*)stack_array)[2].dtype.code) = (uint8_t)2;
  (((DLTensor*)stack_array)[2].dtype.bits) = (uint8_t)32;
  (((DLTensor*)stack_array)[2].dtype.lanes) = (uint16_t)1;
  (((DLTensor*)stack_array)[2].byte_offset) = (uint64_t)0;
  (((DLTensor*)stack_array)[2].device.device_id) = dev_id;
  (((DLTensor*)stack_array)[2].device.device_type) = (DLDeviceType)1;
  (((TVMValue*)stack_value)[0].v_handle) = (((DLTensor*)stack_array) + 0);
  ((int32_t*)stack_tcode)[0] = 7;
  (((TVMValue*)stack_value)[1].v_handle) = (((DLTensor*)stack_array) + 1);
  ((int32_t*)stack_tcode)[1] = 7;
  (((TVMValue*)stack_value)[2].v_handle) = (((DLTensor*)stack_array) + 2);
  ((int32_t*)stack_tcode)[2] = 7;
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (add( (TVMValue*) stack_value , (int*) stack_tcode, 2, &ret_val_1, &ret_type_code_1, NULL) != 0){
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, sid_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return tvmgen_default___tvm_main__(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
