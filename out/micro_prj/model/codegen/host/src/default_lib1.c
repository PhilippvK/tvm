// tvm target: c -keys=cpu -link-params=0
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_T_add = (((TVMValue*)args)[3].v_handle);
  int32_t arg_T_add_code = arg_type_ids[3];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg_placeholder1)[0].data);
  void* arg_placeholder_shape1 = (((DLTensor*)arg_placeholder1)[0].shape);
  void* arg_placeholder_strides1 = (((DLTensor*)arg_placeholder1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg_placeholder2)[0].data);
  void* arg_placeholder_shape2 = (((DLTensor*)arg_placeholder2)[0].shape);
  void* arg_placeholder_strides2 = (((DLTensor*)arg_placeholder2)[0].strides);
  void* T_add = (((DLTensor*)arg_T_add)[0].data);
  void* arg_T_add_shape = (((DLTensor*)arg_T_add)[0].shape);
  void* arg_T_add_strides = (((DLTensor*)arg_T_add)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_T_add_strides == NULL)) {
  }
  float compute_global[1];
  compute_global[0] = 0.000000e+00f;
  for (int32_t k_outer = 0; k_outer < 16; ++k_outer) {
    compute_global[0] = (compute_global[0] + (((float*)placeholder)[k_outer] * ((float*)placeholder1)[k_outer]));
  }
  ((float*)T_add)[0] = (compute_global[0] + ((float*)placeholder2)[0]);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_T_relu = (((TVMValue*)args)[3].v_handle);
  int32_t arg_T_relu_code = arg_type_ids[3];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg_placeholder1)[0].data);
  void* arg_placeholder_shape1 = (((DLTensor*)arg_placeholder1)[0].shape);
  void* arg_placeholder_strides1 = (((DLTensor*)arg_placeholder1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg_placeholder2)[0].data);
  void* arg_placeholder_shape2 = (((DLTensor*)arg_placeholder2)[0].shape);
  void* arg_placeholder_strides2 = (((DLTensor*)arg_placeholder2)[0].strides);
  void* T_relu = (((DLTensor*)arg_T_relu)[0].data);
  void* arg_T_relu_shape = (((DLTensor*)arg_T_relu)[0].shape);
  void* arg_T_relu_strides = (((DLTensor*)arg_T_relu)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_T_relu_strides == NULL)) {
  }
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    float compute_global[8];
    for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
      compute_global[x_c_init] = 0.000000e+00f;
    }
    for (int32_t x_c = 0; x_c < 8; ++x_c) {
      compute_global[x_c] = (compute_global[x_c] + (((float*)placeholder)[0] * ((float*)placeholder1)[((ax1_outer_ax0_outer_fused * 8) + x_c)]));
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 8) + ax1_inner_inner);
      float _1 = compute_global[ax1_inner_inner] + ((float*)placeholder2)[cse_var_1];
      ((float*)T_relu)[cse_var_1] = ((_1) > (0.000000e+00f) ? (_1) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_T_relu = (((TVMValue*)args)[3].v_handle);
  int32_t arg_T_relu_code = arg_type_ids[3];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg_placeholder1)[0].data);
  void* arg_placeholder_shape1 = (((DLTensor*)arg_placeholder1)[0].shape);
  void* arg_placeholder_strides1 = (((DLTensor*)arg_placeholder1)[0].strides);
  void* placeholder2 = (((DLTensor*)arg_placeholder2)[0].data);
  void* arg_placeholder_shape2 = (((DLTensor*)arg_placeholder2)[0].shape);
  void* arg_placeholder_strides2 = (((DLTensor*)arg_placeholder2)[0].strides);
  void* T_relu = (((DLTensor*)arg_T_relu)[0].data);
  void* arg_T_relu_shape = (((DLTensor*)arg_T_relu)[0].shape);
  void* arg_T_relu_strides = (((DLTensor*)arg_T_relu)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_T_relu_strides == NULL)) {
  }
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    float compute_global[8];
    for (int32_t x_c_init = 0; x_c_init < 8; ++x_c_init) {
      compute_global[x_c_init] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 16; ++k_outer) {
      for (int32_t x_c = 0; x_c < 8; ++x_c) {
        compute_global[x_c] = (compute_global[x_c] + (((float*)placeholder)[k_outer] * ((float*)placeholder1)[(((ax1_outer_ax0_outer_fused * 128) + (k_outer * 8)) + x_c)]));
      }
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 8; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 8) + ax1_inner_inner);
      float _1 = compute_global[ax1_inner_inner] + ((float*)placeholder2)[cse_var_1];
      ((float*)T_relu)[cse_var_1] = ((_1) > (0.000000e+00f) ? (_1) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_T_reshape = (((TVMValue*)args)[1].v_handle);
  int32_t arg_T_reshape_code = arg_type_ids[1];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* T_reshape = (((DLTensor*)arg_T_reshape)[0].data);
  void* arg_T_reshape_shape = (((DLTensor*)arg_T_reshape)[0].shape);
  void* arg_T_reshape_strides = (((DLTensor*)arg_T_reshape)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_reshape_strides == NULL)) {
  }
  ((float*)T_reshape)[0] = ((float*)placeholder)[0];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_1(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_T_reshape = (((TVMValue*)args)[1].v_handle);
  int32_t arg_T_reshape_code = arg_type_ids[1];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* T_reshape = (((DLTensor*)arg_T_reshape)[0].data);
  void* arg_T_reshape_shape = (((DLTensor*)arg_T_reshape)[0].shape);
  void* arg_T_reshape_strides = (((DLTensor*)arg_T_reshape)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_reshape_strides == NULL)) {
  }
  for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
    ((float*)T_reshape)[ax1_inner] = ((float*)placeholder)[ax1_inner];
  }
  return 0;
}

