// tvm target: c -keys=cpu -link-params=0
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_T_subtract = (((TVMValue*)args)[2].v_handle);
  int32_t arg_T_subtract_code = arg_type_ids[2];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg_placeholder1)[0].data);
  void* arg_placeholder_shape1 = (((DLTensor*)arg_placeholder1)[0].shape);
  void* arg_placeholder_strides1 = (((DLTensor*)arg_placeholder1)[0].strides);
  void* T_subtract = (((DLTensor*)arg_T_subtract)[0].data);
  void* arg_T_subtract_shape = (((DLTensor*)arg_T_subtract)[0].shape);
  void* arg_T_subtract_strides = (((DLTensor*)arg_T_subtract)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_subtract_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 49; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 10; ++ax2) {
      int32_t cse_var_1 = ((ax0_ax1_fused * 10) + ax2);
      ((int16_t*)T_subtract)[cse_var_1] = (((int16_t)((int8_t*)placeholder)[cse_var_1]) - ((int16_t*)placeholder1)[0]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_avg_pool2d_cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_T_cast = (((TVMValue*)args)[1].v_handle);
  int32_t arg_T_cast_code = arg_type_ids[1];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* T_cast = (((DLTensor*)arg_T_cast)[0].data);
  void* arg_T_cast_shape = (((DLTensor*)arg_T_cast)[0].shape);
  void* arg_T_cast_strides = (((DLTensor*)arg_T_cast)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_cast_strides == NULL)) {
  }
  int32_t tensor[64];
  for (int32_t ax3_init = 0; ax3_init < 64; ++ax3_init) {
    tensor[ax3_init] = 0;
  }
  for (int32_t rv0_rv1_fused = 0; rv0_rv1_fused < 125; ++rv0_rv1_fused) {
    for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
      tensor[ax3] = (tensor[ax3] + ((int32_t*)placeholder)[((rv0_rv1_fused * 64) + ax3)]);
    }
  }
  for (int32_t ax31 = 0; ax31 < 64; ++ax31) {
    ((int8_t*)T_cast)[ax31] = ((int8_t)(tensor[ax31] / 125));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_cb9548905a4a8167_(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_T_multiply = (((TVMValue*)args)[3].v_handle);
  int32_t arg_T_multiply_code = arg_type_ids[3];
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
  void* T_multiply = (((DLTensor*)arg_T_multiply)[0].data);
  void* arg_T_multiply_shape = (((DLTensor*)arg_T_multiply)[0].shape);
  void* arg_T_multiply_strides = (((DLTensor*)arg_T_multiply)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_T_multiply_strides == NULL)) {
  }
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    int32_t compute_global[6];
    for (int32_t x_c_init = 0; x_c_init < 6; ++x_c_init) {
      compute_global[x_c_init] = 0;
    }
    for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
      for (int32_t x_c = 0; x_c < 6; ++x_c) {
        compute_global[x_c] = (compute_global[x_c] + (((int32_t)((int16_t*)placeholder)[k_outer]) * ((int32_t)((int16_t*)placeholder1)[(((ax1_outer_ax0_outer_fused * 384) + (k_outer * 6)) + x_c)])));
      }
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 6; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 6) + ax1_inner_inner);
      int32_t _1 = ((int32_t)(((((0 != 0) ? (((int64_t)(compute_global[ax1_inner_inner] + ((int32_t*)placeholder2)[cse_var_1])) << ((int64_t)0)) : ((int64_t)(compute_global[ax1_inner_inner] + ((int32_t*)placeholder2)[cse_var_1]))) * (int64_t)1617124365) + ((int64_t)1 << ((int64_t)((7 + 31) - 1)))) >> ((int64_t)(7 + 31)))) + 33;
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      ((float*)T_multiply)[cse_var_1] = (((float)(((int32_t)((int8_t)((_2) > (-128) ? (_2) : (-128)))) - 33)) * 1.421776e-01f);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_placeholder3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg_placeholder_code3 = arg_type_ids[3];
  void* arg_placeholder4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg_placeholder_code4 = arg_type_ids[4];
  void* arg_placeholder5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg_placeholder_code5 = arg_type_ids[5];
  void* arg_placeholder6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg_placeholder_code6 = arg_type_ids[6];
  void* arg_T_cast = (((TVMValue*)args)[7].v_handle);
  int32_t arg_T_cast_code = arg_type_ids[7];
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
  void* placeholder3 = (((DLTensor*)arg_placeholder3)[0].data);
  void* arg_placeholder_shape3 = (((DLTensor*)arg_placeholder3)[0].shape);
  void* arg_placeholder_strides3 = (((DLTensor*)arg_placeholder3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg_placeholder4)[0].data);
  void* arg_placeholder_shape4 = (((DLTensor*)arg_placeholder4)[0].shape);
  void* arg_placeholder_strides4 = (((DLTensor*)arg_placeholder4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg_placeholder5)[0].data);
  void* arg_placeholder_shape5 = (((DLTensor*)arg_placeholder5)[0].shape);
  void* arg_placeholder_strides5 = (((DLTensor*)arg_placeholder5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg_placeholder6)[0].data);
  void* arg_placeholder_shape6 = (((DLTensor*)arg_placeholder6)[0].shape);
  void* arg_placeholder_strides6 = (((DLTensor*)arg_placeholder6)[0].strides);
  void* T_cast = (((DLTensor*)arg_T_cast)[0].data);
  void* arg_T_cast_shape = (((DLTensor*)arg_T_cast)[0].shape);
  void* arg_T_cast_strides = (((DLTensor*)arg_T_cast)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_placeholder_strides3 == NULL)) {
  }
  if (!(arg_placeholder_strides4 == NULL)) {
  }
  if (!(arg_placeholder_strides5 == NULL)) {
  }
  if (!(arg_T_cast_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16000, 0, 16);
  if (pad_temp == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 25; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 320) + (i2 * 64)) + i3);
        ((int16_t*)pad_temp)[cse_var_1] = ((int16_t*)placeholder)[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 125; ++ax0_ax1_fused_ax2_fused) {
    int32_t conv2d_nhwc[64];
    for (int32_t ff = 0; ff < 64; ++ff) {
      conv2d_nhwc[ff] = 0;
      for (int32_t rc = 0; rc < 64; ++rc) {
        conv2d_nhwc[ff] = (conv2d_nhwc[ff] + (((int32_t)((int16_t*)pad_temp)[((ax0_ax1_fused_ax2_fused * 64) + rc)]) * ((int32_t)((int16_t*)placeholder1)[((rc * 64) + ff)])));
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t*)placeholder6)[0] + ((int32_t)((((((int64_t)conv2d_nhwc[ax3_inner]) + ((int64_t)((int32_t*)placeholder2)[ax3_inner])) * ((int64_t*)placeholder3)[ax3_inner]) + ((int64_t*)placeholder4)[ax3_inner]) >> ((int64_t*)placeholder5)[ax3_inner]));
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int32_t*)T_cast)[((ax0_ax1_fused_ax2_fused * 64) + ax3_inner)] = ((int32_t)((_5) > (_6) ? (_5) : (_6)));
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_placeholder3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg_placeholder_code3 = arg_type_ids[3];
  void* arg_placeholder4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg_placeholder_code4 = arg_type_ids[4];
  void* arg_placeholder5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg_placeholder_code5 = arg_type_ids[5];
  void* arg_placeholder6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg_placeholder_code6 = arg_type_ids[6];
  void* arg_placeholder7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg_placeholder_code7 = arg_type_ids[7];
  void* arg_T_subtract = (((TVMValue*)args)[8].v_handle);
  int32_t arg_T_subtract_code = arg_type_ids[8];
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
  void* placeholder3 = (((DLTensor*)arg_placeholder3)[0].data);
  void* arg_placeholder_shape3 = (((DLTensor*)arg_placeholder3)[0].shape);
  void* arg_placeholder_strides3 = (((DLTensor*)arg_placeholder3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg_placeholder4)[0].data);
  void* arg_placeholder_shape4 = (((DLTensor*)arg_placeholder4)[0].shape);
  void* arg_placeholder_strides4 = (((DLTensor*)arg_placeholder4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg_placeholder5)[0].data);
  void* arg_placeholder_shape5 = (((DLTensor*)arg_placeholder5)[0].shape);
  void* arg_placeholder_strides5 = (((DLTensor*)arg_placeholder5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg_placeholder6)[0].data);
  void* arg_placeholder_shape6 = (((DLTensor*)arg_placeholder6)[0].shape);
  void* arg_placeholder_strides6 = (((DLTensor*)arg_placeholder6)[0].strides);
  void* placeholder7 = (((DLTensor*)arg_placeholder7)[0].data);
  void* arg_placeholder_shape7 = (((DLTensor*)arg_placeholder7)[0].shape);
  void* arg_placeholder_strides7 = (((DLTensor*)arg_placeholder7)[0].strides);
  void* T_subtract = (((DLTensor*)arg_T_subtract)[0].data);
  void* arg_T_subtract_shape = (((DLTensor*)arg_T_subtract)[0].shape);
  void* arg_T_subtract_strides = (((DLTensor*)arg_T_subtract)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_placeholder_strides3 == NULL)) {
  }
  if (!(arg_placeholder_strides4 == NULL)) {
  }
  if (!(arg_placeholder_strides5 == NULL)) {
  }
  if (!(arg_T_subtract_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1392, 0, 16);
  if (pad_temp == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 58; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 12; ++i2) {
      ((int16_t*)pad_temp)[((i0_i1_fused * 12) + i2)] = (((((4 <= i0_i1_fused) && (i0_i1_fused < 53)) && (1 <= i2)) && (i2 < 11)) ? ((int16_t*)placeholder)[(((i0_i1_fused * 10) + i2) - 41)] : (int16_t)0);
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 125; ++ax0_ax1_fused_ax2_fused) {
    int32_t conv2d_nhwc[64];
    for (int32_t ff = 0; ff < 64; ++ff) {
      conv2d_nhwc[ff] = 0;
      for (int32_t ry = 0; ry < 10; ++ry) {
        for (int32_t rx = 0; rx < 4; ++rx) {
          conv2d_nhwc[ff] = (conv2d_nhwc[ff] + (((int32_t)((int16_t*)pad_temp)[(((((ax0_ax1_fused_ax2_fused / 5) * 24) + (ry * 12)) + ((ax0_ax1_fused_ax2_fused % 5) * 2)) + rx)]) * ((int32_t)((int16_t*)placeholder1)[(((ry * 256) + (rx * 64)) + ff)])));
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t*)placeholder6)[0] + ((int32_t)((((((int64_t)conv2d_nhwc[ax3_inner]) + ((int64_t)((int32_t*)placeholder2)[ax3_inner])) * ((int64_t*)placeholder3)[ax3_inner]) + ((int64_t*)placeholder4)[ax3_inner]) >> ((int64_t*)placeholder5)[ax3_inner]));
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int16_t*)T_subtract)[((ax0_ax1_fused_ax2_fused * 64) + ax3_inner)] = (((int16_t)((_5) > (_6) ? (_5) : (_6))) - ((int16_t*)placeholder7)[0]);
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_placeholder3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg_placeholder_code3 = arg_type_ids[3];
  void* arg_placeholder4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg_placeholder_code4 = arg_type_ids[4];
  void* arg_placeholder5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg_placeholder_code5 = arg_type_ids[5];
  void* arg_placeholder6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg_placeholder_code6 = arg_type_ids[6];
  void* arg_placeholder7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg_placeholder_code7 = arg_type_ids[7];
  void* arg_T_subtract = (((TVMValue*)args)[8].v_handle);
  int32_t arg_T_subtract_code = arg_type_ids[8];
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
  void* placeholder3 = (((DLTensor*)arg_placeholder3)[0].data);
  void* arg_placeholder_shape3 = (((DLTensor*)arg_placeholder3)[0].shape);
  void* arg_placeholder_strides3 = (((DLTensor*)arg_placeholder3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg_placeholder4)[0].data);
  void* arg_placeholder_shape4 = (((DLTensor*)arg_placeholder4)[0].shape);
  void* arg_placeholder_strides4 = (((DLTensor*)arg_placeholder4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg_placeholder5)[0].data);
  void* arg_placeholder_shape5 = (((DLTensor*)arg_placeholder5)[0].shape);
  void* arg_placeholder_strides5 = (((DLTensor*)arg_placeholder5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg_placeholder6)[0].data);
  void* arg_placeholder_shape6 = (((DLTensor*)arg_placeholder6)[0].shape);
  void* arg_placeholder_strides6 = (((DLTensor*)arg_placeholder6)[0].strides);
  void* placeholder7 = (((DLTensor*)arg_placeholder7)[0].data);
  void* arg_placeholder_shape7 = (((DLTensor*)arg_placeholder7)[0].shape);
  void* arg_placeholder_strides7 = (((DLTensor*)arg_placeholder7)[0].strides);
  void* T_subtract = (((DLTensor*)arg_T_subtract)[0].data);
  void* arg_T_subtract_shape = (((DLTensor*)arg_T_subtract)[0].shape);
  void* arg_T_subtract_strides = (((DLTensor*)arg_T_subtract)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_placeholder_strides3 == NULL)) {
  }
  if (!(arg_placeholder_strides4 == NULL)) {
  }
  if (!(arg_placeholder_strides5 == NULL)) {
  }
  if (!(arg_T_subtract_strides == NULL)) {
  }
  void* PaddedInput = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)24192, 0, 16);
  if (PaddedInput == NULL) {
    return -1;
  }
  void* DepthwiseConv2d = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)32000, 0, 32);
  if (DepthwiseConv2d == NULL) {
    return -1;
  }
  void* T_cast = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)64000, 0, 64);
  if (T_cast == NULL) {
    return -1;
  }
  for (int32_t i1 = 0; i1 < 27; ++i1) {
    for (int32_t i2 = 0; i2 < 7; ++i2) {
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        int32_t cse_var_1 = (i2 * 64);
        ((int16_t*)PaddedInput)[(((i1 * 448) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 26)) && (1 <= i2)) && (i2 < 6)) ? ((int16_t*)placeholder)[((((i1 * 320) + cse_var_1) + i3) - 384)] : (int16_t)0);
      }
    }
  }
  for (int32_t i = 0; i < 25; ++i) {
    for (int32_t j = 0; j < 5; ++j) {
      for (int32_t c = 0; c < 64; ++c) {
        ((int32_t*)DepthwiseConv2d)[(((i * 320) + (j * 64)) + c)] = 0;
        for (int32_t di = 0; di < 3; ++di) {
          for (int32_t dj = 0; dj < 3; ++dj) {
            int32_t cse_var_4 = (j * 64);
            int32_t cse_var_3 = (dj * 64);
            int32_t cse_var_2 = (((i * 320) + cse_var_4) + c);
            ((int32_t*)DepthwiseConv2d)[cse_var_2] = (((int32_t*)DepthwiseConv2d)[cse_var_2] + (((int32_t)((int16_t*)PaddedInput)[(((((i * 448) + (di * 448)) + cse_var_4) + cse_var_3) + c)]) * ((int32_t)((int16_t*)placeholder1)[(((di * 192) + cse_var_3) + c)])));
          }
        }
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 25; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 5; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
        int32_t cse_var_5 = (((ax1 * 320) + (ax2 * 64)) + ax3);
        ((int32_t*)DepthwiseConv2d)[cse_var_5] = (((int32_t*)DepthwiseConv2d)[cse_var_5] + ((int32_t*)placeholder2)[ax3]);
      }
    }
  }
  for (int32_t ax11 = 0; ax11 < 25; ++ax11) {
    for (int32_t ax21 = 0; ax21 < 5; ++ax21) {
      for (int32_t ax31 = 0; ax31 < 64; ++ax31) {
        int32_t cse_var_6 = (((ax11 * 320) + (ax21 * 64)) + ax31);
        ((int64_t*)T_cast)[cse_var_6] = ((int64_t)((int32_t*)DepthwiseConv2d)[cse_var_6]);
      }
    }
  }
  for (int32_t ax12 = 0; ax12 < 25; ++ax12) {
    for (int32_t ax22 = 0; ax22 < 5; ++ax22) {
      for (int32_t ax32 = 0; ax32 < 64; ++ax32) {
        int32_t cse_var_7 = (((ax12 * 320) + (ax22 * 64)) + ax32);
        ((int64_t*)T_cast)[cse_var_7] = (((int64_t*)T_cast)[cse_var_7] * ((int64_t*)placeholder3)[ax32]);
      }
    }
  }
  for (int32_t ax13 = 0; ax13 < 25; ++ax13) {
    for (int32_t ax23 = 0; ax23 < 5; ++ax23) {
      for (int32_t ax33 = 0; ax33 < 64; ++ax33) {
        int32_t cse_var_8 = (((ax13 * 320) + (ax23 * 64)) + ax33);
        ((int64_t*)T_cast)[cse_var_8] = (((int64_t*)T_cast)[cse_var_8] + ((int64_t*)placeholder4)[ax33]);
      }
    }
  }
  for (int32_t ax14 = 0; ax14 < 25; ++ax14) {
    for (int32_t ax24 = 0; ax24 < 5; ++ax24) {
      for (int32_t ax34 = 0; ax34 < 64; ++ax34) {
        int32_t cse_var_9 = (((ax14 * 320) + (ax24 * 64)) + ax34);
        ((int64_t*)T_cast)[cse_var_9] = (((int64_t*)T_cast)[cse_var_9] >> ((int64_t*)placeholder5)[ax34]);
      }
    }
  }
  for (int32_t ax15 = 0; ax15 < 25; ++ax15) {
    for (int32_t ax25 = 0; ax25 < 5; ++ax25) {
      for (int32_t ax35 = 0; ax35 < 64; ++ax35) {
        int32_t cse_var_10 = (((ax15 * 320) + (ax25 * 64)) + ax35);
        ((int32_t*)DepthwiseConv2d)[cse_var_10] = ((int32_t)((int64_t*)T_cast)[cse_var_10]);
      }
    }
  }
  for (int32_t ax16 = 0; ax16 < 25; ++ax16) {
    for (int32_t ax26 = 0; ax26 < 5; ++ax26) {
      for (int32_t ax36 = 0; ax36 < 64; ++ax36) {
        int32_t cse_var_11 = (((ax16 * 320) + (ax26 * 64)) + ax36);
        ((int32_t*)DepthwiseConv2d)[cse_var_11] = (((int32_t*)placeholder6)[0] + ((int32_t*)DepthwiseConv2d)[cse_var_11]);
      }
    }
  }
  for (int32_t i11 = 0; i11 < 25; ++i11) {
    for (int32_t i21 = 0; i21 < 5; ++i21) {
      for (int32_t i31 = 0; i31 < 64; ++i31) {
        int32_t cse_var_12 = (((i11 * 320) + (i21 * 64)) + i31);
        int32_t _1 = ((int32_t*)DepthwiseConv2d)[cse_var_12];
        int32_t _2 = (_1) < (127) ? (_1) : (127);
        ((int32_t*)DepthwiseConv2d)[cse_var_12] = ((_2) > (-128) ? (_2) : (-128));
      }
    }
  }
  for (int32_t ax17 = 0; ax17 < 25; ++ax17) {
    for (int32_t ax27 = 0; ax27 < 5; ++ax27) {
      for (int32_t ax37 = 0; ax37 < 64; ++ax37) {
        int32_t cse_var_13 = (((ax17 * 320) + (ax27 * 64)) + ax37);
        ((int8_t*)PaddedInput)[cse_var_13] = ((int8_t)((int32_t*)DepthwiseConv2d)[cse_var_13]);
      }
    }
  }
  for (int32_t i12 = 0; i12 < 25; ++i12) {
    for (int32_t i22 = 0; i22 < 5; ++i22) {
      for (int32_t i32 = 0; i32 < 64; ++i32) {
        int32_t cse_var_14 = (((i12 * 320) + (i22 * 64)) + i32);
        int8_t _3 = ((int8_t*)PaddedInput)[cse_var_14];
        int8_t _4 = (int8_t)127;
        int8_t _5 = (_3) < (_4) ? (_3) : (_4);
        int8_t _6 = (int8_t)-128;
        ((int8_t*)DepthwiseConv2d)[cse_var_14] = ((_5) > (_6) ? (_5) : (_6));
      }
    }
  }
  for (int32_t ax18 = 0; ax18 < 25; ++ax18) {
    for (int32_t ax28 = 0; ax28 < 5; ++ax28) {
      for (int32_t ax38 = 0; ax38 < 64; ++ax38) {
        int32_t cse_var_15 = (((ax18 * 320) + (ax28 * 64)) + ax38);
        ((int16_t*)PaddedInput)[cse_var_15] = ((int16_t)((int8_t*)DepthwiseConv2d)[cse_var_15]);
      }
    }
  }
  for (int32_t ax19 = 0; ax19 < 25; ++ax19) {
    for (int32_t ax29 = 0; ax29 < 5; ++ax29) {
      for (int32_t ax39 = 0; ax39 < 64; ++ax39) {
        int32_t cse_var_16 = (((ax19 * 320) + (ax29 * 64)) + ax39);
        ((int16_t*)T_subtract)[cse_var_16] = (((int16_t*)PaddedInput)[cse_var_16] - ((int16_t*)placeholder7)[0]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, T_cast) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, DepthwiseConv2d) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, PaddedInput) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_placeholder2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg_placeholder_code2 = arg_type_ids[2];
  void* arg_placeholder3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg_placeholder_code3 = arg_type_ids[3];
  void* arg_placeholder4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg_placeholder_code4 = arg_type_ids[4];
  void* arg_placeholder5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg_placeholder_code5 = arg_type_ids[5];
  void* arg_placeholder6 = (((TVMValue*)args)[6].v_handle);
  int32_t arg_placeholder_code6 = arg_type_ids[6];
  void* arg_placeholder7 = (((TVMValue*)args)[7].v_handle);
  int32_t arg_placeholder_code7 = arg_type_ids[7];
  void* arg_T_subtract = (((TVMValue*)args)[8].v_handle);
  int32_t arg_T_subtract_code = arg_type_ids[8];
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
  void* placeholder3 = (((DLTensor*)arg_placeholder3)[0].data);
  void* arg_placeholder_shape3 = (((DLTensor*)arg_placeholder3)[0].shape);
  void* arg_placeholder_strides3 = (((DLTensor*)arg_placeholder3)[0].strides);
  void* placeholder4 = (((DLTensor*)arg_placeholder4)[0].data);
  void* arg_placeholder_shape4 = (((DLTensor*)arg_placeholder4)[0].shape);
  void* arg_placeholder_strides4 = (((DLTensor*)arg_placeholder4)[0].strides);
  void* placeholder5 = (((DLTensor*)arg_placeholder5)[0].data);
  void* arg_placeholder_shape5 = (((DLTensor*)arg_placeholder5)[0].shape);
  void* arg_placeholder_strides5 = (((DLTensor*)arg_placeholder5)[0].strides);
  void* placeholder6 = (((DLTensor*)arg_placeholder6)[0].data);
  void* arg_placeholder_shape6 = (((DLTensor*)arg_placeholder6)[0].shape);
  void* arg_placeholder_strides6 = (((DLTensor*)arg_placeholder6)[0].strides);
  void* placeholder7 = (((DLTensor*)arg_placeholder7)[0].data);
  void* arg_placeholder_shape7 = (((DLTensor*)arg_placeholder7)[0].shape);
  void* arg_placeholder_strides7 = (((DLTensor*)arg_placeholder7)[0].strides);
  void* T_subtract = (((DLTensor*)arg_T_subtract)[0].data);
  void* arg_T_subtract_shape = (((DLTensor*)arg_T_subtract)[0].shape);
  void* arg_T_subtract_strides = (((DLTensor*)arg_T_subtract)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_placeholder_strides1 == NULL)) {
  }
  if (!(arg_placeholder_strides2 == NULL)) {
  }
  if (!(arg_placeholder_strides3 == NULL)) {
  }
  if (!(arg_placeholder_strides4 == NULL)) {
  }
  if (!(arg_placeholder_strides5 == NULL)) {
  }
  if (!(arg_T_subtract_strides == NULL)) {
  }
  void* pad_temp = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16000, 0, 16);
  if (pad_temp == NULL) {
    return -1;
  }
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 25; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 5; ++i2) {
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        int32_t cse_var_1 = (((i0_i1_fused * 320) + (i2 * 64)) + i3);
        ((int16_t*)pad_temp)[cse_var_1] = ((int16_t*)placeholder)[cse_var_1];
      }
    }
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 125; ++ax0_ax1_fused_ax2_fused) {
    int32_t conv2d_nhwc[64];
    for (int32_t ff = 0; ff < 64; ++ff) {
      conv2d_nhwc[ff] = 0;
      for (int32_t rc = 0; rc < 64; ++rc) {
        conv2d_nhwc[ff] = (conv2d_nhwc[ff] + (((int32_t)((int16_t*)pad_temp)[((ax0_ax1_fused_ax2_fused * 64) + rc)]) * ((int32_t)((int16_t*)placeholder1)[((rc * 64) + ff)])));
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 64; ++ax3_inner) {
      int32_t _1 = ((int32_t*)placeholder6)[0] + ((int32_t)((((((int64_t)conv2d_nhwc[ax3_inner]) + ((int64_t)((int32_t*)placeholder2)[ax3_inner])) * ((int64_t*)placeholder3)[ax3_inner]) + ((int64_t*)placeholder4)[ax3_inner]) >> ((int64_t*)placeholder5)[ax3_inner]));
      int32_t _2 = (_1) < (127) ? (_1) : (127);
      int8_t _3 = (int8_t)((_2) > (-128) ? (_2) : (-128));
      int8_t _4 = (int8_t)127;
      int8_t _5 = (_3) < (_4) ? (_3) : (_4);
      int8_t _6 = (int8_t)-128;
      ((int16_t*)T_subtract)[((ax0_ax1_fused_ax2_fused * 64) + ax3_inner)] = (((int16_t)((_5) > (_6) ? (_5) : (_6))) - ((int16_t*)placeholder7)[0]);
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, pad_temp) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_divide_add_clip_round_cast(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_T_cast = (((TVMValue*)args)[1].v_handle);
  int32_t arg_T_cast_code = arg_type_ids[1];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* T_cast = (((DLTensor*)arg_T_cast)[0].data);
  void* arg_T_cast_shape = (((DLTensor*)arg_T_cast)[0].shape);
  void* arg_T_cast_strides = (((DLTensor*)arg_T_cast)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_cast_strides == NULL)) {
  }
  float T_softmax_maxelem[1];
  float T_softmax_exp[12];
  float T_softmax_expsum[1];
  T_softmax_maxelem[0] = -3.402823e+38f;
  for (int32_t k = 0; k < 12; ++k) {
    float _1 = T_softmax_maxelem[0];
    float _2 = ((float*)placeholder)[k];
    T_softmax_maxelem[0] = ((_1) > (_2) ? (_1) : (_2));
  }
  for (int32_t i1 = 0; i1 < 12; ++i1) {
    T_softmax_exp[i1] = expf((((float*)placeholder)[i1] - T_softmax_maxelem[0]));
  }
  T_softmax_expsum[0] = 0.000000e+00f;
  for (int32_t k1 = 0; k1 < 12; ++k1) {
    T_softmax_expsum[0] = (T_softmax_expsum[0] + T_softmax_exp[k1]);
  }
  for (int32_t i11 = 0; i11 < 12; ++i11) {
    T_softmax_exp[i11] = (T_softmax_exp[i11] / T_softmax_expsum[0]);
  }
  for (int32_t ax1 = 0; ax1 < 12; ++ax1) {
    float _3 = (T_softmax_exp[ax1] * 2.560000e+02f) + -1.280000e+02f;
    float _4 = (_3) < (1.270000e+02f) ? (_3) : (1.270000e+02f);
    ((int8_t*)T_cast)[ax1] = ((int8_t)roundf(((_4) > (-1.280000e+02f) ? (_4) : (-1.280000e+02f))));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  void* arg_placeholder = (((TVMValue*)args)[0].v_handle);
  int32_t arg_placeholder_code = arg_type_ids[0];
  void* arg_placeholder1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg_placeholder_code1 = arg_type_ids[1];
  void* arg_T_subtract = (((TVMValue*)args)[2].v_handle);
  int32_t arg_T_subtract_code = arg_type_ids[2];
  void* placeholder = (((DLTensor*)arg_placeholder)[0].data);
  void* arg_placeholder_shape = (((DLTensor*)arg_placeholder)[0].shape);
  void* arg_placeholder_strides = (((DLTensor*)arg_placeholder)[0].strides);
  int32_t dev_id = (((DLTensor*)arg_placeholder)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg_placeholder1)[0].data);
  void* arg_placeholder_shape1 = (((DLTensor*)arg_placeholder1)[0].shape);
  void* arg_placeholder_strides1 = (((DLTensor*)arg_placeholder1)[0].strides);
  void* T_subtract = (((DLTensor*)arg_T_subtract)[0].data);
  void* arg_T_subtract_shape = (((DLTensor*)arg_T_subtract)[0].shape);
  void* arg_T_subtract_strides = (((DLTensor*)arg_T_subtract)[0].strides);
  if (!(arg_placeholder_strides == NULL)) {
  }
  if (!(arg_T_subtract_strides == NULL)) {
  }
  for (int32_t ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
      ((int16_t*)T_subtract)[cse_var_1] = (((int16_t)((int8_t*)placeholder)[cse_var_1]) - ((int16_t*)placeholder1)[0]);
    }
  }
  return 0;
}

