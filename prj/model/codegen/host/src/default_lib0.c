#include <tvm/runtime/crt/module.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_avg_pool2d_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_cb9548905a4a8167_(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_divide_add_clip_round_cast(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)tvmgen_default_fused_cast_subtract,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_avg_pool2d_cast,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_cb9548905a4a8167_,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_softmax_divide_add_clip_round_cast,
    (TVMBackendPackedCFunc)tvmgen_default_fused_reshape_cast_subtract,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\ttvmgen_default_fused_cast_subtract\000tvmgen_default_fused_nn_avg_pool2d_cast\000tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_cb9548905a4a8167_\000tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast\000tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_\000tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1\000tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2\000tvmgen_default_fused_nn_softmax_divide_add_clip_round_cast\000tvmgen_default_fused_reshape_cast_subtract\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;