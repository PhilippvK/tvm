#include <tvm/runtime/crt/module.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_contrib_dense_pack_add,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1,
    (TVMBackendPackedCFunc)tvmgen_default_fused_reshape,
    (TVMBackendPackedCFunc)tvmgen_default_fused_reshape_1,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\005tvmgen_default_fused_nn_contrib_dense_pack_add\000tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu\000tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1\000tvmgen_default_fused_reshape\000tvmgen_default_fused_reshape_1\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;