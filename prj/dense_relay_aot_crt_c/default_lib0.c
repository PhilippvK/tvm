#include <tvm/runtime/crt/module.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_run(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_add(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_matmul(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_get_c_metadata(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)tvmgen_default_run,
    (TVMBackendPackedCFunc)tvmgen_default_fused_add,
    (TVMBackendPackedCFunc)tvmgen_default_fused_nn_matmul,
    (TVMBackendPackedCFunc)tvmgen_default___tvm_main__,
    (TVMBackendPackedCFunc)tvmgen_default_get_c_metadata,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\005\000tvmgen_default_run\000tvmgen_default_fused_add\000tvmgen_default_fused_nn_matmul\000tvmgen_default___tvm_main__\000tvmgen_default_get_c_metadata\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
