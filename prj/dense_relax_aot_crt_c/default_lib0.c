#include <tvm/runtime/crt/module.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t add(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t matmul(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);
static TVMBackendPackedCFunc _tvm_func_array[] = {
    (TVMBackendPackedCFunc)add,
    (TVMBackendPackedCFunc)matmul,
    (TVMBackendPackedCFunc)tvmgen_default___tvm_main__,
    (TVMBackendPackedCFunc)__tvm_main__,
};
static const TVMFuncRegistry _tvm_func_registry = {
    "\004\000add\000matmul\000tvmgen_default___tvm_main__\000__tvm_main__\000",    _tvm_func_array,
};
static const TVMModule _tvm_system_lib = {
    &_tvm_func_registry,
};
const TVMModule* TVMSystemLibEntryPoint(void) {
    return &_tvm_system_lib;
}
;
