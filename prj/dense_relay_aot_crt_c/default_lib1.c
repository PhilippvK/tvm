#include "tvm/runtime/c_runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle);

int32_t tvmgen_default_run(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code, void* resource_handle) {
TVMValue tensors[2];
tensors[0] = ((TVMValue*)args)[0];
tensors[1] = ((TVMValue*)args)[1];
return tvmgen_default___tvm_main__((void*)tensors, type_code, num_args, out_value, out_type_code, resource_handle);
}
#ifdef __cplusplus
}
#endif
;
