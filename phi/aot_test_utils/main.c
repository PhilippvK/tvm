#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>
#include "tvm/runtime/crt/internal/aot_executor/aot_executor.h"
#include "tvm/runtime/crt/stack_allocator.h"

#define WORKSPACE_SIZE (16384*1024)

static uint8_t g_aot_memory[WORKSPACE_SIZE];

const size_t input_data_len = 1;
const size_t output_data_len = 1;

float input_data[] = { /* TODO */ };
float output_data[] = { /* TODO */ };

extern tvm_model_t network;
tvm_workspace_t app_workspace;

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return StackMemoryManager_Free(&app_workspace,ptr);
}

void  TVMPlatformAbort(tvm_crt_error_t code) {}

void TVMLogf(const char* msg, ...) {}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {}

int main() {
    void* inputs[1] = { input_data }
    void* inputs[1] = { output_data }

    StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE)
    tvm_runtime_run(&network, inputs, outputs)

    for (int i = 0; i < output_data_len; i++) {
        printf("%f", output_data[i]
    }

    return 0;
}

