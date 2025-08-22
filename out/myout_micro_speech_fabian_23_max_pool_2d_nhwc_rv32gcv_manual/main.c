// This file is generated. Do not edit.
// Generated on: 2022-04-16 08:32:56.474802

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dlpack/dlpack.h>
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/error_codes.h"

#include "tvm/runtime/crt/stack_allocator.h"

#include "tvmgen_default.h"


// Define data for input and output tensors
char input0_data[15680];
void* inputs[] = {input0_data};
struct tvmgen_default_inputs tvmgen_default_inputs = {
    .input = input0_data,
};
char output0_data[3840];
void* outputs[] = {output0_data};
struct tvmgen_default_outputs tvmgen_default_outputs = {
    .output = output0_data,
};

void TVMLogf(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    printf(msg, args);
    va_end(args);
}

#define WORKSPACE_SIZE (19520 * 2)
static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

#ifdef DEBUG_ARENA_USAGE
size_t max_arena_usage = 0;
#endif

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
#ifdef TVMAOT_DEBUG_ALLOCATIONS
    if (num_bytes > (app_workspace.workspace + app_workspace.workspace_size - app_workspace.next_alloc)) {
      TVMLogf("TVMPlatformMemoryAllocate(%lu): Allocation would overflow arena!\n", num_bytes);
      return kTvmErrorPlatformNoMemory;
    }
#endif
    tvm_crt_error_t ret = StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
#ifdef DEBUG_ARENA_USAGE
  // Use this to estimate the required number of bytes for the arena
  size_t end = app_workspace.next_alloc-app_workspace.workspace;
  if (end > max_arena_usage) {
    max_arena_usage = end;
  }
#endif
    return ret;
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
#ifdef TVMAOT_DEBUG_ALLOCATIONS
    if ((uint8_t*)ptr < app_workspace.workspace || (uint8_t*)ptr >= app_workspace.next_alloc) {
      TVMLogf("TVMPlatformMemoryFree(%p): Invalid Memory region to be free'd!\n", ptr);
      return kTvmErrorPlatformNoMemory;
    }
#endif
    return StackMemoryManager_Free(&app_workspace, ptr);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code) { exit(1); }

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) { return 0; }

void TVMWrap_Init()
{
StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);
}

void *TVMWrap_GetInputPtr(int index)
{
    return inputs[index];
}

size_t TVMWrap_GetInputSize(int index)
{
    size_t sizes[] = { 1960, };

    return sizes[index];
}

size_t TVMWrap_GetNumInputs()
{
    return 1;
}

void TVMWrap_Run()
{
    int ret_val = tvmgen_default_run(&tvmgen_default_inputs, &tvmgen_default_outputs);
    if (ret_val) {
        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }

    #if DEBUG_ARENA_USAGE
        DBGPRINTF("\nAoT executor arena max usage after model invocation: %lu bytes\n", max_arena_usage);
    #endif  // DEBUG_ARENA_USAGE

}

void *TVMWrap_GetOutputPtr(int index)
{
    return outputs[index];
}

size_t TVMWrap_GetOutputSize(int index)
{
    size_t sizes[] = { 4, };

    return sizes[index];
}

size_t TVMWrap_GetNumOutputs()
{
    return 1;
}

int main() {
    TVMWrap_Init();
    TVMWrap_Run();
    return 0;
}
