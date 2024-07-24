// This file is generated. Do not edit.
// Generated on: 2024-07-24 13:06:20.544592

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <dlpack/dlpack.h>
#include "tvm/runtime/crt/error_codes.h"
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/stack_allocator.h"


// Define data for input and output tensors
char input0_data[4];
void* inputs[] = {input0_data};
char output0_data[4];
void* outputs[] = {output0_data};

// void TVMLogf(const char* msg, ...)
// {
//     va_list args;
//     va_start(args, msg);
//     printf(msg, args);
//     va_end(args);
// }

#define WORKSPACE_SIZE (160)
static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

#ifdef DEBUG_ARENA_USAGE
size_t max_arena_usage = 0;
#endif

// tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr)
// {
// #ifdef TVMAOT_DEBUG_ALLOCATIONS
//     if (num_bytes > (app_workspace.workspace + app_workspace.workspace_size - app_workspace.next_alloc))
//     {
//       TVMLogf("TVMPlatformMemoryAllocate(%lu): Allocation would overflow arena!\n", num_bytes);
//       return kTvmErrorPlatformNoMemory;
//     }
// #endif
//     tvm_crt_error_t ret = StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
// #ifdef DEBUG_ARENA_USAGE
//   // Use this to estimate the required number of bytes for the arena
//   size_t end = app_workspace.next_alloc-app_workspace.workspace;
//   if (end > max_arena_usage)
//   {
//     max_arena_usage = end;
//   }
// #endif
//     return ret;
// }
// tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev)
// {
// #ifdef TVMAOT_DEBUG_ALLOCATIONS
//     if ((uint8_t*)ptr < app_workspace.workspace || (uint8_t*)ptr >= app_workspace.next_alloc)
//     {
//       TVMLogf("TVMPlatformMemoryFree(%p): Invalid Memory region to be free'd!\n", ptr);
//       return kTvmErrorPlatformNoMemory;
//     }
// #endif
//     return StackMemoryManager_Free(&app_workspace, ptr);
// }
// int32_t tvmgen_default_run(void* args, void* type_code, int num_args, void* out_value, void* out_type_code, void* resource_handle);
//
// void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code)
// {
//     exit(1);
// }
//
// TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override)
// {
//     return 0;
// }

int TVMWrap_Init()
{
    TVMPlatformInitialize();
    // StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);
    return 0;  // TODO
}

void *TVMWrap_GetInputPtr(int index)
{
    return inputs[index];
}

size_t TVMWrap_GetInputSize(int index)
{
    size_t sizes[] = { 4, };

    return sizes[index];
}

size_t TVMWrap_GetNumInputs()
{
    return 1;
}

int TVMWrap_Run()
{
    static DLDevice fake_device = {kDLCPU, 0};
    static int64_t fake_dims = 0;
    static int64_t fake_shape = {0};

    DLTensor tensors[1 + 1];
    TVMValue values[1 + 1];
    int32_t typeids[1 + 1];

    for (size_t i = 0; i < 1+1; i++)
    {
        tensors[i].device = fake_device;
        tensors[i].data = (i < 1) ? inputs[i] : outputs[i - 1];
        tensors[i].shape = &fake_shape;
        tensors[i].ndim = fake_dims;
        tensors[i].byte_offset = 0;
        tensors[i].strides = NULL;
        values[i].v_handle = &tensors[i];
    }

    int ret_val = tvmgen_default_run(values, typeids, 0, NULL, 0, NULL);
    // int ret_val = tvmgen_default___tvm_main__(values, typeids, 0, NULL, 0, NULL);
    if (ret_val)
    {
        TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }
    return 0;


#if DEBUG_ARENA_USAGE
    printf("\nAoT executor arena max usage after model invocation: %lu bytes\n", max_arena_usage);
#endif  // DEBUG_ARENA_USAGE

    return 0;  // TODO
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
