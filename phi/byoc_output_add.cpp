ga87puy@prakt1:~$ cat src/utvm_staticrt_codegen/examples/out/kernels2.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

// TODO: find out if global information is available to do the following when more than 1 op is used:
// TfLiteTensor *tflTensors[2] = { tflTensors0, tflTensors1};, AllocatePersistentBuffer0/1 -> AllocatePersistentBuffer(int index)
// TODO: support custom ops
// TODO: parse flatbuffer before?
// TODO: other datatypes AND quant!
// TODO: tensor data for constant tensors is currently not static and carried through TVM. Is that okay?
// TODO: decide if non_const variables should be global or local statics in repective function? (May have issues with tensor_arena access for static tensorData)
// TODO: determine kTensorArenaSize0 in codegen!
// TODO: put C++ related things in namespace {} ?
// TODO: Ask Andrew: should the byoc generated kernels be c compatible or require c++?

#include "tensorflow/lite/c/builtin_op_data.h" // TODO: "" or <>?
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

template <int SZ, class T> struct TfArray { // TODO: Problem - this needs C++, use TfLiteTensor? But we need C++ anyway?
  int sz; T elem[SZ];
};

struct TensorInfo_t {
  TfLiteType type;
  /*void* data;*/ // TODO: will likely be required for constant tensors
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteQuantization quantization;
};

struct NodeInfo_t {
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  /*used_operators_e used_op_index;*/
};

/* OP0: Add */

// Non-Constant variables

TfLiteContext ctx0{};
TfLiteTensor tflTensors0[3];
TfLiteEvalTensor evalTensors0[3];
TfLiteRegistration registration0 = nullptr;
TfLiteNode tflNode0 = nullptr;
constexpr int kTensorArenaSize0 = ?; 
uint8_t tensor_arena0[kTensorArenaSize0] ALIGN(16);

// Constant variables

const TfArray<2, int> tensor_dimension0_0 = { 2, { 1,10 } };
const TfArray<2, int> tensor_dimension0_1 = { 2, { 1,10 } };
const TfArray<2, int> tensor_dimension0_2 = { 2, { 1,10 } };

const TensorInfo_t tensorData0[] = {
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_0, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_1, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_2, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
};

const TfArray<2, int> inputs0 = { 2, { 0, 1 } };
const TfArray<1, int> outputs0 = { 1, { 2 } };

const TfLiteAddParams opdata0 = { kTfLiteActNone, true }; // TODO: compare?

const NodeInfo_t nodeData0 = { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), /*OP_ADD*/, };

static void* AllocatePersistentBuffer0(struct TfLiteContext* ctx, // TODO: static?
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena0 + sizeof(tensor_arena0);
  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor0(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return &evalTensors0[tensor_idx];
}

void inline tflitecompiler_0_initialize() {
  ctx0.AllocatePersistentBuffer = &AllocatePersistentBuffer0;
  ctx0.GetEvalTensor = &GetEvalTensor0;
  ctx0.tensors = tflTensors0;
  ctx0.tensors_size = 3;
  for(size_t i = 0; i < 3; ++i) {
    tflTensors0[i].data.data = nullptr; // tensorData[i].data; // TOOO: this gets relavant when using constant tensors again!
    evalTensors0[i].data.data = nullptr; // tensorData[i].data;
    tflTensors0[i].type = tensorData[i].type;
    evalTensors0[i].type = tensorData[i].type;
    tflTensors0[i].is_variable = 0;
    tflTensors0[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors0[i].bytes = tensorData[i].bytes;
    tflTensors0[i].dims = tensorData[i].dims;
    evalTensors0[i].dims = tensorData[i].dims;
    tflTensors0[i].quantization = tensorData[i].quantization;
    if (tflTensors0[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData[i].quantization.params));
      tflTensors0[i].params.scale = quant->scale->data[0];
      tflTensors0[i].params.zero_point = quant->zero_point->data[0];
    }
  }
  registration0 = tflite::ops::micro::Register_ADD()

  tflNode0.inputs = nodeData0.inputs;
  tflNode0.outputs = nodeData0.outputs;
  tflNode0.builtin_data = nodeData0.builtin_data;
  tflNode0.custom_initial_data = nullptr;
  tflNode0.custom_initial_data_size = 0;
  if (registration0.init) {
    tflNode0.user_data = registration0.init(&ctx0, (const char*)tflNode0.builtin_data, 0);
  }

  if (registration0.prepare) {
    TfLiteStatus status = registration0.prepare(&ctx0, &tflNode0);
    if (status != kTfLiteOk) {
      return; //status; // TODO: error handling
    }
  }
  // return kTfLiteOk; // TODO: error handling
}

void tflitecompiler_0_invoke(float* in0, float* in1, float* out0) {
    evalTensors0[0].data.data = in0;
    evalTensors0[1].data.data = in1;
    evalTensors0[2].data.data = out0;
    TfLiteStatus status = registration0.invoke(&ctx0, &tflNode0)
    if (status != kTfLiteOk) {
      return; //status; // TODO: error handling
    }
    // return kTfLiteOk; // TODO: error handling
}

// Op=Add (ADD?) builtin=true dtype=float32, num_inputs=2, num_outputs=1, inout_shapes=(1,10)
void tflitecompiler_0_(float* tflitecompiler_0_i0, float* tflitecompiler_0_i1, float* out0) {
  static bool initialized = 0;
  float* buf_0 = (float*)malloc(4 * 1 * 10); // TODO: figure out if malloc is required? TODO: 8/4/2/1 depending on dtype

  if (!initialized) {
    tflitecompiler_0_initialize();
  }
  tflitecompiler_0_invoke(tflitecompiler_0_i0, tflitecompiler_0_i1, buf_0);
  
  memcpy(out0, buf_0, 4 * 1 * 10);
  free(buf_0);
}

int tflitecompiler_0_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* out0) {
  tflitecompiler_0_((float*)(arg0->data),
  (float*)(arg1->data),
  (float*)(out0->data));
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t tflitecompiler_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  tflitecompiler_0_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif


