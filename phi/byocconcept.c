ga87puy@prakt1:~$ cat src/utvm_staticrt_codegen/examples/out/kernels2.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

// TODOL find out if global information is available to do the following when more than 1 op is used:
// TfLiteTensor *tflTensors[2] = { tflTensors0, tflTensors1};, AllocatePersistentBuffer0/1 -> AllocatePersistentBuffer(int index)

// TODO: support custom ops

// TODO: parse flatbuffer before?

template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};

struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  /*void* data;*/ // TODO: will likely be required for constant tensors
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteQuantization quantization;
};

struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  //used_operators_e used_op_index;
};

// TODO: other datatypes AND quant!
// TODO: tensor data for constant tensors is currently not static and carried through TVM. Is that okay?

TfLiteContext ctx0{};
TfLiteTensor tflTensors0[3];
TfLiteEvalTensor evalTensors[3];
TfLiteRegistration registration0 = nullptr;
TfLiteNode tflNode = nullptr;
constexpr int kTensorArenaSize0 = ?;
uint8_t tensor_arena[kTensorArenaSize0] ALIGN(16);

// TODO: decide if non_const variables should be global or local statics in repective function? (May have issues with tensor_arena access for static tensorData)

const TfArray<2, int> tensor_dimension0_0 = { 2, { 1,10 } };
const TfArray<2, int> tensor_dimension0_0 = { 2, { 1,10 } };
const TfArray<2, int> tensor_dimension0_0 = { 2, { 1,10 } };

const TfArray<1, int> inputs0 = { 2, { 0, 1 } };
const TfArray<1, int> outputs0 = { 1, { 2 } };
const TfLiteAddParams opdata0 = { kTfLiteActNone, true }; // TODO: compare?


const TensorInfo_t tensorData0[] = {
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_0, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_1, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
  { kTfLiteFloat32, /*tensor_arena0 + ?*/, (TfLiteIntArray*)&tensor_dimension0_2, 4 * 1 * 10, {kTfLiteNoQuantization, nullptr },},
};

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
void tflitecompiler_0_(float* ccompiler_0_i0, float* ccompiler_0_i1, float* out0) {
  static bool initialized = 0;
  float* buf_0 = (float*)malloc(4 * 2 * 1 * 10); // TODO: figure out if malloc is required? TODO: 8/4/2/1 depending on dtype

  if (!initialized) {
    tflitecompiler_0_initialize(?);
  }
  tflitecompiler_0_invoke(?);
  
  memcpy(out0, buf_0, 4 * 2);
  free(buf_0);
}

int ccompiler_0_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* out0) {
  ccompiler_0_((float*)(arg0->data),
  (float*)(arg1->data),
  (float*)(out0->data));
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t ccompiler_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  ccompiler_0_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif


#include <stdio.h>
//#include <stdint.h>

#include "tvm/runtime/c_runtime_api.h"
//#include "tvm/runtime/data_type.h"
//#include "tvm/runtime/crt/packed_func.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
//#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
//#include "tensorflow/lite/schema/schema_generated.h"

#ifdef _DEBUG
#include <stdio.h>
#define DBGPRINTF(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DBGPRINTF(format, ...)
#endif


#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif


namespace tvm {
namespace runtime {
TfLiteType DLDataType2TfLiteType(const DLDataType* dtype) {
  if (dtype->lanes != 1) {
  }
  switch (dtype->code) {
    case kDLInt:
      if (dtype->bits == 8) {
        return kTfLiteInt8;
      } else if (dtype->bits == 16) {
        return kTfLiteInt16;
      } else if (dtype->bits == 32) {
        return kTfLiteInt32;
      } else if (dtype->bits == 64) {
        return kTfLiteInt64;
      } else {
        // TODO: Handle Error at the End!
      }
      break;
    case kDLUInt:
      if (dtype->bits == 8) {
        return kTfLiteUInt8;
      } else {
        // TODO: Handle Error at the End!
      }
      break;
    case kDLFloat:
      if (dtype->bits == 16) {
        return kTfLiteFloat16;
      } else if (dtype->bits == 32) {
        return kTfLiteFloat32;
      } else {
        // TODO: Handle Error at the End!
      }
      break;
    default:
      break;
  }
  return kTfLiteFloat32;
}

tflite::ErrorReporter* error_reporter = nullptr;
class DummyReporter : public tflite::ErrorReporter {
 public:
  ~DummyReporter() {}
  int Report(const char*, va_list) override { return 0; }

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

inline TfLiteTensor DLTensor2TfLiteTensor(const DLTensor* tensor) {
  TfLiteType dtype_converted = DLDataType2TfLiteType(&tensor->dtype);
  TfLiteTensor tensor_converted;
  tensor_converted.type = dtype_converted;
  tensor_converted.data.raw =
      reinterpret_cast<char*>(const_cast<void*>(tensor->data));
  // TfLiteIntArray* shape_array = TfLiteIntArrayCreate(tensor->ndim); // TODO:
  // malloc not allowed beause of static memory flag set!
  TfLiteIntArray* shape_array =
      (TfLiteIntArray*)malloc(TfLiteIntArrayGetSizeInBytes(tensor->ndim));
  shape_array->size = tensor->ndim;
  for (size_t dim_index = 0; dim_index < tensor->ndim; dim_index++) {
    shape_array->data[dim_index] = tensor->shape[dim_index];
  }
  tensor_converted.dims = shape_array;
  tensor_converted.bytes = tflite::ElementCount(*shape_array) * sizeof(float);
  tensor_converted.is_variable = false;  // TODO: ?
  if (tensor->strides != NULL) {
    // TODO: handle error!
  }
  return tensor_converted;
}


constexpr int kTensorArenaSize = 300;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);

static void* AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes) {
  static uint8_t* AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

TfLiteEvalTensor evalTensors[10]; // TODO: do not hardcode

static TfLiteEvalTensor* GetEvalTensor(const struct TfLiteContext* context,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}

extern "C" int32_t tflite_extern_wrapper(TVMValue* args, int* type_code,
                                     int num_args, TVMValue* out_value,
                                     int* out_type_code) {
    char* op_name = (char*)args[0].v_handle;
    char is_builtin = (args[1].v_int64 != 0);
    is_builtin = 1; // TODO
    DBGPRINTF("\nOP=%s | IS_BUILTIN=%d\n", op_name, is_builtin);
    int64_t input_tensors_size = args[2].v_int64;
    int64_t output_tensors_size = args[3].v_int64;
    int64_t tensors_size = input_tensors_size + output_tensors_size;

    DLTensor *options_tensor = (DLTensor*)args[4+input_tensors_size+output_tensors_size].v_handle;
    int s = options_tensor->shape[0];
    DBGPRINTF("OPTIONS: ");
    for ( int i = 0 ; i < s ; i++ ) {
        DBGPRINTF("%d,", ((char*)options_tensor->data)[i]);
    }
    DBGPRINTF("\n");

    std::vector<TfLiteTensor> tensors_converted(input_tensors_size+output_tensors_size);

    for (size_t index = 0; index < input_tensors_size; index++) {
      DLTensor* tensor = (DLTensor*)args[4 + index].v_handle;
      TfLiteTensor tensor_converted = DLTensor2TfLiteTensor(tensor);
      tensors_converted[index] = tensor_converted;
    }
    for (size_t index = 0; index < output_tensors_size; index++) {
      DLTensor* tensor = (DLTensor*)args[4 + input_tensors_size + index].v_handle;
      TfLiteTensor tensor_converted = DLTensor2TfLiteTensor(tensor);
      tensors_converted[input_tensors_size + index] = tensor_converted;
    }

    for(size_t i = 0; i < tensors_size; ++i) {
      evalTensors[i].data.data = tensors_converted[i].data.data;
      evalTensors[i].type = tensors_converted[i].type;
      evalTensors[i].dims = tensors_converted[i].dims;
    }

    TfLiteContext context;
    context.tensors_size = tensors_size;
    context.tensors = tensors_converted.data();
    context.recommended_num_threads = 1;
    context.GetEvalTensor = GetEvalTensor;
    context.AllocatePersistentBuffer = AllocatePersistentBuffer;  // TODO: implement!
    static tflite::MicroMutableOpResolver<1> resolver(error_reporter);
    /*if (resolver.AddAdd() != kTfLiteOk) {
      return -1;
    }*/
    if (resolver.AddFullyConnected() != kTfLiteOk) {
      return -1;
    }

    /*const TfLiteRegistration* registration = resolver.FindOp("Add2");
    if (!registration) {
      return -1;
    }*/
    const TfLiteRegistration* registration = resolver.FindOp(tflite::BuiltinOperator_FULLY_CONNECTED);
    if (!registration) {
      return -1;
    }

    void* builtin_data;
    const char* init_data;
    const void* custom_initial_data;
    int custom_initial_data_size;
    if (is_builtin) {
      builtin_data = (void*)options_tensor->data;
      init_data = (const char*)builtin_data;
      custom_initial_data = nullptr;
      custom_initial_data_size = 0;
    } else { // CUSTOM
      builtin_data = nullptr; // TODO
      init_data = nullptr; // TODO
      custom_initial_data = nullptr; // TODO
      custom_initial_data_size = 0; // TODO
    }
    const size_t init_data_size = 0; // TODO: ?
    void* user_data = nullptr;

    if (registration->init) {
      user_data = registration->init(&context, init_data, init_data_size);
    }

    TfLiteIntArray* inputs_array = (TfLiteIntArray*)malloc(
        TfLiteIntArrayGetSizeInBytes(input_tensors_size));
    inputs_array->size = input_tensors_size;
    //inputs_array->data[0] = input_tensors_size;
    for (size_t index = 0; index < input_tensors_size; index++) {
      inputs_array->data[index] = index;
    }
    TfLiteIntArray* outputs_array = (TfLiteIntArray*)malloc(
        TfLiteIntArrayGetSizeInBytes(output_tensors_size));
    outputs_array->size = output_tensors_size;
    //outputs_array->data[0] = output_tensors_size;
    for (size_t index = 0; index < output_tensors_size; index++) {
      outputs_array->data[index] = input_tensors_size + index;
    }

    TfLiteNode node;
    node.inputs = inputs_array;
    node.outputs = outputs_array;
    //node.temporaries = temporaries_array; // TODO: ?
    //node.intermediates = intermediates_array; // TODO: ?
    node.user_data = user_data;
    //node.builtin_data = reinterpret_cast<void*>(&builtin_data);
    node.builtin_data = reinterpret_cast<void*>(builtin_data);
    node.custom_initial_data = nullptr;
    node.custom_initial_data_size = 0;
    //node.delegate = nullptr;

    if (registration->prepare) {
      if (registration->prepare(&context, &node) != kTfLiteOk) {
        return -1;
      }
    }

    if (!registration->invoke) {
      return -1;
    }
    if (registration->invoke(&context, &node) != kTfLiteOk) {
      return -1;
    }

    if (registration->free) {
      registration->free(&context, user_data);
    }

    return 0;
}

}  // namespace runtime
}  // namespace tvm