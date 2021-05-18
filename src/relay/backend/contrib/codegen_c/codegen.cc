/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/attrs/tflite.h>
#include <tvm/runtime/data_type.h>

#include <fstream>
#include <sstream>
#include <string>

#include "../../utils.h"
#include "codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief An example codegen that is only used for quick prototyping and testing
 * purpose. Only several binary options are covered. Users
 * may need to extend them to cover more operators.
 */
class CodegenC : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenC(const std::string& id) { this->ext_func_id_ = id; }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "C codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) final {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
    auto res = VisitExpr(op->tuple);
    ICHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    Output output;
    // Get const: static_cast<float*>(gcc_0_consts[0]->data)
    output.name = CreateDataReference(ext_func_id_, const_idx_);
    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
      std::string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    ICHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
    output.dtype = dtype;

    std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
    const_vars_.push_back(const_var_name);
    const_idx_++;

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

    char is_tflite = 0;
    if (IsOp(call, "add")) {
      macro_stream << "+";
    } else if (IsOp(call, "subtract")) {
      macro_stream << "-";
    } else if (IsOp(call, "multiply")) {
      macro_stream << "*";
    } else if (IsOp(call, "tflite_extern")) {
      LOG(WARNING) << "ITS CODEGEN TIME" << std::endl;
      macro_stream << "%";
      is_tflite = 1;
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    if (is_tflite) {

        /* Find out in which node we are right now for more compact indicies */
        size_t num_offset = ext_func_id_.find("_")+1;
        int num = atoi(ext_func_id_.c_str()+num_offset);

        /* Collect information */
        auto inputs_array = call->args[0].as<TupleNode>()->fields;
        size_t num_inputs = inputs_array.size();
        size_t num_input_tensors = num_inputs - 1; // TODO: this is only requires as long the inputs also carry the options
        size_t num_output_tensors = 1;
        size_t num_tensors = num_input_tensors + num_output_tensors;

        /* Extract attributes */
        const auto attrs = call->attrs.as<TfLiteExternAttrs>();
        std::string op_name = attrs->name;  // No ‘tvm::runtime::String’ here
        for (size_t pos=0; pos<op_name.length(); pos++) {
          /* Workaround to capitalize every op name */
          op_name[pos] = toupper(op_name[pos]);
        }

        bool is_builtin = attrs->is_builtin;
        //ICHECK(is_builtin); // TODO: remove when custom ops are supported
        // DataType out_dtype = attrs->out_dtype;
        // Array<Integer> options = attrs->options;
        // Array<Integer> out_shape_ = attrs->out_shape;

        /* Get information on output tensor */
        const auto* output_type_node = call->checked_type().as<TensorTypeNode>();
        ICHECK(output_type_node);
        const auto& output_type_str = GetDtypeString(output_type_node);
        auto output_shape = GetShape(call->checked_type());
        int output_size = 1;
        for (size_t i = 0; i < output_shape.size(); ++i) {
          output_size *= output_shape[i];
        }

        /* Some verbose logging for debugging */
        /*
        LOG(WARNING) << "PARSED NUM = " << num << std::endl;
        LOG(WARNING) << "CALLargs=" << call->args << std::endl;
        LOG(WARNING) << "array_tuple="  << inputs_array << std::endl;
        LOG(WARNING) << "num_input_tensors="  << num_input_tensors << std::endl;
        LOG(WARNING) << "Input Tensors:" << std::endl;
        for (size_t j = 0 ; j < num_input_tensors ; j++) {
            LOG(WARNING) << "  " << j << ": " << inputs_array[j] << std::endl;
        }
        LOG(WARNING) << "Options Tensor:" << inputs_array[num_input_tensors] << std::endl;
        LOG(WARNING) << "ATTRS=" << attrs << std::endl;
        LOG(WARNING) << "  NAME=" << name << std::endl;
        LOG(WARNING) << "  IS_BUILTIN=" << is_builtin << std::endl;
        LOG(WARNING) << "  OUT_DTYPE=" << out_dtype << std::endl;
        LOG(WARNING) << "  OPTIONS=" << options << std::endl;
        LOG(WARNING) << "  OUT_SHAPE=" << out_shape_ << std::endl;
        */

        /* Temporary workaround to remove buffer contents which have been already written */
        macro_stream.str("");  // TODO: remove this two lines
        macro_stream.clear();

        /* Start writing global variables for node */
        macro_stream << "/* Node: " << num << " - Operator: " << op_name << " */\n";
        macro_stream << "TfLiteContext ctx" << num << "{};\n";
        macro_stream << "TfLiteTensor tflTensors" << num << "[" << num_tensors <<"];\n";
        macro_stream << "TfLiteEvalTensor evalTensors" << num << "[" << num_tensors <<"];\n";
        macro_stream << "TfLiteRegistration registration" << num << ";\n";
        macro_stream << "TfLiteNode tflNode" << num << ";\n";
        macro_stream << "constexpr int kTensorArenaSize" << num << " = DEFAULT_ARENA_SIZE;\n";  // TODO: use actual required arena size!
        macro_stream << "uint8_t tensor_arena" << num << "[kTensorArenaSize" << num << "] ALIGN(16);\n";

        /* Write static dimensions/shapes for input and output tensors*/
        for (size_t input_index = 0; input_index < num_input_tensors; input_index++) {
          /* Get input tensor shape */
          //const auto* input_type_node = inputs_array[input_index]->checked_type().as<TensorTypeNode>(); // TODO: unused?
          auto input_shape = GetShape(inputs_array[input_index]->checked_type());

          /* Write input tensor shape */
          macro_stream << "const TfLiteIntArray tensor_dimension" << num << "_" << input_index
                       << " = { " << input_shape.size() << ", { ";
          for (size_t shape_index = 0; shape_index < input_shape.size()-1; shape_index++) {
            macro_stream << input_shape[shape_index] << ",";
          }
          macro_stream << input_shape[input_shape.size()-1] << " } };\n";
        }
        macro_stream << "const TfLiteIntArray tensor_dimension" << num << "_" << num_input_tensors
                     << " = { " << output_shape.size() << ", { ";
        for (size_t shape_index = 0; shape_index < output_shape.size()-1; ++shape_index) {
          macro_stream << output_shape[shape_index] << ",";
        }
        macro_stream << output_shape[output_shape.size()-1] << " } };\n";

        /* Fill tensorData struct */
        macro_stream << "const TensorInfo_t tensorData" << num << "[] = {\n";
        for (size_t input_index = 0; input_index < num_input_tensors; input_index++) {
          /* Get input tensor type and colulate size */
          const auto* input_type_node = inputs_array[input_index]->checked_type().as<TensorTypeNode>();
          const auto& input_type_str = GetDtypeString(input_type_node);
          auto input_shape = GetShape(inputs_array[input_index]->checked_type());
          int input_size = 1; // TODO: introduce CalcSizeFromShape to have less redundant code
          for (size_t shape_index = 0; shape_index < input_shape.size(); shape_index++) {
            input_size *= input_shape[shape_index];
          }

          /* Write Line of tensorData */
          macro_stream << "  { " << GetTfLiteTypeString(input_type_node) << ", "
                       << "(TfLiteIntArray*)&tensor_dimension" << num << "_" << input_index << ", "
                       <<  GetDtypeSize(input_type_node) << " * " << input_size << ", "
                       << "{ kTfLiteNoQuantization, nullptr }"  // TODO: support quant
                       << " },\n";
        }
        macro_stream << "  { " << GetTfLiteTypeString(output_type_node) << ", "
                     << "(TfLiteIntArray*)&tensor_dimension" << num << "_" << inputs_array.size()-1 << ", "
                     << GetDtypeSize(output_type_node) << " * " << output_size << ", " // TODO: 4 ->sizeof(float)
                     << "{ kTfLiteNoQuantization, nullptr }" // TODO: support quant
                     << " }\n";
        macro_stream << "};\n";

        /* Write TfLiteIntArrays for inputs and outputs */
        macro_stream << "const TfLiteIntArray inputs" << num << " = { " << num_input_tensors << ", { ";
        for (size_t input_index = 0; input_index<num_input_tensors-1; input_index++) {
          macro_stream << input_index << ", ";
        }
        macro_stream << num_input_tensors-1 << " }"
                     << " };\n";
        macro_stream << "const TfLiteIntArray outputs" << num << " = { " << num_output_tensors << ", { ";
        for (size_t output_index = 0; output_index<num_output_tensors-1; output_index++) {
          macro_stream << num_input_tensors+output_index << ", ";
        }
        macro_stream << num_input_tensors+num_output_tensors-1 << " }"
                     << " };\n";

        // TODO: builtins! do not hardcode
        /* Write BuiltinOptions for operator */
        macro_stream << "const TfLiteAddParams opdata" << num << " = { kTfLiteActNone, true };\n";

        /* Create nodeData struct */
        macro_stream << "const NodeInfo_t nodeData" << num << " = {"
                     << "(TfLiteIntArray*)&inputs" << num
                     << ", (TfLiteIntArray*)&outputs" << num
                     << ", (void*)&opdata" << num
                     << " };\n";

        /* Create node-specific AllocatePersistentBuffer and TfLiteEvalTensor functions */
        macro_stream << "static void* AllocatePersistentBuffer" << num << "(struct TfLiteContext* ctx, size_t bytes) {\n"
                     << "  static uint8_t *AllocPtr = tensor_arena" << num << " + sizeof(tensor_arena" << num << ");\n"
                     << "  AllocPtr -= bytes;\n"
                     << "  return AllocPtr;\n"
                     << "}\n";
        macro_stream << "static TfLiteEvalTensor *GetEvalTensor" << num << "(const struct TfLiteContext *context, int tensor_idx) {\n"
                     << "  return &evalTensors0[tensor_idx];\n"
                     << "}\n";

        /* Generate _initialize function for this node */
        macro_stream << "void " << ext_func_id_ << "_initialize() {\n"
                     << "  ctx" << num << ".AllocatePersistentBuffer = &AllocatePersistentBuffer" << num << ";\n"
                     << "  ctx" << num << ".GetEvalTensor = &GetEvalTensor" << num << ";\n"
                     << "  ctx" << num << ".tensors = tflTensors" << num << ";\n"
                     << "  ctx" << num << ".tensors_size = " << num_tensors << ";\n"
                     << "  for(size_t i = 0; i < " << num_tensors << "; ++i) {\n"
                     << "    tflTensors" << num << "[i].data.data = nullptr;\n"
                     << "    evalTensors" << num << "[i].data.data = nullptr;\n"
                     << "    tflTensors" << num << "[i].type = tensorData" << num << "[i].type;\n"
                     << "    evalTensors" << num << "[i].type =  tensorData" << num << "[i].type;\n"
                     << "    tflTensors" << num << "[i].is_variable = 0;\n"
        //           << "    tflTensors" << num << "[i].allocation_type = (tensor_arena" << num << " <= tensorData" << num << "[i].data && tensorData" << num << "[i].data < tensor_arena" << num << " + kTensorArenaSize" << num << ") ? kTfLiteArenaRw : kTfLiteMmapRo;\n"  // TODO: fix allocation_type
                     << "    tflTensors" << num << "[i].allocation_type = kTfLiteArenaRw;\n"
                     << "    tflTensors" << num << "[i].bytes = tensorData" << num << "[i].bytes;\n"
                     << "    tflTensors" << num << "[i].dims = tensorData" << num << "[i].dims;\n"
                     << "    evalTensors" << num << "[i].dims = tensorData" << num << "[i].dims;\n"
                     << "    tflTensors" << num << "[i].quantization = tensorData" << num << "[i].quantization;\n"
                     << "    if (tflTensors" << num << "[i].quantization.type == kTfLiteAffineQuantization) {\n"
                     << "      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData" << num << "[i].quantization.params));\n"
                     << "      tflTensors" << num << "[i].params.scale = quant->scale->data[0];\n"
                     << "      tflTensors" << num << "[i].params.zero_point = quant->zero_point->data[0];\n"
                     << "    }\n"
                     << "  }\n"
                     << "  registration" << num << " = tflite::ops::micro::Register_" << op_name << "();\n" // TODO: upercase op name, TODO: support flat namespace
                     << "  tflNode" << num << ".inputs = nodeData" << num << ".inputs;\n"
                     << "  tflNode" << num << ".outputs = nodeData" << num << ".outputs;\n"
                     << "  tflNode" << num << ".builtin_data = nodeData" << num << ".builtin_data;\n"
                     << "  tflNode" << num << ".custom_initial_data = nullptr;\n"
                     << "  tflNode" << num << ".custom_initial_data_size = 0;\n"
                     << "  if (registration" << num << ".init) {\n"
                     << "    tflNode" << num << ".user_data = registration" << num << ".init(&ctx" << num << ", (const char*)tflNode" << num << ".builtin_data, 0);\n"
                     << "  }\n"
                     << "  if (registration" << num << ".prepare) {\n"
                     << "    TfLiteStatus status = registration" << num << ".prepare(&ctx" << num << ", &tflNode" << num << ");\n"  // TODO: check status!
                     << "  }\n"
                     << "}\n";

        /* Generate _invoke function for this node */
        macro_stream << "void " << ext_func_id_ << "_invoke(";
        for (size_t input_index = 0; input_index < num_input_tensors; input_index++) {
          /* Find out datatype of input tensor */
          const auto* input_type_node = inputs_array[input_index]->checked_type().as<TensorTypeNode>();  // TODO: convert to tflitetype
          const auto& input_type_str = GetDtypeString(input_type_node);

          macro_stream << input_type_str << "* in" << input_index << ", ";
        }
        macro_stream << output_type_str << "* out) {\n";
        for (size_t input_index = 0; input_index < num_input_tensors; input_index++) {
          macro_stream << "  evalTensors" << num << "[" << input_index << "].data.data = in" << input_index << ";\n";
        }
        macro_stream << "  evalTensors" << num << "[" << num_input_tensors << "].data.data = out;\n"
                     << "  TfLiteStatus status = registration" << num << ".invoke(&ctx" << num << ", &tflNode" << num << ");\n";  // TODO: check status!
        macro_stream << "}\n";

        /* Print global variables and functions for current node */
        func_decl_.push_back(macro_stream.str());


        /* Buffer creation */
        std::string out = "buf_" + std::to_string(buf_idx_++);
        buf_stream << output_type_str << "* " << out << " = (" << output_type_str << "*)malloc("
                   << GetDtypeSize(output_type_node) << " * " << output_size << ");";
        buf_decl_.push_back(buf_stream.str());

        /* Print check for initialization */
        decl_stream << "static bool initialized = false;\n"
                   << "  if (!initialized) {\n"
                   << "    " << ext_func_id_ + "_initialize();\n"
                   << "  }\n";

        /* Make function call when visiting arguments */
        bool first = true;
        decl_stream << "  " << ext_func_id_ + "_invoke" << "(";
        for (size_t input_index = 0; input_index < num_input_tensors; input_index++) {
          auto res = VisitExpr(inputs_array[input_index]);
          for (auto out : res) {
            if (!first) {
              decl_stream << ", ";
            }
            first = false;
            decl_stream << out.name;
          }
        }
        decl_stream << ", " << out << ");";
        ext_func_body_.push_back(decl_stream.str());

        /* Set output metadata for JIT generation */
        Output output;
        output.name = out;
        output.dtype = output_type_str;
        output.need_copy = true;  // TODO: Find out if required?
        output.size = GetDtypeSize(output_type_node) * output_size;
        return {output};
    }

    auto in_shape = GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      macro_stream << ", " << in_shape[i];
    }

    const auto* type_node = call->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    macro_stream << ", " << dtype;

    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (auto out : res) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.name;
      }
    }

    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
    buf_decl_.push_back(buf_stream.str());

    decl_stream << ", " << out << ");";
    ext_func_body_.push_back(decl_stream.str());

    // Update output buffer
    // Note C codegen only handles TensorType. Therefore, we don't flatten
    // tuples and only return a single vaule.
    Output output;
    output.name = out;
    output.dtype = dtype;
    output.need_copy = true;
    output.size = GetDtypeSize(type_node) * out_size;
    return {output};
  }

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT(const std::vector<Output>& out) {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  /*! \brief The function id that represents a C source function. */
  std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The index of global constants. */
  int const_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  friend class CSourceCodegen;
};

class CSourceCodegen : public CSourceModuleCodegenBase {
 public:
  std::tuple<Array<String>, String, String> GenCFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    CodegenC builder(GetExtSymbol(func));
    auto out = builder.VisitExpr(func->body);
    return std::make_tuple(builder.const_vars_, builder.ext_func_id_, builder.JIT(out));
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenCFunc(Downcast<Function>(ref));
    Array<String> variables = std::get<0>(res);
    String func_name = std::get<1>(res);

    // Create headers
    code_stream_ << "#include <stdio.h>\n";
    code_stream_ << "#include <stdlib.h>\n";
    code_stream_ << "#include <string.h>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/c_backend_api.h>\n";

    // TfLite Headers
    code_stream_ << "\n";
    code_stream_ << "#include \"tensorflow/lite/c/builtin_op_data.h\"\n";
    code_stream_ << "#include \"tensorflow/lite/c/common.h\"\n";
    code_stream_ << "#include \"tensorflow/lite/micro/kernels/micro_ops.h\"\n";
    if (!variables.empty()) {
      // This segment would be generated in C++ because of the usage
      // of tvm::runtime::Array. This is not ideal, but this to demonstrate
      // constant copying process used packed imports in other external
      // codegen. Moreover, in uTVM we dont expect this part to be generated.
      code_stream_ << "#ifdef __cplusplus\n";
      code_stream_ << "#include <tvm/runtime/ndarray.h>\n";
      code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
      code_stream_ << "#endif\n";
    }

    // Append some common macro for operator definition.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_, p_DTYPE)       \
      void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {    \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                        \
          out[i] = a[i] p_OP_ b[i];                                    \
        }                                                              \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_, p_DTYPE)  \
      void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }
    )op_macro";
    const char* tflite_typedefs = R"tflite_typedefs(
#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

/*template <int SZ, class T> struct TfArray { // TODO: Problem - this needs C++, use TfLiteTensor? But we need C++ anyway?
  int sz; T elem[SZ];
};*/

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

// TODO: remove this :)
#define DEFAULT_ARENA_SIZE 100000
    )tflite_typedefs";
/*    code_stream_ << R"(TfLiteType DLDataType2TfLiteType(const DLDataType* dtype) {
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
})" << "\n\n";*/

    code_stream_ << operator_macro << "\n\n";
    code_stream_ << tflite_typedefs << "\n\n";
    code_stream_ << std::get<2>(res);
    std::string code = code_stream_.str();

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", Array<String>{func_name}, variables);
  }

 private:
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM, so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module CCompiler(const ObjectRef& ref) {
  CSourceCodegen csource;
  return csource.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
