// TODO: header

#include <tvm/relay/attrs/tflite.h>
#include <tvm/relay/op.h>

#include <string>
#include <vector>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {


TVM_REGISTER_GLOBAL("relay.op.contrib._make.tflite_extern")
    .set_body_typed([](Expr inputs, String name, bool is_builtin, Array<Integer> options, DataType out_dtype, Array<Integer> out_shape){
        LOG(WARNING) << "TESTWARNING" << std::endl;
        auto attrs = make_object<TfLiteExternAttrs>();
        attrs->name = std::move(name);
        attrs->is_builtin = std::move(is_builtin);
        attrs->options = std::move(options);
        attrs->out_dtype = out_dtype;
        attrs->out_shape = out_shape;

        static const Op& op = Op::Get("tflite_extern");
        return Call(op, {inputs}, Attrs(attrs), {}); });

TVM_REGISTER_NODE_TYPE(TfLiteExternAttrs);

template <typename AttrType>
bool TfLiteExternRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {

  ICHECK_EQ(types.size(), 2) << "the arity of tflite_extern is 2, not " << types.size();
  /* If we receive a tuple we can continue, if we receive
   * anything but an incomplete type we should signal an
   * error.
   */
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "tflite_extern requires a tuple of tensors as the first argument, found "
        << PrettyPrint(types[0]));
    return false;
  } else if (types[0].as<IncompleteTypeNode>() != nullptr) {
    return false;
  }

  const auto* param = attrs.as<AttrType>();
  std::vector<IndexExpr> oshape;
  for (auto& x: param->out_shape) {
    oshape.push_back(x);
  }
  const auto& out = TensorType(oshape, param->out_dtype);
  reporter->Assign(types[1], out);
  return true;
}

RELAY_REGISTER_OP("tflite_extern")
    .set_num_inputs(1)
    .add_argument("inputs", "Tuple", "The input tensors.")
    .add_type_rel("TfLiteExtern", TfLiteExternRel<TfLiteExternAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .describe("TODO")
    .set_support_level(1);

}  // namespace relay
}  // namespace tvm
