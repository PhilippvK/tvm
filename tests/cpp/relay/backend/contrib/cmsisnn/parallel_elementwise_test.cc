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

#ifdef TVM_USE_CMSISNN

#include <gtest/gtest.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <string>

#include "../../../../../../src/relay/backend/contrib/cmsisnn/compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

runtime::Module TIRToRuntime(IRModule mod, Target target);

// Exercise source emission only; numerical and multicore tests require a platform runtime.
static std::string ElementwiseSource(const std::string& operation, int bits, int size,
                                     int enabled, bool same_input = false) {
  auto context_node = make_object<tvm::transform::PassContextNode>();
  auto config_node = make_object<CMSISNNCompilerConfigNode>();
  if (enabled < 0) {
    config_node->InitBySeq();  // Test the default, not an explicitly supplied false value.
  } else {
    config_node->InitBySeq("experimental_parallel_elementwise", Bool(enabled != 0));
  }
  context_node->config = {{"relay.ext.cmsisnn.options", CMSISNNCompilerConfig(config_node)}};
  tvm::With<tvm::transform::PassContext> scope{tvm::transform::PassContext(context_node)};

  tir::Var lhs("lhs", DataType::Handle());
  tir::Var rhs("rhs", DataType::Handle());
  tir::Var output("output", DataType::Handle());
  bool add = operation == "add";
  int output_arg = add ? 10 : 5;
  int count_arg = add ? 16 : 11;
  Array<PrimExpr> args{tir::StringImm("arm_elementwise_" + operation + "_s" +
                                    std::to_string(bits)),
                       lhs, same_input ? lhs : rhs};
  for (int i = 3; i <= count_arg; ++i) {
    if (i == output_arg) {
      args.push_back(output);
    } else {
      args.push_back(IntImm(DataType::Int(32), i == count_arg ? size : 0));
    }
  }
  tir::PrimFunc function(
      {lhs, rhs, output},
      tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::call_extern(), args)),
      PrimType(DataType::Int(32)), {},
      DictAttrs({{tvm::attr::kGlobalSymbol, String("elementwise_test")}}));
  IRModule module(Map<GlobalVar, BaseFunc>{{GlobalVar("elementwise_test"), function}});
  return TIRToRuntime(module, Target("cmsis-nn"))->GetSource();
}

TEST(CMSISNNParallelElementwise, DefaultAndExplicitDisableStaySerial) {
  for (const auto& operation : {"add", "mul"}) {
    for (int bits : {8, 16}) {
      for (int enabled : {-1, 0}) {
        auto source = ElementwiseSource(operation, bits, 17, enabled);
        EXPECT_EQ(source.find("TVMBackendParallelLaunch("), std::string::npos);
        EXPECT_NE(source.find("arm_elementwise_" + std::string(operation)), std::string::npos);
      }
    }
  }
}

TEST(CMSISNNParallelElementwise, EmitsTwoDisjointSlices) {
  for (const auto& operation : {"add", "mul"}) {
    for (int bits : {8, 16}) {
      for (int size : {2, 16, 17}) {
        for (bool same_input : {false, true}) {
          auto source = ElementwiseSource(operation, bits, size, 1, same_input);
          EXPECT_NE(source.find("TVMBackendParallelLaunch("), std::string::npos);
          EXPECT_NE(source.find("_data, 2) != 0)"), std::string::npos);
          EXPECT_NE(source.find("const int32_t begin = tile == 0 ? 0 : " +
                                std::to_string((size + 1) / 2)), std::string::npos);
          EXPECT_NE(source.find("const int32_t count = tile == 0 ? " +
                                std::to_string((size + 1) / 2) + " : " +
                                std::to_string(size / 2)), std::string::npos);
          EXPECT_NE(source.find("data->input_0 + begin"), std::string::npos);
          EXPECT_NE(source.find("data->input_1 + begin"), std::string::npos);
          EXPECT_NE(source.find("data->output + begin"), std::string::npos);
          EXPECT_NE(source.find("if (penv->num_task >= 2) break;"), std::string::npos);
          EXPECT_NE(source.find("if (status != ARM_CMSIS_NN_SUCCESS) return -1;"),
                    std::string::npos);
        }
      }
    }
  }
}

TEST(CMSISNNParallelElementwise, SingleElementStaysSerial) {
  EXPECT_EQ(ElementwiseSource("add", 8, 1, 1).find("TVMBackendParallelLaunch("),
            std::string::npos);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
