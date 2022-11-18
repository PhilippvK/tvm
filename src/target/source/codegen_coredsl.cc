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

/*!
 * \file codegen_coredsl.cc
 */
#include "codegen_coredsl.h"

#include <string>
#include <vector>

#include "../build_common.h"

namespace tvm {
namespace codegen {

void CodeGenCoreDSL::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);

  // TODO: write header
  this->stream << "// TODO\n\n";
}

void CodeGenCoreDSL::PrintType(DataType t, std::ostream& os) {
  // TODO: bool
  // TODO: do not allow float
  if (t.is_uint()) {
    switch (t.bits()) {
      // case 8:
      //   os << "unsigned char";
      //   break;
      // case 16:
      //   os << "unsigned short";
      //   break;
      // case 32:
      //   os << "unsigned int";
      //   break;
      // case 64:
      //   os << "unsigned long";
      //   break;
      default:
        os << "unsigned<" << t.bits() << ">";
        break;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      // case 8:
      //   os << "char";
      //   break;
      // case 16:
      //   os << "short";
      //   break;
      // case 32:
      //   os << "int";
      //   break;
      // case 64:
      //   os << "long";
      //   break;
      default:
        os << "signed<" << t.bits() << ">";
        break;
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

// TODO: remove
// void CodeGenCoreDSL::PrintFuncPrefix() { stream << "extern \"C\" void"; }

// TODO: remove
// void CodeGenCoreDSL::PreFunctionBody(const PrimFunc& f) {
//   // for (size_t i = 0; i < f->params.size(); ++i) {
//   //   Var v = f->params[i];
//   //   std::string vid = GetVarID(v.get());
//   //   if (v.dtype().is_handle()) {
//   //     this->stream << "#pragma HLS INTERFACE m_axi port=" << vid << "  offset=slave bundle=gmem\n";
//   //   }
//   //   this->stream << "#pragma HLS INTERFACE s_axilite port=" << vid << " bundle=control\n";
//   // }
//   // this->stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n";
// }

// TODO: remove
// template <typename T>
// inline void PrintBinaryExpr(const T* op, const char* opstr,
//                             std::ostream& os,  // NOLINT(*)
//                             CodeGenCoreDSL* p) {
//   os << opstr << '(';
//   p->PrintExpr(op->a, os);
//   os << ", ";
//   p->PrintExpr(op->b, os);
//   os << ')';
// }

// void CodeGenCoreDSL::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
//   // TODO: gen min func
//   const char* opstr = "MIN(";
//   if (op->dtype.is_float()) {
//       // TODO: error
//   }
//
//   PrintBinaryExpr(op, opstr, os, this);
// }
//
// void CodeGenCoreDSL::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
//   // TODO: gen max func
//   const char* opstr = "MAX(";
//   if (op->dtype.is_float()) {
//       // TODO: error
//   }
//
//   PrintBinaryExpr(op, opstr, os, this);
// }

runtime::Module BuildCoreDSL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenCoreDSL cg;

  // Generate source code for get_source().
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenCoreDSL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // TODO: remove?
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenVLHS: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  // Generate source code for compilation.
  Array<Array<runtime::String>> kernel_info;

  // ?
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_REGISTER_GLOBAL("target.build.coredsl").set_body_typed(BuildCoreDSL);

}  // namespace codegen
}  // namespace tvm
