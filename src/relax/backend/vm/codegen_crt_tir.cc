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
 * \file src/relax/backend/vm/codegen_crt_tir.cc
 * \brief TODO
 */
#include <tvm/driver/driver_api.h>
#include <tvm/ir/module.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/name_transforms.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include "../../../relay/backend/utils.h"

#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace relax_vm {

using vm::VMFuncInfo;

using StorageMap =
    std::unordered_map<Expr, tvm::relay::backend::StorageInfo, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

/*!
 * \brief TODO
 *
 * \note Skip CallPacked with special attrs for now, as they can be
 *       further simplified with PrimValue.
 */
class CodeGenCRTTIR : public ExprFunctor<Optional<PrimExpr>(const Expr&)> {
 public:
  explicit CodeGenCRTTIR(relax::ExecBuilder builder, IRModule ctx_mod)
      : builder_(builder), ctx_mod_(ctx_mod) {
    system_lib_prefix_ = ctx_mod_->GetAttr<String>(tvm::attr::kSystemLibPrefix);
  }

  static IRModule Run(relax::ExecBuilder builder, IRModule mod) {
    LOG(INFO) << "Run2" << "\n";
    // create a new copy
    IRModule res_mod = mod;
    res_mod.CopyOnWrite();

    CodeGenCRTTIR codegen(builder, mod);
    // Remove relax function and turn into TIR func.
    for (auto& p : mod->functions) {
      if (auto* func = p.second.as<FunctionNode>()) {
        auto tir_func = codegen.Codegen(GetRef<Function>(func));
        auto gsymbol = tir_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        res_mod->Add(GlobalVar(gsymbol.value()), tir_func);
        res_mod->Remove(p.first);
      }
    }
    return res_mod;
  }

 private:
  int64_t NewRegister() { return registers_num_++; }

  static IntImm ConstInt64(int64_t value) { return IntImm(DataType::Int(64), value); }

  static IntImm ConstInt32(int64_t value) { return IntImm(DataType::Int(32), value); }

  PrimExpr RegListGet(int64_t slot) const {
    // use 128 bits to represent any
    return tir::Call(DataType::Handle(), tir::builtin::anylist_getitem(),
                     {reg_anylist_handle_, ConstInt32(slot)});
  }

  PrimExpr ConstListGet(int64_t slot) const {
    // use 128 bits to represent any
    return tir::Call(DataType::Handle(), tir::builtin::anylist_getitem(),
                     {const_anylist_handle_, ConstInt32(slot)});
  }

  PrimExpr FuncListGet(int64_t slot) const {
    // use 128 bits to represent any
    return tir::Call(DataType::Handle(), tir::builtin::anylist_getitem(),
                     {func_anylist_handle_, ConstInt32(slot)});
  }

  void EmitStmt(tir::Stmt stmt) {
    ICHECK(!stmt_stack_.empty());
    stmt_stack_.back().emplace_back(stmt);
  }

  void EmitCallPacked(String name, const Array<PrimExpr>& args, int64_t dst_anylist_slot = -1) {
    Array<PrimExpr> all_args;
    // negative index indicate return value can be discarded, emit call_packed
    if (dst_anylist_slot >= 0) {
      all_args = {reg_anylist_handle_, ConstInt32(dst_anylist_slot)};
    }
    all_args.push_back(tir::StringImm(name));
    for (PrimExpr arg : args) {
      all_args.push_back(arg);
    }
    if (dst_anylist_slot >= 0) {
      this->EmitStmt(tir::Evaluate(
          tir::Call(DataType::Int(32), tir::builtin::anylist_setitem_call_packed(), all_args)));
    } else {
      this->EmitStmt(
          tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(), all_args)));
    }
  }

  void EmitCallCPacked(const tir::PrimFunc& prim_func, const Array<PrimExpr>& args,
                       int64_t dst_anylist_slot = -1) {
    Optional<String> gsymbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "All functions must have global symbol at this phase";
    Array<PrimExpr> all_args;
    // negative index indicate return value can be discarded, emit call_packed
    if (dst_anylist_slot >= 0) {
      all_args = {reg_anylist_handle_, ConstInt32(dst_anylist_slot)};
    }
    all_args.push_back(tir::StringImm(gsymbol.value()));
    for (PrimExpr arg : args) {
      all_args.push_back(arg);
    }
    // push an empty handle to be compatible with current cpacked convention
    // TODO(tqchen): revisit C Packed convention
    all_args.push_back(tir::make_zero(DataType::Handle()));
    if (dst_anylist_slot >= 0) {
      this->EmitStmt(tir::Evaluate(
          tir::Call(DataType::Int(32), tir::builtin::anylist_setitem_call_cpacked(), all_args)));
    } else {
      this->EmitStmt(
          tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_call_cpacked(), all_args)));
    }
  }

  tir::PrimFunc Codegen(const Function& func) {
    LOG(INFO) << "Codegen" << "\n";
    Optional<String> gsymbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "there should be no local functions in Relax VM codegen phase. "
                                 "Did you forget to apply LambdaLift or AttachGlobalSymbol Pass?";
    // initialize the state
    // stmt_stack_ = {};
    // registers_num_ = 0;
    // var_map_.clear();
    // ctx_ptr_ = tir::Var("ctx_ptr", DataType::Handle());
    // reg_anylist_handle_ = tir::Var("r", DataType::Handle());
    // func_anylist_handle_ = tir::Var("f", DataType::Handle());
    // const_anylist_handle_ = tir::Var("c", DataType::Handle());

    // Array<String> param_names;
    // for (Var param : func->params) {
    //   param_names.push_back(param->name_hint());
    // }
    // Array<tir::Var> tir_params = {};
    for (auto input : func->params) {
      input_vars_.push_back(input);
      std::string input_name = tvm::runtime::SanitizeName(input->name_hint());
      // We dont want the compiler changing input names in the
      // event of a sanitization collision. Therefore, enforcing
      // the var created to use the input_name strictly.
      CreateIOVar(input, input_name, /*use_unique_name = */ false);
    }
    // declare this function.
    builder_->DeclareFunction(gsymbol.value(), vm::VMFuncInfo::FuncKind::kVMTIRFunc);

    for (size_t i = 0; i < func->params.size(); ++i) {
      this->var_map_.insert({func->params[i], GetBufferVarForIO(i)});
    }
    // size_t ret_reg = NewRegister();

    // tir::Stmt body = WithNewScope([&]() {
    //   Optional<PrimExpr> ret = ExprFunctor::VisitExpr(func->body);
    //   if (ret.defined()) {
    //     this->EmitCallPacked("vm.builtin.copy", {ret.value()}, ret_reg);
    //   }
    // });

    // // Mark the function entry internally.
    // builder_->EmitFunction(gsymbol.value(), param_names.size(), param_names,
    //                        VMFuncInfo::FuncKind::kVMTIRFunc, registers_num_);
    // builder_->EndFunction(gsymbol.value());

    Type ret_type = VoidType();
    // Array<tir::Var> tir_params = {ctx_ptr_, reg_anylist_handle_, const_anylist_handle_,
    //                               func_anylist_handle_};
    String tir_func_name = system_lib_prefix_.value_or("") + "__vmtir__" + gsymbol.value();
    // tir::PrimFunc tir_func(tir_params, body, ret_type, {});
    Map<String, ObjectRef> dict_attrs;
    // String run_func_name = runtime::get_name_mangled(mod_name, runtime::symbol::tvm_module_main);
    // dict_attrs.Set("global_symbol", run_func_name);
    // dict_attrs.Set("runner_function", Bool(true));
    // dict_attrs.Set(tvm::attr::kTarget, config_->host_target);
    // tir::Stmt final_body = tir::SeqStmt({device_activations, body, device_deactivations});
    // tir::Stmt body = tir::SeqStmt({});
    std::vector<tir::Stmt> stmts_;
    // tir::Stmt body = tir::SeqStmt::Flatten(stmts_);
    tir::Stmt body = WithNewScope([&]() {
      Optional<PrimExpr> ret = ExprFunctor::VisitExpr(func->body);
      if (ret.defined()) {
        // this->EmitCallPacked("vm.builtin.copy", {ret.value()}, ret_reg);
        // TODO: support tuple ouputs and user defined names
        // CreateIOVar(ret.value().as<Expr>().value(), "output");
      }
    });
    tir::PrimFunc tir_func(main_signature_, body, ret_type, main_buffer_map_, DictAttrs(dict_attrs));
    tir_func = WithAttr(tir_func, "global_symbol", tir_func_name);
    registers_num_ = 0;
    var_map_.clear();
    stmt_stack_.clear();
    return tir_func;
    // for (auto input : lowered_main_func->params) {
    //   input_vars_.push_back(input);
    //   std::string input_name = SanitizeName(input->name_hint());
    //   // We dont want the compiler changing input names in the
    //   // event of a sanitization collision. Therefore, enforcing
    //   // the var created to use the input_name strictly.
    //   CreateIOVar(input, input_name, /*use_unique_name = */ false);
    // }
  }

  Optional<PrimExpr> VisitExpr_(const SeqExprNode* op) final {
    LOG(INFO) << "VisitExpr_(const SeqExprNode* op)" << "\n";
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        Expr expr = GetBoundValue(binding);
        Optional<PrimExpr> value = VisitExpr(expr);

        if (expr.as<Var>() && value.defined()) {
          // For a normalized relax module, there should be one
          // register for each relax::Binding.  This makes the Relax
          // semantics of R.vm.kill_* operate the same as the Python
          // "del" operator.  These bindings may be removable by using
          // relax.transform.CanonicalizeBindings earlier in lowering.
          auto new_reg = NewRegister();
          EmitCallPacked("vm.builtin.copy", {value.value()}, new_reg);
          value = RegListGet(new_reg);
        }

        this->var_map_.insert({binding->var, value});
      }
    }
    return this->VisitExpr(op->body);
  }

  Optional<PrimExpr> VisitExpr_(const CallNode* call_node) final {
    LOG(INFO) << "VisitExpr_(const CallNode* call_node)" << "\n";
    Call call = GetRef<Call>(call_node);

    if (call_node->op == null_value_op_) {
      return tir::Call(DataType::Handle(), tir::builtin::reinterpret(),
                       {IntImm(DataType::Int(64), 0)});
    }
    int64_t dst_reg = HasVoidStructInfo(call) ? -1 : NewRegister();
    if (call->op.as<OpNode>()) {
      return tir::Call(DataType::Handle(), tir::builtin::reinterpret(),
                       {IntImm(DataType::Int(64), 0)});
      // if (call_node->op == call_builtin_with_ctx_op_) {
      //   EmitCallBuiltinWithCtx(call, dst_reg);
      // } else if (call_node->op == alloc_storage_op_) {
      //   EmitAllocStorage(call, dst_reg);
      // } else if (call_node->op == alloc_tensor_op_) {
      //   EmitAllocTensor(call, dst_reg);
      // // } else if (call_node->op == kill_object_op_) {
      // //   dst_reg = EmitKillObject(call);
      // } else if (call_node->op == kill_storage_op_) {
      //   dst_reg = EmitKillObject(call);
      // } else if (call_node->op == kill_tensor_op_) {
      //   dst_reg = EmitKillObject(call);
      // } else if (call_node->op == reshape_op_) {
      //   // TODO: implement!
      //   // dst_reg = EmitReshape(call, dst_reg);
      //   EmitReshape(call, dst_reg);
      // } else {
      //   // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
      //   // ops are handled in a pass when lowering them to TIR.
      //   LOG(FATAL) << "CodeGenVMTIR cannot handle this intrinsic now:\n" << call_node->op;
      // }
    } else {
      EmitNormalCall(call, dst_reg);
    }
    if (dst_reg >= 0) {
      return RegListGet(dst_reg);
    } else {
      return NullOpt;
    }
  }

  Optional<PrimExpr> VisitExpr_(const IfNode* op) final {
    LOG(INFO) << "VisitExpr_(const IfNode* op)" << "\n";
    // Reserve a register for return
    size_t merge_register = NewRegister();
    PrimExpr cond_value = this->VisitExpr(op->cond).value();

    // turn ndarray cond value into scalar.
    cond_value = tir::Cast(DataType::Bool(),
                           tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(),
                                     {tir::StringImm("vm.builtin.read_if_cond"), cond_value}));

    tir::Stmt true_branch = WithNewScope([&]() {
      PrimExpr true_value = this->VisitExpr(op->true_branch).value();
      this->EmitCallPacked("vm.builtin.copy", {true_value}, merge_register);
    });
    tir::Stmt false_branch = WithNewScope([&]() {
      PrimExpr false_value = this->VisitExpr(op->false_branch).value();
      this->EmitCallPacked("vm.builtin.copy", {false_value}, merge_register);
    });
    this->EmitStmt(tir::IfThenElse(cond_value, true_branch, false_branch));
    return RegListGet(merge_register);
  }

  Optional<PrimExpr> VisitExpr_(const VarNode* op) final {
    LOG(INFO) << "VisitExpr_(const VarNode* op)" << "\n";
    Var var = GetRef<Var>(op);
    auto it = this->var_map_.find(var);
    ICHECK(it != this->var_map_.end()) << "Var " << var << " is not defined";
    return it->second;
    // Expr expr = GetRef<Expr>(op);
    // auto var_expr = FindExpr(expr);
    // CopyToOutput(GetBufferVarForIO(input_vars_.size() + output_index), var_expr[0],
    //              /*pack_input*/ false, sinfo->storage_sizes_in_bytes[0]);
    // return GetBufferVarForIO(input_vars_.size() + output_index);
  }

  Optional<PrimExpr> VisitExpr_(const ConstantNode* op) final {
    LOG(INFO) << "VisitExpr_(const ConstantNode* op)" << "\n";
    return ConstListGet(builder_->ConvertConstant(op->data).value());
  }

  Optional<PrimExpr> VisitExpr_(const ShapeExprNode* op) final {
    LOG(INFO) << "VisitExpr_(const ShapeExprNode* op)" << "\n";
    std::vector<int64_t> shape;
    for (PrimExpr e : op->values) {
      if (auto* int_value = e.as<IntImmNode>()) {
        shape.push_back(int_value->value);
      } else {
        LOG(FATAL) << "Should only use constant shape after shape lowering: " << op->values;
      }
    }
    return ConstListGet(builder_->ConvertConstant(ShapeTuple(shape)).value());
  }

  Optional<PrimExpr> VisitExpr_(const PrimValueNode* op) final {
    LOG(INFO) << "VisitExpr_(const PrimValueNode* op)" << "\n";
    return op->value;
  }

  Optional<PrimExpr> VisitExpr_(const StringImmNode* op) final {
    LOG(INFO) << "VisitExpr_(const StringImmNode* op)" << "\n";
    return ConstListGet(builder_->ConvertConstant(op->value).value());
  }

  Optional<PrimExpr> VisitExpr_(const DataTypeImmNode* op) final {
    LOG(INFO) << "VisitExpr_(const DataTypeImmNode* op)" << "\n";
    return ConstListGet(builder_->ConvertConstant(op->value).value());
  }

  Optional<PrimExpr> VisitExpr_(const TupleNode* op) final {
    LOG(INFO) << "VisitExpr_(const TupleNode* op)" << "\n";
    Tuple tuple = GetRef<Tuple>(op);
    Array<PrimExpr> args;
    for (auto arg : tuple->fields) {
      args.push_back(this->VisitExpr(arg).value());
    }
    int32_t dst_register = NewRegister();
    this->EmitCallPacked("vm.builtin.make_tuple", args, dst_register);
    return RegListGet(dst_register);
  }

  Optional<PrimExpr> VisitExpr_(const TupleGetItemNode* op) final {
    LOG(INFO) << "VisitExpr_(const TupleGetItemNode* op)" << "\n";
    TupleGetItem expr = GetRef<TupleGetItem>(op);
    Array<PrimExpr> args = {this->VisitExpr(expr->tuple).value()};

    args.push_back(ConstInt64(expr->index));

    int64_t dst_register = NewRegister();
    this->EmitCallPacked("vm.builtin.tuple_getitem", args, dst_register);
    return RegListGet(dst_register);
  }

  // Lookup the function and see if it matches
  Optional<String> LookupFunction(const Expr& expr, VMFuncInfo::FuncKind* kind) {
    if (auto* ext_func = expr.as<ExternFuncNode>()) {
      *kind = VMFuncInfo::FuncKind::kPackedFunc;
      return ext_func->global_symbol;
    } else if (auto* gvar_ptr = expr.as<GlobalVarNode>()) {
      GlobalVar gvar = GetRef<GlobalVar>(gvar_ptr);
      // Run a look up in the env to see if it maps to an extern func.
      auto it = ctx_mod_->functions.find(gvar);
      if (it != ctx_mod_->functions.end()) {
        BaseFunc func = (*it).second;
        if (auto* efunc = func.as<ExternFuncNode>()) {
          *kind = VMFuncInfo::FuncKind::kPackedFunc;
          return efunc->global_symbol;
        } else if (func.as<FunctionNode>()) {
          *kind = VMFuncInfo::FuncKind::kVMTIRFunc;
          return gvar->name_hint;
        } else if (func.as<tir::PrimFuncNode>()) {
          *kind = VMFuncInfo::FuncKind::kPackedFunc;
          return gvar->name_hint;
        } else {
          *kind = VMFuncInfo::FuncKind::kPackedFunc;
          return gvar->name_hint;
        }
      }
      LOG(WARNING) << "Undefined global var " << gvar->name_hint;
      // undefined global var, consider eliminate later.
      *kind = VMFuncInfo::FuncKind::kPackedFunc;
      return gvar->name_hint;
    } else {
      return NullOpt;
    }
  }
  // Lookup PrimFunc in the same module
  // We can do direct PrimFunc call in such cases
  Optional<tir::PrimFunc> LookupPrimFunc(const String& name) {
    if (!ctx_mod_->ContainGlobalVar(name)) return NullOpt;

    GlobalVar gvar = ctx_mod_->GetGlobalVar(name);
    auto it = ctx_mod_->functions.find(gvar);
    if (it != ctx_mod_->functions.end()) {
      BaseFunc func = (*it).second;
      if (auto* prim_func = func.as<tir::PrimFuncNode>()) {
        return GetRef<tir::PrimFunc>(prim_func);
      }
    }
    return NullOpt;
  }

  Optional<PrimExpr> VisitExpr_(const GlobalVarNode* op) final {
    LOG(INFO) << "VisitExpr_(const GlobalVarNode* op)" << "\n";
    VMFuncInfo::FuncKind kind;
    auto symbol = LookupFunction(GetRef<Expr>(op), &kind);
    ICHECK(symbol.defined());
    builder_->DeclareFunction(symbol.value(), kind);
    return FuncListGet(builder_->GetFunction(symbol.value()).value());
  }

  Optional<PrimExpr> VisitExpr_(const ExternFuncNode* op) final {
    LOG(INFO) << "VisitExpr_(const ExternFuncNode* op)" << "\n";
    builder_->DeclareFunction(op->global_symbol, VMFuncInfo::FuncKind::kPackedFunc);
    return FuncListGet(builder_->GetFunction(op->global_symbol).value());
  }

  void EmitAllocStorage(const Call& call_node, int64_t dst_reg) {
    // Handle args of the call
    Array<PrimExpr> args;
    args.push_back(ctx_ptr_);
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg).value());
    }
    this->EmitCallPacked("vm.builtin.alloc_storage", args, dst_reg);
  }

  void EmitAllocTensor(const Call& call_node, int64_t dst_reg) {
    ICHECK_EQ(call_node->args.size(), 4);
    Array<PrimExpr> args;
    args.reserve(4);
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg).value());
    }
    this->EmitCallPacked("vm.builtin.alloc_tensor", args, dst_reg);
  }

  int64_t EmitKillObject(const Call& call_node) {
    ICHECK_EQ(call_node->args.size(), 1);
    PrimExpr arg = this->VisitExpr(call_node->args[0]).value();

    // Check the arg is a register.
    const auto* tir_call = arg.as<tir::CallNode>();
    ICHECK(tir_call != nullptr);
    ICHECK(tir_call->op == tir::builtin::anylist_getitem());
    ICHECK(tir_call->args.size() == 2);
    ICHECK(tir_call->args[0].same_as(reg_anylist_handle_));
    const auto* p_dst_reg = tir_call->args[1].as<tir::IntImmNode>();
    ICHECK(p_dst_reg != nullptr);
    ICHECK(p_dst_reg->dtype == DataType::Int(32));

    int64_t dst_reg = p_dst_reg->value;
    this->EmitCallPacked("vm.builtin.null_value", {}, dst_reg);
    return dst_reg;
  }

  void EmitReshape(const Call& call_node, int64_t dst_reg) {
    // Handle args of the call
    Array<PrimExpr> args;
    args.push_back(ctx_ptr_);
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg).value());
    }
    this->EmitCallPacked("vm.builtin.reshape", args, dst_reg);
  }

  void EmitCallBuiltinWithCtx(const Call& call_node, int64_t dst_reg) {
    Array<PrimExpr> args;
    // if context is required, pass as first argument.
    args.push_back(ctx_ptr_);
    auto* func = call_node->args[0].as<ExternFuncNode>();
    ICHECK(func) << "CallBuiltin comes with extern func";

    auto tuple_arg = Downcast<Tuple>(call_node->args[1]);

    // Handle args of the call
    for (Expr arg : tuple_arg->fields) {
      args.push_back(this->VisitExpr(arg).value());
    }

    this->EmitCallPacked(func->global_symbol, args, dst_reg);
  }

  void EmitNormalCall(const Call& call_node, int64_t dst_reg) {
    Array<PrimExpr> args = VisitArray(call_node->args);
    // A function can be a closure that comes from parent
    // Do call closure to be safe.
    VMFuncInfo::FuncKind kind;
    auto symbol = LookupFunction(call_node->op, &kind);

    if (symbol.defined() && kind == VMFuncInfo::FuncKind::kPackedFunc) {
      // primfunc in the same module.
      // use cpacked to directly invoke without named based lookup
      if (Optional<tir::PrimFunc> prim_func = LookupPrimFunc(symbol.value())) {
        this->EmitCallCPacked(prim_func.value(), args, dst_reg);
      } else {
        this->EmitCallPacked(symbol.value(), args, dst_reg);
      }
    } else {
      // Default path, leverage function table and invoke as closure
      Array<PrimExpr> all_args;
      all_args.push_back(ctx_ptr_);
      all_args.push_back(this->VisitExpr(call_node->op).value());
      for (auto arg : args) {
        all_args.push_back(arg);
      }
      this->EmitCallPacked("vm.builtin.invoke_closure", all_args, dst_reg);
    }
  }

  template <typename FLambda>
  tir::Stmt WithNewScope(const FLambda& callback) {
    stmt_stack_.push_back({});
    callback();
    tir::Stmt stmt = tir::SeqStmt::Flatten(stmt_stack_.back());
    stmt_stack_.pop_back();
    return stmt;
  }

  Array<PrimExpr> VisitArray(const Array<Expr>& arr) {
    Array<PrimExpr> ret;
    for (size_t i = 0; i < arr.size(); ++i) {
      ret.push_back(this->VisitExpr(arr[i]).value());
    }
    return ret;
  }
  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*! \brief List to ctx_ptr */
  tir::Var ctx_ptr_;
  /*! \brief List to store temp object registers */
  tir::Var reg_anylist_handle_;
  /*! \brief List to store closures */
  tir::Var func_anylist_handle_;
  /*! \brief List to store constants */
  tir::Var const_anylist_handle_;
  /*!
   * \brief Total number of virtual registers allocated.
   * \note The first two registers are reserved for special registers.
   */
  int64_t registers_num_ = 0;
  /*! \brief Stack to build up statements */
  std::vector<std::vector<tir::Stmt>> stmt_stack_;
  /*! \brief Map from var to Expr. */
  std::unordered_map<Var, Optional<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> var_map_;
  /*! \brief the context module. */
  IRModule ctx_mod_;
  /*! \brief system lib prefix */
  Optional<String> system_lib_prefix_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  // const Op& alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op& alloc_storage_op_ = Op::Get("relax.memory.alloc_storage");
  // const Op& alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  const Op& alloc_tensor_op_ = Op::Get("relax.memory.alloc_tensor");
  // const Op& kill_object_op_ = Op::Get("relax.vm.kill_object");
  const Op& kill_storage_op_ = Op::Get("relax.memory.kill_storage");
  const Op& kill_tensor_op_ = Op::Get("relax.memory.kill_tensor");
  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Op& null_value_op_ = Op::Get("relax.null_value");
  const Op& reshape_op_ = Op::Get("relax.reshape");
  ///////////
  /*! \brief list of input expressions (i.e., variable passed by the user) */
  std::vector<Var> input_vars_;
  /*! \brief input and output variables belonging to the main function signature */
  Array<tir::Var> main_signature_;
  /*! \brief input and output variables belonging to the main function signature */
  Map<tir::Var, tir::Buffer> main_buffer_map_;
  /*! \brief maps input and output variables to TensorType which describe them */
  Map<tir::Var, DynTensorType> io_tensor_types_;
  /*! \brief This is per IO var name counter to aid the generating unique names */
  std::unordered_map<std::string, int> io_var_names_;
 /*! \brief plan memory of device result */
  StorageMap storage_device_map_;
  /*! \brief mapping sid -> tir::Var */
  std::unordered_map<int, tir::Var> sids_table_;
  /*! \brief the list of return sids (note that the function might return more then one output */
  std::vector<int> return_sid_;
    /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   *
   * \param expr The expression to evaluate.
   * \param original_name The name of the tir::Var.
   * \param use_unique_name Whether to generate a new unique name where a name conflicts.
   */
  void CreateIOVar(const Expr& expr, const std::string& original_name,
                   bool use_unique_name = true) {
    Array<PrimExpr> shape;
    if (const auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr)) {
      LOG(INFO) << "if1" << "\n";
      if (const ShapeExprNode* shape_expr = tinfo->shape.as<ShapeExprNode>()) {
        LOG(INFO) << "if2" << "\n";
        shape = shape_expr->values;
      }
    }
    // CreateIOVar(expr->checked_type(), original_name, use_unique_name);
    CreateIOVar(expr->checked_type(), shape, original_name, use_unique_name);
  }

  /*!
   * \brief Create tir::Var for input/output while updating the buffer_maps.
   *
   * \param expr The expression to evaluate TODO.
   * \param shape TODO.
   * \param original_name The name of the tir::Var.
   * \param use_unique_name Whether to generate a new unique name where a name conflicts.
   */
  // void CreateIOVar(const Type& type, const std::string& original_name,
  //                  bool use_unique_name = true) {
  void CreateIOVar(const Type& type, Array<PrimExpr> shape, const std::string& original_name,
                   bool use_unique_name = true) {
    if (type->IsInstance<TupleTypeNode>()) {
      TupleType tuple_type = Downcast<TupleType>(type);
      for (unsigned i = 0; i < tuple_type->fields.size(); i++) {
        // CreateIOVar(tuple_type->fields[i], original_name);
        // TODO
      }
    } else {
      std::string name = original_name;
      if (use_unique_name) {
        name = GetUniqueIOVarName(original_name);
      }
      tir::Var var = tir::Var(name, DataType::Handle());
      main_signature_.push_back(var);
      // auto tensor_type = type.as<TensorTypeNode>();
      auto tensor_type = type.as<DynTensorTypeNode>();
      // ICHECK(tensor_type) << "Expected TensorType node but was " << type->GetTypeKey();
      ICHECK(tensor_type) << "Expected DynTensorType node but was " << type->GetTypeKey();
      DataType elem_type = tensor_type->dtype;
      tir::Var buffer_var =
          tir::Var(name + "_buffer_var", PointerType(PrimType(elem_type), "global"));
      // tir::Buffer buffer = tir::Buffer(buffer_var, elem_type, tensor_type->shape, {}, 0,
      //                                  name + "_buffer", 16, 1, tir::BufferType::kDefault);
      tir::Buffer buffer = tir::Buffer(buffer_var, elem_type, shape, {}, 0,
                                       name + "_buffer", 16, 1, tir::BufferType::kDefault);
      main_buffer_map_.Set(var, buffer);
      io_tensor_types_.Set(var, Downcast<DynTensorType>(type));
    }
  }
  /*!
   * \brief Create a unique name for I/O Var
   */
  std::string GetUniqueIOVarName(std::string name) {
    if (io_var_names_.find(name) == io_var_names_.end()) {
      io_var_names_[name] = 1;
      return name;
    } else {
      io_var_names_[name] = io_var_names_[name] + 1;
      return name + std::to_string(io_var_names_[name]);
    }
  }

  /*!
   * \brief Access IO vars using the buffer vars and
   * not the actual var.
   */
  tir::Var GetBufferVarForIO(int index) { return main_buffer_map_[main_signature_[index]]->data; }

  /*!
   * \brief Return a vector of variables that represents the sids for the given Relay Expr
   */
  std::vector<tir::Var> PackSid(Expr expr) {
    std::vector<tir::Var> buffer_vars;

    ICHECK(storage_device_map_.find(expr) != storage_device_map_.end())
        << "Storage map did not contain constant expr";
        // << "Storage map did not contain constant expr " << PrettyPrint(expr);
    tvm::relay::backend::StorageInfo& sinfo = storage_device_map_[expr];

    // Note that an expression can have multiple sids associated with it
    // e.g., returning multiple values from a function
    for (auto sid : sinfo->storage_ids) {
      // Determine if an sid is an output buffer
      auto output_iter = std::find(return_sid_.begin(), return_sid_.end(), sid);
      if (output_iter != return_sid_.end()) {
        int output_index = std::distance(return_sid_.begin(), output_iter);
        buffer_vars.push_back(GetBufferVarForIO(input_vars_.size() + output_index));
        continue;
      }

      auto sid_value = sids_table_[sid];
      buffer_vars.push_back(sid_value);
    }
    return buffer_vars;
  }

 /*!
   * brief Given an expression return the variable(s) associated with that expression
   */
  std::vector<te::Var> FindExpr(Expr arg) {
    auto input_iter = std::find(input_vars_.begin(), input_vars_.end(), arg);
    if (input_iter != input_vars_.end()) {
      // Input variable
      int main_index = std::distance(input_vars_.begin(), input_iter);
      return {GetBufferVarForIO(main_index)};
    } else {
      // Storage identifier (i.e., intermediate memory)
      return PackSid(arg);
    }
  }
};

/*!
 * \brief Create the Relax VM executable from all relax.Function in mod.
 *        and add them to exec_builder. Create extra TIR functions.
 *
 * \param exec_builder Builder to collect executables.
 * \param mod Input module.
 * \return Extra TIR module created.
 */

IRModule CRTTIRCodeGen(ExecBuilder exec_builder, IRModule mod) {
  LOG(INFO) << "CRTTIRCodeGen" << "\n";
  return CodeGenCRTTIR::Run(exec_builder, mod);
}

TVM_REGISTER_GLOBAL("relax.CRTTIRCodeGen").set_body_typed(CRTTIRCodeGen);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
