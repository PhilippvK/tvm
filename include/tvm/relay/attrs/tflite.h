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
 * \file tvm/relay/attrs/nn.h
 * \brief Auxiliary attributes for nn operators.
 */
#ifndef TVM_RELAY_ATTRS_TFLITE_H_
#define TVM_RELAY_ATTRS_TFLITE_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

#include "tvm/runtime/container.h"

namespace tvm {
namespace relay {
// TODO: HEADER

/*! \brief Attributes for tflite_extern operator */
struct TfLiteExternAttrs : public tvm::AttrsNode<TfLiteExternAttrs> {
  tvm::String name;
  bool is_builtin;
  DataType out_dtype;
  Array<Integer> options;
  Array<Integer> out_shape;

  TVM_DECLARE_ATTRS(TfLiteExternAttrs, "relay.attrs.TfLiteExternAttrs") {
    TVM_ATTR_FIELD(name).describe("Custom name");
    TVM_ATTR_FIELD(is_builtin).describe("Builtin Op Flag");
    TVM_ATTR_FIELD(out_dtype).describe("Output Datatype");
    TVM_ATTR_FIELD(options).describe("Custom options");
    TVM_ATTR_FIELD(out_shape).describe("OutputShape");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_H_

