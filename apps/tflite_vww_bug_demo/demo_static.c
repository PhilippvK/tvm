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

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>

#include "bundle.h"

extern const char build_graph_c_json[];
extern unsigned int build_graph_c_json_len;

extern const char build_params_c_bin[];
extern unsigned int build_params_c_bin_len;

#define OUTPUT_LEN 1000

int main(int argc, char** argv) {
  assert(argc == 1 && "Usage: demo_static");

  char* json_data = (char*)(build_graph_c_json);
  char* params_data = (char*)(build_params_c_bin);
  uint64_t params_size = build_params_c_bin_len;

  void* handle = tvm_runtime_create(json_data, params_data, params_size, argv[0]);
  // This is just a bug demonstration, the segfault should occour in the above function call

  tvm_runtime_destroy(handle);

  return 0;
}
