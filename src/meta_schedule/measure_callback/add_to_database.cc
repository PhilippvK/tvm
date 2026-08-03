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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

class AddToDatabaseNode : public MeasureCallbackNode {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const Array<MeasureCandidate>& measure_candidates,
             const Array<BuilderResult>& builder_results,
             const Array<RunnerResult>& runner_results) final {
    // LOG(INFO) << "AddToDatabaseNode::Apply";
    auto _ = Profiler::TimedScope("MeasureCallback/AddToDatabase");
    TuneContext task = task_scheduler->tasks_[task_id]->ctx;
    Target target = task->target.value();
    ICHECK_EQ(runner_results.size(), measure_candidates.size());
    int n = runner_results.size();
    Optional<Workload> workload, workload2;
    if (task->database.defined()) {
      workload = task->database.value()->CommitWorkload(task->mod.value());
    }
    if (task_scheduler->database_.defined()) {
      workload2 = task_scheduler->database_.value()->CommitWorkload(task->mod.value());
    }
    // LOG(INFO) << "task->mod.value()=" << task->mod.value();
    // LOG(INFO) << "workload2=" << workload2;
    if (!workload.defined() && !workload2.defined()) {
      LOG(INFO) << "No database found (skipping)";
      return;
    }
    for (int i = 0; i < n; ++i) {
      RunnerResult result = runner_results[i];
      MeasureCandidate candidate = measure_candidates[i];
      Array<FloatImm> run_secs{nullptr};
      if (result->run_secs.defined()) {
        run_secs = result->run_secs.value();
      } else {
        run_secs = Array<FloatImm>{FloatImm(DataType::Float(32), 1e10)};
      }
      Optional<FloatImm> timestamp;
      if (result->timestamp.defined()) {
          timestamp = result->timestamp.value();
      }
      // LOG(INFO) << "i=" << i;
      // LOG(INFO) << "candidate->sch->trace().value()=" << candidate->sch->trace().value();
      auto rec = TuningRecord(
          /*trace=*/candidate->sch->trace().value(),
          /*workload=*/workload.defined() ? workload.value() : workload2.value(),
          /*run_secs=*/run_secs,
          /*target=*/target,
          /*args_info=*/candidate->args_info,
          /*timestamp=*/timestamp);
      if (task->database.defined()) {
        task->database.value()->CommitTuningRecord(rec);
      }
      if (task_scheduler->database_.defined()) {
        task_scheduler->database_.value()->CommitTuningRecord(rec);
      }
    }
  }

  static constexpr const char* _type_key = "meta_schedule.AddToDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(AddToDatabaseNode, MeasureCallbackNode);
};

MeasureCallback MeasureCallback::AddToDatabase() {
  ObjectPtr<AddToDatabaseNode> n = make_object<AddToDatabaseNode>();
  return MeasureCallback(n);
}

TVM_REGISTER_NODE_TYPE(AddToDatabaseNode);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureCallbackAddToDatabase")
    .set_body_typed(MeasureCallback::AddToDatabase);

}  // namespace meta_schedule
}  // namespace tvm
