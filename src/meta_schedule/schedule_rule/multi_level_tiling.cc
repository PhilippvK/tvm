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
#include "./multi_level_tiling.h"

#include <tvm/meta_schedule/schedule_rule.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const BufferNode* read_buffer = block->reads[i]->buffer.get();
    if (read_buffer != write_buffer) {
      results[i] = read_buffer->shape.size();
    }
  }
  return results;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

TVM_REGISTER_OBJECT_TYPE(StateNode);

State::State(tir::Schedule sch, tir::BlockRV block_rv, Array<Array<tir::LoopRV>> tiles) {
  ObjectPtr<StateNode> node = make_object<StateNode>();
  node->sch = std::move(sch);
  node->block_rv = std::move(block_rv);
  node->tiles = std::move(tiles);
  data_ = std::move(node);
}

State StateNode::Copy() const {
  ObjectPtr<StateNode> node = make_object<StateNode>(*this);
  node->sch = sch->Copy();
  return State(node);
}

// Do nothing; Inherited from ScheduleRuleNode
void MultiLevelTilingNode::InitializeWithTuneContext(const TuneContext& context) {
  if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value()->value;
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }
  if (Optional<String> opt_sm = context->target.value()->GetAttr<String>("arch")) {
    std::string sm = opt_sm.value();
    if (support::StartsWith(sm, "sm_")) {
      sm = sm.substr(3);
      try {
        // only sm_80 or higher supports async memcopy
        if (std::stoi(sm) >= 80) {
          // only stage = 4 & 5 is tested. all integer that is bigger than 2
          // is theoretically feasible, but no guarantee for great performance.
          this->stages = {4, 5};
        }
      } catch (const std::invalid_argument& e) {
        LOG(WARNING) << "ValueError: Unable to parse `target.arch`: " << sm
                     << ". Details: " << e.what();
      }
    }
  }
  logger = context->logger;
}

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  if ((filter_fn_ && filter_fn_.value()(sch, sch->GetSRef(block_rv))) ||
      NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

    Array<Schedule> results;
    for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
      results.push_back(std::move(state->sch));
    }
    return results;
  }
  return {sch};
}

// Inherited from ScheduleRuleNode
ScheduleRule MultiLevelTilingNode::Clone() const {
  ObjectPtr<MultiLevelTilingNode> n = make_object<MultiLevelTilingNode>(*this);
  return ScheduleRule(n);
}

std::vector<State> MultiLevelTilingNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(std::move(state)); });
  states =
      SubRule(std::move(states), [&](State state) { return AddAsyncPipeline(std::move(state)); });
  return states;
}

std::vector<State> MultiLevelTilingNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (Optional<Array<Integer>> ann = tir::GetAnn<Array<Integer>>(
          state->sch->GetSRef(state->block_rv), "meta_schedule.write_cache_level")) {
    req = ReuseType::kMustReuse;
    levels.clear();
    std::transform(ann.value().begin(), ann.value().end(), std::back_inserter(levels),
                   [](auto&& v) { return v.IntValue(); });
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    Array<BlockRV> consumer_rvs = state->sch->GetConsumers(state->block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state->sch->GetSRef(consumer_rvs[0]))) {
      for (int level : levels) {
        State new_state = state->Copy();
        const LoopRV& loop_rv = new_state->tiles[level - 1].back();
        new_state->sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        results.push_back(std::move(new_state));
      }
      state->write_reuse.emplace(0, consumer_rvs[0]);
      results.push_back(state);
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state = state->Copy();
      results.emplace_back(std::move(new_state));
    }
  }

  // Case 3. Add one write cache
  BlockRV write_cache =
      state->sch->CacheWrite(/*block_rv=*/state->block_rv, /*read_buffer_index=*/0,
                             /*storage_scope=*/config.scope);
  state->write_reuse.emplace(0, write_cache);
  for (int level : levels) {
    State new_state = state->Copy();
    const LoopRV& loop_rv = new_state->tiles[level - 1].back();
    new_state->sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

std::pair<Array<tir::ExprRV>, Array<tir::LoopRV>> MultiLevelTilingNode::SplitLoop(
    const Schedule& sch, BlockRV block, LoopRV loop, int n_tiles, IterVarType iter_type,
    int axis_idx) const {
  String axis_key =
      (iter_type == IterVarType::kDataPar ? "S" : "R") + std::to_string(axis_idx);
  auto prefix_it = tile_prefix_products.find(axis_key);
  auto fixed_it = tile_fixed_factors.find(axis_key);
  if (prefix_it != tile_prefix_products.end()) {
    const Array<Integer>& cfg = (*prefix_it).second;
    CHECK_EQ(n_tiles, 4) << "ValueError: tile constraints require four tiles for " << axis_key;
    const auto* extent_imm = sch->Get(loop)->extent.as<IntImmNode>();
    CHECK(extent_imm) << "ValueError: tile constraints require a static extent for " << axis_key;
    int64_t extent = extent_imm->value;
    int64_t product = cfg[1].IntValue();
    CHECK_GT(extent, 0);
    CHECK_EQ(extent % product, 0)
        << "ValueError: prefix product must divide the extent of " << axis_key;
    Array<tir::ExprRV> group_factors{Integer(product)};
    if (fixed_it != tile_fixed_factors.end()) {
      const Array<Integer>& fixed = (*fixed_it).second;
      int64_t suffix[2] = {0, 0};
      for (int i = 0; i < 4; i += 2) {
        suffix[fixed[i].IntValue() - 2] = fixed[i + 1].IntValue();
      }
      int64_t remaining = extent / product;
      CHECK_EQ(remaining % suffix[0], 0)
          << "ValueError: fixed factors do not match the extent of " << axis_key;
      CHECK_EQ(remaining / suffix[0], suffix[1])
          << "ValueError: fixed factors do not match the extent of " << axis_key;
      CHECK(max_innermost_factor == -1 || suffix[1] <= max_innermost_factor)
          << "ValueError: fixed innermost factor exceeds max_innermost_factor";
      group_factors.push_back(Integer(suffix[0]));
      group_factors.push_back(Integer(suffix[1]));
    } else {
      group_factors.push_back(Integer(extent / product));
    }
    Array<LoopRV> groups = sch->Split(loop, {group_factors.begin(), group_factors.end()});
    // The prefix's last factor is not the original axis's innermost factor.
    Array<tir::ExprRV> lhs = sch->SamplePerfectTile(groups[0], 2, -1);
    Array<LoopRV> lhs_loops = sch->Split(groups[0], {lhs[0], lhs[1]});
    if (fixed_it != tile_fixed_factors.end()) {
      return {{lhs[0], lhs[1], group_factors[1], group_factors[2]},
              {lhs_loops[0], lhs_loops[1], groups[1], groups[2]}};
    }
    Array<tir::ExprRV> rhs = sch->SamplePerfectTile(groups[1], 2, max_innermost_factor);
    Array<LoopRV> rhs_loops = sch->Split(groups[1], {rhs[0], rhs[1]});
    return {{lhs[0], lhs[1], rhs[0], rhs[1]},
            {lhs_loops[0], lhs_loops[1], rhs_loops[0], rhs_loops[1]}};
  }
  Array<tir::ExprRV> factors = sch->SamplePerfectTile(
      /*loop=*/loop,
      /*n=*/n_tiles,
      /*max_innermost_factor=*/max_innermost_factor);
  Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                         /*factors=*/{factors.begin(), factors.end()});
  return {factors, splits};
}

std::vector<State> MultiLevelTilingNode::TileLoopNest(State state,
                                                      int tile_inner_most_space_loop_num) const {
  Schedule& sch = state->sch;
  const BlockRV& block_rv = state->block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state->block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;

  int total_spatial_loop_num = 0;
  std::for_each(iter_types.begin(), iter_types.end(), [&](const auto& iter_type) {
    if (iter_type == IterVarType::kDataPar) total_spatial_loop_num++;
  });
  CHECK_GE(total_spatial_loop_num, tile_inner_most_space_loop_num);
  if (tile_inner_most_space_loop_num < 0) tile_inner_most_space_loop_num = total_spatial_loop_num;
  int outer_most_spatial_loop_skipped_num = total_spatial_loop_num - tile_inner_most_space_loop_num;

  Array<LoopRV> skipped_outer_spatial_loops;
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  state->tile_factors.resize(tiles.size());
  std::vector<Array<tir::ExprRV>> tile_factors;
  tile_factors.resize(tiles.size());
  int spatial_axis_idx = 0;
  int reduction_axis_idx = 0;
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;
    IterVarType iter_type = iter_types[i];
    int axis_idx = -1;

    if (iter_type == IterVarType::kDataPar) {
      if (outer_most_spatial_loop_skipped_num > 0) {
        skipped_outer_spatial_loops.push_back(loop);
        outer_most_spatial_loop_skipped_num--;
        spatial_axis_idx++;
        continue;
      }
      idx = &s_indices_;
      axis_idx = spatial_axis_idx++;

      if (spatial_loop_product != -1) {
        if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(loop).get())) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
    } else if (iter_type == IterVarType::kCommReduce) {
      idx = &r_indices_;
      axis_idx = reduction_axis_idx++;
    } else {
      continue;
    }

    const int n_tiles = idx->size();

    if (n_tiles == 1) {
      tiles[idx->at(0)].push_back(loop);
    } else {
      auto [factors, splits] = SplitLoop(sch, block_rv, loop, n_tiles, iter_type, axis_idx);

      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
        tile_factors[idx->at(j)].push_back(factors[j]);
      }
    }
  }
  state->tile_factors = std::move(tile_factors);
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  if (skipped_outer_spatial_loops.size() && n_binds) {
    auto& the_first_tile = tiles[0];
    the_first_tile.insert(the_first_tile.begin(), skipped_outer_spatial_loops.begin(),
                          skipped_outer_spatial_loops.end());
  }
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state->tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_low_inclusive,
                  Integer(low_inclusive));
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_high_inclusive,
                  Integer(high_inclusive));
  }
  return {state};
}

std::vector<State> MultiLevelTilingNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state->block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    State new_state = state->Copy();
    Schedule& sch = new_state->sch;
    const LoopRV& loop_rv = state->tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_buffer_ndims.size(); i < n_reads; ++i) {
      int buffer_ndim = read_buffer_ndims[i];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, config.scope, {block_rv});
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
      sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                              buffer_loops.end()});
      AnnotateCooperativeFetching(&sch, cache_read_block);
      new_state->read_reuse.emplace(i, cache_read_block);
    }
    results.push_back(std::move(new_state));
  }
  return results;
}

std::vector<State> MultiLevelTilingNode::AddAsyncPipeline(State state) const {
  // For arch that does not support async pipeline, this->stages will be an empty vector
  if (r_indices_.size() < 1 || this->stages.empty()) {
    return {state};
  }
  // Current only support default config used by ScheduleRule::DefaultCUDA
  // @see src/meta_schedule/schedule_rule/schedule_rule.cc
  // check the reduce loop contains exactly 3 for loops
  // therefore it matches the notation array size in the following code
  tir::StmtSRef r_loop_sref = state->sch->GetSRef(state->tiles[r_indices_[0]].back());
  const tir::ForNode* r_for_loop = TVM_SREF_TO_FOR(r_loop_sref);
  Array<tir::Stmt> seq = Downcast<tir::SeqStmt>(r_for_loop->body)->seq;
  if (seq.size() != 3) {
    return {state};
  }
  for (auto& stmt : seq) {
    if (!stmt.as<tir::ForNode>()) {
      return {state};
    }
  }

  std::vector<State> ret;
  ret.push_back(state);
  for (int stage : this->stages) {
    State new_state = state->Copy();
    LoopRV r_loop_fused = new_state->sch->Fuse(new_state->tiles[r_indices_[0]]);
    new_state->sch->Annotate(r_loop_fused, tir::attr::software_pipeline_stage,
                             Array<Integer>{0, 0, stage - 2});
    new_state->sch->Annotate(r_loop_fused, tir::attr::software_pipeline_order,
                             Array<Integer>{0, 1, 2});
    new_state->sch->Annotate(r_loop_fused, tir::attr::software_pipeline_async_stages,
                             Array<Integer>{0});
    ret.push_back(std::move(new_state));
  }
  return ret;
}

void MultiLevelTilingNode::AnnotateCooperativeFetching(Schedule* sch,
                                                       const tir::BlockRV& block) const {
  // Filter out invalid vector lanes according to the data type.
  const tir::BlockNode* block_node = (*sch)->GetSRef(block)->StmtAs<tir::BlockNode>();
  ICHECK_EQ(block_node->writes.size(), 1);
  const runtime::DataType dtype = block_node->writes[0]->buffer->dtype;
  std::function<bool(int)> f_filter = nullptr;
  if (dtype == runtime::DataType::Float(32)) {
    f_filter = [&](int vector_len) { return vector_len <= 4; };
  } else if (dtype == runtime::DataType::Float(16)) {
    f_filter = [&](int vector_len) {
      return (vector_len == 1 || vector_len % 2 == 0) && vector_len <= 8;
    };
  } else if (dtype == runtime::DataType::Int(8)) {
    f_filter = [&](int vector_len) { return vector_len <= 16; };
  }
  std::vector<int> valid_vector_lens;
  valid_vector_lens.reserve(vector_load_lens.size());
  if (f_filter != nullptr) {
    std::copy_if(vector_load_lens.begin(), vector_load_lens.end(),
                 std::back_inserter(valid_vector_lens), f_filter);
  } else {
    valid_vector_lens = vector_load_lens;
  }

  if (!valid_vector_lens.empty()) {
    int n = valid_vector_lens.size();
    double prob = 1.0 / n;
    tir::ExprRV vector_load_len = (*sch)->SampleCategorical(
        support::AsArray<int, runtime::Int>(valid_vector_lens), Array<runtime::Float>(n, prob));
    (*sch)->Annotate(block, tir::attr::meta_schedule_cooperative_fetch, vector_load_len);
  }
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(String structure, Optional<Array<String>> tile_binds,
                                            Optional<Integer> max_innermost_factor,
                                            Optional<Array<Integer>> vector_load_lens,
                                            Optional<Map<String, ObjectRef>> reuse_read,
                                            Optional<Map<String, ObjectRef>> reuse_write,
                                            Optional<runtime::PackedFunc> filter_fn,
                                            Map<String, Array<Integer>> tile_prefix_products,
                                            Map<String, Array<Integer>> tile_fixed_factors) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  auto validate_axis = [&](const String& key) {
    std::string axis = key;
    CHECK(axis.size() >= 2 && (axis[0] == 'S' || axis[0] == 'R') &&
          axis.find_first_not_of("0123456789", 1) == std::string::npos &&
          (axis.size() == 2 || axis[1] != '0'))
        << "ValueError: invalid tile constraint axis key: " << key;
    CHECK_EQ(axis[0] == 'S' ? node->s_indices_.size() : node->r_indices_.size(), 4)
        << "ValueError: tile constraints require four tiles for " << key;
  };
  for (const auto& kv : tile_prefix_products) {
    validate_axis(kv.first);
    CHECK_EQ(kv.second.size(), 2) << "ValueError: expected [prefix length, product]";
    CHECK_EQ(kv.second[0].IntValue(), 2) << "ValueError: only prefix length 2 is supported";
    CHECK_GT(kv.second[1].IntValue(), 0) << "ValueError: prefix product must be positive";
  }
  for (const auto& kv : tile_fixed_factors) {
    validate_axis(kv.first);
    CHECK(tile_prefix_products.count(kv.first))
        << "ValueError: fixed factors require a prefix product for " << kv.first;
    const Array<Integer>& cfg = kv.second;
    CHECK_EQ(cfg.size(), 4) << "ValueError: fix both suffix factors 2 and 3";
    CHECK((cfg[0].IntValue() == 2 && cfg[2].IntValue() == 3) ||
          (cfg[0].IntValue() == 3 && cfg[2].IntValue() == 2))
        << "ValueError: fix both suffix factors 2 and 3 exactly once";
    CHECK_GT(cfg[1].IntValue(), 0) << "ValueError: fixed factors must be positive";
    CHECK_GT(cfg[3].IntValue(), 0) << "ValueError: fixed factors must be positive";
  }
  node->tile_prefix_products = std::move(tile_prefix_products);
  node->tile_fixed_factors = std::move(tile_fixed_factors);
  node->filter_fn_ = filter_fn;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTiling")
    .set_body_typed(ScheduleRule::MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
