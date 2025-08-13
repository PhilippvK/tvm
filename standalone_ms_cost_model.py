# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable

import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.logging import get_logger


logging.basicConfig(level=logging.INFO)
get_logger("xgb_model").setLevel(logging.INFO)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR.parent

def create_cost_model():
    # TODO: support other extractions and models
    num_warmup_samples = 0
    extractor = ms.feature_extractor.PerStoreFeature()
    cost_model = ms.cost_model.XGBModel(extractor=extractor, num_warmup_samples=num_warmup_samples)
    print("cost_model", cost_model, dir(cost_model))
    return cost_model


def generate_tasks(samples):
    tasks = []
    candidates = []
    results = []
    target = tvm.target.Target("c")
    space_generator = None
    search_strategy = None
    for sample in samples:
        mod, runtime = sample
        print("mod", mod, type(mod))
        print("runtime", runtime, type(runtime))
        ctx = ms.tune_context.TuneContext(
            mod=mod,
            target=target,
            # space_generator=space,
            # search_strategy=strategy,
            # task_name=task_name,
            # logger=logger,
            # rand_state=rand_state,
            # num_threads=num_tuning_cores,
        )
        tasks.append(ctx)
        sched = tvm.tir.Schedule(mod)
        candidate = ms.MeasureCandidate(sch=sched, args_info=[])
        candidates.append(candidate)
        res = ms.runner.RunnerResult([runtime], "", 0.0)
        results.append(res)
    return tasks, candidates, results




def update_cost_model(cost_model, samples):
    tasks, all_candidates, all_results = generate_tasks(samples)
    for i in range(len(tasks)):
        context = tasks[i]
        candidates = [all_candidates[i]]
        results = [all_results[i]]
        cost_model.update(context, candidates, results)


def test_cost_model(cost_model, samples):
    tasks, _, _ = generate_tasks(samples)
    predictions = []
    expected = []
    for i in range(len(tasks)):
        tune_ctx = tasks[i]
        print("tune_ctx", tune_ctx, dir(tune_ctx))
        sched = tvm.tir.Schedule(tune_ctx.mod)
        print("sched", sched)
        dummy_candidate = ms.MeasureCandidate(sch=sched, args_info=[])
        print("dummy_candidate", dummy_candidate)
        if True:
            extractor = ms.feature_extractor.PerStoreFeature()
            (dummy_feature,) = extractor.extract_from(
                tune_ctx,
                candidates=[dummy_candidate],
            )
            print("dummy_feature", dummy_feature, dir(dummy_feature))
        dummy_predictions = cost_model.predict(tune_ctx, [dummy_candidate])
        predictions.append(dummy_predictions[0])
        _, expected_runtime = samples[i]
        expected.append(expected_runtime)
    print("predictions", predictions)
    print("expected", expected)


def load_tir(tir_path):
    with open(tir_path, "r") as f:
        content = f.read()
    obj = tvm.script.from_source(content)
    print("obj", obj, dir(obj), type(obj))
    if isinstance(obj, tvm.tir.PrimFunc):
        default_name = "main"
        obj = tvm.IRModule({default_name: obj})
    assert isinstance(obj, tvm.IRModule)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or test a TVM MetaSchedule cost model"
    )

    parser.add_argument(
        "--input-model",
        type=Path,
        help="Path to load an existing cost model file"
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help="Path to save the trained cost model file"
    )
    parser.add_argument(
        "--samples",
        # type=float,
        nargs="+",
        help=(
            "List of samples as alternating values: [feature, runtime, feature, runtime, ...]. "
            "Must be even-length."
        )
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the order of samples before splitting"
    )
    parser.add_argument(
        "--split-samples",
        type=float,
        help=(
            "Fraction of samples to use for training (0.0–1.0). "
            "If not set, the same samples will be used for training and testing."
        )
    )

    args = parser.parse_args()
    cost_model = create_cost_model()
    cost_model.num_warmup_samples = 1  # Do not get random predictions. TODO: expose
    if args.input_model is not None:
        cost_model_file = args.input_model
        logging.info("Reading cost model from disk... (%s)", cost_model_file)
        cost_model.load(str(cost_model_file))
    if args.samples is not None and len(args.samples) > 0:
        logging.info("Processing Samples")
        samples = args.samples
        assert len(samples) % 2 == 0
        samples = [(samples[2*i], samples[2*i+1]) for i in range(len(samples) // 2)]
        samples_cnt = len(samples)
        print("samples", samples)
        samples = [(load_tir(x[0]), float(x[1])) for x in samples]
    # task_name = mod.func_name
        if args.randomize:
            raise NotImplementedError("randomize sample order")
        if args.split_samples is not None:
            split = args.split_samples
            assert isinstance(split, float)
            train_cnt = int(samples_cnt * split)
            assert 0 <= train_cnt <= samples_cnt
            test_cnt = samples_cnt - train_cnt
            assert 0 <= test_cnt <= samples_cnt
            train_samples = samples[:train_cnt]
            assert len(train_samples) == train_cnt
            test_samples = samples[train_cnt:]
            assert len(test_samples) == test_cnt
        else:
            train_samples = test_samples = samples
            train_cnt = test_cnt = samples_cnt
        logging.info("sample_cnt: %d, train_cnt: %d, test_cnt: %d", samples_cnt, train_cnt, test_cnt)
    else:
        logging.info("No samples provided")
        train_samples = []
        test_samples = []
    if len(train_samples) > 0:
        logging.info("Training cost model")
        update_cost_model(cost_model, train_samples)
    else:
        logging.info("Skipping training (train_set empty)")
    if args.output_model is not None:
        cost_model_file = args.output_model
        logging.info("Writing cost model to disk... (%s)", cost_model_file)
        cost_model.save(str(cost_model_file))
    if len(test_samples) > 0:
        logging.info("Testing cost model")
        test_cost_model(cost_model, test_samples)
    else:
        logging.info("Skipping testing (test_set empty)")


if __name__ == "__main__":
    main()
