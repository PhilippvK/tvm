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
"""
Provides support to auto-tuning networks using AutoTVM.
"""
import os.path
import logging
import time
import tempfile
import shutil
from copy import deepcopy
from typing import Any, Optional, Dict, List, Union

from urllib.parse import urlparse

import tvm
from tvm import autotvm, auto_scheduler
from tvm.auto_scheduler.search_task import HardwareParams
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner
from tvm import meta_schedule as ms
from tvm.target import Target

from . import TVMCException, composite_target, frontends
from .main import register_parser
from .model import TVMCModel
from .target import target_from_cli, generate_target_args, reconstruct_target_args
from .shape_parser import parse_shape_string
from .transform import convert_graph_layout


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


def add_tune_args(parser, micro=False):
    parser.add_argument(
        "--early-stopping",
        type=int,
        help="minimum number of trials before early stopping",
    )

    # There is some extra processing required to define the actual default value
    # for --min-repeat-ms. This is done in `tune_model`.
    parser.add_argument(
        "--min-repeat-ms",
        default=None,
        type=int,
        help="minimum time to run each trial, in milliseconds. "
        "Defaults to 0 on x86 and 1000 on all other targets",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format",
    )
    parser.add_argument(
        "--number",
        default=1 if micro else 10,
        type=int,
        help="number of runs a single repeat is made of. "
        "The final number of tuning executions is: "
        "(1 + number * repeat)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output file to store the tuning records for the tuning process",
    )
    parser.add_argument(
        "--parallel",
        default=1 if micro else 4 ,
        type=int,
        help="the maximum number of parallel devices to use when tuning",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to repeat each measurement",
    )
    if not micro:
        parser.add_argument(
            "--rpc-key",
            help="the RPC tracker key of the target device. " "Required when --rpc-tracker is provided.",
        )
        parser.add_argument(
            "--rpc-tracker",
            help="hostname (required) and port (optional, defaults to 9090) of the RPC tracker, "
            "e.g. '192.168.0.100:9999'",
        )

    generate_target_args(parser, micro=micro)

    if not micro:
        parser.add_argument(
            "--target-host",
            help="the host compilation target.",
        )

    parser.add_argument("--timeout", type=int, default=10, help="compilation timeout, in seconds")  # TODO: check if increase for micro is required
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="the maximum number of tuning trials to perform",
    )
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        help="path to an auto-tuning log file by AutoTVM.",
    )
    parser.add_argument(
        "--desired-layout",
        # choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    if not micro:
        parser.add_argument(
            "--enable-autoscheduler",
            help="enable tuning the graph through the AutoScheduler tuner",
            action="store_true",
        )
        parser.add_argument(
            "--enable-metascheduler",
            help="enable tuning the graph through the MetaScheduler tuner",
            action="store_true",
        )

    if not micro:
        auto_scheduler_group = parser.add_argument_group(
            "AutoScheduler options",
            "AutoScheduler options, used when --enable-autoscheduler is provided",
        )

        auto_scheduler_group.add_argument(
            "--cache-line-bytes",
            type=int,
            help="the size of cache line in bytes. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--num-cores",
            type=int,
            help="the number of device cores. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--vector-unit-bytes",
            type=int,
            help="the width of vector units in bytes. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--max-shared-memory-per-block",
            type=int,
            help="the max shared memory per block in bytes. "
            "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--max-local-memory-per-block",
            type=int,
            help="the max local memory per block in bytes. "
            "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--max-threads-per-block",
            type=int,
            help="the max number of threads per block. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--max-vthread-extent",
            type=int,
            help="the max vthread extent. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--warp-size",
            type=int,
            help="the thread numbers of a warp. " "If not specified, it will be autoset for the current machine.",
        )
        auto_scheduler_group.add_argument(
            "--include-simple-tasks",
            help="whether to extract simple tasks that do not include complicated ops",
            action="store_true",
        )
        auto_scheduler_group.add_argument(
            "--log-estimated-latency",
            help="whether to log the estimated latency to the file after tuning a task",
            action="store_true",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-verbose",
            default=1,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-strategy",
            choices=["gradient", "round-robin"],
            default="gradient",
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-strategy-gradient-alpha",
            default=0.2,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-strategy-gradient-beta",
            default=2.0,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-strategy-gradient-gamma",
            default=0.5,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-strategy-gradient-backward-window-size",
            default=3,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy",
            choices=["sketch"],
            default="sketch",
            type=str,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-eps-greedy",
            default=0.05,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-retry-search-one-round-on-empty",
            default=1,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-sample-init-min-population",
            default=50,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-sample-init-use-measured-ratio",
            default=0.2,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-evolutionary-search-population",
            default=2048,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-evolutionary-search-num-iters",
            default=4,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-evolutionary-search-mutation-prob",
            default=0.85,
            type=float,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-cpu-multi-level-tiling-structure",
            default="SSRSRS",
            type=str,
            help="",
        )
        # Notice: the default thread bind policy of GPU assumes the tiling structure to have at
        # least 3 spatial tiling levels in outermost
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-gpu-multi-level-tiling-structure",
            default="SSSRRSRS",
            type=str,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-max-innermost-split-factor",
            default=64,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-max-vectorize-size",
            default=16,
            type=int,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-policy-sketch-disable-change-compute-location",
            help="",
            action="store_true",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-model",
            choices=["xgb", "mlp", "random"],
            default="xgb",
            type=str,
            help="",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-model-xgb-adaptive-training",
            help="",
            action="store_true",
        )
        auto_scheduler_group.add_argument(
            "--autoscheduler-num-measures-per-round",
            default=64,
            type=int,
            help="",
        )
        meta_scheduler_group = parser.add_argument_group(
            "MetaScheduler options",
            "MetaScheduler options, used when --enable-metascheduler is provided",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-space",
            choices=["post-order-apply"],  # union is not really useful here
            default="post-order-apply",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-rules",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-postprocs",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-mutator-probs",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model",
            choices=["xgb", "mlp", "random"],
            default="xgb",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-max-depth",
            default=10,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-gamma",
            default=0.001,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-min-child-weight",
            default=0,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-eta",
            default=0.2,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-seed",
            default=43,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-nthread",
            default=None,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-num-warmup_samples",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-verbose-equal",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-average-peak-n",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-adaptive-training",
            help="",
        )
        # meta_scheduler_group.add_argument(
        #     "--metascheduler-model-mlp-",
        #     help="",
        # )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-random-max-range",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy",
            choices=["evolutionaly_search", "replay_trace", "replay_func"],
            default="evolutionaly_search",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-population-size",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-init-measured-ratio",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-init-min-unmeasured",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-num-iters",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-mutate-prob",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-max-fail-count",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-eps-greedy",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-replay-trace-max-fail-count",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-space-post-order-apply-",
            help="",
        )
        meta_scheduler_group = parser.add_argument_group(
            "MetaScheduler options",
            "MetaScheduler options, used when --enable-metascheduler is provided",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-space",
            choices=["post-order-apply"],  # union is not really useful here
            default="post-order-apply",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-rules",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-postprocs",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-mutator-probs",
            default="from-target",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model",
            choices=["xgb", "mlp", "random"],
            default="xgb",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-max-depth",
            default=10,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-gamma",
            default=0.001,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-min-child-weight",
            default=0,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-eta",
            default=0.2,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-seed",
            default=43,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-nthread",
            default=None,
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-num-warmup_samples",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-verbose-equal",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-average-peak-n",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-xgb-adaptive-training",
            help="",
        )
        # meta_scheduler_group.add_argument(
        #     "--metascheduler-model-mlp-",
        #     help="",
        # )
        meta_scheduler_group.add_argument(
            "--metascheduler-model-random-max-range",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy",
            choices=["evolutionaly_search", "replay_trace", "replay_func"],
            default="evolutionaly_search",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-population-size",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-init-measured-ratio",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-init-min-unmeasured",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-num-iters",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-mutate-prob",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-genetic-max-fail-count",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-evolutionaly-search-eps-greedy",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-strategy-replay-trace-max-fail-count",
            help="",
        )
        meta_scheduler_group.add_argument(
            "--metascheduler-space-post-order-apply-",
            help="",
        )
    autotvm_group = parser.add_argument_group(
        "AutoTVM options",
        "AutoTVM options, used when the AutoScheduler or MetaScheduler is not enabled",
    )
    autotvm_group.add_argument(
        "--tuner",
        choices=["ga", "gridsearch", "random", "xgb"],
        default="xgb",
        help="type of tuner to use when tuning with autotvm.",
    )
    autotvm_group.add_argument(
        "--tuner-ga-pop-size",
        default=50,
        type=int,
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-ga-elite-num",
        default=3,
        type=int,
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-ga-mutation-prob",
        default=0.1,
        type=float,
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-xgb-plan-size",
        default=50,
        type=int,
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-xgb-loss-type",
        choices=["rank", "reg"],
        default="rank",
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-xgb-feature-type",
        choices=["itervar", "knob", "curve"],
        default="itervar",
        help="",
    )
    # autotvm_group.add_argument(
    #     "--tuner-xgb-num-threads",
    #     default="logical",
    #     help="",
    # )
    autotvm_group.add_argument(
        "--tuner-xgb-optimizer",
        choices=["sa"],
        default="sa",
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-xgb-log-interval",
        default=50,
        type=int,
        help="",
    )
    autotvm_group.add_argument(
        "--tuner-xgb-diversity-filter-ratio",
        default=None,
        type=float,
        help="",
    )
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file")
    parser.add_argument(
        "--input-shapes",
        help="specify non-generic shapes for model to run, format is "
        '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"',
        type=parse_shape_string,
    )


@register_parser
def add_tune_parser(subparsers, _, json_params):
    """Include parser for 'tune' subcommand"""

    parser = subparsers.add_parser("tune", help="auto-tune a model")
    parser.set_defaults(func=drive_tune)

    add_tune_args(parser)

    for one_entry in json_params:
        parser.set_defaults(**one_entry)


def parse_tuner_options(args):
    """Parser for AutoTVM tuner kwargs

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """
    return {
        "xgb": {
            "loss_type": args.tuner_xgb_loss_type,
            "feature_type": args.tuner_xgb_feature_type,
            # "num_threads": args.tuner_xgb_num_threads,
            "optimizer": args.tuner_xgb_optimizer,
            "log_interval": args.tuner_xgb_log_interval,
            "diversity_filter_ratio": args.tuner_xgb_diversity_filter_ratio,
        },
        "ga": {
            "pop_size": args.tuner_ga_pop_size,
            "elite_num": args.tuner_ga_elite_num,
            "mutation_prob": args.tuner_ga_mutation_prob,
        }
    }


def parse_autoscheduler_options(args):
    verbose = args. autoscheduler_verbose
    strategy = args.autoscheduler_strategy
    strategy_args = {
        "gradient": {
            "alpha": args.autoscheduler_strategy_gradient_alpha,
            "beta": args.autoscheduler_strategy_gradient_beta,
            "gamma": args.autoscheduler_strategy_gradient_gamma,
            "backward_window_size": args.autoscheduler_strategy_gradient_backward_window_size,
        }
    }
    policy = args.autoscheduler_policy,
    policy_args = {
        "eps_greedy": args.autoscheduler_policy_sketch_eps_greedy,
        "retry_search_one_round_on_empty": args.autoscheduler_policy_sketch_retry_search_one_round_on_empty,
        "sample_init_min_population": args.autoscheduler_policy_sketch_sample_init_min_population,
        "sample_init_use_measured_ratio": args.autoscheduler_policy_sketch_sample_init_use_measured_ratio,
        "evolutionary_search_population": args.autoscheduler_policy_sketch_evolutionary_search_population,
        "evolutionary_search_num_iters": args.autoscheduler_policy_sketch_evolutionary_search_num_iters,
        "evolutionary_search_mutation_prob": args.autoscheduler_policy_sketch_evolutionary_search_mutation_prob,
        "cpu_multi_level_tiling_structure": args.autoscheduler_policy_sketch_cpu_multi_level_tiling_structure,
        "gpu_multi_level_tiling_structure": args.autoscheduler_policy_sketch_gpu_multi_level_tiling_structure,
        "max_innermost_split_factor": args.autoscheduler_policy_sketch_max_innermost_split_factor,
        "max_vectorize_size": args.autoscheduler_policy_sketch_max_vectorize_size,
        "disable_change_compute_location": args.autoscheduler_policy_sketch_disable_change_compute_location,
    }
    num_measures_per_round = args.autoscheduler_num_measures_per_round
    model_type = args.autoscheduler_model
    adaptive = args.autoscheduler_model_xgb_adaptive_training
    return verbose, strategy, strategy_args, policy, policy_args, num_measures_per_round, model_type, adaptive


def drive_tune(args):
    """Invoke auto-tuning with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """
    if not os.path.isfile(args.FILE):
        raise TVMCException(
            f"Input file '{args.FILE}' doesn't exist, is a broken symbolic link, or a directory."
        )

    tvmc_model = frontends.load_model(args.FILE, args.model_format, shape_dict=args.input_shapes)

    # Specify hardware parameters, although they'll only be used if autoscheduling.
    hardware_params = auto_scheduler.HardwareParams(
        num_cores=args.num_cores,
        vector_unit_bytes=args.vector_unit_bytes,
        cache_line_bytes=args.cache_line_bytes,
        max_shared_memory_per_block=args.max_shared_memory_per_block,
        max_local_memory_per_block=args.max_local_memory_per_block,
        max_threads_per_block=args.max_threads_per_block,
        max_vthread_extent=args.max_vthread_extent,
        warp_size=args.warp_size,
        target=args.target,
        target_host=args.target_host,
    )

    if args.rpc_tracker:
        parsed_url = urlparse("//%s" % args.rpc_tracker)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

        if not args.rpc_key:
            raise TVMCException("need to provide an RPC tracker key (--rpc-key) for remote tuning")
    else:
        rpc_hostname = None
        rpc_port = None

    autotvm_tuner_options = parse_tuner_options(args)
    autoscheduler_verbose, autoscheduler_strategy, autoscheduler_strategy_args, autoscheduler_policy, autoscheduler_policy_args, autoscheduler_num_measures_per_round, autoscheduler_model_type, autoscheduler_adaptive = parse_autoscheduler_options(args)

    tune_model(
        tvmc_model,
        args.target,
        tuning_records=args.output,
        prior_records=args.tuning_records,
        enable_autoscheduler=args.enable_autoscheduler,
        enable_metascheduler=args.enable_metascheduler,
        rpc_key=args.rpc_key,
        hostname=rpc_hostname,
        port=rpc_port,
        trials=args.trials,
        target_host=args.target_host,
        tuner=args.tuner,
        min_repeat_ms=args.min_repeat_ms,
        early_stopping=args.early_stopping,
        desired_layout=args.desired_layout,
        timeout=args.timeout,
        repeat=args.repeat,
        number=args.number,
        parallel=args.parallel,
        hardware_params=hardware_params,
        include_simple_tasks=args.include_simple_tasks,
        log_estimated_latency=args.log_estimated_latency,
        additional_target_options=reconstruct_target_args(args),
        autotvm_tuner_options=autotvm_tuner_options,
        autoscheduler_verbose=autoscheduler_verbose,
        autoscheduler_strategy=autoscheduler_strategy,
        autoscheduler_strategy_args=autoscheduler_strategy_args.get(autoscheduler_strategy, {}),
        autoscheduler_policy=autoscheduler_policy,
        autoscheduler_policy_args=autoscheduler_policy_args,
        autoscheduler_num_measures_per_round=autoscheduler_num_measures_per_round,
        autoscheduler_model_type=autoscheduler_model_type,
        autoscheduler_adaptive=autoscheduler_adaptive,
    )


def tune_model(
    tvmc_model: TVMCModel,
    target: str,
    tuning_records: Optional[str] = None,
    prior_records: Optional[str] = None,
    enable_autoscheduler: bool = False,
    enable_metascheduler: bool = False,
    rpc_key: Optional[str] = None,
    hostname: Optional[str] = None,
    port: Optional[Union[int, str]] = 9090,
    trials: int = 10000,
    target_host: Optional[str] = None,
    tuner: str = "xgb",
    min_repeat_ms: Optional[int] = None,
    early_stopping: Optional[int] = None,
    desired_layout: Optional[str] = None,
    timeout: int = 10,
    repeat: int = 1,
    number: int = 10,
    parallel: int = 4,
    hardware_params: Optional[HardwareParams] = None,
    include_simple_tasks: bool = False,
    log_estimated_latency: bool = False,
    additional_target_options: Optional[Dict[str, Dict[str, Any]]] = None,
    autotvm_tuner_options: Optional[Dict[str, Dict[str, Any]]] = None,
    autoscheduler_verbose: int = 1,
    autoscheduler_strategy: Optional[str] = None,
    autoscheduler_strategy_args: Optional[Dict[str, Any]] = None,
    autoscheduler_policy: Optional[str] = None,
    autoscheduler_policy_args: Optional[Dict[str, Any]] = None,
    autoscheduler_num_measures_per_round: int = 64,
    autoscheduler_model_type="xgb",  # TODO
    autoscheduler_adaptive=False,  # TODO
    module_loader = None,  # TODO
    build_func = "default",  # TODO
    runtime = None,  # TODO
    build_option : dict = None,
    si_prefix: str = "G",
):
    """Use tuning to automatically optimize the functions in a model.

    Parameters
    ----------
    tvmc_model : TVMCModel
        The model to be optimized.
    target : str
        Compilation target as plain string, inline JSON or path to a JSON file.
    tuning_records: str, optional
        The path to a file that tuning results will be saved to. If not specified,
        a temporary file will be used.
    prior_records: str, optional
        A path to previous tuning results that will be used to hot-start the tuning
        cost model if provided.
    enable_autoscheduler : bool, optional
        When true, use autoscheduling rather than autotvm. This should produce
        faster kernels for compatible model-target pairs.
    enable_metascheduler : bool, optional
        When true, use metascheduling rather than autotvm. This should produce
        faster kernels for compatible model-target pairs.
    rpc_key : str, optional
        The RPC tracker key of the target device. Required when rpc_tracker is provided.
    hostname : str, optional
        The IP address of an RPC tracker, used when benchmarking remotely.
    port : int or str, optional
        The port of the RPC tracker to connect to. Defaults to 9090.
    trials : int, optional
        The number of schedules to try out for the entire model. Note that the default
        value is chosen as a decent average for most models, but larger models may need
        more trials to reach a good result while smaller models will converge with fewer
        trials.
    tuner : str, optional
        The type of tuner to use when tuning with autotvm. Can be one of
        "ga", "gridsearch", "random", "xgb"
    min_repeat_ms : int, optional
        Minimum time to run each trial. Defaults to 0 on x86 and 1000 on other targets.
    early_stopping : int, optional
        When specified, stop tuning after this number of trials if results aren't improving.
    desired_layout : str, optional
        Can be one of "NCHW" or "NHWC". When specified, compatible operations in the graph
        will have their layout set to this format. Tasks will then be tuned using this
        specified layout.
    timeout : int, optional,
        If a kernel trial lasts longer than this duration in seconds, it will be
        considered a failure.
    repeat : int, optional
        How many times each measurement should be repeated.
    number : int, optional
        The number of runs a single repeat is made of.
    parallel : int, optional
        The maximum number of parallel devices to use when tuning.
    hardware_params : auto_scheduler.HardwareParams, optional
        When using the autoscheduler, this object defines the configuration of the target hardware.
    include_simple_tasks : bool, optional
        Whether to extract simple operations or only computationally intensive ones when using
        the autoscheduler.
    log_estimated_latency : bool, optional
        If using the autoscheduler, write the estimated latency at each step of tuning to file.
    additional_target_options: Optional[Dict[str, Dict[str, Any]]]
        Additional target options in a dictionary to combine with initial Target arguments
    autotvm_tuner_options: Optional[Dict[str, Dict[str, Any]]]
        Additional kwsrags for AutoTVM tuner object.
    autoscheduler_verbose : int, optional
        Verbosity level of autoscheduler. 0 equals silent.
    autoscheduler_strategy: Optional[str]
        TODO
    autoscheduler_strategy_args: Optional[Dict[str, Any]]
        TODO
    autoscheduler_policy: Optional[str]
        TODO
    autoscheduler_policy_args: Optional[Dict[str, Any]]
        TODO
    autoscheduler_num_measures_per_round: Optional[int]
        TODO
    autoscheduler_model_type: TODO
        TODO
    autoscheduler_adaptive: TODO
        TODO
    module_loader : TODO, optional
        TODO
    build_func : TODO, optional
        TODO
    runtime : TODO, optional
        TODO
    build_option : dict, optional
        TODO
    si_prefix : str
        SI prefix for FLOPS.

    Returns
    -------
    tuning_records : str
        The path to the produced tuning log file.
    """
    target, extra_targets = target_from_cli(target, additional_target_options)
    target, target_host = Target.canon_target_and_host(target, target_host)
    # TODO(jwfromm) Remove this deepcopy once AlterOpLayout bug that mutates source
    # model is fixed. For now, creating a clone avoids the issue.
    mod = deepcopy(tvmc_model.mod)
    params = tvmc_model.params

    if enable_autoscheduler and enable_metascheduler:
        raise TVMCException(
            "Autoscheduler and Metascheduler can not be enabled at the same time."
        )

    with tvm.transform.PassContext(opt_level=3):
        if tuning_records is None:
            tuning_records = tvmc_model.default_tuning_records_path()

        for codegen_from_cli in extra_targets:
            codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
            partition_function = codegen["pass_pipeline"]
            mod = partition_function(mod, params, **codegen_from_cli["opts"])

        # min_repeat_ms should be:
        # a. the value provided by the user, if any, or
        # b. 0ms in case target is "cpu"; otherwise 1000ms
        if min_repeat_ms is None:
            min_repeat_ms = 0 if target.keys[0] == "cpu" else 1000
            logger.info("Default --min-repeat-ms for this target is %s", min_repeat_ms)

        if rpc_key:
            if hostname is None or port is None:
                raise TVMCException(
                    "You must provide a hostname and port to connect to a remote RPC device."
                )
            if isinstance(port, str):
                port = int(port)

            logger.info("Tuning will be performed on device %s at %s:%d.", rpc_key, hostname, port)

            if enable_autoscheduler:
                runner_ctor = auto_scheduler.RPCRunner
            elif enable_metascheduler:
                runner_ctor = ms.runner.RPCRunner
            else:
                runner_ctor = autotvm.RPCRunner

            if enable_metascheduler:
                rpc_config = ms.runner.RPCConfig(
                    tracker_host=rpc.tracker_host,
                    tracker_port=rpc.tracker_port,
                    tracker_key=rpc.tracker_key,
                    session_priority=1,
                    session_timeout_sec=100,
                )
                evaluator_config = ms.runner.EvaluatorConfig(
                    number=number,
                    repeat=repeat,
                    min_repeat_ms=min_repeat_ms,
                    # enable_cpu_cache_flush=False,
                )
                runner = runner_ctor(rpc_config, evaluator_config)
            else:
                runner = runner_ctor(
                    key=rpc_key,
                    host=hostname,
                    port=port,
                    number=number,
                    repeat=repeat,
                    n_parallel=parallel,
                    timeout=timeout,
                    min_repeat_ms=min_repeat_ms,
                    module_loader=module_loader,
                )

        else:
            logger.info("Starting localhost tuning.")
            if enable_autoscheduler:
                runner_ctor = auto_scheduler.LocalRPCMeasureContext
            elif enable_metascheduler:
                runner_ctor = ms.runner.LocalRunner
            else:
                runner_ctor = autotvm.LocalRunner

            if enable_metascheduler:
                evaluator_config = ms.runner.EvaluatorConfig(
                    number=number,
                    repeat=repeat,
                    min_repeat_ms=min_repeat_ms,
                    # enable_cpu_cache_flush=False,
                )
                local_server = runner_ctor(timeout_sec=timeout, evaluator_config=evaluator_config
                )
            else:
                local_server = runner_ctor(
                    number=number,
                    repeat=repeat,
                    timeout=timeout,
                    min_repeat_ms=min_repeat_ms,
                )

            # For autoscheduling on some devices, we need to maintain a
            # LocalRPCMeasureContext object.
            if enable_autoscheduler:
                runner = local_server.runner
            elif enable_metascheduler:
                runner = local_server
            else:
                runner = local_server

        if enable_autoscheduler:
            tasks, weights = autoscheduler_get_tuning_tasks(
                mod=mod,
                params=params,
                target=target,
                alter_layout=desired_layout,
                hardware_params=hardware_params,
                include_simple_tasks=include_simple_tasks,
                extra_config=build_option,
            )

            # Create the autoscheduler tuning options
            # if build_option is None:
            #     build_option = {}
            builder = auto_scheduler.LocalBuilder(
                # timeout = 1000
                # n_parallel=1,
                # build_kwargs=build_kwargs or {},
                # do_fork=True,
                # do_fork=False,
                build_func=build_func,
                runtime=runtime,
            )
        elif enable_metascheduler:
            tasks = metascheduler_get_tuning_tasks(
                mod=mod,
                params=params,
                target=target,
                alter_layout=desired_layout,
            )
            if prior_records:
                prior_workloads_path = f"{prior_records}_workload.json"
                prior_records_path = f"{prior_records}_tuning_record.json"
                database = ms.database.JSONDatabase(path_workload=prior_workloads_path, path_tuning_record=prior_records_path)
            else:
                database = "json"
            tasks = metascheduler_get_tuning_tasks(
                mod=mod,
                params=params,
                target=target,
                alter_layout=desired_layout,
            )
            builder = ms.builder.LocalBuilder(
                max_workers = None,
                timeout_sec = timeout,
                f_build = None,
                f_export = None,
                initializer = None
            )
        else:
            tasks = autotvm_get_tuning_tasks(
                mod=mod,
                params=params,
                target=target,
                alter_layout=desired_layout,
            )

        if enable_autoscheduler:
            # Create the autoscheduler tuning options
            tuning_options = auto_scheduler.TuningOptions(
                num_measure_trials=trials,
                measure_callbacks=[auto_scheduler.RecordToFile(tuning_records)],
                runner=runner,
                builder=builder,
                early_stopping=early_stopping,
                verbose=autoscheduler_verbose,
                num_measures_per_round=autoscheduler_num_measures_per_round,
                si_prefix=si_prefix,
            )

            logger.info("Autoscheduling with configuration: %s", tuning_options)

            # Schedule the tasks (i.e., produce a schedule for each task)
            schedule_tasks(
                tasks,
                weights,
                tuning_options,
                prior_records,
                log_estimated_latency,
                strategy=autoscheduler_strategy,
                strategy_args=autoscheduler_strategy_args,
                policy=autoscheduler_policy,
                policy_args=autoscheduler_policy_args,
                model_type=autoscheduler_model_type,
                adaptive=autoscheduler_adaptive,
                # num_measures_per_round=?,
            )
        elif enable_metascheduler:
            tuning_options = {
                "trials": trials,
                "space": "post-order-apply",
                "strategy": "evolutionary",
                # "database": "json",  # TODO
                "database": database,  # TODO
                # "builder": "local",  # TODO
                "builder": builder,  # TODO
                # "runner": "local",  # TODO
                "runner": runner,  # TODO
            }
            logger.info("Metascheduling with configuration: %s", tuning_options)
            with tempfile.TemporaryDirectory() as work_dir:
                database_ = schedule_tasks_ms(
                    tasks,
                    work_dir,
                    **tuning_options,
                )

                workloads_path = f"{tuning_records}_workload.json"
                records_path = f"{tuning_records}_tuning_record.json"
                shutil.copyfile(database_.path_tuning_record, records_path)
                shutil.copyfile(database_.path_workload, workloads_path)
        else:

            # In autotvm, trials is specified per task. We can convert the per-model input
            # provided to per-task trials by dividing by the number of tasks.
            trials = int(trials / max(len(tasks), 1))
            logger.info("Autotuning with %d trials per task.", trials)

            builder = autotvm.LocalBuilder(
                n_parallel=max(parallel, 5),
                build_kwargs={"build_option": build_option},
                do_fork=True,
                # do_fork=False,
                build_func=build_func,
                runtime=runtime,
            )

            tuning_options = {
                "tuner": tuner,
                "trials": trials,
                "early_stopping": early_stopping,
                "measure_option": autotvm.measure_option(
                    builder=builder, runner=runner
                ),
                "tuning_records": prior_records,
                "tuner_options": autotvm_tuner_options.get(tuner, {}),
                "si_prefix": si_prefix,
            }
            logger.info("Autotuning with configuration: %s", tuning_options)

            tune_tasks(tasks, tuning_records, **tuning_options)

        return tuning_records



def autotvm_get_tuning_tasks(
    mod: tvm.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: str,
    target_host: Optional[str] = None,
    alter_layout: Optional[str] = None,
    extra_config = None,
):
    """Get the autotvm tuning tasks for a given relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module from which to extract tuning tasks.
    params : dict
        The params for the relay module.
    target : tvm.target.Target
        The compilation target.
    target_host : str, optional
        The compilation target for the host.
    alter_layout : str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.
    extra_config : TODO, optional
        TODO

    Returns
    -------
    tasks : list of autotvm.Tasks
        list of tasks to be tuned
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    if alter_layout:
        mod = convert_graph_layout(mod, alter_layout)

    config = {}
    if extra_config:
        assert isinstance(extra_config, dict)
        config.update(extra_config)
    pass_context = tvm.transform.PassContext(opt_level=3, config=config)  # TODO
    with pass_context:
        tasks = autotvm.task.extract_from_program(
            mod["main"],
            target=target,
            target_host=target_host,
            params=params,
        )

    return tasks


def autoscheduler_get_tuning_tasks(
    mod: tvm.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: str,
    target_host: Optional[str] = None,
    alter_layout: Optional[str] = None,
    hardware_params: Optional[HardwareParams] = None,
    include_simple_tasks: bool = False,
):
    """Get the autoscheduler tuning tasks for a given relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module from which to extract tuning tasks.
    params : dict
        The params for the relay module.
    target : tvm.target.Target
        The compilation target.
    target_host : str, optional
        The compilation target for the host.
    alter_layout : str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used for the search tasks

    Returns
    -------
    tasks : List[auto_scheduler.SearchTask]
        list of tasks to be tuned
    weights : List[int]
        the weight (i.e. the number of appearance) of extracted tasks
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    if alter_layout:
        mod = convert_graph_layout(mod, alter_layout)

    # Extract the tasks
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        target=target,
        hardware_params=hardware_params,
        include_simple_tasks=include_simple_tasks,
    )

    return tasks, task_weights


def metascheduler_get_tuning_tasks(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.NDArray]],
    target: Union[Target, str],
    target_host: Optional[str] = None,
    alter_layout: Optional[str] = None,
):
    """Get the autoscheduler tuning tasks for a given relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module from which to extract tuning tasks.
    params : dict
        The params for the relay module.
    target : tvm.target.Target
        The compilation target.
    target_host : str, optional
        The compilation target for the host.
    alter_layout : str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.

    Returns
    -------
    tasks : List[ms.ExtractedTask]
        list of tasks to be tuned
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    if alter_layout:
        mod = convert_graph_layout(mod, alter_layout)

    # Extract the tasks
    tasks = ms.relay_integration.extract_tasks(
        mod["main"],
        target,
        params,
        # opt_level=?,
        # executor=?,
        # module_equality=?
    )

    return tasks


def schedule_tasks(
    tasks: List[auto_scheduler.SearchTask],
    task_weights: List[float],
    tuning_options: auto_scheduler.TuningOptions,
    prior_records: Optional[str] = None,
    log_estimated_latency: bool = False,
    strategy: str = "gradient",
    strategy_args: Optional[Dict[str, Any]] = None,
    policy="sketch",  # TODO
    policy_args: Optional[Dict[str, Any]] = None,  # TODO
    model_type="xgb",  # TODO
    adaptive=False,  # TODO
    # num_measures_per_round: ? = ?,
):
    """Generate the schedules for the different tasks (i.e., subgraphs) contained in the module.
    Store the schedules in a json file that will be used later by the compiler.

    Parameters
    ----------
    tasks : list
        A list of auto_scheduler.SearchTask to tune.
    task_weights : list
        The weight (i.e. the number of appearance) of extracted tasks
    tuning_options: auto_scheduler.TuningOptions
        The options of tuning
    prior_records : str, optional
        The json file used to preload the autoscheduler
    log_estimated_latency : bool, optional
        If true, writes the estimated runtime of the model during each step of tuning to file.
    TODO
    """
    if not log_estimated_latency:
        callbacks = [auto_scheduler.task_scheduler.PrintTableInfo()]
    else:
        callbacks = [
            auto_scheduler.task_scheduler.PrintTableInfo(),
            auto_scheduler.task_scheduler.LogEstimatedLatency(("total_latency.tsv")),
        ]

    if strategy_args is None:
        strategy_args = {}

    if policy_args is None:
        policy_args = {}

    # Create the scheduler
    tuner = auto_scheduler.TaskScheduler(
        tasks,
        task_weights,
        load_log_file=prior_records,
        strategy=strategy,
        **strategy_args,
        callbacks=callbacks
    )

    # Tune the tasks
    tuner.tune(
        tuning_options,
        search_policy=f"{policy[0]}.{model_type}",
        search_policy_params=policy_args,
        adaptive_training=adaptive,
        # per_task_early_stopping=None
    )


def schedule_tasks_ms(
    tasks: List[ms.ExtractedTask],
    work_dir: str,
    trials: int,
    space: ms.SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: ms.SearchStrategy.SearchStrategyType = "evolutionary",
    database="json",  # TODO
    builder = "local",  # TODO
    runner = "local",  # TODO

):
    """TODO

    Parameters
    ----------
    tasks : list
        A list of meta_schedule.ExtractedTask to tune.
    trials : int
        The number of schedules to try out for the entire model.
    work_dir : TODO
        TODO
    space ; TODO
        TODO
    strategy : TODO
        TODO
    database : TODO
        TODO
    TODO
    """

    # space = "post-order-apply"  # TODO
    # strategy = "evolutionary"  # TODO
    callbacks = "default"  # TODO
    scheduler = "gradient"  # TODO
    cost_model = "xgb"  # TODO

    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
        tasks,
        work_dir,
        space=space,
        strategy=strategy,
    )

    database = ms.tune.tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=trials,
        # max_trials_per_task=None
        # num_trials_per_iter=64
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=callbacks,
        task_scheduler=scheduler,
        # module_equality="structural"
    )
    return database


def tune_tasks(
    tasks: List[autotvm.task.Task],
    log_file: str,
    measure_option: autotvm.measure_option,
    tuner: str,
    trials: int,
    early_stopping: Optional[int] = None,
    tuning_records: Optional[str] = None,
    tuner_options: Optional[dict] = None,
    si_prefix: str = "G",
):
    """Tune a list of tasks and output the history to a log file.

    Parameters
    ----------
    tasks : list
        A list of autotvm.Tasks to tune.
    log_file : str
        A file to output the tuning history, in JSON.
    measure_option : autotvm.measure_option
        Options to build and run a tuning task.
    tuner : str
        Which tuner to use.
    trials : int
        The maximum number of tuning trials to perform.
    early_stopping : int, optional
        The minimum number of tuning trials to perform.
        This will be equal to 'trials' if not specified.
    tuning_records: str, optional
        Path to the file produced by the tuning, to be used during
        tuning.
    tuner_options: dict, optional
    si_prefix : str
        SI prefix for FLOPS.
    """
    if not tasks:
        logger.warning("there were no tasks found to be tuned")
        return

    if not early_stopping:
        early_stopping = trials

    if tuner == "xgb":
        tuner_cls = XGBTuner
    elif tuner == "ga":
        tuner_cls = GATuner
    elif tuner == "random":
        tuner_cls = RandomTuner
    elif tuner == "gridsearch":
        tuner_cls = GridSearchTuner
    else:
        raise TVMCException("invalid tuner: %s " % tuner)

    if tuner_options is None:
        tuner_options = {}

    for i, tsk in enumerate(tasks):
        prefix = "\n[Task %2d/%2d] " % (i + 1, len(tasks))

        # Create a tuner
        tuner_obj = tuner_cls(tsk, **tuner_options)

        # If transfer learning is being used, load the existing results
        if tuning_records and os.path.exists(tuning_records):
            logger.info("loading tuning records from %s", tuning_records)
            start_time = time.time()
            tuner_obj.load_history(autotvm.record.load_from_file(tuning_records))
            logging.info("loaded history in %.2f sec(s)", time.time() - start_time)

        tuner_obj.tune(
            n_trial=min(trials, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(trials, prefix=prefix, si_prefix=si_prefix),
                autotvm.callback.log_to_file(log_file),
            ],
            si_prefix=si_prefix,
        )
