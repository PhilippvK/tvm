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
# pylint: disable=unused-argument
"""
Provides support to auto-tuning networks using AutoTVM.
"""
import os.path
import logging
import time
import tempfile
import shutil
from copy import deepcopy
from pathlib import Path
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
from tvm.relay.backend import Executor, Runtime
from tvm.tir.tensor_intrin.riscv_cpu import *

from . import TVMCException, composite_target, frontends
from .main import register_parser
from tvm.driver.tvmc.registry import generate_registry_args, reconstruct_registry_entity
from .model import TVMCModel
from .target import target_from_cli, generate_target_args, reconstruct_target_args
from .shape_parser import parse_shape_string
from .transform import generate_transform_args, parse_graph_transform_args, apply_graph_transforms
from .compiler import OptionallyDisableLegalize


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_tune_parser(subparsers, _, json_params):
    """Include parser for 'tune' subcommand"""

    parser = subparsers.add_parser("tune", help="auto-tune a model")
    parser.set_defaults(func=drive_tune)
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
        "--disabled-pass",
        help="disable specific passes, comma-separated list of pass names.",
        type=parse_pass_list_str,
        default="",
    )
    parser.add_argument(
        "--number",
        default=10,
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
        "-O",
        "--opt-level",
        default=3,
        type=int,
        choices=range(0, 4),
        metavar="[0-3]",
        help="specify which optimization level to use. Defaults to '3'.",
    )
    generate_registry_args(parser, Executor, "graph")
    generate_registry_args(parser, Runtime, "cpp")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity.")
    parser.add_argument(
        "--parallel",
        default=4,
        type=int,
        help="the maximum number of parallel devices to use when tuning",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to repeat each measurement",
    )
    parser.add_argument(
        "--rpc-key",
        help="the RPC tracker key of the target device. "
        "Required when --rpc-tracker is provided.",
    )
    parser.add_argument(
        "--rpc-tracker",
        help="hostname (required) and port (optional, defaults to 9090) of the RPC tracker, "
        "e.g. '192.168.0.100:9999'",
    )

    generate_target_args(parser)
    parser.add_argument(
        "--target-host",
        help="the host compilation target.",
    )

    parser.add_argument("--timeout", type=int, default=10, help="compilation timeout, in seconds")
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="the maximum number of tuning trials to perform",
    )
    parser.add_argument(
        "--trials-single",
        type=int,
        default=None,
        help="the maximum number of tuning trials to perform per task (MS only)",
    )
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        help="path to an auto-tuning log file by AutoTVM.",
    )
    parser.add_argument(
        "--disable-legalize",
        action="store_true",
        help="TODO",
    )
    generate_transform_args(parser)
    parser.add_argument(
        "--enable-autoscheduler",
        help="enable tuning the graph through the AutoScheduler tuner",
        action="store_true",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="which tasks should be tuned, i.e. 0 0,2 3-5 all list",
    )
    parser.add_argument(
        "--enable-metascheduler",
        help="enable tuning the graph through the MetaScheduler tuner",
        action="store_true",
    )

    auto_scheduler_group = parser.add_argument_group(
        "AutoScheduler options",
        "AutoScheduler options, used when --enable-autoscheduler is provided",
    )

    auto_scheduler_group.add_argument(
        "--cache-line-bytes",
        type=int,
        help="the size of cache line in bytes. "
        "If not specified, it will be autoset for the current machine.",
    )
    auto_scheduler_group.add_argument(
        "--num-cores",
        type=int,
        help="the number of device cores. "
        "If not specified, it will be autoset for the current machine.",
    )
    auto_scheduler_group.add_argument(
        "--vector-unit-bytes",
        type=int,
        help="the width of vector units in bytes. "
        "If not specified, it will be autoset for the current machine.",
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
        help="the max number of threads per block. "
        "If not specified, it will be autoset for the current machine.",
    )
    auto_scheduler_group.add_argument(
        "--max-vthread-extent",
        type=int,
        help="the max vthread extent. "
        "If not specified, it will be autoset for the current machine.",
    )
    auto_scheduler_group.add_argument(
        "--warp-size",
        type=int,
        help="the thread numbers of a warp. "
        "If not specified, it will be autoset for the current machine.",
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
        "--autoscheduler-strategy",
        choices=["gradient", "round-robin"],
        default="gradient",
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
        "--autoscheduler-model",
        choices=["xgb", "random"],
        default="xgb",
        type=str,
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
        "--metascheduler-strategy",
        choices=["evolutionary", "replay_trace", "replay_func"],
        default="evolutionary",
        help="",
    )
    meta_scheduler_group.add_argument(
        "--metascheduler-scheduler",
        choices=["gradient", "round-robin"],
        default="gradient",
        help="",
    )
    meta_scheduler_group.add_argument(
        "--metascheduler-module-equality",
        choices=["structural", "ignore-ndarray", "anchor-block"],
        default="structural",
        help="",
    )
    autotvm_group = parser.add_argument_group(
        "AutoTVM options",
        "AutoTVM options, used when the AutoScheduler or MetaScheduler is not enabled",
    )
    autotvm_group.add_argument(
        "--tuner",
        choices=[
            "ga",
            "gridsearch",
            "random",
            "xgb",
            "xgb_knob",
            "xgb_itervar",
            "xgb_curve",
            "xgb_rank",
            "xgb_rank_knob",
            "xgb_rank_itervar",
            "xgb_rank_curve",
            "xgb_rank_binary",
            "xgb_rank_binary_knob",
            "xgb_rank_binary_itervar",
            "xgb_rank_binary_curve",
        ],
        default="xgb",
        help="type of tuner to use when tuning with autotvm.",
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

    for one_entry in json_params:
        parser.set_defaults(**one_entry)


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

    transform_args = parse_graph_transform_args(args)

    with OptionallyDisableLegalize(args.disable_legalize):
        tune_model(
            tvmc_model,
            args.target,
            opt_level=args.opt_level,
            tuning_records=args.output,
            executor=reconstruct_registry_entity(args, Executor),
            runtime=reconstruct_registry_entity(args, Runtime),
            prior_records=args.tuning_records,
            enable_autoscheduler=args.enable_autoscheduler,
            enable_metascheduler=args.enable_metascheduler,
            rpc_key=args.rpc_key,
            hostname=rpc_hostname,
            port=rpc_port,
            trials=args.trials,
            target_host=args.target_host,
            disabled_pass=args.disabled_pass,
            tuner=args.tuner,
            min_repeat_ms=args.min_repeat_ms,
            early_stopping=args.early_stopping,
            timeout=args.timeout,
            repeat=args.repeat,
            number=args.number,
            parallel=args.parallel,
            hardware_params=hardware_params,
            include_simple_tasks=args.include_simple_tasks,
            log_estimated_latency=args.log_estimated_latency,
            autoscheduler_strategy=args.autoscheduler_strategy,
            autoscheduler_policy=args.autoscheduler_policy,
            autoscheduler_model=args.autoscheduler_model,
            metascheduler_space=args.metascheduler_space,
            metascheduler_rules=args.metascheduler_rules,
            metascheduler_postprocs=args.metascheduler_postprocs,
            metascheduler_mutator_probs=args.metascheduler_mutator_probs,
            metascheduler_model=args.metascheduler_model,
            metascheduler_strategy=args.metascheduler_strategy,
            metascheduler_scheduler=args.metascheduler_scheduler,
            metascheduler_module_equality=args.metascheduler_module_equality,
            additional_target_options=reconstruct_target_args(args),
            tasks_filter=args.tasks,
            **transform_args,
        )


def filter_tasks(
    tasks: Union[List[auto_scheduler.SearchTask], List[autotvm.task.Task]],
    expr: str,
):
    """Utility to filter a list of tasks (AutoTVM or AutoScheduler) based on
    a user-supplied string expression.

    Parameters
    ----------
    tasks: list
        A list of extracted AutoTVM or AutoScheduler tasks.
    expr: str
        User-supplied expression to be used for filtering.
    """
    assert isinstance(expr, str), "Expected filter expression of string type"
    assert len(expr) > 0, "Got empty filter expression"

    # groups of keywords are comma-separated
    splitted = expr.split(",")

    do_list = False
    do_filter = False
    selected = []
    for item in splitted:
        if item in ["list", "help"]:
            do_list = True
        elif item in ["all"]:
            selected = list(range(len(tasks)))
        else:
            do_filter = True
            if "-" in item:
                assert item.count("-") == 1, "Malformed range expression"
                assert len(item) > 1, "Missing lhs or rhs for range expression"
                lhs, rhs = item.split("-")[:2]
                lhs = int(lhs) if lhs else 0
                rhs = int(rhs) if rhs else len(tasks) - 1
                assert 0 <= lhs < len(tasks), "Left-hand side expression out of range"
                assert 0 <= rhs < len(tasks), "Right-hand side expression out of range"
                selected.extend(list(range(lhs, rhs + 1)))
            else:
                assert isinstance(item, str)
                idx = int(item)
                assert 0 <= idx < len(tasks), "Task index out of range"
                selected.append(idx)

    if do_filter:
        # remove duplicates
        selected = list(set(selected))
        tasks = [task for i, task in enumerate(tasks) if i in selected]

    return tasks, do_list


def gen_task_list(
    tasks: Union[List[auto_scheduler.SearchTask], List[autotvm.task.Task]],
    enable_autoscheduler: bool,
):
    """Utility for printing a list of tasks (AutoTVM or AutoScheduler)
    to the terminal.

    Parameters
    ----------
    tasks: list
        A list of extracted AutoTVM or AutoScheduler tasks.
    enable_autoscheduler: bool
        Wether the tasks are extracted with AutoScheduler or AutoTVM.
    """
    ret = "Available Tasks for tuning:\n"

    def _trunc_helper(text, length):
        return text if len(text) < length else text[: length - 3] + "..."

    ret += "\n".join(
        [
            "  {}. {}".format(
                i, _trunc_helper("Unnamed" if len(task.desc) == 0 else task.desc, 100)
            )
            if enable_autoscheduler
            else "  {}. {} (len={})".format(
                i,
                _trunc_helper(str(task), 100),
                "?" if task.config_space is None else len(task.config_space),
            )
            for i, task in enumerate(tasks)
        ]
    )
    return ret


def tune_model(
    tvmc_model: TVMCModel,
    target: str,
    opt_level: int = 3,
    executor: Optional[Executor] = Executor("graph"),
    runtime: Optional[Runtime] = Runtime("cpp"),
    tuning_records: Optional[str] = None,
    prior_records: Optional[str] = None,
    enable_autoscheduler: bool = False,
    enable_metascheduler: bool = False,
    rpc_key: Optional[str] = None,
    hostname: Optional[str] = None,
    port: Optional[Union[int, str]] = 9090,
    trials: int = 10000,
    trials_single: Optional[int] = None,
    target_host: Optional[str] = None,
    disabled_pass: Optional[str] = None,
    tuner: str = "xgb",
    min_repeat_ms: Optional[int] = None,
    early_stopping: Optional[int] = None,
    timeout: int = 10,
    repeat: int = 1,
    number: int = 10,
    parallel: int = 4,
    hardware_params: Optional[HardwareParams] = None,
    include_simple_tasks: bool = False,
    log_estimated_latency: bool = False,
    autoscheduler_strategy: Optional[str] = None,
    autoscheduler_policy: Optional[str] = None,
    autoscheduler_model: Optional[str] = "xgb",
    metascheduler_space: str = "post-order-apply",
    metascheduler_rules: str = "from-target",
    metascheduler_postprocs: str = "from-target",
    metascheduler_mutator_probs: str = "from-target",
    metascheduler_model: str = "xgb",
    metascheduler_strategy: str = "evolutionaly_search",
    metascheduler_scheduler: str = "gradient",
    metascheduler_module_equality: str = "structural",
    additional_target_options: Optional[Dict[str, Dict[str, Any]]] = None,
    tasks_filter: str = "all",
    desired_layout: Optional[str] = None,
    desired_layout_ops: Optional[List[str]] = None,
    mixed_precision: bool = False,
    mixed_precision_ops: Optional[List[str]] = None,
    mixed_precision_calculation_type: Optional[str] = None,
    mixed_precision_acc_type: Optional[str] = None,
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
    disabled_pass: str, optional
        Comma-separated list of passes which needs to be disabled
        during compilation
    trials : int, optional
        The number of schedules to try out for the entire model. Note that the default
        value is chosen as a decent average for most models, but larger models may need
        more trials to reach a good result while smaller models will converge with fewer
        trials.
    trials_single : int, optional
        The number of schedules to try out for the each task.
    tuner : str, optional
        The type of tuner to use when tuning with autotvm. Can be one of
        "ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb_itervar", "xgb_curve",
        "xgb_rank", "xgb_rank_knob", "xgb_rank_itervar", "xgb_rank_binary", "xgb_rank_binary_knob",
        "xgb_rank_binary_itervar" and "xgb_rank_binary_curve".
    min_repeat_ms : int, optional
        Minimum time to run each trial. Defaults to 0 on x86 and 1000 on other targets.
    early_stopping : int, optional
        When specified, stop tuning after this number of trials if results aren't improving.
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
    autoscheduler_strategy: str, optional
        TODO
    autoscheduler_policy: str, optional
        TODO
    autoscheduler_model: str, optional
        TODO
    metascheduler_space: str
        TODO
    metascheduler_rules: str
        TODO
    metascheduler_postprocs: str
        TODO
    metascheduler_mutator_probs: str
        TODO
    metascheduler_model: str
        TODO
    metascheduler_strategy: str
        TODO
    metascheduler_scheduler: str
        TODO
    metascheduler_module_equality: str
        TODO
    additional_target_options: Optional[Dict[str, Dict[str, Any]]]
        Additional target options in a dictionary to combine with initial Target arguments
    tasks_filter : str, optional
        Filter which tasks should be tuned or output a list of the extracted tasks.
        Examples: 0 0,2 3-5 all list
    desired_layout: str, optional
        Can be one of "NCHW" or "NHWC". When specified, compatible operations in the graph
        will have their layout set to this format. Tasks will then be tuned using this
        specified layout.
    desired_layout_ops: list[str], optional
        The list of operators to be transformed with desired layout.
    mixed_precision: bool
        To enable mixed precision transformation.
    mixed_precision_ops: list[str], optional
        The list of operators to be converted to mixed precision.
    mixed_precision_calculation_type: str
        The calculation dtype to be used while mixed precision.
    mixed_precision_acc_type: str
        The accumulation data type to be used while mixed precision.

    Returns
    -------
    tuning_records : str
        The path to the produced tuning log file.
    """
    transform_args = parse_graph_transform_args(locals())
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

    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass):
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
                    tracker_host=hostname,
                    tracker_port=port,
                    tracker_key=rpc_key,
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
                transform_args=transform_args,
                hardware_params=hardware_params,
                include_simple_tasks=include_simple_tasks,
            )
        elif enable_metascheduler:
            tasks = metascheduler_get_tuning_tasks(
                mod=mod,
                params=params,
                target=target,
                transform_args=transform_args,
                executor=executor,
                runtime=runtime,
                opt_level=opt_level,
                module_equality=module_equality,
            )
            if prior_records:
                prior_workloads_path = f"{prior_records}_workload.json"
                prior_records_path = f"{prior_records}_tuning_record.json"
                database = ms.database.JSONDatabase(path_workload=prior_workloads_path, path_tuning_record=prior_records_path)
            else:
                database = "json"
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
                transform_args=transform_args,
            )

        # Filter extracted tasks by provided user expression
        if tasks_filter:
            # if enable_metascheduler:
            #     raise NotImplementedError
            tasks, do_list = filter_tasks(tasks, tasks_filter)
            if do_list:
                print(gen_task_list(tasks, enable_autoscheduler))
                return None
        if len(tasks) == 0:
            logger.info("No tasks have been selected for tuning.")
            return None
        else:
            logger.info("Selected %s tasks for tuning.", len(tasks))

        if enable_autoscheduler:
            assert trials_single is None, "--trials-single is only supported by MS"
            # Create the autoscheduler tuning options
            tuning_options = auto_scheduler.TuningOptions(
                num_measure_trials=trials,
                measure_callbacks=[auto_scheduler.RecordToFile(tuning_records)],
                runner=runner,
                early_stopping=early_stopping,
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
                policy=autoscheduler_policy,
                model_type=autoscheduler_model,
            )
        elif enable_metascheduler:
            tuning_options = {
                "trials": trials,
                "max_trials_per_task": trials_single,
                "num_trials_per_iter": max(parallel, 5),
                "space": metascheduler_space,
                "strategy": metascheduler_strategy,
                "builder": builder,
                "runner": runner,
                "cost_model": metascheduler_model,
                "scheduler": metascheduler_scheduler,
                "rules": metascheduler_rules,
                "postprocs": metascheduler_postprocs,
                "probs": metascheduler_mutator_probs,
                "module_equality": metascheduler_module_equality,
            }
            logger.info("Metascheduling with configuration: %s", tuning_options)
            with tempfile.TemporaryDirectory() as work_dir:
                if prior_records:
                    # Copy prior records to work_dir
                    prior_workloads_path = f"{prior_records}_workload.json"
                    prior_records_path = f"{prior_records}_tuning_record.json"
                    database_workload_path = str(Path(work_dir) / "database_workload.json")
                    database_records_path = str(Path(work_dir) / "database_tuning_record.json")
                    shutil.copyfile(prior_workloads_path, database_workload_path)
                    shutil.copyfile(prior_records_path, database_records_path)
                    database = ms.database.JSONDatabase(
                        path_workload=database_workload_path, path_tuning_record=database_records_path
                    )
                else:
                    database = "json"
                database_ = schedule_tasks_ms(
                    tasks,
                    work_dir,
                    database=database,
                    **tuning_options,
                )
                # Copy new records to desired destination
                workloads_path = f"{tuning_records}_workload.json"
                records_path = f"{tuning_records}_tuning_record.json"
                logs_path = Path(records_path).parent / "logs"
                shutil.copyfile(database_.path_tuning_record, records_path)
                shutil.copyfile(database_.path_workload, workloads_path)
                shutil.copytree(Path(work_dir) / "logs", logs_path)
        else:
            assert trials_single is None, "--trials-single is only supported by MS"
            # In autotvm, trials is specified per task. We can convert the per-model input
            # provided to per-task trials by dividing by the number of tasks.
            trials = int(max(1, trials / max(len(tasks), 1)))
            logger.info("Autotuning with %d trials per task.", trials)

            tuning_options = {
                "tuner": tuner,
                "trials": trials,
                "early_stopping": early_stopping,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
                ),
                "tuning_records": prior_records,
            }
            logger.info("Autotuning with configuration: %s", tuning_options)

            tune_tasks(tasks, tuning_records, **tuning_options)

        return tuning_records


def autotvm_get_tuning_tasks(
    mod: tvm.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: str,
    target_host: Optional[str] = None,
    transform_args: Optional[Dict[str, Any]] = None,
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
    transform_args: dict, optional
        Graph transformation arguments that are applied to the relay module.

    Returns
    -------
    tasks : list of autotvm.Tasks
        list of tasks to be tuned
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    mod = apply_graph_transforms(mod, transform_args)

    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        params=params,
    )

    return tasks


def autoscheduler_get_tuning_tasks(
    mod: tvm.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    target: str,
    target_host: Optional[str] = None,
    transform_args: Optional[Dict[str, Any]] = None,
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
    transform_args: dict, optional
        Graph transformation arguments that are applied to the relay module.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used for the search tasks

    Returns
    -------
    tasks : list of autotvm.Tasks
        list of tasks to be tuned
    weights : List[int]
        the weight (i.e. the number of appearance) of extracted tasks
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    mod = apply_graph_transforms(mod, transform_args)

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
    transform_args: Optional[Dict[str, Any]] = None,
    executor: Optional[Executor] = Executor("graph"),
    runtime: Optional[Runtime] = Runtime("cpp"),
    module_equality: str = "structural",
    opt_level: int = 3,
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
    transform_args: dict, optional
        Graph transformation arguments that are applied to the relay module.

    Returns
    -------
    tasks : List[ms.ExtractedTask]
        list of tasks to be tuned
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    mod = apply_graph_transforms(mod, transform_args)

    # Extract the tasks
    tasks = ms.relay_integration.extract_tasks(
        mod["main"],
        target,
        params,
        opt_level=opt_level,
        executor=executor,
        runtime=runtime,
        module_equality=module_equality,
    )

    return tasks


def schedule_tasks(
    tasks: List[auto_scheduler.SearchTask],
    task_weights: List[float],
    tuning_options: auto_scheduler.TuningOptions,
    prior_records: Optional[str] = None,
    log_estimated_latency: bool = False,
    strategy: str = "gradient",
    policy: str = "sketch",
    model_type: str = "xgb",
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
    strategy : TODO
        TODO
    policy : TODO
        TODO
    """
    if not log_estimated_latency:
        callbacks = [auto_scheduler.task_scheduler.PrintTableInfo()]
    else:
        callbacks = [
            auto_scheduler.task_scheduler.PrintTableInfo(),
            auto_scheduler.task_scheduler.LogEstimatedLatency(("total_latency.tsv")),
        ]

    # Create the scheduler
    tuner = auto_scheduler.TaskScheduler(
        tasks, task_weights, load_log_file=prior_records, callbacks=callbacks, strategy=strategy
    )

    # Tune the tasks
    search_policy = search_policy if policy == "default" else f"{policy}.{model_type}"
    tuner.tune(
        tuning_options,
        search_policy=search_policy,
    )


def schedule_tasks_ms(
    tasks: List[ms.ExtractedTask],
    work_dir: str,
    trials: int,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    space: ms.SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: ms.SearchStrategy.SearchStrategyType = "evolutionary",
    database="json",  # TODO
    builder="local",  # TODO
    runner="local",  # TODO
    cost_model="xgb",
    scheduler="gradient",
    rules="from-target",
    postprocs="from-target",
    probs="from-target",
    module_equality="structural",
    executor: Optional[Executor] = Executor("graph"),
    runtime: Optional[Runtime] = Runtime("cpp"),
):
    """TODO

    Parameters
    ----------
    tasks : list
        A list of meta_schedule.ExtractedTask to tune.
    trials : int
        The number of schedules to try out for the entire model.
    max_trials_per_task : int
        The number of schedules to try for each task.
    num_trials_per_iter : int
        How many trials to submit in parallel.
    work_dir : TODO
        TODO
    space ; TODO
        TODO
    strategy : TODO
        TODO
    database : TODO
        TODO
    cost_model : TODO
    rules : TODO
    postprocs : TODO
    probs : TODO
    TODO
    """

    # space = "post-order-apply"  # TODO
    # strategy = "evolutionary"  # TODO
    callbacks = "default"  # TODO
    scheduler = "gradient"  # TODO

    # rules = "from-target"
    # postprocs = "from-target"
    # probs = "from_target"

    space = ms.space_generator.SpaceGenerator.create(space, sch_rules=rules, postprocs=postprocs, mutator_probs=probs)

    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
        tasks,
        work_dir,
        space=space,
        strategy=strategy,
        num_tuning_cores="logical",  # TODO: expose
    )

    database = ms.tune.tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=trials,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=callbacks,
        task_scheduler=scheduler,
        module_equality=module_equality,
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
    """
    if not tasks:
        logger.warning("there were no tasks found to be tuned")
        return

    if not early_stopping:
        early_stopping = trials

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # Create a tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise TVMCException("invalid tuner: %s " % tuner)

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
                autotvm.callback.progress_bar(min(trials, len(tsk.config_space)), prefix=prefix),
                autotvm.callback.log_to_file(log_file),
            ],
        )
