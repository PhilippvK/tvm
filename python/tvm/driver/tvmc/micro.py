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
Provides support for micro targets (microTVM).
"""
import logging
import argparse
import os
from pathlib import Path
import shutil
import sys

from urllib.parse import urlparse

from tvm import autotvm, auto_scheduler
from tvm.relay.backend import Runtime
from . import TVMCException, frontends
from .shape_parser import parse_shape_string
from .autotuner import tune_model
from .main import register_parser
from .arguments import TVMCSuppressedArgumentParser
from .target import _generate_target_kind_args, reconstruct_target_args
from .project import (
    get_project_options,
    get_and_check_options,
    get_project_dir,
)

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")

try:
    import tvm.micro.project as project
    from tvm.micro import get_microtvm_template_projects, AutoTvmModuleLoader, autotvm_build_func
    from tvm.micro.build import MicroTVMTemplateProjectNotFoundError
    from tvm.micro.project_api.server import ServerError
    from tvm.micro.project_api.client import ProjectAPIServerNotFoundError

    SUPPORT_MICRO = True
except (ImportError, NameError):
    SUPPORT_MICRO = False

def add_micro_tune_args(parser):
    """Include parser for 'tune' subcommand"""

    parser.add_argument(
        "--early-stopping",
        type=int,
        help="minimum number of trials before early stopping",
    )
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
    # parser.add_argument(
    #     "--parallel",
    #     default=4,
    #     type=int,
    #     help="the maximum number of parallel devices to use when tuning",
    # )
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

    # generate_target_args(parser)
    _generate_target_kind_args(parser, "c")
    # for codegen_name in get_codegen_names():
    #     _generate_codegen_args(parser, codegen_name)
    # TODO: --target?
    # parser.add_argument(
    #     "--target-host",
    #     help="the host compilation target, defaults to 'llvm'",
    #     default="llvm",
    # )

    parser.add_argument("--timeout", type=int, default=1000, help="compilation timeout, in seconds")
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
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument(
        "--enable-autoscheduler",
        help="enable tuning the graph through the AutoScheduler tuner",
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
    autotvm_group = parser.add_argument_group(
        "AutoTVM options",
        "AutoTVM options, used when the AutoScheduler is not enabled",
    )
    autotvm_group.add_argument(
        "--tuner",
        choices=["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"],
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

@register_parser
def add_micro_parser(subparsers, main_parser, json_params):
    """Includes parser for 'micro' context and associated subcommands:
    create-project (create), build, and flash.
    """

    if SUPPORT_MICRO is False:
        # Don't create 'tvmc micro' parser.
        return

    # Probe available default platform templates.
    templates = {}
    for p in ("zephyr", "arduino"):
        try:
            templates[p] = get_microtvm_template_projects(p)
        except MicroTVMTemplateProjectNotFoundError:
            pass

    micro = subparsers.add_parser("micro", help="select micro context.")
    micro.set_defaults(func=drive_micro)

    micro_parser = micro.add_subparsers(title="subcommands")
    # Selecting a subcommand under 'micro' is mandatory
    micro_parser.required = True
    micro_parser.dest = "subcommand"

    # 'create_project' subcommand
    create_project_parser = micro_parser.add_parser(
        "create-project",
        aliases=["create"],
        help="create a project template of a given type or given a template dir.",
    )
    create_project_parser.set_defaults(subcommand_handler=create_project_handler)
    create_project_parser.add_argument(
        "project_dir",
        help="project dir where the new project based on the template dir will be created.",
    )
    create_project_parser.add_argument("MLF", help="Model Library Format (MLF) .tar archive.")
    create_project_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force project creating even if the specified project directory already exists.",
    )

    # 'build' subcommand
    build_parser = micro_parser.add_parser(
        "build",
        help="build a project dir, generally creating an image to be flashed, e.g. zephyr.elf.",
    )
    build_parser.set_defaults(subcommand_handler=build_handler)
    build_parser.add_argument("project_dir", help="project dir to build.")
    build_parser.add_argument("-f", "--force", action="store_true", help="Force rebuild.")

    # 'flash' subcommand
    flash_parser = micro_parser.add_parser(
        "flash", help="flash the built image on a given micro target."
    )
    flash_parser.set_defaults(subcommand_handler=flash_handler)
    flash_parser.add_argument("project_dir", help="project dir where the built image is.")


    # 'tune' subcommand
    tune_parser = micro_parser.add_parser(
        "tune",
        help="Tune a model using a MicroTVM device.",
    )
    tune_parser.set_defaults(subcommand_handler=tune_handler)
    add_micro_tune_args(tune_parser)

    # For each platform add arguments detected automatically using Project API info query.

    # Create subparsers for the platforms under 'create-project', 'build', and 'flash' subcommands.
    help_msg = (
        "you must select a platform from the list. You can pass '-h' for a selected "
        "platform to list its options."
    )
    create_project_platforms_parser = create_project_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )
    build_platforms_parser = build_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )
    flash_platforms_parser = flash_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )
    tune_platforms_parser = tune_parser.add_subparsers(
        title="platforms", help=help_msg, dest="platform"
    )

    subcmds = {
        # API method name    Parser associated to method      Handler func to call after parsing
        "generate_project": [create_project_platforms_parser, create_project_handler],
        "build": [build_platforms_parser, build_handler],
        "flash": [flash_platforms_parser, flash_handler],
        "tune": [tune_platforms_parser, tune_handler],
    }

    # Helper to add a platform parser to a subcmd parser.
    def _add_parser(parser, platform):
        platform_name = platform[0].upper() + platform[1:] + " platform"
        platform_parser = parser.add_parser(
            platform, add_help=False, help=f"select {platform_name}."
        )
        platform_parser.set_defaults(platform=platform)
        return platform_parser

    parser_by_subcmd = {}
    for subcmd, subcmd_parser_handler in subcmds.items():
        subcmd_parser = subcmd_parser_handler[0]
        subcmd_parser.required = True  # Selecting a platform or template is mandatory
        parser_by_platform = {}
        for platform in templates:
            new_parser = _add_parser(subcmd_parser, platform)
            parser_by_platform[platform] = new_parser

        # Besides adding the parsers for each default platform (like Zephyr and Arduino), add a
        # parser for 'template' to deal with adhoc projects/platforms.
        new_parser = subcmd_parser.add_parser(
            "template", add_help=False, help="select an adhoc template."
        )
        new_parser.add_argument(
            "--template-dir", required=True, help="Project API template directory."
        )
        new_parser.set_defaults(platform="template")
        parser_by_platform["template"] = new_parser

        parser_by_subcmd[subcmd] = parser_by_platform

    disposable_parser = TVMCSuppressedArgumentParser(main_parser)
    try:
        known_args, _ = disposable_parser.parse_known_args()
    except TVMCException:
        return

    try:
        subcmd = known_args.subcommand
        platform = known_args.platform
    except AttributeError:
        # No subcommand or platform, hence no need to augment the parser for micro targets.
        return

    # Augment parser with project options.

    print("known_args", known_args)
    if platform == "template":
        # adhoc template
        template_dir = str(Path(known_args.template_dir).resolve())
    else:
        # default template
        template_dir = templates[platform]

    try:
        template = project.TemplateProject.from_directory(template_dir)
    except ProjectAPIServerNotFoundError:
        sys.exit(f"Error: Project API server not found in {template_dir}!")

    template_info = template.info()
    # print("template_info", template_info)

    options_by_method = get_project_options(template_info)
    # print("options_by_method", options_by_method)

    # TODO(gromero): refactor to remove this map.
    subcmd_to_method = {
        "create-project": "generate_project",
        "create": "generate_project",
        "build": "build",
        "flash": "flash",
        "tune": "tune",
        # "tune": ["generate_project"],
    }

    method = subcmd_to_method[subcmd]
    # print("method", method)
    parser_by_subcmd_n_platform = parser_by_subcmd[method][platform]
    # print("parser_by_subcmd_n_platform", parser_by_subcmd_n_platform)
    _, handler = subcmds[method]
    # print("handler", handler)

    # TODO: get rid of this workaround
    options_by_method["tune"] = sum([options_by_method[method] for method in ["generate_project", "build", "flash", "open_transport"]], [])

    parser_by_subcmd_n_platform.formatter_class = (
        # Set raw help text so help_text format works
        argparse.RawTextHelpFormatter
    )
    parser_by_subcmd_n_platform.set_defaults(
        subcommand_handler=handler,
        valid_options=options_by_method[method],
        template_dir=template_dir,
    )

    required = any([opt["required"] for opt in options_by_method[method]])
    # print("required", required)
    nargs = "+" if required else "*"

    help_text_by_option = [opt["help_text"] for opt in options_by_method[method]]
    help_text = "\n\n".join(help_text_by_option) + "\n\n"

    parser_by_subcmd_n_platform.add_argument(
        "--project-option", required=required, metavar="OPTION=VALUE", nargs=nargs, help=help_text
    )

    parser_by_subcmd_n_platform.add_argument(
        "-h",
        "--help",
        "--list-options",
        action="help",
        help="show this help message which includes platform-specific options and exit.",
    )

    for one_entry in json_params:
        micro.set_defaults(**one_entry)


def drive_micro(args):
    # Call proper handler based on subcommand parsed.
    args.subcommand_handler(args)


def create_project_handler(args):
    """Creates a new project dir."""
    project_dir = get_project_dir(args.project_dir)

    if os.path.exists(project_dir):
        if args.force:
            shutil.rmtree(project_dir)
        else:
            raise TVMCException(
                "The specified project dir already exists. "
                "To force overwriting it use '-f' or '--force'."
            )

    template_dir = str(Path(args.template_dir).resolve())
    if not os.path.exists(template_dir):
        raise TVMCException(f"Template directory {template_dir} does not exist!")

    mlf_path = str(Path(args.MLF).resolve())
    if not os.path.exists(mlf_path):
        raise TVMCException(f"MLF file {mlf_path} does not exist!")

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        project.generate_project_from_mlf(template_dir, project_dir, mlf_path, options)
    except ServerError as error:
        print("The following error occurred on the Project API server side: \n", error)
        sys.exit(1)


def build_handler(args):
    """Builds a firmware image given a project dir."""
    project_dir = get_project_dir(args.project_dir)

    if not os.path.exists(project_dir):
        raise TVMCException(f"{project_dir} doesn't exist.")

    if os.path.exists(project_dir + "/build"):
        if args.force:
            shutil.rmtree(project_dir + "/build")
        else:
            raise TVMCException(
                f"There is already a build in {project_dir}. "
                "To force rebuild it use '-f' or '--force'."
            )

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        prj = project.GeneratedProject.from_directory(project_dir, options=options)
        prj.build()
    except ServerError as error:
        print("The following error occurred on the Project API server side: ", error)
        sys.exit(1)


def flash_handler(args):
    """Flashes a firmware image to a target device given a project dir."""

    project_dir = get_project_dir(args.project_dir)

    if not os.path.exists(project_dir + "/build"):
        raise TVMCException(f"Could not find a build in {project_dir}")

    options = get_and_check_options(args.project_option, args.valid_options)

    try:
        prj = project.GeneratedProject.from_directory(project_dir, options=options)
        prj.flash()
    except ServerError as error:
        print("The following error occurred on the Project API server side: ", error)
        sys.exit(1)


def tune_handler(args):
    """Tunes a model using the chosen target device.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """
    tvmc_model = frontends.load_model(args.FILE, args.model_format, shape_dict=args.input_shapes)

    # Specify hardware parameters, although they'll only be used if autoscheduling.
    hardware_params = auto_scheduler.HardwareParams(
        num_cores=1,
        # vector_unit_bytes=0,  # VLEN on riscv?
        # cache_line_bytes=0,
        max_shared_memory_per_block=0,
        max_local_memory_per_block=0,
        max_threads_per_block=0,
        max_vthread_extent=0,
        warp_size=0,
        # num_cores=None,
        # vector_unit_bytes=None,
        # cache_line_bytes=None,
        # max_shared_memory_per_block=None,
        # max_local_memory_per_block=None,
        # max_threads_per_block=None,
        # max_vthread_extent=None,
        # warp_size=None,
        target="c",
        # target_host="?",
        target_host=None,
        # target_host="llvm",
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

    options = get_and_check_options(args.project_option, args.valid_options)

    runtime = Runtime("crt", {"system-lib": True})

    module_loader = AutoTvmModuleLoader(
        template_project_dir=args.template_dir,
        project_options=options,
    )

    # print("options", options)
    # print("args", args)
    # drive_tune(args, micro=True, module_loader=module_loader)

    tune_model(
        tvmc_model,
        "c",
        tuning_records=args.output,
        prior_records=args.tuning_records,
        enable_autoscheduler=args.enable_autoscheduler,
        rpc_key=args.rpc_key,
        hostname=rpc_hostname,
        port=rpc_port,
        trials=args.trials,
        # target_host=args.target_host,
        # target_host="?",
        target_host=None,
        # target_host="llvm",
        tuner=args.tuner,
        min_repeat_ms=args.min_repeat_ms,
        early_stopping=args.early_stopping,
        desired_layout=args.desired_layout,
        timeout=args.timeout,
        repeat=args.repeat,
        number=args.number,
        # parallel=args.parallel,
        parallel=5,
        hardware_params=hardware_params,
        include_simple_tasks=False,
        log_estimated_latency=False,
        additional_target_options=reconstruct_target_args(args),
        module_loader=module_loader,
        runtime=runtime,
        build_func=autotvm_build_func,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        si_prefix="M",  # Display MFLOPS instead of GFLOPS
    )
