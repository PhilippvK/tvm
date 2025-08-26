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
"""RPC Runner Micro"""

from contextlib import contextmanager
from typing import Callable, List, Optional, Union
from collections import namedtuple
import signal
import random
from pathlib import Path

from tvm import micro
from tvm import nd
from tvm.contrib.popen_pool import PopenPoolExecutor
from tvm.rpc.server import Server
from tvm.rpc.tracker import Tracker
from tvm.meta_schedule.logging import get_logger
from tvm.meta_schedule.utils import cpu_count, derived_object
from tvm.meta_schedule.runner.config import EvaluatorConfig, RPCConfig
from tvm.meta_schedule.runner import PyRunner, RunnerFuture, RunnerInput
from tvm.meta_schedule.runner.rpc_runner import RPCRunnerFuture
from tvm.meta_schedule.runner.utils import T_ARG_INFO_JSON_OBJ_LIST

logger = get_logger(__name__)  # pylint: disable=invalid-name


@derived_object
class RPCRunnerMicroMem(PyRunner):
    """RPC based runner for tuning micro models."""

    def __init__(
        self,
        platform: str = "crt",
        project_options: Optional[dict] = None,
        rpc_configs: Optional[List[RPCConfig]] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
        max_workers: Optional[int] = None,
        initializer: Optional[Callable[[], None]] = None,
        session_timeout_sec: int = 300,
    ) -> None:
        """Constructor

        Parameters
        ----------
        platform: str
            The platform used for project generation.
        project_options: dict
            The options for the generated micro project.
        rpc_config: RPCConfig
            The rpc configuration.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        max_workers: Optional[int] = None
            The maximum number of connections. Defaults to number of logical CPU cores.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        session_timeout_sec: int
            The session timeout, including the pending time. if the number of candidates sent to runner is larger
            than the runner workers, increase the timeout.
        """
        super().__init__()
        self.platform = platform
        if project_options is None:
            project_options = {}
        self.project_options = project_options
        self.rpc_configs = rpc_configs
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.session_timeout_sec = session_timeout_sec

        if max_workers is None:
            max_workers = cpu_count(logical=True)
        logger.info("RPCRunner: max_workers = %d", max_workers)
        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=session_timeout_sec,
            initializer=initializer,
        )

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []

        for runner_input in runner_inputs:
            future = RPCRunnerFuture(
                future=self.pool.submit(
                    _worker_func_mem,
                    self.platform,
                    self.project_options or {},
                    self.rpc_configs,
                    self.evaluator_config,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.session_timeout_sec,
            )
            results.append(future)  # type: ignore
        return results


def parse_elf(path):
    """Extract static memory usage details from ELF file by mapping each segment."""
    from elftools.elf import elffile
    # TODO: check if this is generic anough for multiple platforms (riscv, arm, x86)
    # TODO: comare results with `riscv32-unknown-elf-size`
    m = {}
    m["rom_rodata"] = 0
    m["rom_code"] = 0
    m["rom_misc"] = 0
    m["ram_data"] = 0
    m["ram_zdata"] = 0

    ignoreSections = [
        "",
        ".stack",
        ".comment",
        ".riscv.attributes",
        ".strtab",
        ".stabstr",
        ".shstrtab",
        ".symtab",
        ".eh_frame",
        ".stab",
        ".heap",  # ?
        # The following are x86 only:
        ".interp",
        ".dynsym",
        ".dynstr",
        ".dynamic",
        ".got",
        ".data.rel.ro",
        # Espressif
        ".flash.appdesc",
        ".iram0.text_end",  # ?
        # QEMU
        ".htif",
        # Zephyr
        ".mcuboot_header",
        ".metadata",
        "ctors",
        "initlevel",
        "devices",
        "device_handles",
        "sw_isr_table",
        "device_states",
        ".mcuboot_header",
        ".metadata",
        "ctors",
        "initlevel",
        "devices",
        "device_handles",
        "sw_isr_table",
        "device_states",
        ".xt.prop",
        ".xt.lit",
        "k_heap_area",
        "datas",
        # Pulp
        ".data_tiny_fc",
        ".data_tiny_l1",
        ".l1cluster_g",
        ".heap_l2_shared",
        ".Pulp_Chip.Info",
        # ARM (corstone300)
        ".ddr",
        # cv32e40p
        ".debugger_stack",
        # ara
        ".l2",
        # vicuna (ram)
        ".user_align",
    ]
    ignorePrefixes = [
        ".gcc_except",
        ".sdata2",
        ".debug_",
        # ARM only:
        ".ARM",
        # The following are x86 only:
        ".note",
        ".gnu",
        ".rela",
        ".plt",
    ]
    ignoreSuffixes = [
        ".table",
        "dummy",
        "heap_start",
        "rom_start",
        ".info",
    ]

    with open(path, "rb") as f:
        e = elffile.ELFFile(f)

        for s in e.iter_sections():
            if s.name.startswith(".text") or s.name.endswith(".text") or s.name == "text":
                m["rom_code"] += s.data_size
            elif s.name.startswith(".srodata"):
                m["rom_rodata"] += s.data_size
            elif s.name.startswith(".sdata"):
                m["ram_data"] += s.data_size
            elif s.name.endswith(".rodata") or s.name == "rodata":
                m["rom_rodata"] += s.data_size
            elif s.name in [
                ".vectors",
                "iram0.vectors",
                ".iram0.vectors",
                ".init_array",
                ".fini_array",
                ".fini",
                ".init",
                ".eh_frame",
                ".eh_frame_hdr",
            ]:
                m["rom_misc"] += s.data_size
            elif s.name.endswith(".data"):
                m["ram_data"] += s.data_size
            elif (
                s.name == ".bss"
                or s.name == "bss"
                or s.name == ".sbss"
                or s.name == ".shbss"
                or s.name == ".bss.noinit"
                or s.name.endswith(".bss")
                or s.name.startswith(".bss")
                or s.name.startswith(".sbss")
                or s.name == "noinit"
            ):
                m["ram_zdata"] += s.data_size
            elif s.name in ignoreSections:
                pass
            elif any(s.name.startswith(prefix) for prefix in ignorePrefixes):
                pass
            elif any(s.name.endswith(suffix) for suffix in ignoreSuffixes):
                pass
            elif s.data_size == 0:
                pass  # No warning for empty sections
            else:
                logger.warning("ignored: %s / size: %d", s.name, s.data_size)

    return m


def extract_mem(build_result):
    import tarfile
    import tempfile
    # import subprocess
    import json
    ret = []
    fname = build_result.filename
    print("fname", fname)
    # with tempfile.TemporaryDirectory() as dest:
    dest = tempfile.TemporaryDirectory().name
    if True:
        print("dest", dest)
        with tarfile.open(fname) as f:
            f.extractall(dest)
            metadata_path = Path(dest) / "metadata.json"
            with open(metadata_path, "r") as f2:
                metadata = json.load(f2)
                # metadata_str = f2.read()
                # print("metadata_str", metadata_str)
                print("metadata", metadata)
            const_bytes = metadata["const_bytes"]
            const_kb = const_bytes / 1e3
            workspace_bytes = metadata["workspace_bytes"]
            workspace_kb = workspace_bytes / 1e3

            # lib0_path = Path(dest) / "codegen" / "host" / "lib" / "lib0.o"
            lib1_path = Path(dest) / "codegen" / "host" / "lib" / "lib1.o"
            assert lib1_path.is_file(), "lib1.o does not exist"
            parsed = parse_elf(lib1_path)
            print("parsed", parsed)
            # out = subprocess.check_output(["size", lib0_path]).decode("utf-8")
            # print("out0", out)
            # out = subprocess.check_output(["size", lib1_path]).decode("utf-8")
            # print("out1", out)
            # out = out.strip().splitlines()
            # assert len(out) == 2
            # out = out[1].strip()
            # out = out.split(" ", 1)[0]
            # print("out", out)
            # text_b = int(out)
            text_b = parsed["rom_code"]
            text_kb = text_b / 1e3
            rodata_b = parsed["rom_rodata"]
            rodata_kb = rodata_b / 1e3
            print("text_kb", text_kb)
            print("rodata_kb", rodata_kb)
            print("const_kb", const_kb)
            print("workspace_kb", workspace_kb)
            ret.append(text_kb)
            ret.append(rodata_kb)
            ret.append(const_kb)
            ret.append(workspace_kb)
    # import time
    # time.sleep(5)
    return ret


def _worker_func_mem(
    platform: str,
    project_options: dict,
    rpc_configs: List[RPCConfig],
    evaluator_config: EvaluatorConfig,
    artifact_path: str,
    device_type: str,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
) -> List[float]:
    print("_worker_func")
    if platform not in micro.build.MicroTVMTemplateProject.list():
        # lookup via path
        if not Path(platform).is_dir():
            raise ValueError(f"platform {platform} not found")
        template_project_dir = platform
    else:
        template_project_dir=micro.get_microtvm_template_projects(platform)

    module_loader = micro.AutoTvmModuleLoader(
        template_project_dir=template_project_dir,
        project_options=project_options,
    )

    rpc_config = random.choice(rpc_configs)
    remote_kw = {
        "device_key": rpc_config.tracker_key,
        "host": rpc_config.tracker_host,
        "port": rpc_config.tracker_port,
        "priority": 0,
        "timeout": 100,
    }

    build_result = namedtuple("BuildResult", ["filename"])(artifact_path)
    print("build_result", build_result)
    mem = extract_mem(build_result)

    with module_loader(remote_kw, build_result) as (remote, mod):
        dev = remote.device(device_type, 0)
        # print("mod", mod, dir(mod), mod.imported_modules)
        f_prepare = ""
        if evaluator_config.enable_cpu_cache_flush:
            f_prepare = "cache_flush_cpu_non_first_arg"
        # TODO: PROFILER
        # mod.save("/tmp/saved.tar")
        # mod.export_library("/tmp/saved.o")
        # mod.get_source()
        # print("!")
        time_f = mod.time_evaluator(
            mod.entry_name,
            dev,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc=f_prepare,
        )

        random_fill = remote.get_function("tvm.contrib.random.random_fill")
        args = [nd.empty(x[2], x[1], dev) for x in args_info]
        for arg in args:
            random_fill(arg)
        dev.sync()

        costs = time_f(*args).results
    print("costs", costs)
    # mem = [12.12]
    print("mem", mem)
    # import time
    # time.sleep("60")
    # input(">")
    # return costs
    costs_ = {"run_secs": costs, "mem": mem}
    return costs_


@contextmanager
def get_rpc_runner_micro_mem(
    platform,
    options,
    evaluator_config: EvaluatorConfig = None,
    tracker_host: Optional[str] = None,
    tracker_port: Union[None, int, str] = None,
    session_timeout_sec: int = 300,
    rpc_timeout_sec: int = 10,
    serial_numbers: List[str] = None,
):
    """Parameters
    ----------
    platform: str
        The platform used for project generation.
    options: dict
        The options for the generated micro project.
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    tracker_host: Optional[str]
        The host url of the rpc server.
    tracker_port: Union[None, int, str]
        The TCP port to bind to
    session_timeout_sec: int
        The session timeout. if the number of candidates sent to runner is larger
        than the runner workers, increase the timeout.
    rpc_timeout_sec:
        The rpc session timeout.
    serial_numbers:
        List of board serial numbers to be used during tuning.
        For "CRT" and "QEMU" platforms the serial numners are not used,
        but the length of the list determines the number of runner instances.
    """

    if evaluator_config is None:
        evaluator_config = EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )

    if tracker_host is None:
        tracker_host = "127.0.0.1"

    if tracker_port is None:
        tracker_port = 9000
    else:
        tracker_port = int(tracker_port)
    tracker_port_end = tracker_port + 1000

    if not (serial_numbers):
        serial_numbers = ["$local$device"]

    tracker = Tracker(
        port=tracker_port,
        port_end=tracker_port_end,
        silent=True,
        reuse_addr=True,
        timeout=60,
    )

    servers = []
    rpc_configs = []
    for serial_number in serial_numbers:
        key = serial_number
        rpc_config = RPCConfig(
            tracker_host=tracker_host,
            tracker_port=tracker_port,
            tracker_key=key,
            session_priority=0,
            session_timeout_sec=rpc_timeout_sec,
        )
        rpc_configs.append(rpc_config)

        server = Server(
            port=tracker_port,
            port_end=tracker_port_end,
            key=key,
            silent=True,
            tracker_addr=(tracker_host, tracker_port),
            reuse_addr=True,
            timeout=60,
        )
        servers.append(server)

    def terminate():
        tracker.terminate()
        for server in servers:
            server.terminate()

    def handle_SIGINT(signal, frame):
        terminate()
        raise KeyboardInterrupt("Received SIGINT")

    signal.signal(signal.SIGINT, handle_SIGINT)

    try:
        yield RPCRunnerMicroMem(
            platform=platform,
            project_options=options,
            rpc_configs=rpc_configs,
            evaluator_config=evaluator_config,
            session_timeout_sec=session_timeout_sec,
        )
    finally:
        terminate()
