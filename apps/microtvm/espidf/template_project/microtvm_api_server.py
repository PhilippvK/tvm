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


# TODO(@PhilippvK): get rid of unused imports

import atexit
import collections
import collections.abc
import enum
import fcntl
import json
import logging
import os
import os.path
import pathlib
import queue
import re
import select
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import usb

import serial
import serial.tools.list_ports
import yaml

from tvm.micro.project_api import server


_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR / "build"


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


BOARDS = API_SERVER_DIR / "boards.json"

# Used to check Zephyr version installed on the host.
# We only check two levels of the version.
ESPIDF_VERSION = 4.4

IDF_CMD = default = "idf.py"

IDF_PATH = os.getenv("IDF_PATH")
IDF_TOOLS_PATH = os.getenv("IDF_TOOLS_PATH")

# Data structure to hold the information microtvm_api_server.py needs
# to communicate with each of these boards.
try:
    with open(BOARDS) as boards:
        BOARD_PROPERTIES = json.load(boards)
except FileNotFoundError:
    raise FileNotFoundError(f"Board file {{{BOARDS}}} does not exist.")


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


def check_output(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_output(cmd_args, *args, **kwargs)


CACHE_ENTRY_RE = re.compile(r"(?P<name>[^:]+):(?P<type>[^=]+)=(?P<value>.*)")


CMAKE_BOOL_MAP = dict(
    [(k, True) for k in ("1", "ON", "YES", "TRUE", "Y")]
    + [(k, False) for k in ("0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "")]
)


class CMakeCache(collections.abc.Mapping):
    def __init__(self, path):
        self._path = path
        self._dict = None

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key):
        if self._dict is None:
            self._dict = self._read_cmake_cache()

        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def _read_cmake_cache(self):
        """Read a CMakeCache.txt-like file and return a dictionary of values."""
        entries = collections.OrderedDict()
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                m = CACHE_ENTRY_RE.match(line.rstrip("\n"))
                if not m:
                    continue

                if m.group("type") == "BOOL":
                    value = CMAKE_BOOL_MAP[m.group("value").upper()]
                else:
                    value = m.group("value")

                entries[m.group("name")] = value

        return entries


CMAKE_CACHE = CMakeCache(BUILD_DIR / "CMakeCache.txt")


class BoardError(Exception):
    """Raised when an attached board cannot be opened (i.e. missing /dev nodes, etc)."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the board= given to ZephyrCompiler."""


# def _get_flash_runner():
#     flash_runner = CMAKE_CACHE.get("ZEPHYR_BOARD_FLASH_RUNNER")
#     if flash_runner is not None:
#         return flash_runner
#
#     with open(CMAKE_CACHE["ZEPHYR_RUNNERS_YAML"]) as f:
#         doc = yaml.load(f, Loader=yaml.FullLoader)
#     return doc["flash-runner"]


# def _get_device_args(options):
#     flash_runner = _get_flash_runner()
#
#     if flash_runner == "nrfjprog":
#         return _get_nrf_device_args(options)
#
#     if flash_runner == "openocd":
#         return _get_openocd_device_args(options)
#
#     raise BoardError(
#         f"Don't know how to find serial terminal for board {CMAKE_CACHE['BOARD']} with flash "
#         f"runner {flash_runner}"
#     )


def generic_find_serial_port(serial_number=None):
    """Find a USB serial port based on its serial number or its VID:PID.

    This method finds a USB serial port device path based on the port's serial number (if given) or
    based on the board's idVendor and idProduct ids.

    Parameters
    ----------
    serial_number : str
        The serial number associated to the USB serial port which the board is attached to. This is
        the same number as shown by 'lsusb -v' in the iSerial field.

    Returns
    -------
    Path to the USB serial port device, for example /dev/ttyACM1.
    """
    if serial_number:
        regex = serial_number
    else:
        prop = BOARD_PROPERTIES[CMAKE_CACHE["IDF_TARGET"]]
        device_id = ":".join([prop["vid_hex"], prop["pid_hex"]])
        regex = device_id

    serial_ports = list(serial.tools.list_ports.grep(regex))

    # Workaround on MacOS
    if len(serial_ports) > 0:
        serial_ports = list(filter(lambda x: "wch" not in x.name and "SLAB" not in x.name, serial_ports))

    if len(serial_ports) == 0:
        raise Exception(f"No serial port found for board {prop['board']}!")

    if len(serial_ports) != 1:
        ports_lst = ""
        for port in serial_ports:
            ports_lst += f"Serial port: {port.device}, serial number: {port.serial_number}\n"

        raise Exception(f"Expected 1 serial port, found multiple ports:\n {ports_lst}")

    return serial_ports[0].device


PROJECT_TYPES = []
if IS_TEMPLATE:
    for d in (API_SERVER_DIR / "src").iterdir():
        if d.is_dir():
            PROJECT_TYPES.append(d.name)


PROJECT_OPTIONS = [
    server.ProjectOption(
        "extra_files_tar",
        optional=["generate_project"],
        type="str",
        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
    ),
    server.ProjectOption(
        "project_type",
        choices=tuple(PROJECT_TYPES),
        required=["generate_project"],
        type="str",
        help="Type of project to generate.",
    ),
    server.ProjectOption(
        "verbose",
        optional=["build"],
        type="bool",
        help="Run build with verbose output.",
    ),
    # server.ProjectOption(
    #     "idf_cmd",
    #     optional=["build"],
    #     default=IDF_CMD,
    #     type="str",
    #     help=(
    #         "Path to the idf.py tool. If given, supersedes both the idf_path/idf_tools_path "  # TODO(@PhilippvK)
    #         "option and IDF_PATH/IDF_TOOLS_PATH environment variable."
    #     ),
    # ),
    server.ProjectOption(
        "idf_path",
        required=(["generate_project", "open_transport"] if not IDF_PATH else None),
        optional=(["generate_project", "open_transport", "build"] if IDF_PATH else ["build"]),
        default=IDF_PATH,
        type="str",
        help="Path to the ESP-IDF clone directory.",
    ),
    server.ProjectOption(
        "idf_tools_path",
        required=(["generate_project", "open_transport"] if not IDF_TOOLS_PATH else None),
        optional=(["generate_project", "open_transport", "build"] if IDF_TOOLS_PATH else ["build"]),
        default=IDF_TOOLS_PATH,
        type="str",
        help="Path to the ESP-IDF install directory.",
    ),
    server.ProjectOption(
        "idf_target",
        required=["generate_project", "build", "flash", "open_transport"],
        choices=list(BOARD_PROPERTIES),
        type="str",
        help="Name of the Espressif board to build for.",
    ),
    server.ProjectOption(
        "warning_as_error",
        optional=["generate_project"],
        type="bool",
        help="Treat warnings as errors and raise an Exception.",
    ),
    server.ProjectOption(
        "compile_definitions",
        optional=["generate_project"],
        type="str",
        help="Extra definitions added project compile.",
    ),
]


def get_idf_path(options: dict):
    """Returns ESP-IDF base path"""
    idf_path = options.get("idf_path", IDF_PATH)
    assert idf_path, "'idf_path' option not passed and not found by default!"
    return idf_path


def get_idf_tools_path(options: dict):
    """Returns ESP-IDF install path"""
    idf_tools_path = options.get("idf_tools_path", IDF_TOOLS_PATH)
    assert idf_tools_path, "'idf_tools_path' option not passed and not found by default!"
    return idf_tools_path


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="espidf",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
            project_options=PROJECT_OPTIONS,
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "Makefile", "src")  # TODO(@PhilippvK)

    # Creates extra lines added to sdkconfig.defaults file
    EXTRA_PRJ_CONF_DIRECTIVES = {
        # "CONFIG_TIMER_RANDOM_GENERATOR=y": (
        #     "qemu_x86",
        #     "qemu_riscv32",
        #     "qemu_cortex_r5",
        #     "qemu_riscv64",
        # ),
        # "CONFIG_ENTROPY_GENERATOR=y": (
        #     "mps2_an521",
        #     "nrf5340dk_nrf5340_cpuapp",
        #     "nucleo_f746zg",
        #     "nucleo_l4r5zi",
        #     "stm32f746g_disco",
        # ),
    }

    def _create_prj_conf(self, project_dir, options):
        with open(project_dir / "sdkconfig.defaults", "w") as f:
            f.write(
                "# For UART used from main().\n"  # TODO(@PhilippvK)
                "CONFIG_RING_BUFFER=y\n"
                "CONFIG_UART_CONSOLE=n\n"
                "CONFIG_UART_INTERRUPT_DRIVEN=y\n"
                "\n"
            )
            f.write("# For TVMPlatformAbort().\n" "CONFIG_REBOOT=y\n" "\n")

            if options["project_type"] == "host_driven":
                f.write(
                    "# For RPC server C++ bindings.\n"
                    "CONFIG_CPLUSPLUS=y\n"  # TODO(@PhilippvK)
                    "CONFIG_LIB_CPLUSPLUS=y\n"
                    "\n"
                )

            f.write("# For math routines\n" "CONFIG_NEWLIB_LIBC=y\n" "\n")

            # if self._has_fpu(options["zephyr_board"]):
            #     f.write("# For models with floating point.\n" "CONFIG_FPU=y\n" "\n")

            # Set main stack size, if needed.
            # if options.get("config_main_stack_size") is not None:
            #     f.write(f"CONFIG_MAIN_STACK_SIZE={options['config_main_stack_size']}\n")

            # f.write("# For random number generation.\n" "CONFIG_TEST_RANDOM_GENERATOR=y\n")

            f.write("\n# Extra sdkconfig.defaults directives\n")
            for line, board_list in self.EXTRA_PRJ_CONF_DIRECTIVES.items():
                if options["idf_target"] in board_list:
                    f.write(f"{line}\n")

            f.write("\n")

    API_SERVER_CRT_LIBS_TOKEN = "<API_SERVER_CRT_LIBS>"

    CRT_LIBS_BY_PROJECT_TYPE = {
        "host_driven": "microtvm_rpc_server microtvm_rpc_common common",
        "aot_demo": "memory microtvm_rpc_common common",
    }

    def _get_platform_version(self, zephyr_base: str) -> float:
        idf_args = [IDF_CMD, "--version"]
        env = os.environ.copy()
        # env["IDF_TOOLS_PATH"] = ?  # TODO(@PhilippvK)
        out = check_output(idf_args, env=env).decode("utf-8")
        version_str = re.search(r'v(\d+.\d+)', out).group(1)
        try:
            version = float(version_str)
        except ValueError:
            # Unable to detect version
            version = None
        return version

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Check ESP-IDF version
        version = self._get_platform_version(get_idf_path(options))
        if version != ESPIDF_VERSION:
            message = f"ESP-IDF version found is not supported: found {version}, expected {ESPID_VERSION}."
            if options.get("warning_as_error") is not None and options["warning_as_error"]:
                raise server.ServerError(message=message)
            _LOG.warning(message)

        project_dir = pathlib.Path(project_dir)
        # Make project directory.
        project_dir.mkdir()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Copy boards.json file to generated project.
        shutil.copy2(BOARDS, project_dir / BOARDS.name)

        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_tar_path = project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        shutil.copy2(model_library_format_path, project_model_library_format_tar_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = os.path.splitext(project_model_library_format_tar_path)[0]
        with tarfile.TarFile(project_model_library_format_tar_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        # Populate CRT.
        crt_path = project_dir / "crt"
        crt_path.mkdir()
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Populate Makefile.
        with open(API_SERVER_DIR / "CMakeLists.txt.template", "r") as cmake_template_f:
            with open(project_dir / "CMakeLists.txt", "w") as cmake_f:
                for line in cmake_template_f:
                    if self.API_SERVER_CRT_LIBS_TOKEN in line:
                        crt_libs = self.CRT_LIBS_BY_PROJECT_TYPE[options["project_type"]]
                        line = line.replace("<API_SERVER_CRT_LIBS>", crt_libs)

                    cmake_f.write(line)

                if options.get("compile_definitions"):
                    flags = options.get("compile_definitions")
                    for item in flags:
                        cmake_f.write(f"target_compile_definitions(app PUBLIC {item})\n")

        self._create_prj_conf(project_dir, options)

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        )

        # Populate src/
        src_dir = project_dir / "main"
        shutil.copytree(API_SERVER_DIR / "src" / options["project_type"], src_dir)

        # Populate extra_files
        if options.get("extra_files_tar"):
            with tarfile.open(options["extra_files_tar"], mode="r:*") as tf:
                tf.extractall(project_dir)

    def configure(self, options):
        idf_args = [IDF_CMD, "set-target", options["idf_target"]]
        env = os.environ.copy()
        env["IDF_TOOLS_PATH"] = options.get("IDF_TOOLS_PATH", "")
        check_call(idf_args, cwd=API_SERVER_DIR, env=env)

    def build(self, options):
        if not BUILD_DIR.is_dir():
            self.configure(options)

        idf_args = [IDF_CMD, "build"]
        if options.get("verbose"):
            idf_args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE")

        check_call(idf_args, cwd=API_SERVER_DIR)


    def flash(self, options):
        idf_target = options["idf_target"]

        idf_args = [IDF_CMD, "flash"]  # TODO(@PhilippvK): set serial port and baud?
        check_call(idf_args, cwd=API_SERVER_DIR)


    def open_transport(self, options):
        transport = EspidfSerialTransport(options)

        to_return = transport.open()
        self._transport = transport
        atexit.register(lambda: self.close_transport())
        return to_return

    def close_transport(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def read_transport(self, n, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.read(n, timeout_sec)

    def write_transport(self, data, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.write(data, timeout_sec)


class EspidfSerialTransport:

    @classmethod
    def _find_serial_port(cls, options):
        flash_runner = "espidf"  # TODO(@PhilippvK): Support standalone esptool as well?

        serial_number = options.get("idf_serial_port")
        return generic_find_serial_port(serial_number=None)

        raise RuntimeError(f"Don't know how to deduce serial port for flash runner {flash_runner}")

    @classmethod
    def _lookup_baud_rate(cls, options):
        return options.get("idf_serial_baud")

    def __init__(self, options):
        self._options = options
        self._port = None

    def open(self):
        port_path = self._find_serial_port(self._options)
        self._port = serial.Serial(port_path, baudrate=self._lookup_baud_rate(self._options))
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=5.0,
            session_established_timeout_sec=5.0,
        )

    def close(self):
        self._port.close()
        self._port = None

    def read(self, n, timeout_sec):
        self._port.timeout = timeout_sec
        to_return = self._port.read(n)
        if not to_return:
            raise server.IoTimeoutError()

        return to_return

    def write(self, data, timeout_sec):
        self._port.write_timeout = timeout_sec
        bytes_written = 0
        while bytes_written < len(data):
            n = self._port.write(data)
            data = data[n:]
            bytes_written += n


if __name__ == "__main__":
    server.main(Handler())
