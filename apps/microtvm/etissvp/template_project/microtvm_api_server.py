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

import atexit
import collections
import collections.abc
import enum
import fcntl
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
import json
import signal

import yaml

from tvm.micro.project_api import server


_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR / "build"


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()

def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.debug("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


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


PROJECT_TYPES = []
if IS_TEMPLATE:
    for d in (API_SERVER_DIR / "src").iterdir():
        if d.is_dir():
            PROJECT_TYPES.append(d.name)


PROJECT_OPTIONS = [
    server.ProjectOption(
        "extra_files_tar",
        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
    ),
    server.ProjectOption(
        "etiss_path",
        help="Path to the installed ETISS directory.",
    ),
    server.ProjectOption(
        "riscv_path",
        help="Path to the installed RISCV GCC directory.",
    ),
    server.ProjectOption(
        "etissvp_script",
        help="Path to script which sets up the environment and starts running the provided target software binary on the vp.",
    ),
    server.ProjectOption(
        "etissvp_script_args",
        help="Additional arguments to etissvp_script.",
    ),
    server.ProjectOption(
        "project_type",
        help="Type of project to generate.",
        choices=tuple(PROJECT_TYPES),
    ),
    server.ProjectOption("verbose", help="Run build with verbose output.", choices=(True, False)),
    server.ProjectOption("debug", help="Run build in DEBUG mode.", choices=(True, False)),
    server.ProjectOption("transport", help="Run build in DEBUG mode.", choices=(True, False)),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="etissvp",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
            project_options=PROJECT_OPTIONS,
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "Makefile", "src")

    API_SERVER_CRT_LIBS_TOKEN = "<API_SERVER_CRT_LIBS>"

    CRT_LIBS_BY_PROJECT_TYPE = {
        "host_driven": "microtvm_rpc_server microtvm_rpc_common common",
        "aot_demo": "memory microtvm_rpc_common common",
    }

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        project_dir = pathlib.Path(project_dir)
        # Make project directory.
        project_dir.mkdir()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

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

        #self._create_prj_conf(project_dir, options)

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        )

        # Populate src/
        src_dir = project_dir / "src"
        shutil.copytree(API_SERVER_DIR / "src" / options["project_type"], src_dir)
        #print(src_dir)
        #input()

        # Populate extra_files
        if options.get("extra_files_tar"):
            with tarfile.open(options["extra_files_tar"], mode="r:*") as tf:
                tf.extractall(project_dir)

    def build(self, options):
        BUILD_DIR.mkdir()

        cmake_args = ["cmake", ".."]

        if options.get("etiss_path"):
            cmake_args.append("-DETISS_DIR=" + options["etiss_path"])
        else:
            raise RuntimeError("Project Config 'etiss_path' undefined!")

        if options.get("riscv_path"):
            cmake_args.append("-DRISCV_ELF_GCC_PREFIX=" + options["riscv_path"])
        else:
            raise RuntimeError("Project Config 'riscv_path' undefined!")

        if options.get("debug"):
            cmake_args.append("-DCMAKE_BUILD_TYPE=DEBUG")

        if options.get("verbose"):
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE")

        #print("BUILD", BUILD_DIR)
        #input()
        if options.get("verbose"):
            check_call(cmake_args, cwd=BUILD_DIR)
        else:
            check_call(cmake_args, cwd=BUILD_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        args = ["make", "-j2"]
        if options.get("verbose"):
            args.append("VERBOSE=1")
            check_call(args, cwd=BUILD_DIR)
        else:
            check_call(args, cwd=BUILD_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def flash(self, options):
        if options.get("transport"):
            return  # NOTE: etissvp requires no flash step--it is launched from open_transport.
        else:
            transport = ETISSVPTransport(options)
            to_return = transport.open()
            self._transport = transport
            transport._wait_for_etissvp()
            transport._wait_for_etissvp()
            self.close_transport()
            return

    def open_transport(self, options):
        transport = ETISSVPTransport(options)

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


def _set_nonblock(fd):
    flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
    new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"


class ETISSVPMakeResult(enum.Enum):
    ETISSVP_STARTED = "etissvp_started"
    ETISSVP_ENDED = "etissvp_ended"
    MAKE_FAILED = "make_failed"
    EOF = "eof"


class ETISSVPTransport:
    """The user-facing ETISSVP transport class."""

    def __init__(self, options):
        self.options = options
        self.proc = None
        self.pipe_dir = None
        self.read_fd = None
        self.write_fd = None
        self._queue = queue.Queue()

    def open(self):
        #self.pipe_dir = pathlib.Path(tempfile.mkdtemp())
        self.pipe_dir = pathlib.Path(os.path.join(BUILD_DIR, ".tmp"))
        os.mkdir(self.pipe_dir)
        self.write_pipe = self.pipe_dir / "uartdevicefifoin"
        self.write_pipe2 = self.pipe_dir / "uartdevicefifoin2"
        self.read_pipe = self.pipe_dir / "uartdevicefifoout"
        
        #os.mkfifo(self.read_pipe)

        #print("RUN", BUILD_DIR)
        #input()
        os.mkfifo(self.write_pipe2)
        if not self.options.get("etissvp_script"):
            raise RuntimeError("Project Config 'etissvp_script' undefined!")
        etissvp_env = os.environ.copy()
        etissvp_env["ETISS_DIR"] = self.options["etiss_path"]
        self.proc = subprocess.Popen(
            [self.options["etissvp_script"], "app", *self.options["etissvp_script_args"].split()],
            #["make", "run", f"UART_PIPE={self.pipe}"],
            cwd=BUILD_DIR,
            stdout=subprocess.PIPE,
            env=etissvp_env,
            preexec_fn=os.setsid,
        )
        #input()
        #self._wait_for_etissvp()

        # NOTE: although each pipe is unidirectional, open both as RDWR to work around a select
        # limitation on linux. Without this, non-blocking I/O can't use timeouts because named
        # FIFO are always considered ready to read when no one has opened them for writing.
        #while not (os.path.exists(self.read_pipe) and os.path.exists(self.write_pipe) and os.path.exists(self.write_pipe2)):
        while not (os.path.exists(self.read_pipe) and os.path.exists(self.write_pipe)):
            time.sleep(1)
        time.sleep(1)

        self.read_fd = os.open(self.read_pipe, os.O_RDWR | os.O_NONBLOCK)
        self.write_fd = os.open(self.write_pipe, os.O_RDWR | os.O_NONBLOCK)
        self.write_fd2 = os.open(self.write_pipe2, os.O_RDWR | os.O_NONBLOCK)
        _set_nonblock(self.read_fd)
        _set_nonblock(self.write_fd)
        _set_nonblock(self.write_fd2)

        #self._wait_for_etissvp()

        #return server.TransportTimeouts(
        #    session_start_retry_timeout_sec=1.0,
        #    #session_start_timeout_sec=200.0, #20.0,
        #    session_start_timeout_sec=400.0, #20.0,
        #    #session_established_timeout_sec=10.0,
        #    session_established_timeout_sec=200.0, #20.0,
        #)
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=1,
            #session_start_timeout_sec=200.0, #20.0,
            session_start_timeout_sec=10, #20.0,
            #session_established_timeout_sec=10.0,
            session_established_timeout_sec=10, #20.0,
        )
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=0,
            session_start_timeout_sec=0,
            session_established_timeout_sec=0,
        )

    def close(self):
        did_write = False
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

        if self.write_fd2 is not None:
            os.close(self.write_fd2)
            self.write_fd2 = None

        if self.proc:
            # Killing the ETISS subprocess seems to need this workaround...
            self.proc.send_signal(signal.SIGINT)
            time.sleep(1)
            self.proc.kill()
            pgrp = os.getpgid(self.proc.pid)
            os.killpg(pgrp, signal.SIGKILL)

        if self.read_fd:
            os.close(self.read_fd)
            self.read_fd = None

        if self.pipe_dir is not None:
            if os.path.exists(self.pipe_dir):
                shutil.rmtree(self.pipe_dir)
            self.pipe_dir = None

    def read(self, n, timeout_sec):
        return server.read_with_timeout(self.read_fd, n, timeout_sec)

    def write(self, data, timeout_sec):
        to_write = bytearray(data)

        while to_write:
            num_written = server.write_with_timeout(self.write_fd, to_write, timeout_sec, dbg_fd=self.write_fd2)
            to_write = to_write[num_written:]

    def _etissvp_check_stdout(self):
        for line in self.proc.stdout:
            line = str(line.decode(errors="ignore"))
            _LOG.info("%s", line.replace("\n", ""))
            if "=== Simulation start ===" in line:  # TODO
                self._queue.put(ETISSVPMakeResult.ETISSVP_STARTED)
            elif "=== Simulation end ===" in line:  # TODO
                self._queue.put(ETISSVPMakeResult.ETISSVP_ENDED)
            else:
                line = re.sub("[^a-zA-Z0-9 \n]", "", line)
                pattern = r"recipe for target (\w*) failed"
                if re.search(pattern, line, re.IGNORECASE):
                    self._queue.put(ETISSVPMakeResult.MAKE_FAILED)
        self._queue.put(ETISSVPMakeResult.EOF)

    def _wait_for_etissvp(self):
        threading.Thread(target=self._etissvp_check_stdout, daemon=True).start()
        while True:
            try:
                item = self._queue.get(timeout=120)
            except Exception:
                raise TimeoutError("ETISSVP setup timeout.")

            if item == ETISSVPMakeResult.ETISSVP_STARTED:
                break

            if item == ETISSVPMakeResult.ETISSVP_ENDED:
                break

            #if item in [ETISSVPMakeResult.MAKE_FAILED, ETISSVPMakeResult.EOF]:
            #    raise RuntimeError("ETISSVP setup failed.")

            #raise ValueError(f"{item} not expected.")


if __name__ == "__main__":
    server.main(Handler())
