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
"""Dummy Runner"""
import logging
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Union
import subprocess

import tvm

# from ...contrib.popen_pool import PopenPoolExecutor
from ...runtime import Device, Module
from ..logging import get_logger
from ..profiler import Profiler
from ..utils import derived_object, get_global_func_with_default_on_worker
from .config import EvaluatorConfig
from .runner import PyRunner, PyRunnerFuture, RunnerFuture, RunnerInput, RunnerResult
from .utils import (
    T_ARG_INFO_JSON_OBJ_LIST,
    T_ARGUMENT_LIST,
    alloc_argument_common,
    run_evaluator_common,
)
from .local_runner import LocalRunnerFuture, T_ALLOC_ARGUMENT, T_RUN_EVALUATOR, T_CLEANUP, default_cleanup

logger = get_logger(__name__)  # pylint: disable=invalid-name


def _worker_func_dummy(
    _f_alloc_argument: Optional[str],
    _f_run_evaluator: Optional[str],
    _f_cleanup: Optional[str],
    evaluator_config: EvaluatorConfig,
    alloc_repeat: int,
    artifact_path: str,
    device_type: str,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
) -> List[float]:
    f_cleanup: T_CLEANUP = get_global_func_with_default_on_worker(_f_cleanup, default_cleanup)

    @contextmanager
    def resource_handler():
        try:
            yield
        finally:
            # Final step. Always clean up
            with Profiler.timeit("LocalRunner/cleanup"):
                f_cleanup()
    with resource_handler():
        cnt = evaluator_config.number
    return [1.0] * cnt


@derived_object
class DummyRunner(PyRunner):
    """Dummy runner

    Parameters
    ----------
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    cooldown_sec: float
        The cooldown in seconds.
    alloc_repeat: int
        The number of times to repeat the allocation.
    f_alloc_argument: Optional[str, Callable]
        The function name to allocate the arguments or the function itself.
    f_run_evaluator: Optional[str, Callable]
        The function name to run the evaluator or the function itself.
    f_cleanup: Optional[str, Callable]
        The function name to cleanup the session or the function itself.
    pool: PopenPoolExecutor
        The popen pool executor.

    Attributes
    ----------
    T_ALLOC_ARGUMENT : typing._GenericAlias
        The signature of the function `f_alloc_argument`, which is:

        .. code-block:: python

        def default_alloc_argument(
            device: Device,
            args_info: T_ARG_INFO_JSON_OBJ_LIST,
            alloc_repeat: int,
        ) -> List[T_ARGUMENT_LIST]:
            ...

    T_RUN_EVALUATOR : typing._GenericAlias
        The signature of the function `f_run_evaluator`, which is:

        .. code-block:: python

        def default_run_evaluator(
            rt_mod: Module,
            device: Device,
            evaluator_config: EvaluatorConfig,
            repeated_args: List[T_ARGUMENT_LIST],
        ) -> List[float]:
            ...

    T_CLEANUP : typing._GenericAlias
        The signature of the function `f_cleanup`, which is:

        .. code-block:: python

        def default_cleanup() -> None:
            ...
    """

    timeout_sec: float
    evaluator_config: EvaluatorConfig
    cooldown_sec: float
    alloc_repeat: int

    f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None]
    f_run_evaluator: Union[T_RUN_EVALUATOR, str, None]
    f_cleanup: Union[T_CLEANUP, str, None]

    # pool: PopenPoolExecutor

    def __init__(
        self,
        timeout_sec: float = 30,
        evaluator_config: Optional[EvaluatorConfig] = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None] = None,
        f_run_evaluator: Union[T_RUN_EVALUATOR, str, None] = None,
        f_cleanup: Union[T_CLEANUP, str, None] = None,
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        timeout_sec: float
            The timeout setting.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        cooldown_sec: float
            The cooldown in seconds.
        alloc_repeat: int
            The number of times to random fill the allocation.
        f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None]
            The function name to allocate the arguments or the function itself.
        f_run_evaluator: Union[T_RUN_EVALUATOR, str, None]
            The function name to run the evaluator or the function itself.
        f_cleanup: Union[T_CLEANUP, str, None]
            The function name to cleanup the session or the function itself.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        """
        super().__init__()
        self.timeout_sec = timeout_sec
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        self.f_alloc_argument = f_alloc_argument
        self.f_run_evaluator = f_run_evaluator
        self.f_cleanup = f_cleanup

        err_path = subprocess.DEVNULL
        if logger.root.level <= logging.DEBUG:
            err_path = subprocess.STDOUT

        # logger.info("DummyRunner: max_workers = 1")
        # self.pool = PopenPoolExecutor(
        #     max_workers=1,  # one dummy worker (TODO: allow many?)
        #     timeout=timeout_sec,
        #     initializer=initializer,
        #     stderr=err_path,  # suppress the stderr output
        # )
        # self._sanity_check()

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []
        for runner_input in runner_inputs:
            func = _worker_func_dummy
            try:
                result = func(
                    self.f_alloc_argument,
                    self.f_run_evaluator,
                    self.f_cleanup,
                    self.evaluator_config,
                    self.alloc_repeat,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                )
                error_message = None
            except Exception as exception:  # pylint: disable=broad-except
                result = None
                error_message = "DummyRunner: An exception occurred\n" + str(exception)
                # raise exception
            local_future = LocalRunnerFuture(res=result, error_message=error_message)
            results.append(local_future)  # type: ignore
        return results

    # def _sanity_check(self) -> None:
    #     def _check(
    #         f_alloc_argument,
    #         f_run_evaluator,
    #         f_cleanup,
    #     ) -> None:
    #         get_global_func_with_default_on_worker(name=f_alloc_argument, default=None)
    #         get_global_func_with_default_on_worker(name=f_run_evaluator, default=None)
    #         get_global_func_with_default_on_worker(name=f_cleanup, default=None)

    #     value = self.pool.submit(
    #         _check,
    #         self.f_alloc_argument,
    #         self.f_run_evaluator,
    #         self.f_cleanup,
    #     )
    #     value.result()


@tvm.register_func("meta_schedule.runner.get_dummy_runner")
def get_dummy_runner() -> DummyRunner:
    """Get the dummy Runner.

    Returns
    -------
    runner: DummyRunner
        The dummy runner
    """
    return DummyRunner()
