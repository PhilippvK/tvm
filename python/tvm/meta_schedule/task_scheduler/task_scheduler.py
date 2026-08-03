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
"""Auto-tuning Task Scheduler"""

from typing import Callable, List, Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..builder import Builder, BuilderResult, BuilderInput
from ..cost_model import CostModel
from ..database import Database
from ..logging import get_logger, get_logging_func
from ..measure_callback import MeasureCallback
from ..runner import Runner, RunnerResult, RunnerFuture, RunnerInput
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext

logger = get_logger(__name__)  # pylint: disable=invalid-name


def get_run_ms_median(runner_result: RunnerResult):
    run_secs = runner_result.run_secs
    assert len(run_secs) > 0
    v = sorted(list(run_secs))
    n = len(v)
    if n % 2 == 0:
        return (v[n // 2 - 1] + v[n // 2]) * 0.5 * 1000.0
    else:
        return v[n // 2] * 1000.0


@register_object("meta_schedule.TaskRecord")
class TaskRecord(Object):
    """The running record of a task."""

    ctx: TuneContext
    task_weight: float
    flop: float
    is_terminated: bool
    build_error_count: int
    run_error_count: int
    measure_candidates: List[MeasureCandidate]
    builder_results: List[BuilderResult]
    runner_results: List[RunnerResult]

    # def __init__(
    #     self,
    #     ctx: TuneContext,
    #     task_weight: float,
    # ):
    #     """Constructor."""

    #     self.__init_handle_by_constructor__(
    #         _ffi_api.TaskRecordPyTaskRecord,  # type: ignore # pylint: disable=no-member
    #         ctx,
    #         task_weight,
    #     )


class TaskRecord2(Object):
    """The running record of a task."""

    ctx: TuneContext
    task_weight: float
    flop: float
    is_terminated: bool
    build_error_count: int
    run_error_count: int
    latency_ms: List[float]
    measure_candidates: Optional[List[MeasureCandidate]]
    builder_results: Optional[List[BuilderResult]]
    runner_results: Optional[List[RunnerResult]]
    runner_futures: Optional[List[RunnerFuture]]

    def __init__(
        self,
        ctx: TuneContext,
        task_weight: float,
    ):
        self.ctx = ctx
        self.task_weight = task_weight
        self.flop = 1.0
        self.is_terminated = False
        self.build_error_count = 0
        self.run_error_count = 0
        self.latency_ms = []
        self.measure_candidates = None
        self.builder_results = None
        self.runner_results = None
        self.runner_futures = None
        if self.ctx.search_strategy is not None:
            self.ctx.search_strategy._initialize_with_tune_context(self.ctx)
        if self.ctx.space_generator is not None:
            self.ctx.space_generator._initialize_with_tune_context(self.ctx)
        from tvm.tir.analysis import estimate_tir_flops

        self.flops = estimate_tir_flops(self.ctx.mod)

    def send_to_builder(self, builder):
        candidates = self.measure_candidates
        target = self.ctx.target
        inputs = []
        for candidate in candidates:
            inputs.append(BuilderInput(candidate.sch.mod, target))
        self.builder_results = builder.build(inputs)

    def send_to_runner(self, runner):
        candidates = self.measure_candidates
        builder_results = self.builder_results
        target = self.ctx.target
        assert len(candidates) == len(builder_results)
        n = len(candidates)
        n_build_errors = 0
        inputs = []
        for i in range(n):
            # TODO: use enumerate
            candidate = candidates[i]
            builder_result = builder_results[i]
            if builder_result.error_msg is not None:
                n_build_errors += 1
                continue
            inputs.append(RunnerInput(builder_result.artifact_path, target.kind.name, candidate.args_info))
        futures = runner.run(inputs)
        if n_build_errors == 0:
            self.runner_futures = futures
            return
        results = []
        j = 0
        for i in range(n):
            # TODO: use enumerate
            builder_result = builder_results[i]

            # TODO: check if pickable?
            def f_result():
                timestamp = None
                # TODO: timestamp
                return RunnerResult(None, builder_result.error_msg, timestamp)

            if builder_result.error_msg is not None:
                results.append(RunnerFuture(f_done=lambda: True, f_result=f_result))
            else:
                results.append(futures[j])
                j += 1
        self.runner_futures = results

    def cleanup(self, task_id: int, results: List[RunnerResult]):
        assert len(self.builder_results) == len(results)
        assert len(self.runner_futures) == len(results)
        n = len(results)
        name = self.ctx.task_name
        # TODO: logger
        for i in range(n):
            builder_result = self.builder_results[i]
            candidate = self.measure_candidates[i]
            runner_result = results[i]
            error_msg = None
            trials = len(self.latency_ms) + 1
            run_ms = 1.0e9
            if builder_result.error_msg:
                error_msg = builder_result.error_msg
                self.build_error_count += 1
            elif runner_result.error_msg:
                error_msg = runner_result.error_msg
                self.run_error_count += 1
            else:
                run_ms = get_run_ms_median(runner_result)
            self.latency_ms.append(run_ms)
            if error_msg is not None:
                # TODO: logging
                print(f"Error: {error_msg}")
            else:
                # TODO: logging
                best_ms = min(self.latency_ms)
                print(
                    f"[Task #{task_id}: {name}] Trial #{trials}: GFLOPs: {self.flop / run_ms / 1e6}. Time: {run_ms * 1e3}. Best GFLOPs: {self.flop / best_ms / 1e6}"
                )
        self.measure_candidates = None
        self.builder_results = None
        self.runner_futures = None


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """The abstract task scheduler interface."""

    tasks_: List[TaskRecord]
    measure_callbacks_: List[MeasureCallback]
    database_: Optional[Database]
    cost_model_: Optional[CostModel]
    remaining_tasks_: int

    TaskSchedulerType = Union["TaskScheduler", Literal["gradient", "round-robin"]]

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # type: ignore # pylint: disable=no-member

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.

        Returns
        -------
        results : List[RunnerResult]
            The list of results.
        """
        return _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        num_trials_per_iter: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
        design_spaces_mask: List[int] = [],
    ) -> None:
        """Auto-tuning.

        Parameters
        ----------
        tasks : List[TuneContext]
            The list of tuning contexts as tasks.
        task_weights : List[float]
            The list of task weights.
        max_trials_global : int
            The maximum number of trials globally.
        max_trials_per_task : int
            The maximum number of trials per task.
        num_trials_per_iter : int
            The number of trials per iteration.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        measure_callbacks : List[MeasureCallback]
            The list of measure callbacks.
        database : Optional[Database]
            The database.
        cost_model : Optional[CostModel]
            The cost model.
        design_spaces_mask : TODO
        """
        task_weights = [float(w) for w in task_weights]
        print("mmoodd", tasks[0].mod)
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self,
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            num_trials_per_iter,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
            design_spaces_mask,
        )

    def terminate_task(self, task_id: int) -> None:
        """Terminate the task

        Parameters
        ----------
        task_id : int
            The task id to be terminated.
        """
        _ffi_api.TaskSchedulerTerminateTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def touch_task(self, task_id: int) -> None:
        """Touch the task and update its status

        Parameters
        ----------
        task_id : int
            The task id to be checked.
        """
        _ffi_api.TaskSchedulerTouchTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def print_tuning_statistics(self) -> None:
        """Print out a human-readable format of the tuning statistics."""
        return _ffi_api.TaskSchedulerPrintTuningStatistics(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Literal["round-robin", "gradient"] = "gradient",
        *args,
        **kwargs,
    ) -> "TaskScheduler":
        """Create a task scheduler."""
        from . import (  # pylint: disable=import-outside-toplevel
            GradientBased,
            RoundRobin,
        )

        if kind == "round-robin":
            return RoundRobin(*args, **kwargs)  # type: ignore
        if kind == "gradient":
            return GradientBased(*args, **kwargs)
        raise ValueError(f"Unknown TaskScheduler name: {kind}")


create = TaskScheduler.create  # pylint: disable=invalid-name


@register_object("meta_schedule.PyTaskScheduler")
class _PyTaskScheduler(TaskScheduler):
    """
    A TVM object task scheduler to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyTaskScheduler
    """

    def __init__(
        self,
        f_next_task_id: Callable,
        f_join_running_task: Callable,
        f_tune: Callable,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerPyTaskScheduler,  # type: ignore # pylint: disable=no-member
            get_logging_func(logger),
            f_next_task_id,
            f_join_running_task,
            f_tune,
        )


class PyTaskScheduler:
    """
    An abstract task scheduler with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyTaskScheduler,
        "fields": [],
        "methods": ["next_task_id", "join_running_task", "tune"],
    }

    def __init__(self): ...

    def tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
    ) -> None:
        """Auto-tuning."""
        # Using self._outer to replace the self pointer
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self._outer(),  # type: ignore # pylint: disable=no-member
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        raise NotImplementedError

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.TaskSchedulerJoinRunningTask(self._outer(), task_id)  # type: ignore # pylint: disable=no-member
