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
"""Gradient Based Task Scheduler"""
from typing import List
from tvm._ffi import register_object

from .. import _ffi_api
from ..logging import get_logger, get_logging_func
from .task_scheduler import TaskScheduler, TaskRecord2
from ..runner import RunnerResult

logger = get_logger(__name__)  # pylint: disable=invalid-name


from rich.console import Console
from rich.table import Table

from .. import _ffi_api

_console = Console()


def rich_print_tuning_statistics(scheduler):
    rows = _ffi_api.TaskSchedulerTaskStats(scheduler)

    table = Table(title="Meta-Schedule Tuning")
    for col in [
        "ID",
        "Name",
        "Group",
        "FLOP",
        "Weight",
        "Speed GFLOPS",
        "Latency us",
        "Weighted us",
        "Trials",
        "Errors",
        "Spaces",
        "Done",
    ]:
        table.add_column(col)

    total_trials = 0
    total_weighted_latency = 0.0

    for row in rows:
        flop = float(row["flop"])
        weight = float(row["weight"])
        best_ms = float(row["best_latency_ms"])
        trials = int(row["trials"])
        mask = row["mask"]
        spaces_str = str(mask)  # TODO
        space_idxs = [i for i in range(len(mask)) if mask[i]]
        spaces_str = "{" + ",".join(map(str, space_idxs)) + "}"

        total_trials += trials

        if best_ms >= 1e9:
            speed = latency_us = weighted_us = "N/A"
        else:
            latency_us_f = best_ms * 1000.0
            speed_f = flop / best_ms / 1e6
            weighted_us_f = latency_us_f * weight
            total_weighted_latency += weighted_us_f

            speed = f"{speed_f:.4f}"
            latency_us = f"{latency_us_f:.4f}"
            weighted_us = f"{weighted_us_f:.4f}"

        table.add_row(
            str(int(row["id"])),
            str(row["name"]),
            str(row["group"]),
            f"{flop:.0f}",
            f"{weight:.2f}",
            speed,
            latency_us,
            weighted_us,
            str(trials),
            f'{int(row["build_errors"])}/{int(row["run_errors"])}',
            spaces_str,
            "Y" if bool(row["done"]) else "",
        )

    _console.clear()
    _console.print(table)
    _console.print(f"Total trials: {total_trials}")
    _console.print(f"Total weighted latency: {total_weighted_latency:.4f} us")


def custom_tune(
    self,
    ctxs,
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
):
    use_default = True
    # use_default = False
    if use_default:
        return _ffi_api.TaskSchedulerTuneDefault(
            self,
            ctxs,
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

    # custom algorithm
    n_tasks = remaining_tasks = len(ctxs)
    self.measure_callbacks_ = measure_callbacks;
    database_ = database
    cost_model_ = cost_model
    self.round_robin_rounds_ = 0
    best_latency_history_ = [None for _ in range(n_tasks)]
    _ffi_api.TaskSchedulerInitTasks(
        self,
        ctxs,
        task_weights,
        measure_callbacks,
        database,
        cost_model,
    )
    # self.tasks_ = []
    for i in range(n_tasks):
        ctx = ctxs[i]
        weight = task_weights[i]
        print(f"Initializing Task #{i}: {ctx.task_name}")
        # task_rec = TaskRecord2(ctx, weight)
        # self.tasks_.append(task_rec)
        design_spaces = ctx.space_generator.generate_design_space(ctx.mod)
        print("len(design_spaces)", len(design_spaces))
        for j, design_space in enumerate(design_spaces):
            sch = design_space
            trace = sch.trace
            python_str = trace.as_python(False)
            print(f"Design space #{j}: {sch.mod}\n{python_str}\n")
        ctx.search_strategy.pre_tuning(max_trials_per_task, num_trials_per_iter, design_spaces, database, cost_model)
    num_trials_already = 0;
    while num_trials_already < max_trials_global:
        task_id = self.next_task_id()
        print("task_id", task_id)
        if task_id < 0:
            print("break")
            break
        # task = self.tasks_[task_id]
        # print(f"TaskScheduler picks Task #{task_id}: {task.ctx.task_name}")
        # assert not task.is_terminated
        # assert task.runner_futures is None
        # if len(task.latency_ms) >= max_trials_per_task:
        #     self.terminate_task(task_id)
        #     continue
        # candidates = task.ctx.search_strategy.generate_measure_candidates()
        num_candidates = _ffi_api.TaskSchedulerGenerateMeasureCandidates(self, task_id)
        # if candidates is None:
        if num_candidates == 0:
            # self.terminate_task(task_id)
            _ffi_api.TaskSchedulerTerminateTask(self, task_id)
        else:
            # task.measure_candidates = candidates
            num_candidates = len(candidates)
            num_trials_already += num_candidates
            print(f"Sending {num_candidates} sample(s) to builder")
            # task.send_to_builder(builder)
            _ffi_api.TaskSchedulerSendToBuilder(self, task_id, builder)
            print(f"Sending {num_candidates} sample(s) to runner")
            # task.send_to_runner(runner)
            _ffi_api.TaskSchedulerSendToRunner(self, task_id, runner)
    for task_id in range(n_tasks):
        task = self.tasks_[task_id]
        if not task.is_terminated:
            if task.runner_futures is not None:
                self.join_running_task(task_id)
            # self.terminate_task(task_id)
            _ffi_api.TaskSchedulerTerminateTask(self, task_id)
        task.ctx.search_strategy.post_tuning()
    raise NotImplementedError("custom algo")


@register_object("meta_schedule.GradientBased")
class GradientBased(TaskScheduler):
    """Gradient Based Task Scheduler"""

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        window_size: int = 3,
        seed: int = -1,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        alpha : float = 0.2
            The parameter alpha in gradient computation.
        window_size : int = 3
            The parameter to control backward window size in gradient computation.
        seed : int = -1
            The random seed.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerGradientBased,  # type: ignore # pylint: disable=no-member
            get_logging_func(logger),
            alpha,
            window_size,
            seed,
        )
        _ffi_api.TaskSchedulerSetPrintTuningStatisticsFunc(
            self,
            rich_print_tuning_statistics,
        )
        _ffi_api.TaskSchedulerSetTuneFunc(
            self,
            custom_tune,
        )

    def next_task_id(self) -> int:
        print("next_task_id")
        n_tasks = len(self.tasks_)
        print("n_tasks", n_tasks)
        if self.round_robin_rounds_ == 0:
            rich_print_tuning_statistics(self)
        if self.round_robin_rounds_ < n_tasks:
            to_ret = self.round_robin_rounds_
            self.round_robin_rounds_ += 1
            return to_ret
        if self.round_robin_rounds_ == n_tasks:
            for i in range(n_tasks):
                if self.tasks_[i].runner_futures is not None:
                    self.join_running_task(i)
            self.round_robin_rounds_ += 1
        tasks_alive = []
        for i in range(n_tasks):
            self.touch_task(i)
            if not self.tasks_[i].is_terminated:
                tasks_alive.append(i)
        if len(tasks_alive) == 0:
            return -1
        raise NotImplementedError("next_task_id custom")

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        # task = self.tasks_[task_id]
        # assert task.runner_futures is not None
        # results = []
        # futures = task.runner_futures
        # for future in futures:
        #     result = future.result()
        #     results.append(result)
        # assert task.measure_candidates is not None
        # task.ctx.search_strategy.notify_runner_results(task.measure_candidates, results)
        # assert task.builder_results is not None
        # assert len(results) == len(task.measure_candidates)
        # assert len(results) == len(task.builder_results)
        # for callback in self.measure_callbacks_:
        #     callback.apply(self, task_id, task.measure_candidates, task.builder_results, results)
        # task.cleanup(task_id, results)
        # rich_print_tuning_statistics(self)
        # return results
        results = _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)
        best = _ffi_api.TaskSchedulerTaskBestLatency(self, task_id)
        if best < 1e9:
            self.best_latency_history_[task_id].append(best)
        return results




# class GradientBasedPy(PyTaskScheduler):
#     """Gradient Based Task Scheduler."""
# 
#     def __init__(
#         self,
#         *,
#         alpha: float = 0.2,
#         window_size: int = 3,
#         seed: int = -1,
#     ) -> None:
#         self.alpha = alpha
#         self.window_size = window_size
#         self.rand = random.Random(None if seed == -1 else seed)
# 
#         self.round_robin_rounds = 0
#         self.best_latency_history: List[List[float]] = []
# 
#         super().__init__(
#             f_next_task_id=self.next_task_id,
#             f_join_running_task=self.join_running_task,
#             f_tune=self.tune,
#             logger=get_logging_func(logger),
#         )
# 
#     def tune(
#         self,
#         tasks,
#         task_weights,
#         max_trials_global,
#         max_trials_per_task,
#         num_trials_per_iter,
#         builder,
#         runner,
#         measure_callbacks,
#         database,
#         cost_model,
#     ):
#         n_tasks = len(tasks)
#         self.round_robin_rounds = 0
#         self.best_latency_history = [[] for _ in range(n_tasks)]
# 
#         _ffi_api.TaskSchedulerTune(
#             self,
#             tasks,
#             task_weights,
#             max_trials_global,
#             max_trials_per_task,
#             num_trials_per_iter,
#             builder,
#             runner,
#             measure_callbacks,
#             database,
#             cost_model,
#         )
# 
#     def next_task_id(self) -> int:
#         n_tasks = _ffi_api.TaskSchedulerNumTasks(self)
# 
#         if self.round_robin_rounds == 0:
#             _ffi_api.TaskSchedulerPrintTuningStatistics(self)
# 
#         if self.round_robin_rounds < n_tasks:
#             task_id = self.round_robin_rounds
#             self.round_robin_rounds += 1
#             return task_id
# 
#         if self.round_robin_rounds == n_tasks:
#             for task_id in range(n_tasks):
#                 if _ffi_api.TaskSchedulerTaskHasRunnerFutures(self, task_id):
#                     self.join_running_task(task_id)
#             self.round_robin_rounds += 1
# 
#         tasks_alive = []
#         for task_id in range(n_tasks):
#             _ffi_api.TaskSchedulerTouchTask(self, task_id)
#             if not _ffi_api.TaskSchedulerTaskIsTerminated(self, task_id):
#                 tasks_alive.append(task_id)
# 
#         if not tasks_alive:
#             return -1
# 
#         grads = []
#         for task_id in tasks_alive:
#             hist = self.best_latency_history[task_id]
#             n = len(hist)
#             w = self.window_size
#             weight = _ffi_api.TaskSchedulerTaskWeight(self, task_id)
# 
#             if n > 0 and hist[-1] < 1e9:
#                 best = hist[-1]
#                 g1 = (hist[n - 1 - w] - best) / w if n >= 1 + w else 0.0
#                 g2 = best / n
#                 g = self.alpha * g1 + (1.0 - self.alpha) * g2
#                 grads.append(g * weight)
#             else:
#                 grads.append(-1e9)
# 
#         if max(grads) == min(grads):
#             task_id = self.rand.choice(tasks_alive)
#         else:
#             task_id = tasks_alive[grads.index(max(grads))]
# 
#         if _ffi_api.TaskSchedulerTaskHasRunnerFutures(self, task_id):
#             self.join_running_task(task_id)
# 
#         return task_id
# 
#     def join_running_task(self, task_id: int):
#         results = _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)
# 
#         best = _ffi_api.TaskSchedulerTaskBestLatency(self, task_id)
#         if best < 1e9:
#             self.best_latency_history[task_id].append(best)
# 
#         return results
