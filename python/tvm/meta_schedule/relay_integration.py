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
"""MetaSchedule-Relay integration"""

from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on
import numpy as np  # type: ignore

from tvm import nd
from tvm._ffi import get_global_func
from tvm.ir import IRModule, transform
from tvm.ir.instrument import PassInstrument
from tvm.runtime import NDArray
from tvm.target import Target

from .builder import Builder
from .cost_model import CostModel
from .database import Database, MemoryDatabase, JSONDatabase
from .extracted_task import ExtractedTask
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .profiler import Profiler
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext
from .utils import fork_seed

if TYPE_CHECKING:
    from tvm import relay

_extract_task = get_global_func(  # pylint: disable=invalid-name
    "relay.backend.MetaScheduleExtractTask",
    allow_missing=True,
)


@contextmanager
def _autotvm_silencer():
    """A context manager that silences autotvm warnings."""
    from tvm import autotvm  # pylint: disable=import-outside-toplevel

    silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True
    try:
        yield
    finally:
        autotvm.GLOBAL_SCOPE.silent = silent


def _normalize_params(
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    pass_config: Mapping[str, Any],
    executor: Optional["relay.backend.Executor"],
    runtime: Optional["relay.backend.Runtime"],
) -> Tuple[
    IRModule,
    Target,
    Dict[str, NDArray],
    Dict[str, Any],
    Optional["relay.backend.Executor"],
    Optional["relay.backend.Runtime"],
]:
    from tvm import relay  # pylint: disable=import-outside-toplevel

    if isinstance(mod, relay.Function):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)
    if params is None:
        params = {}
    relay_params = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = nd.array(param)
        relay_params[name] = param

    if executor is None:
        executor = relay.backend.Executor("graph")

    if runtime is None:
        runtime = relay.backend.Runtime("cpp")

    if mod.get_attr("executor") is None:
        mod = mod.with_attr("executor", executor)
    else:
        executor = mod.get_attr("executor")

    pass_config = dict(pass_config)
    return mod, target, relay_params, pass_config, executor, runtime


def extract_tasks(
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    *,
    opt_level: int = 3,
    pass_config: Mapping[str, Any] = MappingProxyType(
        {
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "default",
        }
    ),
    executor: Optional["relay.backend.Executor"] = None,
    runtime: Optional["relay.backend.Runtime"] = None,
    module_equality: str = "structural",
    disabled_pass: Optional[Union[List[str], Set[str], Tuple[str]]] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
    include_simple_tasks: bool = False,
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod : IRModule
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    opt_level : int
        The optimization level of the compilation
    pass_config : Mapping[str, Any]
        The pass configuration
    executor : Optional[relay.backend.Executor]
        The executor to use
    runtime : Optional[relay.backend.Runtime]
        The runtime to use
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-ndarray" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tir/analysis/analysis.py.
    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of disabled passes
    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.
    include_simple_tasks : bool
        TODO

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """
    # pylint: disable=import-outside-toplevel
    from tvm import autotvm

    # pylint: enable=import-outside-toplevel
    mod, target, params, pass_config, _ex, _rt = _normalize_params(
        mod,
        target,
        params,
        pass_config,
        executor,
        runtime,
    )
    if target.kind.name != "cuda" and isinstance(
        autotvm.DispatchContext.current, autotvm.FallbackContext
    ):
        tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.utils.EmptyContext()
    with Profiler.timeit("TaskExtraction"):
        with target, _autotvm_silencer(), tophub_context:
            with transform.PassContext(
                opt_level=opt_level,
                config=pass_config,
                disabled_pass=disabled_pass,
                instruments=instruments,
            ):
                multi_dispatch = True
                ret = list(_extract_task(mod, target, params, module_equality, multi_dispatch))
                assert len(ret) > 0
    if not include_simple_tasks:
        # ret = [task for task in ret if task.flops >= 100]
        temp = []
        for task in ret:
            from tvm.tir.analysis import estimate_tir_flops
            # assert len(task.dispatched) == 1
            flops = estimate_tir_flops(task.dispatched[0])
            if flops >= 100:
                temp.append(task)
        ret = temp
    return ret


def extracted_tasks_to_tune_contexts(
    extracted_tasks: List[ExtractedTask],
    work_dir: str,
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
    mask_mode: Optional[str] = None,
    database="json",
    module_equality="ignore-ndarray",
) -> Tuple[List[TuneContext], List[float]]:
    """Convert ExtractedTask to TuneContext.

    Parameters
    ----------
    tasks : List[ExtractedTask]
        The tasks to be converted
    work_dir : str
        The working directory to store logs and databases
    space : SpaceGenerator.SpaceGeneratorType
        The space generator to use.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy to use.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The random seed to use.
    mask_mode : Optional[str]
        TODO
    database : Optional[str]
        TODO
    module_equality : Optional[str]
        TODO

    Returns
    -------
    tasks : List[TuneContext]
        The converted tasks
    task_weights : List[float]
        The weights of the tasks
    """

    def split_tasks_per_space(tasks, task_weights, mask_mode: str, database="json", module_equality="ignore-ndarray"):
        print("split_tasks_per_space", tasks, task_weights, mask_mode)
        assert mask_mode in ["split", "union", "all"]
        # if mask_mode == "split":
        # if mask_mode == "union":
        # if mask_mode == "all":
        all_ctx_kwargs = []
        all_task_names = []
        for i, task in enumerate(tasks):
            weight = task_weights[i]

            # space_generator2space_idxs = None
            spaces_ = []
            if mask_mode == "union":
                from .space_generator import SpaceGeneratorUnion

                if isinstance(task.space_generator, SpaceGeneratorUnion):

                    spaces_ = []
                    for k, space_generator in enumerate(task.space_generator.space_generators):
                        spaces__ = task.space_generator.generate_design_space(task.mod)
                        space_idxs = []
                        for m, space in enumerate(spaces__):
                            space_idx = len(spaces_)
                            spaces_.append(space)
                            space_idxs.append(space_idx)
                    space_groups = space_idxs
                    print("space_groups", space_groups)
                else:
                    spaces_ = task.space_generator.generate_design_space(task.mod)
                    mask_mode = "all"
            else:
                spaces_ = task.space_generator.generate_design_space(task.mod)
            print("spaces_", spaces_)

            num_spaces = len(spaces_)
            if mask_mode == "all":
                space_groups = [[idx for idx in range(num_spaces)]]
                assert len(space_groups) == 1
            elif mask_mode == "split":
                space_groups = [[idx] for idx in range(num_spaces)]
                assert len(space_groups) == num_spaces
            else:
                assert num_spaces == sum(len(idxs) for idxs in space_groups)

            print("space_groups", space_groups)

            for j, space_idxs in enumerate(space_groups):
                # print("i,j", i, j)
                # print("space", space, dir(space))
                # print("space.mod", space.mod)
                # print("space.trace", space.trace)
                # input("%%%")
                mask = [0] * num_spaces
                for space_idx in space_idxs:
                    mask[space_idx] = 1
                # print("mask", mask)
                # group = f"T{i}_M{j}"
                group = task.group
                if group:
                    group = task.group
                else:
                    group = f"T{i}"
                # new_task_name = f"{task.task_name}_{group}"
                new_task_name = f"{task.task_name}_M{j}"
                all_task_names.append(new_task_name)
                # new_rand_state = fork_seed(seed, n=1)[0]
                # print("new_task_name", new_task_name)
                work_dir_ = f"{work_dir}/{group}_M{j}"
                from pathlib import Path

                Path(work_dir_).mkdir(exist_ok=True)
                if database is None:
                    database = "memory"
                if isinstance(database, Database):
                    if isinstance(database, JSONDatabase):
                        database = "json"
                    else:
                        assert isinstance(database, MemoryDatabase)
                        database = "memory"
                assert isinstance(database, str)
                if database == "json":
                    database = Database.create(database, work_dir=work_dir_, module_equality=module_equality)
                else:
                    assert database == "memory"
                    database = Database.create(database, module_equality=module_equality)
                ctx_kwargs = dict(
                    mod=task.mod,
                    target=task.target,
                    space_generator=task.space_generator,
                    search_strategy=task.search_strategy,
                    task_name=new_task_name,
                    group=group,
                    # logger=new_logger,
                    # rand_state=new_rand_state,
                    design_spaces_mask=mask,
                    database=database,
                )
                all_ctx_kwargs.append(ctx_kwargs)
                # new_logger = get_loggers_from_work_dir(work_dir, [new_task_name])[0]
                # new_logger.debug("DEBUG")
                # new_logger.info("INFO")
                # print("new_logger", new_logger)
                # print("new_logger.root", new_logger.root)
                # print("new_logger.handlers", new_logger.handlers)
                # print("dir(new_logger)", dir(new_logger))
        ret_tasks = []
        ret_weights = []
        print("all_ctx_kwargs", all_ctx_kwargs)
        print("all_task_names", all_task_names)
        print("all_task_names", all_task_names)
        for ctx_kwargs, logger, rand_state in zip(
            all_ctx_kwargs,
            get_loggers_from_work_dir(work_dir, all_task_names),
            fork_seed(seed, n=len(all_task_names)),
        ):
            ctx_kwargs["logger"] = logger
            ctx_kwargs["rand_state"] = rand_state
            new_task = TuneContext(**ctx_kwargs)
            new_task = new_task.clone()
            ret_tasks.append(new_task)
            ret_weights.append(weight)
        return ret_tasks, ret_weights

    tasks: List[TuneContext] = []
    task_weights: List[float] = []
    for task, logger, rand_state in zip(
        extracted_tasks,
        get_loggers_from_work_dir(work_dir, [t.task_name for t in extracted_tasks]),
        fork_seed(seed, n=len(extracted_tasks)),
    ):
        # TODO: multi-dispatch?
        multi_dispatch = True
        dispatched = task.dispatched
        assert len(dispatched) >= 1
        if not multi_dispatch:
            dispatched = dispatched[:1]
        for d, disp in enumerate(dispatched):
            if multi_dispatch:
                task_name = f"{task.task_name}_D{d}"
                task_idx = len(tasks)
                group = f"T{task_idx}"
            else:
                task_name = task.task_name

            print("disp", disp)

            tasks.append(
                TuneContext(
                    mod=disp,
                    target=task.target,
                    space_generator=space,
                    search_strategy=strategy,
                    task_name=task_name,
                    group=group,
                    logger=logger,
                    rand_state=rand_state,
                    num_threads=num_tuning_cores,
                ).clone()
            )
            task_weights.append(task.weight)
    if mask_mode is not None:
        tasks_per_space, task_weights_per_space = split_tasks_per_space(
            tasks, task_weights, mask_mode=mask_mode, database=database, module_equality=module_equality
        )
        tasks = tasks_per_space
        task_weights = task_weights_per_space
        print("tasks", tasks)
        print("task_weights", task_weights)
        # input("!!!")
    return tasks, task_weights


def tune_relay(
    mod: IRModule,
    params: Dict[str, NDArray],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    *,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    seed: Optional[int] = None,
    module_equality: str = "structural",
    num_tuning_cores: Union[Literal["physical", "logical"], int] = "physical",
    disabled_pass: Optional[Union[List[str], Set[str], Tuple[str]]] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
    opt_level: int = 3,
    pass_config: Mapping[str, Any] = MappingProxyType({}),
) -> Database:
    """Tune a Relay program.

    Parameters
    ----------
    mod : Union[IRModule, tir.PrimFunc]
        The module or function to tune
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    target : Union[Target, str]
        The compilation target
    work_dir : str
        The working directory to store the tuning records
    max_trials_global : int
        The maximum number of trials to run
    max_trials_per_task : Optional[int]
        The maximum number of trials to run for each task
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : BuilderType
        The builder to use
    runner : RunnerType
        The runner to use
    database : DatabaseType
        The database to use
    cost_model : CostModelType
        The cost model to use
    measure_callbacks : CallbackListType
        The measure callbacks to use
    task_scheduler : TaskSchedulerType
        The task scheduler to use
    space : SpaceGeneratorType
        The space generator to use
    strategy : SearchStrategyType
        The search strategy to use
    seed : Optional[int]
        The random seed
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-ndarray" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tir/analysis/analysis.py.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    opt_level : int
        The optimization level of the compilation
    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of disabled passes during tasks extraction
    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.

    Returns
    -------
    database : Database
        The database that contains the tuning records
    """
    tasks, task_weights = extracted_tasks_to_tune_contexts(
        extracted_tasks=extract_tasks(
            mod,
            target,
            params,
            opt_level=opt_level,
            module_equality=module_equality,
            pass_config=pass_config,
            disabled_pass=disabled_pass,
            instruments=instruments,
        ),
        work_dir=work_dir,
        space=space,
        strategy=strategy,
        seed=seed,
        num_tuning_cores=num_tuning_cores,
    )
    pass_config = dict(pass_config)
    with transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
        instruments=instruments,
    ):
        return tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=max_trials_global,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            builder=builder,
            runner=runner,
            database=database,
            cost_model=cost_model,
            measure_callbacks=measure_callbacks,
            task_scheduler=task_scheduler,
            module_equality=module_equality,
        )


def compile_relay(
    database: Database,
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    *,
    backend: Literal["graph", "vm"] = "graph",
    opt_level: int = 3,
    pass_config: Mapping[str, Any] = MappingProxyType(
        {
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "default",
        }
    ),
    executor: Optional["relay.backend.Executor"] = None,
    disabled_pass: Optional[Union[List[str], Set[str], Tuple[str]]] = None,
    runtime: Optional["relay.backend.Runtime"] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
):
    """Compile a relay program with a MetaSchedule database.

    Parameters
    ----------
    database : Database
        The database to use
    mod : IRModule
        The Relay program to be compiled
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    backend : str
        The backend to use. Builtin backends:
            - "graph"
            - "vm"
    opt_level : int
        The optimization level of the compilation
    pass_config : Mapping[str, Any]
        The pass configuration
    executor : Optional[relay.backend.Executor]
        The executor to use in relay.build. It is not supported by RelayVM.
    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of disabled passes
    runtime : Optional[relay.backend.Runtime]
        The runtime to use in relay.build. It is not supported by RelayVM.
    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.

    Returns
    -------
    lib : Union[Module, tvm.runtime.vm.Executable]
        The built runtime module or vm Executable for the given relay workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    # pylint: enable=import-outside-toplevel
    mod, target, params, pass_config, executor, runtime = _normalize_params(
        mod, target, params, pass_config, executor, runtime
    )
    if database is None:
        database = MemoryDatabase()
    else:
        pass_config.setdefault("relay.backend.use_meta_schedule_dispatch", True)
    with Profiler.timeit("PostTuningCompilation"):
        with target, _autotvm_silencer(), database:
            with transform.PassContext(
                opt_level=opt_level,
                config=pass_config,
                disabled_pass=disabled_pass,
                instruments=instruments,
            ):
                if backend == "graph":
                    return relay.build(
                        mod, target=target, params=params, executor=executor, runtime=runtime
                    )
                elif backend == "vm":
                    return relay.vm.compile(mod, target=target, params=params)
                else:
                    raise ValueError(f"Unknown backend: {backend}")


def is_meta_schedule_enabled() -> bool:
    """Return whether the meta-schedule is enabled.

    Returns
    -------
    enabled: bool
        Whether the meta schedule is enabled
    """
    return transform.PassContext.current().config.get(
        "relay.backend.use_meta_schedule",
        False,
    )
