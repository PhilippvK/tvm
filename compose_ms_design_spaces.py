import re
import sys
import yaml
import pickle
import random
import argparse
import tempfile
from pathlib import Path
from types import MappingProxyType
from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule import postproc, schedule_rule

# from process_experiment import build_rules_df
from rule_utils import generate_rules
from mod_utils import get_dense_relay_module, get_conv2d_relay_module
from estimate_utils import estimate_size

import time
from rich import print, box
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.text import Text


def next_power_of_two(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def load_history_yaml(history_yaml):
    history_yaml = Path(history_yaml)
    assert history_yaml.is_file()
    with open(history_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    # print("yaml_data", yaml_data)

    rules_data = yaml_data["rules"]
    steps_data = yaml_data["steps"]
    return rules_data, steps_data


def process_rules_data(rules_data, postprocs="from-target", mutator_probs="from-target"):
    rule2space_generator = {}
    rule_ids = []
    for rule_data in rules_data:
        rule_id = rule_data.pop("rule_id")
        rule_ids.append(rule_id)
        # print("rule_id", rule_id)
        rule_kwargs = rule_data
        # print("rule_kwargs", rule_kwargs)
        sch_rules = generate_rules(**rule_kwargs)
        # print("sch_rules", sch_rules, len(sch_rules))
        space_generator = ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=postprocs,
            mutator_probs=mutator_probs,
        )
        # print("space_generator", space_generator)
        rule2space_generator[rule_id] = space_generator
        # input("!!!")
    return rule2space_generator, rule_ids


def process_steps_data(steps_data, rule2space_generator, rule_ids):
    step_space_generators = {}
    step_union_space_generator = {}
    step_total_size = {}
    step_space_generator_masks = {}
    step_ids = []

    for step_data in steps_data:
        # print("step_data", step_data)
        step_id = step_data["step"]
        step_ids.append(step_id)
        added_rules = step_data["add"]
        assert len(added_rules) > 0
        dropped_rules = step_data["drop"]
        assert len(dropped_rules) == 0
        active_rules = step_data["active"]
        assert len(active_rules) > 0
        metrics = step_data["metrics"]
        total_size = metrics["total_size"]
        step_total_size[step_id] = total_size
        cur_space_generators = [rule2space_generator[rule_id] for rule_id in active_rules]
        # print("cur_space_generators", cur_space_generators)
        space_generator_mask = [1 if rule_id in active_rules else 0 for rule_id in rule_ids]
        # print("space_generator_mask", space_generator_mask)
        step_space_generator_masks[step_id] = space_generator_mask
        step_space_generators[step_id] = cur_space_generators
        union_space_generator = ms.space_generator.SpaceGeneratorUnion(cur_space_generators)
        # print("union_design_space", union_design_space)
        step_union_space_generator[step_id] = union_space_generator
        # input("!!!2")
    return step_space_generators, step_union_space_generator, step_total_size, step_space_generator_masks, step_ids


def run(
    mod,
    params,
    target,
    space_generator,
    num_trials_per_iter = 1_000_000,
    max_trials_per_task = 10_000_000,
    opt_level = 3,
    pass_config = MappingProxyType({}),
    module_equality = "structural",
    disabled_pass = None,
    instruments = None,
    seed = None,
    num_tuning_cores = "physical",
    pop_size = 128,
    min_pop_size = 128,
    step_space_generator_masks = None,
    step_ids = None,
    rule_ids = None,
    step_sizes = None,
):
    strategy_kwargs = dict(
        population_size=pop_size,
        init_measured_ratio=0.2,
        init_min_unmeasured=50,
        max_fail_count=1,
        genetic_num_iters=6,
        genetic_mutate_prob=0.85,
        genetic_max_fail_count=5,
        eps_greedy=0.25,
    )
    strategy = ms.search_strategy.EvolutionarySearch(
        **strategy_kwargs,
    )
    print("strategy", strategy)
    database = ms.database.MemoryDatabase()
    print("database", database)
    cost_model = ms.cost_model.DummyModel()
    print("cost_model", cost_model)
    with tempfile.TemporaryDirectory() as work_dir, ms.Profiler() as profiler:
        extracted_tasks = ms.relay_integration.extract_tasks(
            mod,
            target,
            params,
            opt_level=opt_level,
            module_equality=module_equality,
            pass_config=pass_config,
            disabled_pass=disabled_pass,
            instruments=instruments,
        )
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=extracted_tasks,
            work_dir=work_dir,
            # space=space,
            space=space_generator,
            strategy=strategy,
            seed=seed,
            num_tuning_cores=num_tuning_cores,
        )
        # print("tasks", tasks, len(tasks))
        # print("task_weights", task_weights, len(task_weights))
        num_tasks = len(tasks)
        assert num_tasks > 0
        assert num_tasks == 1
        context = tasks[0]
        # print("context", context)
        step_spaces_masks = None
        assert isinstance(context.space_generator, ms.space_generator.SpaceGeneratorUnion)
        spaces = context.space_generator.generate_design_space(context.mod)
        # print("spaces", spaces, len(spaces))
        num_spaces = len(spaces)
        if step_space_generator_masks is not None:
            space_ids = list(range(num_spaces))
            step_spaces_masks = []
            space_generators = context.space_generator.space_generators
            assert rule_ids is not None
            num_steps = len(step_ids)
            num_rules = len(rule_ids)
            assert len(rule_ids) == len(space_generators)
            # print("rule_ids", rule_ids, len(rule_ids))
            assert step_ids[0] == 0
            assert step_ids[-1] == (num_steps - 1)
            rule2space_idxs = {}
            cur = 0
            for rule_idx, rule_id in enumerate(rule_ids):
                rule_space_generator = space_generators[rule_idx]
                rule_spaces = rule_space_generator.generate_design_space(context.mod)
                num_rule_spaces = len(rule_spaces)
                # print("num_rule_spaces", num_rule_spaces)
                space_idxs = list(range(cur, cur + num_rule_spaces))
                # print("space_idxs", space_idxs)
                rule2space_idxs[rule_id] = space_idxs
                cur += num_rule_spaces
            assert step_ids is not None
            assert len(step_ids) == len(step_space_generator_masks)
            # print("rule2space_idxs", rule2space_idxs)
            for step_id in step_ids:
                step_space_generator_mask = step_space_generator_masks[step_id]
                enabled_space_idxs = []
                for rule_idx, rule_en in enumerate(step_space_generator_mask):
                    # print("rule_idx", rule_idx)
                    rule_id = rule_ids[rule_idx]
                    # print("rule_id", rule_id)
                    # print("rule_en", rule_en)
                    if rule_en:
                        space_idxs = rule2space_idxs[rule_id]
                        enabled_space_idxs += space_idxs
                spaces_mask = [1 if space_id in enabled_space_idxs else 0 for space_id in space_ids]
                step_spaces_masks.append(spaces_mask)
            # print("step_spaces_masks", step_spaces_masks, len(step_spaces_masks))
        # input("!")
        # rule_design_spaces = {rule_id: rule2space[rule_id] for rule_id in rule_ids}
        # count = 0
        # counts = defaultdict(int)
        # for rule_id, design_space_ in rule_design_spaces.items():
        #     print("rule_id", rule_id)
        #     print("design_space_", design_space_)
        #     spaces_ = design_space_.generate_design_space(context.mod)
        #     print("spaces_", spaces_, len(spaces_))
        #     for space_ in spaces_:
        #         print("space_", space_, dir(space_))
        #         print(space_.show())
        #         # space_.show()
        #         # print("mod", space_.mod, dir(space_.mod))
        #         # print("trace", space_.trace, dir(space_.trace))
        #         import hashlib
        #         m = hashlib.sha256()
        #         m.update(str(space_.mod).encode())
        #         mod_hash = m.hexdigest()
        #         m = hashlib.sha256()
        #         m.update(str(space_.trace).encode())
        #         trace_hash = m.hexdigest()
        #         print("mod_hash", mod_hash)
        #         print("trace_hash", trace_hash)
        #         key = (mod_hash, trace_hash)
        #         count += 1
        #         counts[key] += 1
        #         # input("!")
        # print("count", count)
        # print("counts", counts)
        # input("!!!")
        # for ii, space in enumerate(spaces):
        #     print("ii", ii)
        #     print("space", space)
        # cost_model = ms.cost_model.XGBModel()
        strategy.pre_tuning(
            max_trials=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=spaces,
            database=database,
            cost_model=cost_model,
        )
        # if design_spaces_mask is not None:
        #     strategy.mask_design_spaces(design_spaces_mask)
        rel_thr = 0.05
        task_trials = 0
        sizes_hist = []
        # i = 0
        MAX_ITERS = 10
        if step_ids is not None:
            num_steps = len(step_ids)
            assert step_ids[0] == 0
            assert step_ids[-1] == (num_steps - 1)
            iters = list(range(num_steps + MAX_ITERS))
        else:
            iters = list(range(MAX_ITERS))
        print("iters", iters, len(iters))
        for i in iters:
            print("while", i)
            # if i < num_spaces:
            if step_spaces_masks is not None:
                if i < num_steps:
                    new_mask = step_spaces_masks[i]
                    spaces_mask = new_mask
                    assert step_sizes is not None
                    assert len(step_sizes) == num_steps
                    new_pop_size = step_sizes[i]
                else:
                    assert all(spaces_mask)
                    new_pop_size = max(min_pop_size, task_trials * 2)
                print("spaces_mask", spaces_mask)
                strategy.mask_design_spaces(spaces_mask)
            else:
                new_pop_size = max(min_pop_size, task_trials * 2)
            print("new_pop_size", new_pop_size)
            strategy.update_population_size(new_pop_size)
            candidates = strategy.generate_measure_candidates()
            # print("candidates", candidates, len(candidates))
            print("len(candidates)", len(candidates))
            # input("!!!")
            # print("context.mod", context.mod)
            workload = database.commit_workload(context.mod)
            # print("workload", workload)
            # if not database.has_workload(context.mod):
            #     print("commit_workload")
            #     workload = database.commit_workload(context.mod)
            num_candidates = len(candidates)
            sizes_hist.append(num_candidates)
            if task_trials > 0:
                num_candidates_new_rel = num_candidates / task_trials;
                print("num_candidates_new_rel", num_candidates_new_rel)
                task_trials += num_candidates
                if (step_spaces_masks is None or i >= num_steps) and (num_candidates_new_rel <= rel_thr):
                    print("break (rel_thr)")
                    break
            else:
                task_trials = num_candidates
            for candidate in candidates:
                # print("candidate", candidate, dir(candidate))
                sch = candidate.sch
                # print("sch", sch, dir(sch))
                # print("sch.trace", sch.trace)
                record = ms.database.TuningRecord(
                    # _create_schedule(mod, _schedule_matmul).trace,
                    sch.trace,
                    workload,
                    [1.0],
                    target,
                    ms.arg_info.ArgInfo.from_prim_func(func=context.mod["main"]),
                )
                # print("record", record)
                # print("commit_record")
                database.commit_tuning_record(record)
        print("task_trials", task_trials)
        print("sizes_hist", sizes_hist)
        search_space_size, is_estimate = estimate_size(sizes_hist)
        print("search_space_size", search_space_size)
        print("is_estimate", is_estimate)
        # TODO: pick random db candidates instead of top?


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_yaml")
    # parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--masked", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    rules_data, steps_data = load_history_yaml(args.history_yaml)

    rule2space_generator, rule_ids = process_rules_data(rules_data)
    step_space_generators, step_union_space_generator, step_total_size, step_space_generator_masks, step_ids = process_steps_data(steps_data, rule2space_generator, rule_ids)

    # union_space_generator = step_union_space_generator[last_step]
    # print("union_space_generator", union_space_generator, dir(union_space_generator))
    # print("union_space_generator.space_generators", union_space_generator.space_generators, len(union_space_generator.space_generators))

    print("step_space_generator_masks", step_space_generator_masks)
    all_space_generators = list(rule2space_generator.values())
    union_space_generator = ms.space_generator.SpaceGeneratorUnion(all_space_generators)
    # input("?")

    min_pop_size = 128
    if args.pop_size is not None:
        pop_size = args.pop_size
    else:
        if args.masked:
            step = step_ids[0]
        else:
            step = step_ids[-1]
        print("step", step)
        est_total_size = step_total_size[step]
        print("est_total_size", est_total_size)
        next_pow2_pop_size = next_power_of_two(est_total_size)
        pop_size = max(min_pop_size, next_pow2_pop_size)
    print("pop_size", pop_size)
    step_sizes = list(step_total_size.values())

    space_generator = union_space_generator
    print("space_generator", space_generator)
    mod, params = get_conv2d_relay_module(h=32, w=32, kw=3, kh=3, cin=16, cout=16, dtype="int8", data_layout="NHWC", kernel_layout="HWOI")
    print("mod", mod)
    target = tvm.target.Target("llvm -num-cores=1")

    tuning_table = Table()
    tuning_table.add_column("Task", justify="right", style="cyan", no_wrap=True)
    tuning_table.add_column("Space", justify="right", style="cyan", no_wrap=True)
    tuning_table.add_column("Space Size")
    tuning_table.add_column("Subspaces")
    tuning_table.add_column("Masked Size")
    tuning_table.add_column("Latency (ms)")
    tuning_table.add_column("Performance (GFLOPS)")
    tuning_table.add_column("Trials")
    tuning_table.add_column("Coverage [Masked]")
    tuning_table.add_column("Status")
    tuning_table.add_row("T0", "S0", "~11256", "8/22", "6644 (59%)", "0.2325", "11.65", "105", "1.0% [1.6%]", Text("✓", style="green"))
    tuning_table.add_row("T0", "S1", "~11256", "3/3", "11256 (100%)", "0.2325", "11.65", "105", "1.0% [1.6%]", Spinner("dots", style="orange"))
    tuning_table.add_row("T0", "S2", "~11256", "3/3", "11256 (100%)", "N/A", "N/A", "0", "0.0% [0.0%]", Spinner("clock", style="orange", speed=0.1))
    with Live(Panel(tuning_table, title="Tuning", border_style="blue"), refresh_per_second=5) as live:
        run(
            mod,
            params,
            target,
            space_generator,
            pop_size=pop_size,
            min_pop_size=min_pop_size,
            step_space_generator_masks=step_space_generator_masks if args.masked else None,
            step_ids=step_ids if args.masked else None,
            rule_ids=rule_ids if args.masked else None,
            step_sizes=step_sizes if args.masked else None,
        )


if __name__ == "__main__":
    with pd.option_context(
        'display.max_rows', 20,
        'display.max_columns', None,
        'display.precision', 3,
    ):
        main()
