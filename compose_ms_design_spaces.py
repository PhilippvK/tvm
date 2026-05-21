import re
import sys
import yaml
import pickle
import random
import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_yaml")
    # parser.add_argument("--out", "-o", default=None)
    args = parser.parse_args()

    history_yaml = Path(args.history_yaml)
    assert history_yaml.is_file()
    with open(history_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    print("yaml_data", yaml_data)

    rules_data = yaml_data["rules"]
    steps_data = yaml_data["steps"]

    rule2space = {}

    for rule_data in rules_data:
        rule_id = rule_data.pop("rule_id")
        print("rule_id", rule_id)
        rule_kwargs = rule_data
        print("rule_kwargs", rule_kwargs)
        sch_rules = generate_rules(**rule_kwargs)
        postprocs = "from-target"
        mutator_probs = "from-target"
        print("sch_rules", sch_rules, len(sch_rules))
        space = ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=postprocs,
            mutator_probs=mutator_probs,
        )
        print("space", space)
        rule2space[rule_id] = space
        # input("!!!")

    step_design_spaces = {}
    step_union_design_space = {}

    for step_data in steps_data:
        print("step_data", step_data)
        step_id = step_data["step"]
        added_rules = step_data["add"]
        assert len(added_rules) > 0
        dropped_rules = step_data["drop"]
        assert len(dropped_rules) == 0
        active_rules = step_data["active"]
        assert len(active_rules) > 0
        cur_design_spaces = [rule2space[rule_id] for rule_id in active_rules]
        print("cur_design_spaces", cur_design_spaces)
        step_design_spaces[step_id] = cur_design_spaces
        union_design_space = ms.space_generator.SpaceGeneratorUnion(cur_design_spaces)
        print("union_design_space", union_design_space)
        step_union_design_space[step_id] = union_design_space
        # input("!!!2")

    # test
    space_generator = union_design_space
    print("space_generator", space_generator)
    mod, params = get_conv2d_relay_module(h=32, w=32, kw=3, kh=3, cin=16, cout=16, dtype="int8", data_layout="NHWC", kernel_layout="HWOI")
    print("mod", mod)
    num_trials_per_iter = 7
    max_trials_per_task = 20

    strategy_kwargs = dict(
        # population_size=512,
        # population_size=128,
        # population_size=1024,
        # population_size=1024 * 16 * 16,
        # population_size=1024 * 16,
        # population_size=1024 * 16 * 2,
        population_size=1024 * 16 // 2,
        # population_size=16,
        init_measured_ratio=0.2,
        # init_measured_ratio=0.9,
        init_min_unmeasured=50,
        # init_min_unmeasured=10,
        # max_fail_count=5,
        max_fail_count=1,
        # genetic_num_iters=4,
        genetic_num_iters=6,
        genetic_mutate_prob=0.85,
        # genetic_max_fail_count=10,
        genetic_max_fail_count=5,
        # eps_greedy=0.05,
        eps_greedy=0.25,
    )
    strategy = ms.search_strategy.EvolutionarySearch(
        **strategy_kwargs,
    )
    print("strategy", strategy)
    target = tvm.target.Target("llvm -num-cores=1")
    context = ms.TuneContext(
        target=target,
        mod=mod,
        space_generator=space_generator,
        search_strategy=strategy,
    )
    print("context", context)
    spaces = context.space_generator.generate_design_space(context.mod)
    print("spaces", spaces, len(spaces))
    database = ms.database.MemoryDatabase()
    print("database", database)
    cost_model = ms.cost_model.XGBModel()
    print("cost_model", cost_model)
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=spaces,
        database=database,
        cost_model=cost_model,
    )
    candidates = strategy.generate_measure_candidates()
    print("candidates", candidates, len(candidates))


if __name__ == "__main__":
    with pd.option_context(
        'display.max_rows', 20,
        'display.max_columns', None,
        'display.precision', 3,
    ):
        main()
