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

from process_experiment import build_rules_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file")
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--hist-out", default=None)
    parser.add_argument("--agg-hist-out", default=None)
    parser.add_argument("--history-out", default=None)
    parser.add_argument("--rules-dir", default=None)
    parser.add_argument("--spaces-dir", default=None)
    parser.add_argument("--intrins", default=None)
    parser.add_argument("--fold-order", action="store_true")
    args = parser.parse_args()
    pkl_file = Path(args.pkl_file)
    assert pkl_file.is_file()
    
    with open(pkl_file, "rb") as f:
        G = pickle.load(f)
    print("G", G)
    
    if args.rules_dir is None:
        # space_dir = pkl_file.parent
        # assert space_dir.is_dir()
        # spaces_dir = space_dir.parent
        # assert spaces_dir.is_dir()
        # task_dir = spaces_dir.parent
        task_dir = pkl_file.parent
        assert task_dir.is_dir()
        tasks_dir = task_dir.parent
        assert tasks_dir.is_dir()
        mod_dir = tasks_dir.parent
        assert mod_dir.is_dir()
        mods_dir = mod_dir.parent
        print("mods_dir", mods_dir)
        assert mods_dir.is_dir()
        exp_dir = mods_dir.parent
        assert exp_dir.is_dir()
        rules_dir = exp_dir / "rules"
        assert rules_dir.is_dir()
    else:
        rules_dir = args.rules_dir
    rules_df = build_rules_df(rules_dir)
    print("rules_df", rules_df)

    if args.spaces_dir is None:
        task_dir = pkl_file.parent
        assert task_dir.is_dir()
        spaces_dir = task_dir / "space"
    else:
        spaces_dir = Path(args.spaces_dir)
    assert spaces_dir.is_dir()
    if args.intrins is not None:
        keep_intrins = args.intrins.split(",")
        print("keep_intrins", keep_intrins)
        assert len(keep_intrins) > 0
        keep_nodes = set()
        for intrin in keep_intrins:
            intrin_rows = rules_df[rules_df["intrin"] == intrin]
            print("intrin_rows", intrin_rows)
            intrin_space_ids = list(map(int, intrin_rows["rule_id"].astype(int).values))
            print("intrin_space_ids", intrin_space_ids)
            keep_nodes.update(intrin_space_ids)
        drop_nodes = set(list(G.nodes)) - keep_nodes
        print("drop_nodes", drop_nodes)
        if len(drop_nodes) > 0:
            print(f"Dropping {len(drop_nodes)} nodes...")
            G.remove_nodes_from(drop_nodes)
    # input("!!!")
    used_space_ids = list(G.nodes)
    print("used_space_ids", used_space_ids, len(used_space_ids))
    space2shashs = {}
    unique_shashs = set()
    for space_id in used_space_ids:
        space_dir = spaces_dir / str(space_id)
        assert space_dir.is_dir()
        shashs_txt = space_dir / "shashs.txt"
        assert shashs_txt.is_file()
        with open(shashs_txt, "r") as f:
            shashs = set(list(map(lambda x: x.strip(), f.readlines())))
        space2shashs[space_id] = shashs
        unique_shashs.update(shashs)
    num_unique_shashs = len(unique_shashs)
    print("num_unique_shashs", num_unique_shashs)

    THRESHOLD = 0.98
    def jaccard(a, b):
        return len(a & b) / len(a | b)
    keep = []
    seen = []

    for idx, s in space2shashs.items():
        redundant = False
        for prev in seen:
            if jaccard(s, prev) >= THRESHOLD:
                redundant = True
                break
        if not redundant:
            keep.append(idx)
            seen.append(s)


    print("keep", keep, len(keep))
    # print("seen", seen, len(seen))
    drop_nodes = set(list(G.nodes)) - set(keep)
    print("drop_nodes", drop_nodes)
    if len(drop_nodes) > 0:
        print(f"Dropping {len(drop_nodes)} nodes...")
        G.remove_nodes_from(drop_nodes)
    input("!!")


    
    G_reduced2 = nx.transitive_reduction(G)
    G_reduced = G.copy()
    to_drop = []
    for edge in G_reduced.edges:
        u, v = edge
        if edge not in G_reduced2.edges:
            to_drop.append(edge)
    for edge in to_drop:
        u, v = edge
        G_reduced.remove_edge(u, v)
    # print("G_reduced2", G_reduced2)
    
    redundant_nodes = [
        n
        for n, attrs in G_reduced.nodes(data=True)
        if attrs["redundant"]
    ]
    for n in redundant_nodes:
        G_reduced.remove_node(n)
    
    
    
    print("G_reduced", G_reduced)
    
    # FOLD = True
    FOLD = False
    assert FOLD is False, "Fold currently unsupported"
    
    if FOLD:
        def fold_small_spaces(G, min_size):
            """
            Fold chains of small spaces into groups.
        
            Returns:
                groups: list[list[node]]
                representative_of: dict[node -> representative]
            """
            # print("fold_small_spaces", G, min_size)
        
            G_gen = G.reverse(copy=True)
            # print("G_gen", G_gen)
        
            visited = set()
        
            groups = []
            representative_of = {}
        
            roots = [n for n in G_gen.nodes if G_gen.in_degree(n) == 0]
            # print("roots", roots)
        
            for root in roots:
                # print("root", root)
                cur = root
        
                group = []
        
                while True:
                    # print("loop")
                    # print("cur", cur)
                    if cur in visited:
                        # print("break")
                        break
        
                    visited.add(cur)
                    group.append(cur)
        
                    size = G.nodes[cur]["size"]
                    # print("size", size)
        
                    succs = list(G_gen.successors(cur))
                    # print("succs", succs, len(succs))
        
                    # stop conditions
                    stop = False
        
                    if size >= min_size:
                        # print("size >= min_size")
                        stop = True
        
                    elif len(succs) > 1:
                        # print("multiple succs")
                        stop = True
                    elif len(succs) == 0:
                        # print("no succs")
                        stop = True
        
                    # print("stop", stop)
        
                    if stop:
                        rep = cur
        
                        groups.append(group)
        
                        for n in group:
                            representative_of[n] = rep
        
                        # continue traversal upward if possible
                        if len(succs) == 1:
                            cur = succs[0]
                            group = []
                            continue
        
                        break
        
                    cur = succs[0]
        
            return groups, representative_of
        
        
        MIN_SIZE = 100
        
        groups, representative_of = fold_small_spaces(G_reduced, MIN_SIZE)
        print("groups", groups)
        print("representative_of", representative_of)
        # input("!")
        
        
        def build_contracted_graph(G, representative_of):
            """
            Build a quotient graph where nodes are representatives of folded groups.
            """
        
            H = nx.DiGraph()
        
            # invert mapping: rep -> members
            groups = defaultdict(list)
            for node, rep in representative_of.items():
                groups[rep].append(node)
        
            # add nodes with aggregated attributes
            for rep, members in groups.items():
                data = G.nodes[rep]
                H.add_node(
                    rep,
                    **data,
                    # size=sum(G.nodes[n]["size"] for n in members),
                    # num_nodes=len(members),
                    # members=members,
                )
        
            # add edges (collapsed)
            for u, v, data in G.edges(data=True):
                ru = representative_of.get(u)
                rv = representative_of.get(v)
        
                if ru is None or rv is None:
                    continue
        
                if ru == rv:
                    continue  # internal edge inside collapsed group
        
                # H.add_edge(ru, rv)
                H.add_edge(ru, rv, **data)
        
            # optional: remove duplicates (nx already handles, but safe)
            H = nx.DiGraph(H)
        
            return H
        
        G_small = build_contracted_graph(G_reduced, representative_of)
        print("G_small", G_small)
        G_reduced = G_small
    
    G_gen = G_reduced.reverse(copy=True)
    
    print("G_gen", G_gen)
    print("G_gen.nodes", G_gen.nodes)
    print("G_gen.edges", G_gen.edges)
    
    G_cur = G_gen.copy()
    
    
    def get_available(G):
        available = [
            n for n in G.nodes
            if G.in_degree(n) == 0
        ]
        return available
    
    
    available = get_available(G_cur)
    
    print("available", available, len(available))
    
    def pick_best(available, G, strategy="random", current_shashs=None):
        print("pick_best", available, strategy)
        if strategy == "random":
            best = random.choice(available)
        elif strategy == "smallest":
            sorted_ids = list(sorted(available, key=lambda n: G.nodes[n]["size"]))
            best = sorted_ids[0]
        elif strategy == "min_delta":
            assert current_shashs is not None
            space2delta = {space_id: len(space2shashs[space_id] - current_shashs) for space_id in available}
            print("space2delta", space2delta)
            sorted_ids = list(sorted(available, key=lambda n: space2delta[n]))
            print("sorted_ids", sorted_ids)
            sorted_deltas = [space2delta[idx] for idx in sorted_ids]
            print("sorted_deltas", sorted_deltas)
            
            # input("!")
            best = sorted_ids[0]
        else:
            raise NotImplementedError(f"strategy={strategy}")
        # input("1")
        return best
    
    order = []
    sizes = []
    current_shashs = set()
    
    # strategy = "smallest"
    strategy = "min_delta"
    
    while available:
        best = pick_best(available, G_cur, strategy=strategy, current_shashs=current_shashs)
        print("best", best)
    
        order.append(best)
        space_id = best
        new_shashs = space2shashs[space_id]
        current_shashs.update(new_shashs)
        new_size = len(current_shashs)
        print("new_size", new_size)
        sizes.append(int(new_size))
        print("sizes", sizes)
        # if len(sizes) == 0:
        #     new_size = G.nodes[best]["size"]
        #     sizes.append(int(new_size))
        # else:
        #     generated = set(order)
        #     # in_edges = list(G_gen.in_edges(best))
        #     in_edges = [
        #         e for e in G_gen.in_edges(best)
        #         if e[0] in generated
        #     ]
        #     print("in_edges", in_edges)
        #     # out_edges = list(G_gen.out_edges(best))
        #     # print("out_edges", out_edges)
        #     old_size = sizes[-1]
        #     print("old_size", old_size)
    
        #     if len(in_edges) == 0:
        #         added_size = G_gen.nodes[best]["size"]
        #         print("added_size", added_size)
        #         new_size = old_size + added_size
        #         print("new_size", new_size)
        #     elif len(in_edges) == 1:
        #         in_edge = in_edges[0]
        #         print("in_edge", in_edge)
        #         smaller = in_edge[0]
        #         print("smaller", smaller)
        #         edge = G_gen.edges[in_edge]
        #         print("edge", edge)
        #         containment_ratio = edge["containment_ratio"]
        #         print("containment_ratio", containment_ratio)
        #         sz = G_gen.nodes[best]["size"]
        #         print("sz", sz)
        #         added_size = round(sz * (1 - containment_ratio), 2)
        #         print("added_size", added_size)
        #         # assert int(added_size) == added_size
        #         added_size = int(added_size)
        #         new_size = old_size + added_size
        #         print("new_size", new_size)
        #     elif len(in_edges) > 0:
        #         raise NotImplementedError
        #     sizes.append(int(new_size))
    
    
        G_cur.remove_node(best)
    
        available = get_available(G_cur)
        print("available", available, len(available))
    
    print("order", order)
    print("sizes", sizes)
    # sizes2 = [int(G.nodes[n]["size"]) for n in order]
    # print("sizes2", sizes2)
    
    active = set()
    history = []
    
    for n in order:
        # all smaller spaces contained in n
        contained = nx.descendants(G, n)
    
        # remove contained active spaces
        active -= contained
    
        # add new maximal space
        active.add(n)
    
        # snapshot
        history.append(set(active))

    
    print("history =", history)

    # FOLD_ORDER = True
    FOLD_ORDER = args.fold_order
    if FOLD_ORDER:
        FOLD_ORDER_MIN_DELTA = 20
        FOLD_ORDER_MIN_DELTA_REL = 0.1
        folded_order = []
        folded_history = []
        folded_sizes = []
        new_space_ids = set()
        for i, space_id in enumerate(order):
            print("i", i)
            print("space_id", space_id)
            new_space_ids.add(space_id)
            old_size = folded_sizes[-1] if len(folded_sizes) > 0 else 0
            print("old_size", old_size)
            new_size = sizes[i]
            print("new_size", new_size)
            delta_size = new_size - old_size
            print("delta_size", delta_size)
            delta_size_rel = delta_size / old_size if old_size > 0 else None
            print("delta_size_rel", delta_size_rel)
            if (delta_size_rel is None or delta_size_rel >= FOLD_ORDER_MIN_DELTA_REL) and (delta_size >= FOLD_ORDER_MIN_DELTA):
                print("> delta")
                folded_order.append(new_space_ids)
                old_active = folded_history[-1] if len(folded_history) > 0 else set()
                all_active = new_space_ids | old_active
                folded_history.append(all_active)
                folded_sizes.append(new_size)
                new_space_ids = set()
        if len(new_space_ids) > 0:
            print("tail")
            folded_order.append(new_space_ids)
            old_active = folded_history[-1] if len(folded_history) > 0 else set()
            all_active = new_space_ids | old_active
            folded_history.append(all_active)
            folded_sizes.append(new_size)
            # new_space_ids = set()


            # input(".")
        print("folded order =", folded_order)
        print("folded history =", folded_history)
        print("folded sizes =", folded_sizes)
        order = folded_order
        history = folded_history
        sizes = folded_sizes
    else:
        order = [{x} for x in order]

    # input("!")
    all_features = set()
    features_hist = []
    # for i in range(len(order)):
    for i, current in enumerate(order):
        print("step", i)
        # space_id = order[i]
        # print("space_id", space_id)
        active_sets = sorted(list(history[i]))
        # active_sets = sorted(list(active))
        print("active_sets", active_sets)
        DELTA_MODE = True
        # DELTA_MODE = False
        # assert DELTA_MODE is False, "DELTA_MODE currently unsupported"
        if DELTA_MODE:
            active_sets = current
        print("active_sets", active_sets)
        features = []
        for space_id_ in active_sets:
            rule_rows = rules_df[rules_df["rule_id"] == space_id_]
            print("rule_rows", rule_rows)
            assert len(rule_rows) == 1
            rule_row = rule_rows.iloc[0]
            print("rule_row", rule_row, dir(rule_row))
            for key, val in rule_row.items():
                if key == "rule_id":
                    continue
                key_lower = key.lower()
                print("key_lower", key_lower)
                # print("val", val, type(val))
                if isinstance(val, np.float64):
                    val = float(val)
                if isinstance(val, np.int64):
                    val = int(val)
                if isinstance(val, np.bool):
                    val = bool(val)
                if isinstance(val, (list, tuple)):
                    val = list(val)
                    for item in val:
                        item_lower = str(item).lower()
                        key2 = f"{key_lower}_{item_lower}"
                        feature = f"{key2}"
                        features.append(feature)
                elif pd.isna(val) or val is None:
                    continue
                elif isinstance(val, bool):
                    if val:
                        feature = key_lower
                        features.append(feature)
                else:
                    assert isinstance(val, (int, float, str)), f"Got: {type(val)}"
                    if isinstance(val, float):
                        if int(val) == val:
                            val = int(val)
                    feature = f"{key_lower}={val}"
                    features.append(feature)

        print("features", features)
        all_features.update(set(features))
        features_hist.append(features)
    print("features_hist", features_hist)
    print("all_features", all_features)
    def natural_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]
    sorted_features = sorted(list(all_features), key=natural_key)
    print("sorted_features", sorted_features)
    # num_steps = len(order)
    num_steps = len(history)
    print("num_steps", num_steps)
    num_features = len(all_features)
    features_matrix = np.zeros((num_features, num_steps), dtype=bool)
    agg_features_matrix = np.zeros((num_features, num_steps), dtype=int)
    for i, features in enumerate(features_hist):
        if i > 0:
            agg_features_matrix[:, i] = agg_features_matrix[:, i-1]
        for feature in features:
            feature_idx = sorted_features.index(feature)
            features_matrix[feature_idx, i] = True
            agg_features_matrix[feature_idx, i] += 1
    print("features_matrix", features_matrix, features_matrix.dtype, features_matrix.shape)
    print("agg_features_matrix", agg_features_matrix, agg_features_matrix.dtype, agg_features_matrix.shape)

    def plot_features_hist(features_matrix, sorted_features, hist_out_file):
        z = features_matrix.astype(int)
        iterations = list(range(z.shape[1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=iterations,
                y=sorted_features,
                colorscale=[
                    [0.0, "white"],
                    [1.0, "darkblue"],
                ],
                showscale=False,
                hoverongaps=False,
                text=z,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title="Active Features per Iteration",
            xaxis_title="Iteration",
            yaxis_title="Feature",
            width=1800,
            height=max(600, 25 * len(sorted_features)),
        )
        # fig.update_yaxes(autorange="reversed")
        fig.write_html(hist_out_file)

    assert args.hist_out is not None
    hist_out_file = Path(args.hist_out)
    plot_features_hist(features_matrix, sorted_features, hist_out_file)
    assert args.agg_hist_out is not None
    agg_hist_out_file = Path(args.agg_hist_out)
    plot_features_hist(agg_features_matrix, sorted_features, agg_hist_out_file)
    # input("!")


    def plot_sequence(order, sizes, sets, out):
        # Compute incremental growth
        incremental = [sizes[0]]
        for i in range(1, len(sizes)):
            incremental.append(sizes[i] - sizes[i - 1])
    
        # Build hover labels
        hover_text = []
        for i in range(len(order)):
            txt = (
                f"step={i}<br>"
                f"space_ids={order[i]}<br>"
                f"total_size={sizes[i]}<br>"
                f"increment={incremental[i]}<br>"
                f"active_sets={sorted(list(sets[i]))}"
            )
            hover_text.append(txt)
        
        # Create figure
        fig = go.Figure()
        
        # Total explored space
        fig.add_trace(
            go.Scatter(
                x=list(range(len(order))),
                y=sizes,
                mode="lines+markers",
                name="Total explored space",
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
        
        # Incremental additions
        fig.add_trace(
            go.Bar(
                x=list(range(len(order))),
                y=incremental,
                name="Incremental growth",
                opacity=0.4,
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
        
        # Layout
        fig.update_layout(
            title="Explored Search Space Over Time",
            xaxis_title="Generation Step",
            yaxis_title="Search Space Size",
            hovermode="closest",
            template="plotly_white",
        )
        
        # Write HTML
        fig.write_html(out)
    
        print(f"Wrote {out}")
    
    plot_out_file = args.out
    plot_sequence(order, sizes, history, plot_out_file)
    def space2rule_data(rules_df, space_id):
        rule_rows = rules_df[rules_df["rule_id"] == space_id]
        print("rule_rows", rule_rows)
        assert len(rule_rows) == 1
        rule_row = rule_rows.iloc[0]
        print("rule_row", rule_row)
        rule_data = rule_row.to_dict()
        def fix_val(val):
            if isinstance(val, set):
                val = sorted(list(val))
            if isinstance(val, tuple):
                val = list(val)
            if isinstance(val, list):
                val = [fix_val(x) for x in val]
            elif pd.isna(val):
                val = None
            return val
        rule_data = {key: fix_val(val) for key, val in rule_data.items()}
        print("rule_data", rule_data)
        return rule_data
    print("args.history_out", args.history_out)
    if args.history_out is not None:
        rules_data = []
        used_spaces = set.union(*order)
        print("used_spaces", used_spaces, len(used_spaces))
        for space_id in used_spaces:
            rule_data = space2rule_data(rules_df, space_id)
            rules_data.append(rule_data)
        steps_data = []
        for i, current in enumerate(order):
            print("step", i)
            current = sorted(list(current))
            print("current", current)
            active_sets = sorted(list(history[i]))
            print("active_sets", active_sets)
            to_drop = []  # TODO: identify dropped rules?
            prev_size = sizes[i-1] if i > 0 else 0
            total_size = sizes[i]
            increment = total_size - prev_size
            n_active = len(active_sets)
            n_add = len(current)
            n_drop = len(to_drop)
            step_metrics = {"total_size": total_size, "increment": increment, "n_active": n_active, "n_add": n_add, "n_drop": n_drop}
            step_data = {"step": i, "add": current, "drop": to_drop, "active": active_sets, "metrics": step_metrics}
            steps_data.append(step_data)
        yaml_data = {
            "rules": rules_data,
            "steps": steps_data,
        }
        print("yaml_data", yaml_data)
        with open(args.history_out, "w") as f:
            yaml.dump(yaml_data, f)


if __name__ == "__main__":
    with pd.option_context(
        'display.max_rows', 20,
        'display.max_columns', None,
        'display.precision', 3,
    ):
        main()
