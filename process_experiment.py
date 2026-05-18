import argparse
from pathlib import Path

import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_rules_df(rules_dir):
    rows = []

    for rule_file in sorted(rules_dir.glob("*.yaml"), key=lambda p: int(p.stem)):
        with open(rule_file, "r") as f:
            data = yaml.safe_load(f)

        data["rule_id"] = int(rule_file.stem)

        # Optional: make rules easier to analyze
        data["rules"] = tuple(data.get("rules", []))

        rows.append(data)

    df = pd.DataFrame(rows)

    # Nice column ordering
    first_cols = ["rule_id", "rules"]
    remaining_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + remaining_cols]

    return df


def try_read_csv(*args, **kwargs):
    try:
        return pd.read_csv(*args, **kwargs)
    except pd.errors.EmptyDataError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="+")
    parser.add_argument("--out", "-o", default=None)
    # TODO: expose out dir
    args = parser.parse_args()
    exp_dirs = args.exp_dir
    merged_dfs = []
    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        exp_name = exp_dir.name
        assert exp_dir.is_dir()
        rules_dir = exp_dir / "rules"
        assert rules_dir.is_dir()
        rules_df = build_rules_df(rules_dir)
        print("rules_df", rules_df)
        rule_ids = list(map(int, list(rules_df["rule_id"].unique())))
        print("rule_ids", rule_ids)

        mods_dir = exp_dir / "mods"
        assert mods_dir.is_dir()
        mod_ids = sorted(map(lambda p: int(p.name), mods_dir.glob("*")))
        print("mod_ids", mod_ids)
        for mod_id in mod_ids:
            print("mod_id", mod_id)
            mod_dir = mods_dir / str(mod_id)
            assert mod_dir.is_dir()
            metrics_csv = mod_dir / "metrics.csv"
            assert metrics_csv.is_file()
            metrics_df = try_read_csv(metrics_csv)
            assert metrics_df is not None
            print("metrics_df", metrics_df)
            summary_csv = mod_dir / "summary.csv"
            assert summary_csv.is_file()
            summary_df = try_read_csv(summary_csv)
            assert summary_df is not None
            print("summary_df", summary_df)
            merged_df = pd.merge(summary_df, rules_df, left_on="space_id", right_on="rule_id", how="left")
            print("merged_df", merged_df)
            # merged_df["exp_name"] = exp_name
            merged_df.insert(0, "exp_name", exp_name)
            tasks_dir = mod_dir / "tasks"
            assert tasks_dir.is_dir()
            task_ids = sorted(map(lambda p: int(p.name), tasks_dir.glob("*")))
            print("task_ids", task_ids)
            shashs_data = []
            for task_id in task_ids:
                print("task_id", task_id)
                task_dir = tasks_dir / str(task_id)
                assert task_dir.is_dir()
                spaces_dir = task_dir / "space"
                assert spaces_dir.is_dir()
                for space_id in rule_ids:
                    space_dir = spaces_dir / str(space_id)
                    if not space_dir.is_dir():
                        continue
                    assert space_dir.is_dir()
                    # assert shashs_txt.is_file()
                    shashs_txt = space_dir / "shashs.txt"
                    if shashs_txt.is_file():
                        with open(shashs_txt, "r") as f:
                            shashs = set(list(map(lambda x: x.strip(), f.readlines())))
                        # print("shashs", shashs, len(shashs))
                        new = {"space_id": space_id, "task_id": task_id, "shashs": shashs}
                        shashs_data.append(new)
                    annotation_hist_csv = space_dir / "annotation_hist.csv"
                    # print("annotation_hist_csv", annotation_hist_csv)
                    if annotation_hist_csv.is_file():
                        assert annotation_hist_csv.is_file()
                        annotation_hist_df = try_read_csv(annotation_hist_csv)
                        annotation_val_hist_csv = space_dir / "annotation_val_hist.csv"
                        assert annotation_val_hist_csv.is_file()
                        annotation_val_hist_df = try_read_csv(annotation_val_hist_csv)
                        inst_hist_csv = space_dir / "inst_hist.csv"
                        assert inst_hist_csv.is_file()
                        inst_hist_df = try_read_csv(inst_hist_csv)
            shashs_df = pd.DataFrame(shashs_data)
            print("shashs_df", shashs_df)
            merged_df = pd.merge(merged_df, shashs_df, on=["space_id", "task_id"], how="left")
            print("merged_df", merged_df)
            # input("!")
            merged_dfs.append(merged_df)
    print("len(merged_dfs)", merged_dfs)
    merged_df = pd.concat(merged_dfs)
    merged_df.reset_index(inplace=True)
    FILTER = False
    if FILTER:
        merged_df = merged_df[merged_df.apply(lambda x: x["intrin"] == "none" or "MultiLevelTiling" not in x["rules"], axis=1)]
    print("merged_df", merged_df)
    # input("!")
    # for group, task_df in merged_df.groupby(["task_name", "task_args", "task_args_hash"]):
    #     task_name, task_args, _ = group
    #     print("task_name", task_name)
    #     print("task_args", task_args)
    #     for intrin, intrin_df in task_df.groupby("intrin"):
    #         print("intrin", intrin)
    #         # print("intrin_df", intrin_df)
    #         search_space_sizes = intrin_df["search_space_size"].values
    #         print("search_space_sizes", search_space_sizes)
    #         mean_search_space_size = intrin_df["search_space_size"].mean()
    #         max_search_space_size = intrin_df["search_space_size"].max()
    #         print("len(intrin_df)", len(intrin_df))
    #         print("mean_search_space_size", mean_search_space_size)
    #         print("max_search_space_size", max_search_space_size)
    if args.out:
        plot_helper(merged_df, args.out)

def plot_helper(merged_df, out):
    # fig = go.Figure()
    fig = make_subplots(
        rows=3,
        cols=1,
        # shared_xaxes=True,
        shared_xaxes=False,
        vertical_spacing=0.12,
        # specs=[
        #     [{}],
        #     [{"secondary_y": True}],
        # ],
        subplot_titles=(
            "Search Space Size Distribution",
            "Number of Spaces",
            "Unique Candidates",
        ),
    )

    from plotly.colors import DEFAULT_PLOTLY_COLORS

    intrins = sorted(merged_df["intrin"].unique())
    intrin_colors = {
        intrin: DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        for i, intrin in enumerate(intrins)
    }

    for group, task_df in merged_df.groupby(["task_name", "task_args", "task_args_hash"]):
        task_name, task_args, _ = group
        print("task_name", task_name)
        print("task_args", task_args)
        workload = f"{task_name}\n{task_args}"
        print("workload", workload)
        # keep track of legend entries already shown
        shown_legends = set()

        for intrin, intrin_df in task_df.groupby("intrin"):
            print("intrin", intrin)
            exp_names = list(intrin_df["exp_name"].unique())
            print("exp_names", exp_names)
            assert len(exp_names) == 1

            space_ids = list(intrin_df["space_id"].unique())
            num_spaces = len(space_ids)
            print("num_spaces", num_spaces)
            assert num_spaces == len(intrin_df)
            unique_shashs = set().union(*intrin_df["shashs"])
            # print("unique_shashs", unique_shashs, len(unique_shashs))
            num_unique_candidates = len(unique_shashs)
            print("num_unique_candidates", num_unique_candidates)

            # print("intrin_df", intrin_df)
            search_space_sizes = intrin_df["search_space_size"].values
            print("search_space_sizes", search_space_sizes)
            mean_search_space_size = intrin_df["search_space_size"].mean()
            max_search_space_size = intrin_df["search_space_size"].max()
            print("len(intrin_df)", len(intrin_df))
            print("mean_search_space_size", mean_search_space_size)
            print("max_search_space_size", max_search_space_size)

            showlegend = intrin not in shown_legends

            if showlegend:
                shown_legends.add(intrin)

            # One box trace per intrin/workload combination
            fig.add_trace(
                go.Box(
                    x=[workload] * len(search_space_sizes),
                    y=search_space_sizes,
                    name=intrin,
                    boxpoints="all",
                    legendgroup=intrin,
                    marker_color=intrin_colors[intrin],
                    showlegend=showlegend,
                    # showlegend=(
                    #     intrin
                    #     not in {
                    #         trace.name
                    #         for trace in fig.data
                    #     }
                    # ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=[workload],
                    y=[num_spaces],
                    name=intrin,
                    legendgroup=intrin,
                    marker_color=intrin_colors[intrin],
                    opacity=0.7,
                    showlegend=False,
                ),
                row=2,
                col=1,
                # secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=[workload],
                    y=[num_unique_candidates],
                    name=intrin,
                    legendgroup=intrin,
                    mode="markers+lines",
                    marker=dict(
                        color=intrin_colors[intrin],
                        size=10,
                    ),
                    showlegend=False,
                ),
                row=3,
                col=1,
                # secondary_y=True,
            )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.update_layout(
        title="Search Space Size Distribution",
        xaxis_title="Workload",
        yaxis_title="Search Space Size",
        boxmode="group",
        width=1600,
        height=1000,
    )

    # ------------------------------------------------------------------
    # Write HTML
    # ------------------------------------------------------------------

    fig.write_html(out)


if __name__ == "__main__":
    with pd.option_context(
        'display.max_rows', 20,
        'display.max_columns', None,
        'display.precision', 3,
    ):
        main()
