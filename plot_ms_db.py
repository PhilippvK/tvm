import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from tvm import meta_schedule as ms
from tvm.tir.analysis import estimate_tir_flops


parser = argparse.ArgumentParser(description="TODO", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("database", default=None, help="TODO")
parser.add_argument("--out", "-o", default=None, help="TODO")
parser.add_argument("--topk-bars", nargs=2, type=int, default=None, help="TODO")
parser.add_argument("--zero-time", action="store_true", help="TODO")
parser.add_argument("--print-df", action="store_true", help="TODO")
parser.add_argument("--invert", action="store_true", help="TODO")
parser.add_argument("--no-scatter", action="store_true", help="TODO")
parser.add_argument("--compare-scatter", action="store_true", help="TODO")
parser.add_argument("--quantile", type=float, default=0.95, help="TODO")
parser.add_argument("--batch-size", type=int, default=1, help="TODO")
parser.add_argument("--ref-y", type=float, default=None, help="TODO")
parser.add_argument("-x", default="Time", choices=["Time", "Trial", "Batch"], help="TODO")  # TODO: allow seconds, datetime,...
parser.add_argument("-y", default="Max. MFLOP/s", choices=["Min. Runtime", "Max. FLOP/s", "Max. MFLOP/s"], help="TODO")
parser.add_argument("--scatter-y", default="MFLOP/s", choices=["Runtime", "FLOP/s", "MFLOP/s"], help="TODO")
parser.add_argument("--compare", nargs="+", help="TODO")
args = parser.parse_args()


def database_to_df(db_path):
    if not isinstance(db_path, Path):
        db_path = Path(db_path)
    path_workload = db_path / "database_workload.json"
    path_tuning_record = db_path / "database_tuning_record.json"
    database = ms.database.JSONDatabase(path_workload=str(path_workload), path_tuning_record=str(path_tuning_record))
    records = database.get_all_tuning_records()
    workload2records = defaultdict(list)
    for i, record in enumerate(records):
        workload = record.workload
        workload2records[workload].append(records[i])
    # print("workload2records", workload2records)
    main_workloads = [workload for workload, records in workload2records.items() if len(records) > 2]
    # print("main_workloads", main_workloads)
    if len(main_workloads) == 0:
        raise RuntimeError(f"Empty DB encountered: {db_path}")
    assert len(main_workloads) == 1
    main_workload = main_workloads[0]
    records_ = workload2records[main_workload]
    assert len(records) > 0
    records_by_time = sorted(records_, key=lambda x: x.timestamp)
    # print("records_by_time", records_by_time)
    if args.zero_time:
        start = records_by_time[0].timestamp.value
    else:
        # TODO: convert Time col to datetime type?
        start = 0
    times = [record.timestamp.value - start for record in records_by_time]
    idxs = list(range(len(records_by_time)))
    secs = [record.run_secs[0].value for record in records_by_time]
    workload_flops = estimate_tir_flops(main_workload.mod)
    flops = [workload_flops] * len(records_by_time)
    df = pd.DataFrame({"Time": times, "Trial": idxs, "Runtime": secs, "FLOPS": flops})
    if args.batch_size > 1:
        df["Batch"] = df["Trial"] // args.batch_size
    df["FLOP/s"] = df["FLOPS"] / df["Runtime"]
    df["MFLOP/s"] = df["FLOP/s"] / 1e6
    df["Min. Runtime"] = df["Runtime"].rolling(len(df), min_periods=1).min()
    df["Max. FLOP/s"] = df["FLOP/s"].rolling(len(df), min_periods=1).max()
    df["Max. MFLOP/s"] = df["MFLOP/s"].rolling(len(df), min_periods=1).max()
    return df


df = database_to_df(args.database)
if args.print_df:
    print("DataFrame:")
    print(df)


compare_dfs = []
if args.compare:
    for compare_path in args.compare:
        df_ = database_to_df(compare_path)
        compare_dfs.append(df_)
    # print("compare_dfs", compare_dfs)


def get_thr_val(df, x="Time", y="MFLOPS", thr=0.95, invert=False):
    # x_thr = df[df[y] > df[y].quantile(thr)].head(1)[x].iloc[0]
    assert 0 < thr < 1.0
    if invert:
        y_thr = df[y].min() * (2 - thr)
        x_thr = df[df[y] < y_thr].head(1)[x].iloc[0]
    else:
        y_thr = df[y].max() * thr
        x_thr = df[df[y] > y_thr].head(1)[x].iloc[0]
    return x_thr, y_thr


x_thr, y_thr = None, None
if args.quantile > 0:
    x_thr, y_thr = get_thr_val(df, args.x, args.y, args.quantile, invert=args.invert)
    print(f"{args.quantile*100}% Quantile:")
    print(f"{args.y}={y_thr} @ {args.x}={x_thr} ")


if args.out:
    if args.topk_bars is not None:
        topk, total = args.topk_bars
        total = max(min(len(df), total), 0)
        assert 0 <= topk <= total
        topk_df = df.sort_values(args.scatter_y, ascending=False).head(total)[[args.x, args.scatter_y]]
        topk_df[f"top{topk}"] = False
        topk_df["k"] = list(range(total))
        topk_df.iloc[range(topk), 2] = True
        topk_df["desc"] = topk_df[f"top{topk}"].apply(lambda x: "topk" if x else "rest")
        # print("topk_df", topk_df)
        fig = px.bar(topk_df, x="k", y=args.scatter_y, color="desc")
    else:
        # df = px.data.iris()
        fig = make_subplots(rows=1, cols=1)  # TODO: ?
        fig.update_layout(
            title="Tuning",  # TODO
            autosize=True,
            # height=950,
            xaxis_title=args.x,
            yaxis_title=args.y,
        )
        fig.update_xaxes(rangeslider=dict(visible=False))

        if len(compare_dfs) > 0:
            full_compare_df = pd.DataFrame()
            for j, df_ in enumerate(compare_dfs):
                temp = df_.copy()
                temp["compare"] = j
                full_compare_df = pd.concat([full_compare_df, temp])
            full_compare_df["compare"] = full_compare_df["compare"].astype(str)
            print("full_compare_df", full_compare_df)
            # input(">")
            # cm = px.colors.sequential.gray
            # cm = px.colors.qualitative.Plotly
            # cm = ["gray", "lightgray", "darkgray"]
            cm = ["#444444", "#555555", "#666666", "#777777", "#888888", "#999999"]

            fig2 = px.line(full_compare_df, x=args.x, y=args.y, color="compare", color_discrete_sequence=cm)
            for data in fig2.data:
                fig.add_trace(data)
            if not args.no_scatter and args.compare_scatter:
                fig1 = px.scatter(full_compare_df, x=args.x, y=args.scatter_y, title="Title", color="compare", color_discrete_sequence=cm)
                for data in fig1.data:
                    fig.add_trace(data)

        fig2 = px.line(df, x=args.x, y=args.y)  # , color_discrete_sequence=["blue"])
        fig.add_trace(fig2.data[0])

        if not args.no_scatter:
            fig1 = px.scatter(df, x=args.x, y=args.scatter_y, title="Title")  # , color_discrete_sequence=["gray"])
            fig.add_trace(fig1.data[0])
        if y_thr:
            fig.add_hline(y=y_thr, line_width=1, line_dash="dash")
        if x_thr:
            fig.add_vline(x=x_thr, line_width=1, line_dash="dash")

        if args.ref_y:
            x0 = list(df[args.x].head(1))[0]
            ymax = df[args.y].max()
            if args.invert:
                raise NotImplementedError
            else:
                speedup = ymax / args.ref_y
                text = f"Speedup: {speedup:.2f} x"
            fig.add_scatter(x=[x0], y=[args.ref_y], marker=dict(color="red", size=8, symbol="x"), name="ref")
            fig.add_annotation(x=x0, y=ymax, text=text, showarrow=True)
            fig.add_scatter(x=[x0], y=[ymax], marker=dict(color="red", size=8, symbol="square"), name="best")
    dest = Path(args.out)
    fmt = dest.suffix[1:].lower()
    if fmt in ["pdf", "png", "jpg"]:
        fig.write_image(dest)
    elif fmt in ["html"]:
        fig.write_html(dest)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
