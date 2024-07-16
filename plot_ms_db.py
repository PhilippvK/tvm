import sys
import argparse
import tempfile
from pathlib import Path
from types import MappingProxyType
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

import tvm
from tvm import relay, tir
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.relay.backend import Executor
from tvm.tir.analysis import estimate_tir_flops


parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("database", default=None, help="TODO")
parser.add_argument("--out", "-o", default=None, help="TODO")
parser.add_argument("--zero-time", action="store_true", help="TODO")
parser.add_argument("--invert", action="store_true", help="TODO")
parser.add_argument("--quantile", type=float, default=0.95, help="TODO")
parser.add_argument("--batch-size", type=int, default=1, help="TODO")
parser.add_argument("-x", default="Time", help="TODO", choices=["Time", "Trial", "Batch"])  # TODO: allow seconds, datetime,...
parser.add_argument("-y", default="Max. MFLOP/s", help="TODO", choices=["Min. Runtime", "Max. FLOP/s", "Max. MFLOP/s"])
parser.add_argument("--scatter-y", default="MFLOP/s", help="TODO", choices=["Runtime", "FLOP/s", "MFLOP/s"])
args = parser.parse_args()


trials = []  # stores tuples (mode, record, trial_idx)
db_path = Path(args.database)
path_workload = db_path / "database_workload.json"
path_tuning_record = db_path / "database_tuning_record.json"
database = ms.database.JSONDatabase(path_workload=str(path_workload), path_tuning_record=str(path_tuning_record))
records = database.get_all_tuning_records()
workload2records = defaultdict(list)
for i, record in enumerate(records):
    workload = record.workload
    workload2records[workload].append(records[i])
print("workload2records", workload2records)
main_workloads = [workload for workload, records in workload2records.items() if len(records) > 1]
print("main_workloads", main_workloads)
assert len(main_workloads) == 1
main_workload = main_workloads[0]
records_ = workload2records[main_workload]
assert len(records) > 0
records_by_time = sorted(records_, key=lambda x: x.timestamp)
print("records_by_time", records_by_time)
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
print("df")
print(df)
# input("1")


def get_thr_val(df, x="Time", y="MFLOPS", thr=0.95, invert=False):
    # x_thr = df[df[y] > df[y].quantile(thr)].head(1)[x].iloc[0]
    if invert:
        y_thr = df[y].min() * (2 - thr)
        x_thr = df[df[y] < y_thr].head(1)[x].iloc[0]
    else:
        y_thr = df[y].max() * thr
        x_thr = df[df[y] > y_thr].head(1)[x].iloc[0]
    return x_thr, y_thr


x_thr, y_thr = get_thr_val(df, args.x, args.y, args.quantile, invert=args.invert)
print("x_thr", x_thr)
print("y_thr", y_thr)

if args.out:
    # df = px.data.iris()
    fig = make_subplots(rows=2, cols=1)  # TODO: ?
    fig.update_layout(
        title='TODO: Title',  # TODO
        autosize=True,
        height=750
    )
    fig.update_xaxes(rangeslider=dict(visible=False))

    fig1 = px.scatter(df, x=args.x, y=args.scatter_y, color_discrete_sequence=["gray"])
    fig.add_trace(fig1.data[0])
    fig.add_hline(y=y_thr, line_width=1, line_dash="dash")
    fig.add_vline(x=x_thr, line_width=1, line_dash="dash")
    fig2 = px.line(df, x=args.x, y=args.y)  # , color_discrete_sequence=["blue"])
    fig.add_trace(fig2.data[0])
    fig.write_image(args.out)
