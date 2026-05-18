import logging
import argparse
import tempfile
import tarfile
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tvm import meta_schedule as ms
from tvm.tir.analysis import estimate_tir_flops

from .db_utils import load_ms_db_wrapper


def analyze_ms_db(in_db):
    # print("DB", in_db, dir(in_db))
    recs = in_db.get_all_tuning_records()
    # print("recs", recs, len(recs))
    workloads = []
    targets = []
    annotation_hist = defaultdict(int)
    annotation_val_hist = defaultdict(lambda: defaultdict(int))
    inst_hist = defaultdict(int)
    # TODO: handle postproc

    for rec in recs:
        # print("rec.args_info", rec.args_info)
        args_info = rec.args_info
        # print("rec.as_json()", rec.as_json())
        # print("rec.run_secs", rec.run_secs)
        # print("rec.timestamp", rec.timestamp)
        # input("!")
        target = rec.target
        target_str = str(target)
        print("target", target, dir(target), type(target))
        workload = rec.workload
        print("workload", workload, dir(workload))
        if workload not in workloads:
            workloads.append(workload)
            # workload2args[workload] = args_info
            # flops = estimate_tir_flops(workload.mod)
            # workload2flops[workload] = flops
        # workload2recs[workload].append(rec)
        if target_str not in targets:
            targets.append(target_str)
        # target2recs[target_str].append(rec)
        print("rec.trace", rec.trace, dir(rec.trace))
        # print("rec.trace.insts", rec.trace.insts, dir(rec.trace.insts))
        for i, inst in enumerate(rec.trace.insts):
            print("i", i)
            print("inst", inst)
            # print("inst", inst, dir(inst))
            # print("inst.attrs", inst.attrs, dir(inst.attrs))
            # print("inst.inputs", inst.inputs, dir(inst.inputs))
            # print("inst.inputs", inst.outputs, dir(inst.outputs))
            # print("inst.kind", inst.kind, dir(inst.kind))
            # print("inst.kind.name", inst.kind.name, dir(inst.kind.name))
            kind = inst.kind.name
            inst_hist[kind] += 1
            if kind == "Annotate":
                assert len(inst.attrs) == 1
                key = inst.attrs[0]
                # print("key", key)
                annotation_hist[key] += 1
                assert len(inst.inputs) > 0
                val = inst.inputs[-1]
                # print("val", val)
                annotation_val_hist[key][val] += 1

        # target2workloads[target_str].add(workload)
    print("len(workloads)", len(workloads))
    print("len(targets)", len(targets))
    print("annotation_hist", annotation_hist)
    print("annotation_val_hist", annotation_val_hist)
    print("inst_hist", inst_hist)
    return annotation_hist, annotation_val_hist, inst_hist


def analyze_ms_db_wrapper(db_arg):
    db = load_ms_db_wrapper(db_arg)
    # print("db", db)
    assert isinstance(db, ms.Database)
    _ = analyze_ms_db(db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str, help="input file/dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    analyze_ms_db_wrapper(args.db)
