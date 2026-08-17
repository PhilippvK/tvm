import logging
import argparse
import tempfile
import tarfile
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import tvm  # avoid?
from tvm import meta_schedule as ms
from tvm import tir
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
        # print("target", target, dir(target), type(target))
        workload = rec.workload
        # print("workload", workload, dir(workload))
        if workload not in workloads:
            workloads.append(workload)
            # workload2args[workload] = args_info
            # flops = estimate_tir_flops(workload.mod)
            # workload2flops[workload] = flops
        # workload2recs[workload].append(rec)
        # print("workload.mod", workload.mod, dir(workload.mod))
        # lowered_mod = tvm.lower(workload.mod)
        # print("lowered_mod", lowered_mod)
        sch = tir.Schedule(workload.mod)
        # print("sch", sch, dir(sch))
        # print("sch.mod", sch.mod)
        rec.trace.apply_to_schedule(
            sch,
            remove_postproc=False,
        )
        # print("sch", sch, dir(sch))
        # print("sch.mod", sch.mod)
        # lowered_mod = tvm.lower(sch.mod)
        # print("lowered_mod", lowered_mod)
        # TODO: refactor mod analysis to other func/file
        # input("!")
        if target_str not in targets:
            targets.append(target_str)
        # target2recs[target_str].append(rec)
        # print("rec.trace", rec.trace, dir(rec.trace))
        # print("rec.trace.insts", rec.trace.insts, dir(rec.trace.insts))
        # print("decisions", rec.trace.decisions)
        output_decisions = {}
        for k2, v2 in rec.trace.decisions.items():
            # print("k2", k2, type(k2), dir(k2))
            # print("v2", v2, type(v2), dir(v2))
            outputs = k2.outputs
            # print("outputs", outputs)
            assert len(outputs) > 0
            if len(outputs) == 1:
                outp = outputs[0]
                output_decisions[outp] = v2
            else:
                assert len(v2) == len(outputs)
                for j, outp in enumerate(outputs):
                    # print("outp", outp, type(outp), dir(outp))
                    output_decisions[outp] = v2[j]
        # print("output_decisions", output_decisions)

        for i, inst in enumerate(rec.trace.insts):
            # print("i", i)
            # print("inst", inst)
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
                # print("key", key, dir(key))
                annotation_hist[key] += 1
                assert len(inst.inputs) > 0
                val = inst.inputs[-1]
                # print("val", val, dir(val), type(val))
                # if "unroll" in key:
                if isinstance(val, tir.expr.Var):
                    assert val in output_decisions
                    val = output_decisions[val]
                    # print("val_new", val)
                    # print("inst", inst)
                    # input("$$$")
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
