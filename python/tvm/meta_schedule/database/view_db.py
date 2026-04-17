import logging
import argparse
import tempfile
import tarfile
from collections import defaultdict
from pathlib import Path
from tvm import meta_schedule as ms

from .db_utils import load_ms_db_wrapper


def view_ms_db(in_db):
    # print("DB", in_db, dir(in_db))
    recs = in_db.get_all_tuning_records()
    # print("recs", recs, len(recs))
    workloads = []
    workload2args = {}
    workload2recs = defaultdict(list)
    workload2targets = defaultdict(set)
    workload2secs = defaultdict(list)
    targets = []
    target2recs = defaultdict(list)
    target2workloads = defaultdict(set)

    for rec in recs:
        # print("rec", rec, dir(rec))
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
            workload2args[workload] = args_info
        workload2recs[workload].append(rec)
        if target_str not in targets:
            targets.append(target_str)
        target2recs[target_str].append(rec)
        target2workloads[target_str].add(workload)
        workload2targets[workload].add(target_str)
        workload2secs[workload].extend(rec.run_secs)
    # print("workloads", workloads, len(workloads))
    print("--- MS Database ---")
    print("## Workloads ##")
    num_workloads = len(workloads)
    print(f"Count: {num_workloads}")
    for workload in workloads:
        # print("workload", workload, dir(workload))
        workload_str = str(workload)
        workload_str = workload_str.replace("meta_schedule.", "")
        # print("workload.mod", workload.mod, dir(workload.mod))
        workload_hash = workload.as_json()[0]
        workload_args = workload2args[workload]
        workload_args = str(workload_args).replace("TensorInfo", "")
        workload_args = workload_args.replace('"', "")
        workload_args = workload_args.replace(" ", "")
        print(f"- {workload_str} ({workload_hash}) {workload_args}")
        # input("!")
    print()
    print("## Targets ##")
    num_targets = len(targets)
    print(f"Count: {num_targets}")
    for target in targets:
        print(f"- {target}")
    print()
    print("## Records ##")
    num_recs = len(recs)
    print(f"Count: {num_recs}")

    # TODO: flops,...

    print()
    print("## Records by Workload ##")
    for workload, workload_recs in workload2recs.items():
        workload_str = str(workload)
        workload_str = workload_str.replace("meta_schedule.", "")
        num_workload_recs = len(workload_recs)
        num_workload_targets = len(workload2targets[workload])
        print(f"- {workload_str}: #recs={num_workload_recs} #targets={num_workload_targets}")

    print()
    print("## Measures by Workload ##")
    for workload, workload_secs in workload2secs.items():
        # print("workload_secs", workload_secs)
        workload_str = str(workload)
        workload_str = workload_str.replace("meta_schedule.", "")
        invalid_secs = [secs for secs in workload_secs if secs > 1000.0]
        num_invalid = len(invalid_secs)
        valid_secs = [secs for secs in workload_secs if secs <= 1000.0]
        min_secs = float(min(valid_secs))
        max_secs = float(max(valid_secs))
        mean_secs = float(sum(valid_secs) / len(valid_secs))
        num_workload_secs = len(workload_secs)
        print(
            f"- {workload_str}: #measures={num_workload_secs} #invalid={num_invalid} min={min_secs:.5f}s max={max_secs:.5f}s mean={mean_secs:.5f}s"
        )
        # TODO: GFLOP/s
        # TODO: make secs,... optional via cli

    print()
    print("## Records by Target ##")
    for target, target_recs in target2recs.items():
        num_target_recs = len(target_recs)
        num_target_workloads = len(target2workloads[target])
        print(f"- {target}: #recs={num_target_recs} #workloads={num_target_workloads}")


# def view_ms_db_dir(in_db_dir):
#     in_db_path = Path(in_db_dir)
#     path_workload = in_db_path / "database_workload.json"
#     path_tuning_record = in_db_path / "database_tuning_record.json"
#     in_db = ms.database.JSONDatabase(
#         path_workload=str(path_workload),
#         path_tuning_record=str(path_tuning_record),
#         allow_missing=False,
#     )
#     view_ms_db(in_db)
#
#
# def view_ms_db_file(in_db_file):
#     in_db_path = Path(in_db_file)
#     assert in_db_path.is_file()
#     suffix = in_db_path.suffix
#     assert suffix == ".json"
#     if "workload" in in_db_path.stem:
#         path_workload = in_db_path
#         path_tuning_record = in_db_path.parent / in_db_path.name.replace("workload", "tuning_record")
#     elif "tuning_record" in in_db_path.stem:
#         path_tuning_record = in_db_path
#         path_workload = in_db_path.parent / in_db_path.name.replace("tuning_record", "workload")
#     else:
#         raise ValueError("Invalid MS DB file name: {in_db_path.name}")
#     in_db = ms.database.JSONDatabase(
#         path_workload=str(path_workload),
#         path_tuning_record=str(path_tuning_record),
#         allow_missing=False,
#     )
#     view_ms_db(in_db)
#
#
# def view_ms_db_archive(in_db_archive):
#     in_db_path = Path(in_db_archive)
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         if tarfile.is_tarfile(in_db_path):
#             temp_in_db_path = Path(tmpdirname) / "in_db"
#             with tarfile.open(in_db_path) as f:
#                 f.extractall(path=temp_in_db_path)
#             has_workdir = False
#             if (temp_in_db_path / "work_dir").is_dir():
#                 has_workdir = True
#                 temp_in_db_path = temp_in_db_path / "work_dir"
#             view_ms_db_dir(temp_in_db_path)
#         else:
#             raise ValueError(f"Unsupported format")


def view_ms_db_wrapper(db_arg):
    db = load_ms_db_wrapper(db_arg)
    print("db", db)
    assert isinstance(db, ms.Database)
    _ = view_ms_db(db)
    # if isinstance(db_arg, ms.Database):
    #     _ = view_ms_db(db_arg)
    #     return
    # in_path = Path(db_arg)
    # assert in_path.exists()
    # if in_path.is_dir():
    #     view_ms_db_dir(in_path)
    #     return
    # if in_path.is_file():
    #     if in_path.suffix == ".json":
    #         view_ms_db_file(in_path)
    #     assert in_path.suffix in [".tar"]
    #     view_ms_db_archive(in_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str, help="input file/dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    view_ms_db_wrapper(args.db)
