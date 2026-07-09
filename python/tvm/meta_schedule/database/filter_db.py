import logging
import argparse
import tarfile
import tempfile
from collections import defaultdict
from typing import List, Optional, Union
from pathlib import Path
from tvm import meta_schedule as ms

from .db_utils import load_ms_db_wrapper, db_to_json_db
from tvm.tir.tensor_intrin.riscv_cpu import *


def drop_duplicate_recs_helper(recs):
    hash2recs = defaultdict(list)
    num = 0
    ret = []
    for rec in recs:
        rec_json = str(rec.as_json()).encode()
        import hashlib

        m = hashlib.sha256()
        m.update(rec_json)
        rec_hash = m.hexdigest()
        if rec_hash in hash2recs:
            num += 1
            continue
            # print("hash2recs[rec_hash]", hash2recs[rec_hash])
            # print("duplicate!")
            # input("!!!")
        hash2recs[rec_hash].append(rec)
        ret.append(rec)
    # print("num_duplicates", num)
    return ret


def drop_duplicate_candidates_helper(recs, lower: bool = False):
    hash2recs = defaultdict(list)
    num = 0
    from tvm.meta_schedule.utils import shash2hex

    for rec in recs:
        # print("rec", rec, dir(rec))
        measure_candidate = rec.as_measure_candidate()
        sch = measure_candidate.sch
        if lower:
            mod = tvm.lower(sch.mod)
        else:
            mod = sch.mod
        # print("sch", sch)
        # print("sch.mod", sch.mod)
        # print("sch.trace", sch.trace)
        shash = shash2hex(mod)
        # print("shash", shash)
        # TODO: apply trace? -> not needed
        # TODO: also check args_info/space_idx?
        # candidate_hash = m.hexdigest()
        if shash in hash2recs:
            num += 1
            continue
            # print("hash2recs[shash]", hash2recs[shash])
            # print("duplicate!")
            # input("!!!")
        hash2recs[shash].append(rec)
        # ret.append(rec)
    # print("hash2recs", hash2recs, len(hash2recs))
    # TODO: keep best rec per candidate
    # TODO: compare post-lowering tir?
    # TODO: analyze recs per candidate -> view_db?
    # print("num_duplicates", num)
    ret = []  # TODO
    for shash, recs in hash2recs.items():
        # print("shash", shash)
        # print("recs", recs, len(recs))
        rec_run_secs = [rec.run_secs for rec in recs]
        # print("rec_run_secs", rec_run_secs)
        rec_mean_run_secs = list(map(lambda x: sum(x) / len(x), rec_run_secs))
        # TODO: median?
        # print("rec_mean_run_secs", rec_mean_run_secs)
        best_rec_idx = min(range(len(rec_mean_run_secs)), key=lambda x: rec_mean_run_secs[x])
        # print("best_rec_idx", best_rec_idx)
        best_rec = recs[best_rec_idx]
        # print("best_rec", best_rec)
        ret.append(best_rec)
    # print("ret", ret, len(ret))
    # print("len(ret)", len(ret))
    # input("$")
    return ret


def filter_ms_db(
    in_db: List[ms.database.Database],
    filter_topk: Optional[int] = None,
    filter_target_str: Optional[str] = None,
    filter_target_kind: Optional[str] = None,
    filter_target_mcpu: Optional[str] = None,
    filter_target_model: Optional[str] = None,
    filter_target_tag: Optional[str] = None,
    filter_target_device: Optional[str] = None,
    filter_target_num_cores: Optional[str] = None,
    filter_target_keys: Optional[Union[str, List[str]]] = None,
    filter_target_mattr: Optional[Union[str, List[str]]] = None,
    filter_timestamp_min: Optional[float] = None,
    filter_timestamp_max: Optional[float] = None,
    filter_tensor_intrin: Optional[str] = None,
    drop_failing: bool = False,
    drop_non_failing: bool = False,
    drop_duplicate_recs: bool = False,
    drop_duplicate_candidates: bool = False,
    drop_duplicate_lowered_candidates: bool = False,
    module_equality: str = "structural",
) -> ms.database.MemoryDatabase:
    assert not (drop_failing and drop_non_failing), "drop_failing and drop_non_failing can only be used exclusively"
    # print("filter_ms_db")
    # print("in_db", in_db, len(in_db))
    out_db = ms.database.MemoryDatabase(module_equality=module_equality)
    workloads = set()
    recs = in_db.get_all_tuning_records()
    for rec in recs:
        workloads.add(rec.workload)
    all_topk_recs = set()
    drop_hist = defaultdict(int)
    if filter_topk:
        for workload in workloads:
            topk_recs = in_db.get_top_k(workload, filter_topk)
            all_topk_recs.update(topk_recs)
        num_before = len(recs)
        recs = [rec for rec in recs if rec in all_topk_recs]  # TODO: use filter()
        num_topk = len(recs)
        num_dropped = num_before - num_topk
        drop_hist["topk"] += num_dropped
        print(f"Selected {num_topk} topk records")
    if drop_duplicate_recs:
        len_before = len(recs)
        recs = drop_duplicate_recs_helper(recs)
        len_after = len(recs)
        num_duplicates = len_before - len_after
        drop_hist["duplicate_rec"] += num_duplicates
        print(f"Dropped {num_duplicates} duplicate records")
    if drop_duplicate_candidates:
        len_before = len(recs)
        recs = drop_duplicate_candidates_helper(recs, lower=False)
        len_after = len(recs)
        num_duplicates = len_before - len_after
        drop_hist["duplicate_candidate"] += num_duplicates
        print(f"Dropped {num_duplicates} duplicate candidates")
    if drop_duplicate_lowered_candidates:
        len_before = len(recs)
        recs = drop_duplicate_candidates_helper(recs, lower=True)
        len_after = len(recs)
        num_duplicates = len_before - len_after
        drop_hist["duplicate_lowered_candidate"] += num_duplicates
        print(f"Dropped {num_duplicates} duplicate lowered candidates")
    for rec in recs:
        # print("rec.target", rec.target, dir(rec.target))
        # print("rec.target.keys", rec.target.keys)
        # print("rec.target.kind", rec.target.kind)
        # print("rec.target.mattr", rec.target.mattr)
        # print("rec.target.mcpu", rec.target.mcpu)
        # print("rec.target.model", rec.target.model)
        # print("rec.target.tag", rec.target.tag)
        # target_str = str(rec.target)
        if drop_failing or drop_non_failing:
            is_failing = False
            # run_secs = rec.run_secs
            run_secs = [rec.run_secs[i] for i in range(len(rec.run_secs))]
            assert run_secs is not None
            assert len(run_secs) > 0
            min_run_secs = min(run_secs)
            # max_run_secs = max(run_secs)
            if min_run_secs >= 10000000000.0:
                is_failing = True
            if is_failing and drop_failing:
                drop_hist["failing"] += 1
                continue
            if not is_failing and drop_non_failing:
                drop_hist["non_failing"] += 1
                continue
        if filter_target_str:
            if str(rec.target) != filter_target_str:
                drop_hist["target_str"] += 1
                continue
        if filter_target_kind:
            if rec.target.kind != filter_target_kind:
                drop_hist["target_kind"] += 1
                continue
        if filter_target_mcpu:
            if rec.target.mcpu != filter_target_mcpu:
                drop_hist["target_mcpu"] += 1
                continue
        if filter_target_model:
            if rec.target.model != filter_target_model:
                drop_hist["target_model"] += 1
                continue
        if filter_target_tag:
            if rec.target.tag != filter_target_tag:
                drop_hist["target_tag"] += 1
                continue
        if filter_target_device:
            if rec.target.attrs.get("device") != filter_target_device:
                continue
        if filter_target_num_cores:
            if rec.target.attrs.get("num-cores") != filter_target_num_cores:
                continue
        if filter_target_keys:
            if isinstance(filter_target_keys, str):
                filter_target_keys = filter_target_keys.split(",")
            if set(filter_target_keys) != set(rec.target.keys):
                drop_hist["target_keys"] += 1
                continue
        if filter_target_mattr:
            if isinstance(filter_target_mattr, str):
                filter_target_mattr = filter_target_mattr.split(",")
            if set(filter_target_mattr) != set(rec.target.mattr):
                drop_hist["target_mattr"] += 1
                continue
        if filter_timestamp_min is not None:
            if rec.timestamp is None:
                drop_hist["timestamp_none"] += 1
                continue
            if rec.timestamp < filter_timestamp_min:
                drop_hist["timestamp_min"] += 1
                continue
        if filter_timestamp_max is not None:
            if rec.timestamp is None:
                drop_hist["timestamp_none"] += 1
                continue
            if rec.timestamp > filter_timestamp_max:
                drop_hist["timestamp_max"] += 1
                continue
        if filter_tensor_intrin is not None:
            has_tensorize = "sch.tensorize" in str(rec.trace)
            # print("filter_tensor_intrin", filter_tensor_intrin)
            used_intrins = []
            if has_tensorize:
                trace = rec.trace
                inst = trace.pop()
                while inst is not None:
                    if "sch.tensorize" in str(inst):
                        intrin_name = str(inst).split("tensor_intrin=", 1)[1].split(",", 1)[0]
                        used_intrins.append(intrin_name)
                    inst = trace.pop()
                assert len(used_intrins) > 0
            # print("used_intrins", used_intrins)
            assert isinstance(filter_tensor_intrin, list)
            keep_all = "all" in filter_tensor_intrin
            # print("keep_all", keep_all)
            filtered_intrins = [name for name in used_intrins if name in filter_tensor_intrin or keep_all]
            # print("filtered_intrins", filtered_intrins)

            if filter_tensor_intrin == "none" and len(filtered_intrins) > 0:
                drop_hist["tensor_intrin_used"] += 1
                # print("tensor_intrin_used!")
                continue
            elif len(filtered_intrins) == 0:
                drop_hist["tensor_intrin_unused"] += 1
                # print("tensor_intrin_unused!")
                continue
        if not out_db.has_workload(rec.workload.mod):
            out_db.commit_workload(rec.workload.mod)
        out_db.commit_tuning_record(rec)
    # print("out_db", out_db, len(out_db))
    return out_db, drop_hist


def filter_ms_db_wrapper(
    in_arg,
    out_arg,
    module_equality: str = "structural",
    append: bool = False,
    filter_topk: Optional[int] = None,
    filter_target_str: Optional[str] = None,
    filter_target_kind: Optional[str] = None,
    filter_target_mcpu: Optional[str] = None,
    filter_target_model: Optional[str] = None,
    filter_target_tag: Optional[str] = None,
    filter_target_device: Optional[str] = None,
    filter_target_num_cores: Optional[str] = None,
    filter_target_keys: Optional[Union[str, List[str]]] = None,
    filter_target_mattr: Optional[Union[str, List[str]]] = None,
    filter_timestamp_min: Optional[float] = None,
    filter_timestamp_max: Optional[float] = None,
    filter_tensor_intrin: Optional[str] = None,
    drop_failing: bool = False,
    drop_non_failing: bool = False,
    drop_duplicate_recs: bool = False,
    drop_duplicate_candidates: bool = False,
    drop_duplicate_lowered_candidates: bool = False,
):
    filter_kwargs = dict(
        filter_topk=filter_topk,
        filter_target_str=filter_target_str,
        filter_target_kind=filter_target_kind,
        filter_target_mcpu=filter_target_mcpu,
        filter_target_model=filter_target_model,
        filter_target_tag=filter_target_tag,
        filter_target_device=filter_target_device,
        filter_target_num_cores=filter_target_num_cores,
        filter_target_keys=filter_target_keys,
        filter_target_mattr=filter_target_mattr,
        filter_timestamp_min=filter_timestamp_min,
        filter_timestamp_max=filter_timestamp_max,
        filter_tensor_intrin=filter_tensor_intrin,
        drop_failing=drop_failing,
        drop_non_failing=drop_non_failing,
        drop_duplicate_recs=drop_duplicate_recs,
        drop_duplicate_candidates=drop_duplicate_candidates,
        drop_duplicate_lowered_candidates=drop_duplicate_lowered_candidates,
    )
    assert out_arg is not None
    in_db = load_ms_db_wrapper(in_arg)
    num_recs_before = len(in_db)
    if out_arg.startswith("s3://"):
        raise NotImplementedError("S3 output")
    else:
        out_path = Path(out_arg)
        if out_path.suffix == ".json":  # file
            raise NotImplementedError("JSON output")
        elif out_path.suffix in [".tar"]:  # archive
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_out_path = Path(tmpdirname)
                out_db = ms.database.JSONDatabase(
                    work_dir=str(temp_out_path),
                    module_equality=module_equality,
                )
                filtered_db, drop_hist = filter_ms_db(
                    in_db,
                    **filter_kwargs,
                )
                num_recs_after = len(filtered_db)
                db_to_json_db(filtered_db, out_db, append=append)
                with tarfile.open(out_path, mode="w") as archive:
                    archive.add(temp_out_path, recursive=True, arcname=".")
        else:  # directory
            out_path.mkdir(exist_ok=True)
            out_db = ms.database.JSONDatabase(
                work_dir=str(out_path),
                module_equality=module_equality,
            )
            filtered_db, drop_hist = filter_ms_db(
                in_db,
                **filter_kwargs,
            )
            num_recs_after = len(filtered_db)
            db_to_json_db(filtered_db, out_db, append=append)
    num_filtered_recs = num_recs_before - num_recs_after
    num_filtered_recs_rel = num_filtered_recs / num_recs_before
    print(f"Filtered DB ({num_recs_before} -> {num_recs_after} [{num_filtered_recs_rel*100:.1f}%] records)")
    if drop_hist:
        print("Reasons:")
        for reason, freq in drop_hist.items():
            freq_rel = freq / num_recs_before
            print(f"- {reason}: {freq} [{freq_rel*100:.1f}%]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_db", type=str, help="input db files/dirs")
    parser.add_argument("--filter-topk", type=int, default=None, help="filter by topk recs (per workload)")
    parser.add_argument("--filter-target-str", type=str, default=None, help="filter by full quoted target str")
    parser.add_argument("--filter-target-kind", type=str, default=None, help="filter by target kind")
    parser.add_argument("--filter-target-mcpu", type=str, default=None, help="filter by target mcpu")
    parser.add_argument("--filter-target-model", type=str, default=None, help="filter by target model")
    parser.add_argument("--filter-target-tag", type=str, default=None, help="filter by target tag")
    parser.add_argument("--filter-target-device", type=str, default=None, help="filter by target device")
    parser.add_argument("--filter-target-num-cores", type=int, default=None, help="filter by target num-cores")
    parser.add_argument("--filter-target-keys", type=str, default=None, help="filter by target keys")
    parser.add_argument("--filter-target-mattr", type=str, default=None, help="filter by target mattr")
    parser.add_argument("--filter-timestamp-min", type=float, default=None, help="filter by min timestamp")
    parser.add_argument("--filter-timestamp-max", type=float, default=None, help="filter by max timestamp")
    parser.add_argument(
        "--filter-tensor-intrin",
        nargs="?",
        type=str,
        default=None,
        const="all",
        help="filter by used tensor intrin (name ; name1,name2,... ; none ; all)",
    )
    parser.add_argument("--drop-failing", action="store_true", help="Drop all failing records")
    parser.add_argument("--drop-non-failing", action="store_true", help="Drop all non-failing records")
    parser.add_argument("--output", "-o", type=str, default=None, help="output file", required=True)
    parser.add_argument("--append", action="store_true", help="Append to existing non-empty out dbs")
    parser.add_argument(
        "--drop-duplicate-recs",
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate Records (same JSON, including timestamp)",
    )
    parser.add_argument(
        "--drop-duplicate-candidates", action="store_true", help="drop duplicate candidates (same module shash)"
    )
    parser.add_argument(
        "--drop-duplicate-lowered-candidates",
        action="store_true",
        help="drop duplicate lowered candidates (same module shash after tvm.lower)",
    )
    # parser.add_argument("--allow-empty", action="store_true", help="Allow empty out_db")  # TODO
    parser.add_argument("--module-equality", type=str, default="structural", help="module equality")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: handle dir/files/archive
    filter_ms_db_wrapper(
        args.in_db,
        args.output,
        filter_topk=args.filter_topk,
        filter_target_str=args.filter_target_str,
        filter_target_kind=args.filter_target_kind,
        filter_target_mcpu=args.filter_target_mcpu,
        filter_target_model=args.filter_target_model,
        filter_target_tag=args.filter_target_tag,
        filter_target_keys=args.filter_target_keys,
        filter_target_mattr=args.filter_target_mattr,
        filter_target_device=args.filter_target_device,
        filter_target_num_cores=args.filter_target_num_cores,
        filter_timestamp_min=args.filter_timestamp_min,
        filter_timestamp_max=args.filter_timestamp_max,
        filter_tensor_intrin=args.filter_tensor_intrin.split(",") if args.filter_tensor_intrin is not None else None,
        drop_failing=args.drop_failing,
        drop_non_failing=args.drop_non_failing,
        drop_duplicate_recs=args.drop_duplicate_recs,
        drop_duplicate_candidates=args.drop_duplicate_candidates,
        drop_duplicate_lowered_candidates=args.drop_duplicate_lowered_candidates,
        module_equality=args.module_equality,
        append=args.append,
    )
