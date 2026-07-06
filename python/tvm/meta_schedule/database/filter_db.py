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


def drop_duplicate_recs(recs):
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
    print("num_duplicates", num)
    return ret


def filter_ms_db(
    in_db: List[ms.database.Database],
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
    drop_duplicates: bool = False,
    module_equality: str = "structural",
) -> ms.database.MemoryDatabase:
    # print("filter_ms_db")
    # print("in_db", in_db, len(in_db))
    out_db = ms.database.MemoryDatabase(module_equality=module_equality)
    recs = in_db.get_all_tuning_records()
    if drop_duplicates:
        len_before = len(recs)
        recs = drop_duplicate_recs(recs)
        len_after = len(recs)
        num_duplicates = len_after - len_before
        print("Dropped {num_duplciates} duplicate records")
    for rec in recs:
        # print("rec.target", rec.target, dir(rec.target))
        # print("rec.target.keys", rec.target.keys)
        # print("rec.target.kind", rec.target.kind)
        # print("rec.target.mattr", rec.target.mattr)
        # print("rec.target.mcpu", rec.target.mcpu)
        # print("rec.target.model", rec.target.model)
        # print("rec.target.tag", rec.target.tag)
        # target_str = str(rec.target)
        if filter_target_str:
            if str(rec.target) != filter_target_str:
                continue
        if filter_target_kind:
            if rec.target.kind != filter_target_kind:
                continue
        if filter_target_mcpu:
            if rec.target.mcpu != filter_target_mcpu:
                continue
        if filter_target_model:
            if rec.target.model != filter_target_model:
                continue
        if filter_target_tag:
            if rec.target.tag != filter_target_tag:
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
                continue
        if filter_target_mattr:
            if isinstance(filter_target_mattr, str):
                filter_target_mattr = filter_target_mattr.split(",")
            if set(filter_target_mattr) != set(rec.target.mattr):
                continue
        if filter_timestamp_min is not None:
            if rec.timestamp is None:
                continue
            if rec.timestamp < filter_timestamp_min:
                continue
        if filter_timestamp_max is not None:
            if rec.timestamp is None:
                continue
            if rec.timestamp > filter_timestamp_max:
                continue
        if not out_db.has_workload(rec.workload.mod):
            out_db.commit_workload(rec.workload.mod)
        out_db.commit_tuning_record(rec)
    # print("out_db", out_db, len(out_db))
    return out_db


def filter_ms_db_wrapper(
    in_arg,
    out_arg,
    module_equality: str = "structural",
    append: bool = False,
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
    drop_duplicates: bool = False,
):
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
                filtered_db = filter_ms_db(
                    in_db,
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
                    drop_duplicates=drop_duplicates,
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
            filtered_db = filter_ms_db(in_db, filter_target_str=filter_target_str)
            num_recs_after = len(filtered_db)
            db_to_json_db(filtered_db, out_db, append=append)
    print(f"Filtered DB ({num_recs_before} -> {num_recs_after} rercords)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_db", type=str, help="input db files/dirs")
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
    parser.add_argument("--output", "-o", type=str, default=None, help="output file", required=True)
    parser.add_argument("--append", action="store_true", help="Append to existing non-empty out dbs")
    parser.add_argument(
        "--drop-duplicates", action="store_true", help="Drop duplicates (same JSON, including timestamp)"
    )
    # parser.add_argument("--allow-empty", action="store_true", help="Allow empty out_db")  # TODO
    parser.add_argument("--module-equality", type=str, default="structural", help="module equality")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: handle dir/files/archive
    filter_ms_db_wrapper(
        args.in_db,
        args.output,
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
        drop_duplicates=args.drop_duplicates,
        module_equality=args.module_equality,
        append=args.append,
    )
