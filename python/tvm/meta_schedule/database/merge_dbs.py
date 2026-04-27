import logging
import argparse
import tarfile
import tempfile
from typing import List
from pathlib import Path
from tvm import meta_schedule as ms

from .db_utils import load_ms_db_wrapper, db_to_json_db
from tvm.meta_schedule.database.s3_json_database import S3JSONDatabase
from tvm.tir.tensor_intrin.riscv_cpu import *


def merge_ms_dbs(in_dbs: List[ms.database.Database], ordered: bool = False):
    union_db_cls = ms.database.OrderedUnionDatabase if ordered else ms.database.UnionDatabase
    out_db = union_db_cls(*in_dbs)
    return out_db


# def load_ms_db_dir(in_db_dir, module_equality: str = "structural"):
#     in_db_path = Path(in_db_dir)
#     path_workload = in_db_path / "database_workload.json"
#     path_tuning_record = in_db_path / "database_tuning_record.json"
#     in_db = ms.database.JSONDatabase(
#         # TODO: use work_dir=
#         path_workload=str(path_workload),
#         path_tuning_record=str(path_tuning_record),
#         module_equality=module_equality,
#     )
#     return in_db
#
#
# def load_ms_db_file(in_db_file, module_equality: str = "structural"):
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
#         module_equality=module_equality,
#     )
#     return in_db
#
#
# def load_ms_db_archive(in_db_archive, module_equality: str = "structural"):
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
#             return load_ms_db_dir(temp_in_db_path)
#             # TODO: check if this works with tempdir? yield?
#         else:
#             raise ValueError(f"Unsupported format")
#
#
# def load_ms_db_wrapper(in_arg, module_equality: str = "structural"):
#     if isinstance(in_arg, ms.Database):
#         return in_arg
#     in_path = Path(in_arg)
#     assert in_path.exists()
#     if in_path.is_dir():
#         return load_ms_db_dir(in_path, module_equality=module_equality)
#     if in_path.is_file():
#         if in_path.suffix == ".json":
#             return load_ms_db_file(in_path, module_equality=module_equality)
#         assert in_path.suffix in [".tar"]
#         return load_ms_db_archive(in_path, module_equality=module_equality)


def merge_ms_dbs_wrapper(in_args, out_arg, module_equality: str = "structural", append: bool = False):
    assert out_arg is not None
    in_dbs = list(map(load_ms_db_wrapper, in_args))
    num_dbs = len(in_dbs)
    print(f"Merging {num_dbs} MS databases...")
    # print("in_dbs", in_dbs, len(in_dbs))
    if out_arg.startswith("s3://"):
        import time

        out_db = S3JSONDatabase(
            out_arg,
            module_equality=module_equality,
            auto_upload=False,
        )
        union_db = merge_ms_dbs(in_dbs)
        num_recs = len(union_db)
        db_to_json_db(union_db, out_db, append=append)
        out_db.sync()
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
                union_db = merge_ms_dbs(in_dbs)
                num_recs = len(union_db)
                db_to_json_db(union_db, out_db, append=append)
                with tarfile.open(out_path, mode="w") as archive:
                    archive.add(temp_out_path, recursive=True, arcname=".")
        else:  # directory
            out_path.mkdir(exist_ok=True)
            out_db = ms.database.JSONDatabase(
                work_dir=str(out_path),
                module_equality=module_equality,
            )
            union_db = merge_ms_dbs(in_dbs)
            num_recs = len(union_db)
            db_to_json_db(union_db, out_db, append=append)
    print(f"Merge completed ({num_recs} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_db", type=str, nargs="+", help="input db files/dirs")
    parser.add_argument("--output", "-o", type=str, default=None, help="output file", required=True)
    parser.add_argument("--append", action="store_true", help="Append to existing non-empty out dbs")
    parser.add_argument("--module-equality", type=str, default="structural", help="module equality")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: handle dir/files/archive
    merge_ms_dbs_wrapper(args.in_db, args.output, module_equality=args.module_equality, append=args.append)
