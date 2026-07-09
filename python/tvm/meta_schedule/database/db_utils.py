import tarfile
import tempfile
from typing import Union
from pathlib import Path

from tvm import meta_schedule as ms
from tvm.meta_schedule.database.s3_json_database import S3JSONDatabase
from tvm.tir.tensor_intrin.riscv_cpu import *
from ime.tensor_intrin_ime import *
from wca.tensor_intrin_esp_pie import *


def load_ms_db_dir(in_db_dir, module_equality: str = "structural"):
    in_db_path = Path(in_db_dir)
    path_workload = in_db_path / "database_workload.json"
    path_tuning_record = in_db_path / "database_tuning_record.json"
    in_db = ms.database.JSONDatabase(
        work_dir=in_db_dir,
        # path_workload=str(path_workload),
        # path_tuning_record=str(path_tuning_record),
        module_equality=module_equality,
    )
    return in_db


def load_ms_db_s3(in_db_url, module_equality: str = "structural"):
    in_db = ms.database.s3_json_database.S3JSONDatabase(
        in_db_url,
        # readonly=True,
        module_equality=module_equality,
    )
    return in_db


def load_ms_db_file(in_db_file, module_equality: str = "structural"):
    in_db_path = Path(in_db_file)
    assert in_db_path.is_file()
    suffix = in_db_path.suffix
    assert suffix == ".json"
    if "workload" in in_db_path.stem:
        path_workload = in_db_path
        path_tuning_record = in_db_path.parent / in_db_path.name.replace("workload", "tuning_record")
    elif "tuning_record" in in_db_path.stem:
        path_tuning_record = in_db_path
        path_workload = in_db_path.parent / in_db_path.name.replace("tuning_record", "workload")
    else:
        raise ValueError("Invalid MS DB file name: {in_db_path.name}")
    in_db = ms.database.JSONDatabase(
        path_workload=str(path_workload),
        path_tuning_record=str(path_tuning_record),
        module_equality=module_equality,
    )
    return in_db


def load_ms_db_archive(in_db_archive, module_equality: str = "structural"):
    in_db_path = Path(in_db_archive)
    with tempfile.TemporaryDirectory() as tmpdirname:
        if tarfile.is_tarfile(in_db_path):
            temp_in_db_path = Path(tmpdirname) / "in_db"
            with tarfile.open(in_db_path) as f:
                f.extractall(path=temp_in_db_path)
            has_workdir = False
            if (temp_in_db_path / "work_dir").is_dir():
                has_workdir = True
                temp_in_db_path = temp_in_db_path / "work_dir"
            db = load_ms_db_dir(temp_in_db_path)
            # TODO: check if this works with tempdir? yield?
            # TODO: convert to memory db!
            return db
            # yield db
        else:
            raise ValueError(f"Unsupported format")


def load_ms_db_wrapper(in_arg, module_equality: str = "structural"):
    print("Loading MS database:", in_arg)
    if isinstance(in_arg, ms.Database):
        return in_arg
    # print("in_arg", in_arg)
    if str(in_arg).startswith("s3://"):
        return load_ms_db_s3(in_arg, module_equality=module_equality)
    in_path = Path(in_arg)
    assert in_path.exists()
    if in_path.is_dir():
        return load_ms_db_dir(in_path, module_equality=module_equality)
    if in_path.is_file():
        if in_path.suffix == ".json":
            return load_ms_db_file(in_path, module_equality=module_equality)
        assert in_path.suffix in [".tar"]
        return load_ms_db_archive(in_path, module_equality=module_equality)
        # with load_ms_db_archive(in_path, module_equality=module_equality) as db:
        #     yield db


def db_to_json_db(db: Union[ms.database.UnionDatabase, ms.database.MemoryDatabase], json_db, append: bool = False):
    if len(json_db) > 0:
        assert append, "Out DB is non-empty. Remove first or use append=True"
    for rec in db.get_all_tuning_records():
        if not json_db.has_workload(rec.workload.mod):
            json_db.commit_workload(rec.workload.mod)
        json_db.commit_tuning_record(rec)
    # return json_db
