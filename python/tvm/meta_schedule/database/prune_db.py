import logging
import argparse
import tarfile
import tempfile
from pathlib import Path
from tvm import meta_schedule as ms


def prune_ms_db(in_db, out_db=None, module_equality: str = "structural", append: bool = False):
    if out_db is None:
        out_db = ms.database.MemoryDatabase(module_equality=module_equality)
    else:
        if len(out_db) > 0:
            if isinstance(out_db, ms.database.MemoryDatabase):
                out_db = ms.database.MemoryDatabase(module_equality=module_equality)
            else:
                assert append, "Out DB is non-empty. Use append=True or MemoryDatabase"
    in_db.dump_pruned(out_db)
    return out_db


def prune_ms_db_dir(in_db_dir, out_db_dir, module_equality: str = "structural", append: bool = False):
    in_db_path = Path(in_db_dir)
    out_db_path = Path(out_db_dir)
    path_workload = in_db_path / "database_workload.json"
    path_tuning_record = in_db_path / "database_tuning_record.json"
    path_workload_ = out_db_path / "database_workload.json"
    path_tuning_record_ = out_db_path / "database_tuning_record.json"
    in_db = ms.database.JSONDatabase(
        path_workload=str(path_workload),
        path_tuning_record=str(path_tuning_record),
        module_equality=module_equality,
    )
    out_db_path.mkdir(exist_ok=True)
    out_db = ms.database.JSONDatabase(
        path_workload=str(path_workload_),
        path_tuning_record=str(path_tuning_record_),
        module_equality=module_equality,
    )
    if len(out_db) > 0:
        assert append, "Out DB is non-empty. Remove first or use append=True"
    prune_ms_db(in_db, out_db)


def prune_ms_db_file(in_db_file, out_db_file, module_equality: str = "structural", append: bool = False):
    in_db_path = Path(in_db_file)
    out_db_path = Path(out_db_file)
    assert in_db_path.is_file()
    suffix = in_db_path.suffix
    assert suffix == ".json"
    if "workload" in in_db_path.stem:
        assert "workload" in out_db_path.stem
        path_workload = in_db_path
        path_workload_ = out_db_path
        path_tuning_record = in_db_path.parent / in_db_path.name.replace("workload", "tuning_record")
        path_tuning_record_ = out_db_path.parent / out_db_path.name.replace("workload", "tuning_record")
    elif "tuning_record" in in_db_path.stem:
        assert "tuning_record" in out_db_path.stem
        path_tuning_record = in_db_path
        path_tuning_record_ = out_db_path
        path_workload = in_db_path.parent / in_db_path.name.replace("tuning_record", "workload")
        path_workload_ = out_db_path.parent / out_db_path.name.replace("tuning_record", "workload")
    else:
        raise ValueError("Invalid MS DB file name: {in_db_path.name}")
    in_db = ms.database.JSONDatabase(
        path_workload=str(path_workload),
        path_tuning_record=str(path_tuning_record),
        module_equality=module_equality,
    )
    out_db = ms.database.JSONDatabase(
        path_workload=str(path_workload_),
        path_tuning_record=str(path_tuning_record_),
        module_equality=module_equality,
    )
    if len(out_db) > 0:
        assert append, "Out DB is non-empty. Remove first or use append=True"
    prune_ms_db(in_db, out_db)


def prune_ms_db_archive(in_db_archive, out_db_archive, module_equality: str = "structural", append: bool = False):
    in_db_path = Path(in_db_archive)
    out_db_path = Path(out_db_archive)
    with tempfile.TemporaryDirectory() as tmpdirname:
        if tarfile.is_tarfile(in_db_path):
            assert in_db_path.suffix == out_db_archive.suffix
            temp_in_db_path = Path(tmpdirname) / "in_db"
            with tarfile.open(in_db_path) as f:
                f.extractall(path=temp_in_db_path)
            has_workdir = False
            if (temp_in_db_path / "work_dir").is_dir():
                has_workdir = True
                temp_in_db_path = temp_in_db_path / "work_dir"
            path_workload = out_db_path / "database_workload.json"
            path_tuning_record = out_db_path / "database_tuning_record.json"
            temp_out_db_path = Path(tmpdirname) / "out_db"
            temp_out_db_path.mkdir()
            if has_workdir:
                temp_out_db_path_ = temp_out_db_path / "work_dir"
            else:
                temp_out_db_path_ = temp_out_db_path
            prune_ms_db_dir(temp_in_db_path, temp_out_db_path_, module_equality=module_equality, append=append)
            with tarfile.open(out_db_path, mode="w") as archive:
                archive.add(temp_out_db_path, recursive=True)
        else:
            raise ValueError(f"Unsupported format")


def prune_ms_db_wrapper(in_arg, out_arg, module_equality: str = "structural", append: bool = False):
    if isinstance(in_arg, ms.Database):
        assert isinstance(out_arg, ms.Database)
        _ = prune_ms_db(in_arg, out_arg, module_equality=module_equality, append=append)
        return
    in_path = Path(in_arg)
    out_path = Path(out_arg) if out_arg is not None else None
    assert in_path.exists()
    if in_path.is_dir():
        if out_path is None:
            out_path = in_path.parent / (in_path.name + ".pruned")
        else:
            assert out_path.is_dir()
        prune_ms_db_dir(in_path, out_path, module_equality=module_equality, append=append)
        return
    if in_path.is_file():
        if in_path.suffix == ".json":
            if out_path is None:
                out_path = in_path.parent / (in_path.stem + ".pruned" + in_path.suffix)
            prune_ms_db_file(in_path, out_path, module_equality=module_equality, append=append)
        assert in_path.suffix in [".tar"]
        if out_path is None:
            out_path = in_path.parent / (in_path.stem + ".pruned" + in_path.suffix)
        prune_ms_db_archive(in_path, out_path, module_equality=module_equality, append=append)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="input file/dir",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="output file/dir")
    parser.add_argument("--append", action="store_true", help="Append to existing non-empty out dbs")
    parser.add_argument("--module-equality", type=str, default="structural", help="module equality")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: handle dir/files/archive
    prune_ms_db_wrapper(args.input, args.output, module_equality=args.module_equality, append=args.append)
