import os
import tempfile
from urllib.parse import urlparse, parse_qs
from typing import Callable, List, Optional

import boto3
from botocore.client import Config

import tvm
from tvm import meta_schedule as ms
from tvm import relay, tir
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import TuningRecord, Workload
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule


@ms.utils.derived_object
class S3JSONDatabase(ms.database.PyDatabase):
    def __init__(self, url: str, readonly: bool = False, module_equality: str = "structural"):
        super().__init__()

        self.url = url
        # print("self.url", self.url)
        self.readonly = readonly

        parsed = urlparse(url)
        # print("parsed", parsed)
        self.bucket = parsed.netloc
        # print("self.bucket", self.bucket)
        self.prefix = parsed.path.lstrip("/")
        # print("self.prefix", self.prefix)
        query = parse_qs(parsed.query)
        # print("query", query)

        endpoint = query.get("endpoint", [None])[0]
        # print("endpoint", endpoint)
        region = query.get("region", ["us-east-1"])[0]
        # print("region", region)

        session = boto3.Session()
        creds = session.get_credentials()

        # print("Access key:", creds.access_key)
        # print("Secret key:", "SET" if creds.secret_key else None)

        # self.s3 = boto3.client("s3")
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint,
        )
        # print(self.s3.list_buckets())
        # print(self.s3.list_objects_v2(Bucket="tophub"))

        # tempdir
        self.tmpdir = tempfile.TemporaryDirectory()
        self.local_dir = self.tmpdir.name
        # print("self.local_dir", self.local_dir)

        self.path_workload = os.path.join(self.local_dir, "database_workload.json")
        self.path_records = os.path.join(self.local_dir, "database_tuning_record.json")

        self._download()
        # print("download_done")
        # input("!")

        # internal JSONDatabase
        self._db = ms.database.JSONDatabase(
            path_workload=self.path_workload,
            path_tuning_record=self.path_records,
            allow_missing=True,
            module_equality=module_equality,
        )

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _download_file(self, key, local_path):
        try:
            self.s3.download_file(self.bucket, key, local_path)
        except self.s3.exceptions.ClientError as ex:
            raise ex

    def _download(self):
        self._download_file(self._key("database_workload.json"), self.path_workload)
        self._download_file(self._key("database_tuning_record.json"), self.path_records)

    def _upload(self):
        if self.readonly:
            return

        self.s3.upload_file(self.path_workload, self.bucket, self._key("database_workload.json"))
        self.s3.upload_file(self.path_records, self.bucket, self._key("database_tuning_record.json"))

    def has_workload(self, mod: IRModule) -> bool:
        return self._db.has_workload(mod)

    def commit_workload(self, mod: IRModule):
        wl = self._db.commit_workload(mod)
        self._upload()
        return wl

    def commit_tuning_record(self, record):
        self._db.commit_tuning_record(record)
        self._upload()

    def get_top_k(self, workload, top_k):
        return self._db.get_top_k(workload, top_k)

    def get_all_tuning_records(self):
        return self._db.get_all_tuning_records()

    def __len__(self):
        return len(self._db)


# if __name__ == "__main__":
#     url = "s3://tophub/tvm-db?endpoint=http://ryloth.eda.cit.tum.de:3900&region=garage"
#     read_only = False
#     db = S3JSONDatabase(url, readonly=read_only)
#     print("db", db)
#     print("len(db)", len(db))
#     recs = db.get_all_tuning_records()
#     print("recs", recs, len(recs))
#     workloads = []
#     workloads.append(recs[0].workload)
#     print("workloads", workloads, len(workloads))
#     db.commit_workload(workloads[0].mod)
#     topk = db.get_top_k(workloads[0], 2)
#     print("topk", topk, len(topk))
