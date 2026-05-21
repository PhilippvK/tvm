import time
import pickle
import datetime
import argparse
from pathlib import Path
import tempfile
import hashlib
from types import MappingProxyType
from collections import defaultdict

from tvm import meta_schedule as ms
from tvm.meta_schedule import postproc, schedule_rule

def generate_rules(rules=[], rfactor_max_innermost_factor=64, mlt_structure="SSRSRS", mlt_max_innermost_factor=64, unroll_max_steps=[0, 2, 4, 8, 16, 32, 64], unroll_explicit=True, intrin=None, swap_mlt_rules=None):
    # TODO: change rule order?
    mlt_rule = ms.schedule_rule.MultiLevelTiling(
            structure=mlt_structure,
            tile_binds=None,
            max_innermost_factor=int(mlt_max_innermost_factor),
            # max_innermost_factor=4,
            vector_load_lens=None,
            reuse_read=None,
            reuse_write=ms.schedule_rule.ReuseType(
                req="may",
                levels=[1, 2],
                scope="global",
            )
    ) if "MultiLevelTiling" in rules else None
    if intrin is not None and intrin != "none":
        intrins = gen_intrins(intrin)
        # TODO: different structure?
        mlti_structure = mlt_structure if mlt_structure is not None else "SR"
        mlti_rules = [
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                intrin,
                structure=mlti_structure,
                tile_binds=None,
                max_innermost_factor=64,  # TODO: expose
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            )
            for intrin in intrins
        ]
    else:
        mlti_rules = []
    sch_rules = [
        *([ms.schedule_rule.ApplyCustomRule()] if "ApplyCustomRule" in rules else []),
        *([ms.schedule_rule.InlineConstantScalars()] if "InlineConstantScalars" in rules else []),
        *([ms.schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        )] if "AutoInline" in rules else []),
        *([ms.schedule_rule.AddRFactor(max_jobs_per_core=1, max_innermost_factor=int(rfactor_max_innermost_factor))] if "AddRFactor" in rules else []),
        *([mlt_rule] if mlt_rule is not None and swap_mlt_rules else []),
        *mlti_rules,
        *([mlt_rule] if mlt_rule is not None and not swap_mlt_rules else []),
        *([ms.schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=unroll_max_steps,
            # unroll_max_steps=[0, 2],
            unroll_explicit=unroll_explicit,
            # unroll_explicit=False,
        )] if "ParallelizeVectorizeUnroll" in rules else []),
        *([ms.schedule_rule.RandomComputeLocation()] if "RandomComputeLocation" in rules else []),
    ]
    return sch_rules
