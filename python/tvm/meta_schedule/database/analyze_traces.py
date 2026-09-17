import logging
import argparse
from pathlib import Path
from typing import Union, Optional
from collections import defaultdict

from tqdm import tqdm

import tvm
from tvm import meta_schedule as ms
from tvm import tir

# from tvm.tir.analysis import estimate_tir_flops

from .db_utils import load_ms_db_wrapper


from tvm.tir import stmt_functor


def structure_features(structure):
    first_r = structure.index("R")
    return {
        "num_s": structure.count("S"),
        "num_r": structure.count("R"),
        "s_before_first_r": structure[:first_r].count("S"),
        "s_after_first_r": structure[first_r + 1 :].count("S"),
    }


def trace_before_postproc(trace):
    for inst in trace.insts:
        if inst.kind.name == "EnterPostproc":
            break
        yield inst


def detect_write_reuse(trace):
    info = {
        "used": False,
        "cache_write_count": 0,
        "placement": [],
    }

    cache_write_outputs = set()

    for inst in trace_before_postproc(trace):
        kind = inst.kind.name

        if kind == "CacheWrite":
            info["used"] = True
            info["cache_write_count"] += 1

            for out in inst.outputs:
                cache_write_outputs.add(out)

        elif kind in ("ReverseComputeAt", "ComputeAt"):
            if len(inst.inputs) > 0:
                block_rv = inst.inputs[0]

                if block_rv in cache_write_outputs:
                    info["placement"].append(kind)

    return info


def as_int(expr):
    if isinstance(expr, tir.IntImm):
        return int(expr)
    return None


def analyze_final_mod(mod):
    result = {
        "unroll_loops": [],
        "vectorized_loops": [],
        "buffer_loads": [],
        "buffer_stores": [],
        "loops": [],
    }

    for gv, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue

        def visit(node):
            if isinstance(node, tir.For):
                extent = as_int(node.extent)

                loop_info = {
                    "var": node.loop_var.name,
                    "extent": extent,
                    "kind": str(node.kind),
                    "annotations": dict(node.annotations),
                }

                result["loops"].append(loop_info)

                # Explicit vectorized loop
                if node.kind == tir.ForKind.VECTORIZED:
                    result["vectorized_loops"].append(loop_info)

                # Unroll-related annotations
                anns = node.annotations

                if "pragma_unroll_explicit" in anns or "pragma_auto_unroll_max_step" in anns:
                    result["unroll_loops"].append(loop_info)

            elif isinstance(node, tir.BufferLoad):
                result["buffer_loads"].append(
                    {
                        "buffer": node.buffer.name,
                        "indices": [str(x) for x in node.indices],
                    }
                )

            elif isinstance(node, tir.BufferStore):
                result["buffer_stores"].append(
                    {
                        "buffer": node.buffer.name,
                        "indices": [str(x) for x in node.indices],
                        "value": str(node.value),
                    }
                )

        stmt_functor.post_order_visit(func.body, visit)

    return result


def get_axis_info(sch, block_name, func_name="main"):
    block_rv = sch.get_block(block_name, func_name=func_name)
    block = sch.get(block_rv)

    result = []

    spatial_idx = 0
    reduction_idx = 0

    for iter_var in block.iter_vars:
        if iter_var.iter_type == tir.IterVar.DataPar:
            result.append(
                {
                    "label": f"S{spatial_idx}",
                    "axis_type": "S",
                }
            )
            spatial_idx += 1

        elif iter_var.iter_type == tir.IterVar.CommReduce:
            result.append(
                {
                    "label": f"R{reduction_idx}",
                    "axis_type": "R",
                }
            )
            reduction_idx += 1

        else:
            result.append(
                {
                    "label": f"U{len(result)}",
                    "axis_type": "U",
                }
            )

    return result


def get_axis_labels(sch, block_name, func_name="main"):
    block_rv = sch.get_block(block_name, func_name=func_name)
    block = sch.get(block_rv)

    labels = []
    spatial_idx = 0
    reduction_idx = 0

    for iter_var in block.iter_vars:
        if iter_var.iter_type == tir.IterVar.DataPar:
            labels.append(f"S{spatial_idx}")
            spatial_idx += 1
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            labels.append(f"R{reduction_idx}")
            reduction_idx += 1
        else:
            labels.append(f"?{len(labels)}")

    return labels


def find_removable_tiling_levels(structure, tiles):
    """
    Find tiling-structure positions that are unit across all axes
    of the corresponding type.

    Example:
        structure = "SSRSRS"
        tiles = [
            {"axis": "S0", "axis_type": "S", "decision": (1, 16, 1, 8)},
            {"axis": "S1", "axis_type": "S", "decision": (4, 2, 16, 1)},
            {"axis": "R0", "axis_type": "R", "decision": (128, 1)},
        ]

    Returns information about removable positions.
    """

    spatial_tiles = [t["decision"] for t in tiles if t["axis_type"] == "S"]
    reduction_tiles = [t["decision"] for t in tiles if t["axis_type"] == "R"]

    num_s_levels = structure.count("S")
    num_r_levels = structure.count("R")

    # Sanity checks
    for decision in spatial_tiles:
        assert len(decision) == num_s_levels, (
            f"Spatial tile {decision} has {len(decision)} factors, "
            f"but structure {structure} has {num_s_levels} S levels"
        )

    for decision in reduction_tiles:
        assert len(decision) == num_r_levels, (
            f"Reduction tile {decision} has {len(decision)} factors, "
            f"but structure {structure} has {num_r_levels} R levels"
        )

    removable = []

    s_idx = 0
    r_idx = 0

    for structure_pos, kind in enumerate(structure):
        if kind == "S":
            factors = [decision[s_idx] for decision in spatial_tiles]

            is_removable = len(factors) > 0 and all(int(x) == 1 for x in factors)

            if is_removable:
                removable.append(
                    {
                        "position": structure_pos,
                        "kind": "S",
                        "level": s_idx,
                        "factors": tuple(int(x) for x in factors),
                    }
                )

            s_idx += 1

        elif kind == "R":
            factors = [decision[r_idx] for decision in reduction_tiles]

            is_removable = len(factors) > 0 and all(int(x) == 1 for x in factors)

            if is_removable:
                removable.append(
                    {
                        "position": structure_pos,
                        "kind": "R",
                        "level": r_idx,
                        "factors": tuple(int(x) for x in factors),
                    }
                )

            r_idx += 1

        else:
            raise ValueError(f"Unsupported tiling structure character: {kind}")

    return removable


def remove_tiling_levels(structure, tiles, removable):
    remove_positions = {x["position"] for x in removable}

    reduced_structure = "".join(c for i, c in enumerate(structure) if i not in remove_positions)

    remove_s_levels = {x["level"] for x in removable if x["kind"] == "S"}
    remove_r_levels = {x["level"] for x in removable if x["kind"] == "R"}

    reduced_tiles = []

    for tile in tiles:
        axis_type = tile["axis_type"]
        decision = tile["decision"]

        if axis_type == "S":
            remove_levels = remove_s_levels
        elif axis_type == "R":
            remove_levels = remove_r_levels
        else:
            reduced_tiles.append(dict(tile))
            continue

        new_decision = tuple(int(x) for i, x in enumerate(decision) if i not in remove_levels)

        new_tile = dict(tile)
        new_tile["decision"] = new_decision
        reduced_tiles.append(new_tile)

    return reduced_structure, reduced_tiles


def detect_rule_usage_from_trace(trace):
    kinds = [inst.kind.name for inst in trace_before_postproc(trace)]
    kind_set = set(kinds)
    # print("kind_set", kind_set)

    return {
        "AddRFactor": {
            "used": "RFactor" in kind_set,
            "matching_insts": [k for k in kinds if k == "RFactor"],
        },
        "AutoInline": {
            "used": bool({"ComputeInline", "ReverseComputeInline"} & kind_set),
            "matching_insts": [k for k in kinds if k in ("ComputeInline", "ReverseComputeInline")],
        },
    }


def get_trace_block_names(trace):
    block_names = []

    for inst in trace.insts:
        if inst.kind.name == "GetBlock":
            # GetBlock attrs are typically:
            #   [block_name, func_name]
            block_name = str(inst.attrs[0])

            if block_name != "root" and block_name not in block_names:
                block_names.append(block_name)

    return block_names


def analyze_ms_db(
    in_db,
    print_mod: bool = False,
    print_trace: bool = False,
    print_info: bool = False,
    dump_dir: Optional[Union[str, Path]] = None,
):
    # print("DB", in_db, dir(in_db))
    recs = in_db.get_all_tuning_records()
    # print("recs", recs, len(recs))
    workloads = []
    targets = []
    annotation_hist = defaultdict(int)
    annotation_val_hist = defaultdict(lambda: defaultdict(int))
    inst_hist = defaultdict(int)
    # TODO: handle postproc
    original_mod_strs = []
    pre_postproc_mod_strs = []
    final_mod_strs = []
    lowered_mod_strs = []

    progress = True  # TODO: expose
    # TODO: process_pool
    for rec in tqdm(recs, disable=not progress):
        # TEMP
        # if len(lowered_mod_strs) > 10:
        #     break
        # print("rec.args_info", rec.args_info)
        # args_info = rec.args_info
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
        sch_original = tir.Schedule(workload.mod)
        trace_block_names = get_trace_block_names(rec.trace)
        if print_mod:
            print("sch.mod before apply", sch_original.mod)
        original_mod_strs.append(str(sch_original.mod))
        sch_pre_postproc = tir.Schedule(workload.mod)
        rec.trace.apply_to_schedule(
            sch_pre_postproc,
            remove_postproc=True,
        )
        # print("sch", sch, dir(sch))
        if print_mod:
            print("sch.mod before postproc", sch_pre_postproc.mod)
        pre_postproc_mod_strs.append(str(sch_pre_postproc.mod))
        # TODO: do not hardcode block name!
        # axis_labels = {
        #     "T_matmul_NT": get_axis_labels(sch, "T_matmul_NT"),
        # }
        # axis_info = {
        #     "T_matmul_NT": get_axis_info(sch_original, "T_matmul_NT"),
        # }
        axis_info = {}
        for block_name in trace_block_names:
            try:
                axis_info[block_name] = get_axis_info(
                    sch_original,
                    block_name,
                )
            except tir.ScheduleError:
                # Some blocks may only exist after transformations.
                pass

        # print("axis_labels", axis_labels)
        if print_info:
            print("axis_info", axis_info)
        sch_final = tir.Schedule(workload.mod)
        rec.trace.apply_to_schedule(
            sch_final,
            remove_postproc=False,
        )
        final_mod_strs.append(str(sch_final.mod))
        # print("sch", sch, dir(sch))
        if print_mod:
            print("sch.mod after apply", sch_final.mod)
        lowered_mod = tvm.lower(sch_final.mod)
        if print_mod:
            print("lowered_mod", lowered_mod)
        lowered_mod_strs.append(str(lowered_mod))
        # TODO: refactor mod analysis to other func/file
        # input("!")
        if target_str not in targets:
            targets.append(target_str)
        # target2recs[target_str].append(rec)
        if print_trace:
            print("rec.trace", rec.trace, dir(rec.trace))
        # print("rec.trace.insts", rec.trace.insts, dir(rec.trace.insts))
        # print("decisions", rec.trace.decisions)
        output_decisions = {}
        # decision_map = {}
        raw_decision_map = {}
        for inst, decision in rec.trace.decisions.items():
            raw_decision_map[inst] = decision
            # print("k2", k2, type(k2), dir(k2))
            # print("v2", v2, type(v2), dir(v2))
            outputs = inst.outputs
            # print("outputs", outputs)
            assert len(outputs) > 0

            resolved_decision = decision

            if inst.kind.name == "SampleCategorical":
                # decision is the index into candidates.
                #
                # For:
                #   candidates=[0, 16, 64, 512]
                #   decision=3
                #
                # resolved_decision = 512
                candidates = inst.attrs[0]
                resolved_decision = candidates[int(decision)]
            if len(outputs) == 1:
                outp = outputs[0]
                output_decisions[outp] = resolved_decision
            else:
                assert len(decision) == len(outputs)
                for j, outp in enumerate(outputs):
                    # print("outp", outp, type(outp), dir(outp))
                    output_decisions[outp] = decision[j]
        if print_info:
            print("output_decisions", output_decisions)

        # Map LoopRV -> descriptive name
        # loop_names = {}
        block_info = {}
        loop_info = {}

        # Map BlockRV -> descriptive name
        block_names = {}

        tiles = []
        candidate_annotations = {}
        structures = set()
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

            if kind == "GetBlock":
                # Usually attrs contains block name + func name.
                # For your trace this corresponds to:
                # sch.get_block(name="T_matmul_NT", func_name="main")
                block_name = str(inst.attrs[0])
                for out in inst.outputs:
                    block_info[out] = {
                        "block": block_name,
                        "source_block": block_name,
                    }
                for out in inst.outputs:
                    block_names[out] = block_name

            elif kind == "GetLoops":
                assert len(inst.inputs) == 1
                block_rv = inst.inputs[0]
                info = block_info.get(block_rv)
                block_name = info["source_block"] if info else None
                # block_name = block_names.get(block_rv, "<unknown_block>")

                # labels = axis_labels.get(block_name)
                infos = axis_info.get(block_name)

                for loop_idx, loop_rv in enumerate(inst.outputs):
                    if infos is not None and loop_idx < len(infos):
                        info = infos[loop_idx]

                        loop_info[loop_rv] = {
                            "block": block_name,
                            "axis": info["label"],
                            "axis_type": info["axis_type"],
                        }
                    else:
                        loop_info[loop_rv] = {
                            "block": block_name,
                            "axis": f"L{loop_idx}",
                            "axis_type": "U",
                        }

            elif kind == "Split":
                src_loop = inst.inputs[0]
                src_info = loop_info.get(src_loop)

                if src_info is not None:
                    for out_loop in inst.outputs:
                        loop_info[out_loop] = dict(src_info)
            elif kind == "Blockize":
                target_loop = inst.inputs[0]
                new_block = inst.outputs[0]

                src = loop_info.get(target_loop)

                if src is not None:
                    block_info[new_block] = {
                        "block": "<blockized>",
                        "source_block": src["block"],
                    }
            elif kind == "SamplePerfectTile":
                decision = raw_decision_map.get(inst)
                if decision is None:
                    continue
                # tiles.append(tuple(int(x) for x in decision))
                loop_rv = inst.inputs[0]
                # loop_name = loop_names.get(loop_rv, str(loop_rv))
                info = loop_info.get(
                    loop_rv,
                    {
                        "block": "<unknown>",
                        "axis": str(loop_rv),
                        "axis_type": "U",
                    },
                )

                tile = tuple(x for x in decision)

                tiles.append(
                    {
                        "block": info["block"],
                        "axis": info["axis"],
                        "axis_type": info["axis_type"],
                        "decision": tile,
                    }
                )
            elif kind == "Annotate":
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
                candidate_annotations[key] = val
                annotation_val_hist[key][val] += 1
                if key == "meta_schedule.tiling_structure":
                    structures.add(val)

        # target2workloads[target_str].add(workload)
        feats = {}
        for structure in structures:
            feats_ = structure_features(structure)
            feats[structure] = feats_

        write_reuse = detect_write_reuse(rec.trace)
        parallel = candidate_annotations.get("meta_schedule.parallel")
        vectorize = candidate_annotations.get("meta_schedule.vectorize")
        unroll = candidate_annotations.get("meta_schedule.unroll_explicit")
        removable = find_removable_tiling_levels(structure, tiles)
        if print_info:
            print("structures", structures)
            print("feats", feats)
            print("write_reuse", write_reuse)
            print("parallel", parallel)
            print("vectorize", vectorize)
            print("unroll", unroll)
            print("tiles", tiles)
            print("removable", removable)
        if len(removable) > 0:

            new_structure, new_tiles = remove_tiling_levels(
                structure,
                tiles,
                removable,
            )

            if print_info:
                print("new_structure", new_structure)
                print("new_tiles", new_tiles)
        max_spatial_inner_factor = max(
            (t["decision"][-1] for t in tiles if t["axis_type"] == "S"),
            default=None,
        )

        max_reduction_inner_factor = max(
            (t["decision"][-1] for t in tiles if t["axis_type"] == "R"),
            default=None,
        )

        result = analyze_final_mod(sch_final.mod)
        unroll_extents = [x["extent"] for x in result["unroll_loops"] if x["extent"] is not None]

        max_unroll_loop_extent = max(unroll_extents, default=None)
        vector_extents = [x["extent"] for x in result["vectorized_loops"] if x["extent"] is not None]

        max_vectorized_extent = max(vector_extents, default=None)
        rule_usage = detect_rule_usage_from_trace(rec.trace)
        if print_info:
            print("max_spatial_inner_factor", max_spatial_inner_factor)
            print("max_reduction_inner_factor", max_reduction_inner_factor)
            print("res", result)
            print("max_unroll_loop_extent", max_unroll_loop_extent)
            print("max_vectorized_extent", max_vectorized_extent)
            print("rule_usage", rule_usage)
            print("AddRFactor used:", rule_usage["AddRFactor"]["used"])
            print("AutoInline used:", rule_usage["AutoInline"]["used"])

    print("len(workloads)", len(workloads))
    print("len(targets)", len(targets))
    print("annotation_hist", annotation_hist)
    print("annotation_val_hist", annotation_val_hist)
    print("inst_hist", inst_hist)
    # ---
    # print("original_mod_strs[0]", original_mod_strs[0])
    # print("original_mod_strs[-1]", original_mod_strs[-1])
    # print("pre_postproc_mod_strs[0]", pre_postproc_mod_strs[0])
    # print("pre_postproc_mod_strs[-1]", pre_postproc_mod_strs[-1])
    # print("final_mod_strs[0]", final_mod_strs[0])
    # print("final_mod_strs[-1]", final_mod_strs[-1])
    # print("lowered_mod_strs[0]", lowered_mod_strs[0])
    # print("lowered_mod_strs[-1]", lowered_mod_strs[-1])

    print("original_mod_strs", len(original_mod_strs), len(set(original_mod_strs)))
    print("pre_postproc_mod_strs", len(pre_postproc_mod_strs), len(set(pre_postproc_mod_strs)))
    print("final_mod_strs", len(final_mod_strs), len(set(final_mod_strs)))
    print("lowered_mod_strs", len(lowered_mod_strs), len(set(lowered_mod_strs)))
    if dump_dir:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(exist_ok=True)
        lowered_mods_file = dump_dir / "lowered_mods.json"
        import json

        with open(lowered_mods_file, "w") as f:
            json.dump(lowered_mod_strs, f, indent=0)
    return annotation_hist, annotation_val_hist, inst_hist


def analyze_ms_db_wrapper(
    db_arg,
    print_mod: bool = False,
    print_trace: bool = False,
    print_info: bool = False,
    dump_dir: Optional[Union[str, Path]] = None,
):
    db = load_ms_db_wrapper(db_arg)
    # print("db", db)
    assert isinstance(db, ms.Database)
    _ = analyze_ms_db(db, print_mod=print_mod, print_trace=print_trace, print_info=print_info, dump_dir=dump_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str, help="input file/dir")
    parser.add_argument("--dump", type=str, default=None, help="TODO")
    parser.add_argument("--print-mod", action="store_true", help="TODO")
    parser.add_argument("--print-trace", action="store_true", help="TODO")
    parser.add_argument("--print-info", action="store_true", help="TODO")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    analyze_ms_db_wrapper(
        args.db, print_mod=args.print_mod, print_trace=args.print_trace, print_info=args.print_info, dump_dir=args.dump
    )
