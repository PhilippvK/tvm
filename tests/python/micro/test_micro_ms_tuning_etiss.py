# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
from types import MappingProxyType
from typing import Optional
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
# from tvm.contrib import graph_executor
from tvm.contrib import utils
from tvm import meta_schedule as ms
from tvm.driver import tvmc

###
import tvm.micro.testing
from tvm.meta_schedule.runner import EvaluatorConfig

###
# from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

import logging
logging.basicConfig(level=logging.ERROR)

from tvm.meta_schedule.logging import get_logger
get_logger("xgb_model").setLevel(logging.ERROR)

DIR = Path(__file__).parent.resolve()
BASE_DIR = DIR / "../../../../"


MS_DISPATCH = 1  # silent?
# MS_DISPATCH = 2  # verbose
# MS_DISPATCH = ?  # error


import numpy as np
import pytest
from types import MappingProxyType
import pathlib
import json
import tvm
import tvm.testing
from tvm import relay
from tvm import transform
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor, utils
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Schedule, Trace
from tvm.tir.tensor_intrin.rocm import AMDGPU_SDOT4_INTRIN
from tvm.tir.tensor_intrin.arm_cpu import DP4A_S8S8S32_INTRIN
from tvm.tir.tensor_intrin.hexagon import VRMPY_i8i8i32_INTRIN
from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_NEON_INTRIN
# from tvm.tir.tensor_intrin.cfu import CFU_MAC_i8i8i32_INTRIN

from tvm.tir.schedule.analysis import has_block

C_CODE = """
#ifndef CFU_KERNEL_CODE
#define CFU_KERNEL_CODE
#include <stdint.h>

#define MODE MODE_EMUL

#if MODE == MODE_EMUL
typedef struct {
    uint32_t word0;
    uint32_t word1;
} weights_t;

typedef struct {
    int8_t x[4];
} codebook_t;

static weights_t current_weights = {.word0 = 0, .word1 = 0};
// static codebook_t current_codebook = {.byte0 = 0, .byte1 = 0, .byte2 = 0, .byte3 = 0};
static codebook_t current_codebook = {.x = {0, 0, 0, 0}};

static int32_t acc = 0;
#endif  // MODE

#if MODE == MODE_EMUL
static void __attribute__((always_inline)) inline push_weights_4b(uint32_t word0, uint32_t word1) {
    // printf("push_weights_4b\\n");
    current_weights.word0 = word0;
    current_weights.word1 = word1;
}
#elif MODE == MODE_CFU
static int32_t __attribute__((always_inline)) inline push_weights_4b(uint32_t word0, uint32_t word1) {
    // printf("push_weights_4b\\n");
#ifdef SEAL5
    return __builtin_riscv_xcfu_cfu0_push_weights_4b(word0);
#else
    cfu_op0_hw(CFU_OPCODE_PUSH_WEIGHTS_4B, word0, word1);
#endif  // SEAL5
}
#endif  // MODE

static int32_t __attribute__((always_inline)) inline alu_mac(uint32_t word0, uint32_t word1) {
    // printf("alu_mac\\n");
#if MODE == MODE_EMUL
    acc += current_weights.word0 * word0;
    acc += current_weights.word1 * word1;
#elif MODE == MODE_CFU
#ifdef SEAL5
    int32_t acc = __builtin_riscv_xcfu_cfu0_alu_mac(word0, word1);
#else
    cfu_op0_hw(CFU_OPCODE_ALU_MAC, word0, word1);
#endif  // SEAL5
#endif  // MODE
    // TODO: if non-zero?
    return acc;
}

static void __attribute__((always_inline)) inline alu_rst() {
    // printf("alu_rst\\n");
#if MODE == MODE_EMUL
    acc = 0;
#elif MODE == MODE_CFU
#ifdef SEAL5
    __builtin_riscv_xcfu_cfu0_alu_rst();
#else
    cfu_op0_hw(CFU_OPCODE_ALU_RST, 0, 0);
#endif  // SEAL5
#endif  // MODE
}

static int32_t __attribute__((always_inline)) inline get_acc() {
    // printf("get_acc\\n");
#if MODE == MODE_EMUL
    return acc;
#elif MODE == MODE_CFU
#ifdef SEAL5
    return __builtin_riscv_xcfu_cfu0_alu_mac(0, 0);
#else
    return cfu_op0_hw(CFU_OPCODE_ALU_MAC, 0, 0);  // TODO: opcode for load?
#endif  // SEAL5
#endif  // MODE
}

int32_t cfu_kernel_32x(int8_t* data_ptr, int8_t* weights_ptr, int32_t* acc) {
    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t code_word1 = *((uint32_t*)(weights_ptr + 4));
    uint32_t* act_words = (uint32_t*)data_ptr;

    alu_rst();
    // cfu_op0(CFU_FUNCT7_PUSH_WEIGHTS, code_word0, code_word1);
    push_weights_4b(code_word0, code_word1);
    for (int i = 0; i < 4; i++) {
        // cfu_op0(CFU_FUNCT7_ALU_MAC, act_words[2 * i], act_words[2 * i + 1]);
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }
    *acc = get_acc();
    return 42;
}
#endif  // CFU_KERNEL_CODE
"""

@derived_object
class ImportCPostprocess(ms.postproc.PyPostproc):
    """A postproc that always fails."""

    def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
        pass

    def apply(self, sch: Schedule) -> bool:
        # print("apply", sch)
        # return False
        # has = has_block(sch, "block")
        has = has_block(sch, "root")
        # print("has", has)
        if has:
            # block = sch.get_block("block")
            block = sch.get_block("root")
            # print("block", block)
            sch.annotate(block, "foo", "bar")
            sch.annotate(block, "pragma_import_c", C_CODE)
        # print("sch", sch)
        # input(">")
        return True

    def clone(self) -> "ImportCPostprocess":
        return ImportCPostprocess()

    def __str__(self) -> str:
        return "ImportCPostprocess"


def get_tuning_config(enable_intrin: bool = False):
    def _get_sch_rules(intrin: Optional[str] = None):
        structure_lookup = {
            AMDGPU_SDOT4_INTRIN: "SSSRRSRS",
            VRMPY_i8i8i32_INTRIN: "SRSRS",
            DP4A_S8S8S32_INTRIN: "SR",
            # ARM_DOT_4x4_i8_NEON_INTRIN: "SR",
            ARM_DOT_4x4_i8_NEON_INTRIN: "RS",
        }
        return [
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.InlineConstantScalars(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            # ms.schedule_rule.AddRFactor(max_jobs_per_core=1, max_innermost_factor=64),
            *([ms.schedule_rule.MultiLevelTilingWithIntrin(
                    intrin,
                    structure=structure_lookup[intrin],
                    # tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                    # max_innermost_factor=32,
                    # vector_load_lens=[1, 2, 3, 4],
                    # reuse_read=ms.schedule_rule.ReuseType(
                    #     req="must",
                    #     levels=[4],
                    #     scope="shared",
                    # ),
                    # reuse_write=ms.schedule_rule.ReuseType(
                    #     req="must",
                    #     levels=[3],
                    #     scope="local",
                    # ),
                )] if intrin is not None else []),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
                # structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 2, 4, 8, 16, 32, 64],
                unroll_explicit=True,
                # unroll_explicit=False,
            ),
            ms.schedule_rule.RandomComputeLocation(),
        ]

    def _get_postprocs():
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(),
            # ms.postproc.RewriteTensorize(vectorize_init_loop=True),
            *([ImportCPostprocess()] if enable_intrin else []),
        ]

    def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
            # ms.mutator.Parallel(): 0.02,
        }

    default_intrin = DP4A_S8S8S32_INTRIN
    intrin = default_intrin if enable_intrin else None
    sch_rules = _get_sch_rules(intrin)
    postprocs = _get_postprocs()
    mutator_probs = _get_mutator_probs()
    return sch_rules, postprocs, mutator_probs


def _schedule_dummy():

    def schedule_fn(sch, block=None) -> bool:
        return True

    return schedule_fn


def create_relay_module():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    np.random.seed(seed=1234)
    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info
###


@pytest.mark.parametrize("alter_op", [
    # False,
    True
])
@pytest.mark.parametrize("toolchain", [
    "gcc",
    # "llvm",
])
@pytest.mark.parametrize("target", [
    # "c -num-cores 1",
    "c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc",
    # "llvm -num-cores 1 -mcpu generic-rv64 -mtriple=riscv64-unknown-elf -mabi lp64d -mattr=+d,+f,+m,+64bit -model=etiss-rv64gc -global-isel=1 -global-isel-abort=2 -basic-block-sections=1",
])
@pytest.mark.parametrize("num_trials_per_iter,max_trials_per_task,max_trials_global", [
    # (0, 0, 1000000),
    # (1, 1, 1000000),
    # (5, 10, 1000000),
    (5, 20, 1000000),
    # (5, 50, 1000000),
    # (5, 100, 1000000),
    # (5, 200, 1000000),
    # (5, 400, 1000000),
    # (5, 800, 1000000),
    # (5, 1600, 1000000),
])
@pytest.mark.parametrize("enable_custom,enable_intrin", [
    # (False, False),
    # (True, False),
    (True, True),
])
@pytest.mark.parametrize("module_equality", ["ignore-ndarray"])
@pytest.mark.parametrize("model", [
    # "default",
    # "resnet_clustered",
    # "resnet_clustered_layer14",
    "resnet_clustered_layer10",
])
@pytest.mark.parametrize("transform_layout", [
    # False,
    True,
])
@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule(alter_op, toolchain, target, num_trials_per_iter, max_trials_per_task, max_trials_global, enable_custom, enable_intrin, module_equality, model, transform_layout):
    print()
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    import pathlib
    platform = DIR / "../../../../microtvm-etiss-template/template_project"
    print("platform", platform)
    options = {
        "verbose": True,
        "quiet": True,
        "gcc_prefix": str(BASE_DIR / "install/rv32gc_ilp32d"),
        "gcc_name": "riscv32-unknown-elf",
        "llvm_dir": str(BASE_DIR / "install/seal5_llvm"),
        "toolchain": toolchain,
        "etiss_script": str(BASE_DIR / "etiss/build/install/bin/run_helper.sh"),
        "etiss_args": "",
        "arch": "rv32gc_zicsr_zifencei",
        "abi": "ilp32d",
        "cpu_arch": "RV32IMACFD",
        "cpu_freq": 100000000,
    }
    opt_level = 3
    pass_config = {
        "tir.disable_vectorize": True
    }
    disabled_pass = []
    if not alter_op:
        disabled_pass += ["AlterOpLayout"]

    KEEP = True
    if KEEP:
        base_dir = Path("/tmp/base")
        now = datetime.now()
        ts = now.strftime("%Y%m%dT%H%M%S")
        def sanitize(x):
            if not isinstance(x, str):
                x = str(x)
            x = x.replace(" ", "").replace(",", "").replace("/", "").replace(";", "").replace("=", "-").replace("+", "")
            return x
        fields = [target, toolchain, alter_op, num_trials_per_iter, max_trials_per_task, max_trials_global, ts, opt_level, enable_custom, enable_intrin, module_equality, model, *sum(map(list, pass_config.items()), []), *[f"no{x}" for x in disabled_pass]]
        label = "-".join([sanitize(x) for x in fields])
        work_dir_path = base_dir / label
    else:
        work_dir = utils.tempdir()
        work_dir_path = work_dir.path
    print("work_dir_path", work_dir_path)
    # MODEL = "default"
    # MODEL = "resnet_clustered"
    # MODEL = "resnet_clustered_layer14"
    MODEL = "resnet_clustered_layer10"

    if MODEL == "default":
        # input("1")
        mod, params, model_info = create_relay_module()
        input_name = model_info["in_tensor"]
        input_shape = model_info["in_shape"]
        input_dtype = model_info["in_dtype"]
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered":
        model = tvmc.load(
            str(BASE_DIR / "models/pretrainedResnet_clustered_quant_remap.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer10":  # conv2d(1x16x16x32, 64x1x1x32, 1x8x8x64)
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer10.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 16, 16, 32]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer11":  # add
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer11.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer12":  # avg pool
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer12.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer13":  # rehape
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer13.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer14":  # fully connected
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer14.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    elif MODEL == "resnet_clustered_layer15":  # softmax
        model = tvmc.load(
            str(BASE_DIR / "models/layers/pretrainedResnet_clustered_quant_remap_packed_layer15.tflite")
        )
        mod = model.mod
        params = model.params

        input_shape = [1, 32, 32, 3]
        input_dtype = "int8"
        data_sample = np.random.rand(*input_shape).astype(input_dtype)
    else:
        assert False, f"Unsupported Model: {MODEL}"

    if transform_layout:
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config=pass_config,
            disabled_pass=disabled_pass,
        ):
            desired_layouts = {"qnn.conv2d": ["NHWC", "HWOI"]}
            # desired_layouts = {"qnn.conv2d": ["NHWC", "OHWI"]}

            # Convert the layout of the graph where possible.
            seq = transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

    print("model.mod", model.mod)
    link_params = True

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": link_params})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    # SKIP_TUNING = True
    SKIP_TUNING = False
    builder = get_local_builder_micro()


    with ms.Profiler() as profiler:
        if not SKIP_TUNING:
            # print("a1")
            if enable_custom:
                sch_rules, postprocs, mutator_probs = get_tuning_config(enable_intrin)
                space = ms.space_generator.PostOrderApply(
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs=mutator_probs,
                )
                strategy = "evolutionary"
            else:
                space = "post-order-apply"
                strategy = "evolutionary"
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            with get_rpc_runner_micro(
                platform=platform, options=options, session_timeout_sec=120, evaluator_config=evaluator_config,
            ) as runner:
                # print("runner", runner)
                # if True:
                if max_trials_global > 0:
                    db: ms.Database = ms.relay_integration.tune_relay(
                        mod=mod,
                        params=params,
                        target=target,
                        builder=builder,
                        runner=runner,
                        strategy=strategy,
                        space=space,
                        # num_trials_per_iter=2,
                        num_trials_per_iter=num_trials_per_iter,
                        # max_trials_per_task=10,
                        max_trials_per_task=max_trials_per_task,
                        # max_trials_global=100,
                        max_trials_global=max_trials_global,
                        work_dir=str(work_dir_path),
                        module_equality=module_equality,
                        pass_config=MappingProxyType(
                            pass_config,
                            # {
                            #     "tir.disable_vectorize": True,
                            #     # "tir.enable_debug": True,
                            # }
                        ),
                        disabled_pass=disabled_pass,
                    )
                else:
                    # db = ms.database.MemoryDatabase()
                    db = ms.database.ScheduleFnDatabase(
                        _schedule_dummy()
                    )

            #  Build model using meta_schedule logs
            ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
                database=db,
                mod=mod,
                target=target,
                params=params,
                pass_config=MappingProxyType(
                    {
                        **pass_config,
                        "relay.backend.use_meta_schedule": True,
                        "relay.backend.tir_converter": "default",
                        "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
                        # "tir.enable_debug": True,
                    }
                ),
                disabled_pass=disabled_pass,
                executor=executor,
                runtime=runtime,
            )
    print(profiler.table())
    import time
    # print("sleeping")
    # time.sleep(10)
    non_ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
        None,
        mod=mod,
        target=target,
        params=params,
        pass_config=MappingProxyType(
            {
                **pass_config,
                "relay.backend.use_meta_schedule_dispatch": MS_DISPATCH,
            }
        ),
        disabled_pass=disabled_pass,
        executor=executor,
        runtime=runtime,
    )

    if not SKIP_TUNING:
        # TUNED
        # TODO: wrap in helper
        project = tvm.micro.generate_project(
            # str(tvm.micro.get_microtvm_template_projects(platform)),
            str(platform),
            ms_mod,
            str(work_dir_path / "project"),
            options=options,
        )
        project.build()
        project.flash()
        with tvm.micro.Session(project.transport()) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            # aot_executor.get_input(0).copyfrom(data_sample)
            # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            result = aot_executor.module.time_evaluator("run", session.device, number=1)()
            print("result", result)
            print("mean: ", result.mean)
            # output = aot_executor.get_output(0).numpy()

    # UNTUNED
    # TODO: wrap in helper
    project = tvm.micro.generate_project(
        # str(tvm.micro.get_microtvm_template_projects(platform)),
        str(platform),
        non_ms_mod,
        str(work_dir_path / "project2"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        # aot_executor.get_input(0).copyfrom(data_sample)
        # result = aot_executor.module.time_evaluator("run", session.device, number=3)()
        result2 = aot_executor.module.time_evaluator("run", session.device, number=1)()
        print("result2", result2)
        print("mean2:", result2.mean)
        # import time
        # time.sleep(100)
        # output = aot_executor.get_output(0).numpy()
    if not SKIP_TUNING:
        rel = result.mean / result2.mean
        print("rel:  ", rel)

    # Build reference model (without tuning)
    # dev = tvm.cpu()
    # target = tvm.target.target.micro(model="host")
    # with tvm.transform.PassContext(
    #     opt_level=opt_level,
    #     config=pass_config,
    #     disabled_pass=disabled_pass,
    # ):
    #     ref_mod = relay.build(
    #         mod,
    #         target=target,
    #         params=params,
    #         runtime=runtime,
    #     )
    # ref_mod.export_library(work_dir / "compiled_lib2.so")
    # mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    # graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    # graph_mod.set_input(input_name, data_sample)
    # graph_mod.run()
    # ref_output = graph_mod.get_output(0).numpy()

    # assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    # if not KEEP:
    #     work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
