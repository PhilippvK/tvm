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
"""

@derived_object
class AlwaysFailPostproc(ms.postproc.PyPostproc):
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

    def clone(self) -> "AlwaysFailPostproc":
        return AlwaysFailPostproc()

    def __str__(self) -> str:
        return "AlwaysFailPostproc"

ENABLE_INTRIN = True

def _get_sch_rules(intrin):
    structure_lookup = {
        AMDGPU_SDOT4_INTRIN: "SSSRRSRS",
        VRMPY_i8i8i32_INTRIN: "SRSRS",
        DP4A_S8S8S32_INTRIN: "SR",
        # ARM_DOT_4x4_i8_NEON_INTRIN: "SR",
        ARM_DOT_4x4_i8_NEON_INTRIN: "RS",
    }
    return [
        ms.schedule_rule.ApplyCustomRule(),
        ms.schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        ),
        ms.schedule_rule.AddRFactor(max_jobs_per_core=1, max_innermost_factor=64),
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
            )] if ENABLE_INTRIN else []),
        # structure="SRSRS",
        # tile_binds=None,
        # max_innermost_factor=64,
        # vector_load_lens=None,
        # reuse_read=None,
        # reuse_write=schedule_rule.ReuseType(
        #     req="may",
        #     levels=[1, 2],
        #     scope="global",
        # ),
        # ms.schedule_rule.AutoInline(
        #     into_producer=True,
        #     into_consumer=True,
        #     inline_const_tensor=True,
        #     disallow_if_then_else=False,
        #     require_injective=False,
        #     require_ordered=False,
        #     disallow_op=None,
        # ),
        # ms.schedule_rule.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
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


# SCH_RULES = _get_sch_rules(AMDGPU_SDOT4_INTRIN)
# SCH_RULES = _get_sch_rules(VRMPY_i8i8i32_INTRIN)
SCH_RULES = _get_sch_rules(DP4A_S8S8S32_INTRIN)
# SCH_RULES = _get_sch_rules(ARM_DOT_4x4_i8_NEON_INTRIN)
# SCH_RULES = _get_sch_rules(CFU_MAC_i8i8i32_INTRIN)

POSTPROCS = [
    ms.postproc.DisallowDynamicLoop(),
    # ms.postproc.RewriteCooperativeFetch(),
    # ms.postproc.RewriteUnboundBlock(),
    ms.postproc.RewriteParallelVectorizeUnroll(),
    ms.postproc.RewriteReductionBlock(),
    ms.postproc.RewriteTensorize(),
    # ms.postproc.RewriteTensorize(vectorize_init_loop=True),
    # ms.postproc.VerifyGPUCode(),
    AlwaysFailPostproc(),
]

MUTATOR_PROBS = {
    ms.mutator.MutateTileSize(): 0.9,
    ms.mutator.MutateComputeLocation(): 0.05,
    ms.mutator.MutateUnroll(): 0.03,
    # ms.mutator.Parallel(): 0.02,
}


# @pytest.mark.skip(reason="flaky test")
@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule():
    from tests.micro.zephyr.test_ms_tuning import create_relay_module
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    platform = "crt"
    # target = tvm.target.target.micro(model="host")
    target = "c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1"
    options = {}

    work_dir = utils.tempdir()
    # mod, params, model_info = create_relay_module()
    from tvm.driver import tvmc

    model = tvmc.load(
        "/work/git/cfu/docker-cfu/fau-tum-cfu-collab_new/tflite_models/weight_clustering/wca/pretrainedResnet_clustered_quant_remap.tflite"
    )  # Step 1: Load
    print("model", model, dir(model))
    print("model.mod", model.mod)
    print("model.summary", model.summary())
    mod = model.mod
    params = model.params

    # input_dtype = model_info["in_dtype"]
    input_shape = [1, 32, 32, 3]
    input_dtype = "int8"
    data_sample = np.random.rand(*input_shape).astype(input_dtype)

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    opt_level = 3
    module_equality = "ignore-ndarray"
    disabled_pass = []
    instruments = []
    strategy = "evolutionary"
    seed = None
    num_tuning_cores = 1
    num_trials_per_iter = 2
    # max_trials_per_task = 10
    max_trials_per_task = 100
    max_trials_global = 100
    task_scheduler = "gradient"
    sch_rules = SCH_RULES
    postprocs = POSTPROCS
    mutator_probs = MUTATOR_PROBS
    config = {
        "tir.disable_vectorize": True,
    }

    mod = mod.with_attr("executor", executor)
    with tvm.transform.PassContext(
        opt_level=opt_level,
        config=config,
        disabled_pass=disabled_pass,
        instruments=instruments,
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

    builder = get_local_builder_micro()
    # task_idx = None
    # task_idx = -2
    # task_idx = -4
    task_idx = -5
    # task_idx = -6

    with ms.Profiler() as profiler:
        with get_rpc_runner_micro(platform=platform, options=options, session_timeout_sec=120) as runner:
            tune_tasks = ms.relay_integration.extract_tasks(
                mod,
                target,
                params,
                opt_level=opt_level,
                module_equality=module_equality,
                disabled_pass=disabled_pass,
                instruments=instruments,
            )
            if task_idx is not None:
                tune_tasks = [tune_tasks[task_idx]]
            tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
                extracted_tasks=tune_tasks,
                work_dir=str(work_dir.path),
                space=ms.space_generator.PostOrderApply(
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs=mutator_probs,
                ),
                strategy=strategy,
                seed=seed,
                num_tuning_cores=num_tuning_cores,
            )
            # print(work_dir, str(work_dir.path))
            input(">")
            db: ms.Database = ms.tune.tune_tasks(
                tasks=tasks,
                task_weights=task_weights,
                work_dir=str(work_dir.path),
                num_trials_per_iter=num_trials_per_iter,
                max_trials_per_task=max_trials_per_task,
                max_trials_global=max_trials_global,
                builder=builder,
                runner=runner,
                module_equality=module_equality,
                task_scheduler=task_scheduler,
            )
            print("db", db)
            print(work_dir, str(work_dir.path))
            input(">")
            # db: ms.Database = ms.relay_integration.tune_relay(
            #     mod=mod,
            #     params=params,
            #     target=target,
            #     builder=builder,
            #     runner=runner,
            #     strategy=strategy,
            #     num_trials_per_iter=num_trials_per_iter,
            #     max_trials_per_task=max_trials_per_task,
            #     max_trials_global=max_trials_global,
            #     work_dir=str(work_dir.path),
            #     module_equality=module_equality,
            #     task_scheduler=task_scheduler,
            # )

        #  Build model using meta_schedule logs
        ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
            database=db,
            mod=mod,
            target=target,
            params=params,
            pass_config=MappingProxyType(
                {
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.tir_converter": "default",
                    ** config,
                }
            ),
            executor=executor,
            runtime=runtime,
        )
    print(profiler.table())

    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects(platform)),
        ms_mod,
        str(work_dir / "project"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        aot_executor.get_input(0).copyfrom(data_sample)
        result = aot_executor.module.time_evaluator("run", session.device, number=3)()
        output = aot_executor.get_output(0).numpy()

    # Build reference model (without tuning)
    dev = tvm.cpu()
    target = tvm.target.target.micro(model="host")
    with tvm.transform.PassContext(
        opt_level=opt_level, config=config, disabled_pass=["AlterOpLayout"]
    ):
        ref_mod = relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
        )
    ref_mod.export_library(work_dir / "compiled_lib2.so")
    mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    input_name = "serving_default_input_1:0"
    graph_mod.set_input(input_name, data_sample)
    graph_mod.run()
    ref_output = graph_mod.get_output(0).numpy()

    assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
