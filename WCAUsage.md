# WCA TVM Integration

## Prerequisites

Make sure to have a clone of `fau-tum-cfu-collab` already available.

Instructions of setting up ETISS,... are found in `fau-tum-cfu-collab/README.md`.

## Setup

If not already insteadd via `fau-tum-cfu-collab`, the following steps are required to setup the TVM environment:

```sh
cd tvm
git submodule update --init --recursive
export PYTHONPATH=$(pwd)/python
python3.10 -m venv venv
source venv/bin/activate
python3 python/gen_requirements.py
pip install -r python/requirements/core.txt
pip install -r python/requirements/importer-tflite.txt
pip install "xgboost==2.0.3"
pip install pandas
mkdir build
cp cmake/config.cmake build
vi build/config.cmake
# set USE_MICRO to ON and USE_LLVM to the path to llvm-config-14 or newer
cmake -S . -B build
cmake --build build/ -j$(nproc)
```

## Configuration

The following environment variables can be used to change the default directories:

```sh
BASE_DIR=/path/to/tvm/.. # directory of cloned fau-tum-cfu-collab repo (tumeda branch)
BASE_OUT_DIR=/tmp/base/  # destination for session artifacts if --out is undefined
```

## Usage

Hint: Don't forget `export PYTHONPATH=$(pwd)/python`


### Run untuned Resnet Model via MicroTVM on ETISS (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap \
    --skip-tuning \
    --template etiss --out outputs/bench_resnet_baseline_etiss_etiss
```

### Tune Resnet Model on ETISS (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap \
    --num-trials-per-iter 5 --max-trials-per-task 50 --skip-bench \
    --template etiss --out outputs/tune_resnet_baseline_etiss
```

### Tune and Run Resnet Model on ETISS (WCA enabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap \
    --num-trials-per-iter 5 --max-trials-per-task 50 \
    --enable-custom --enable-intrin --cfu-mode=MODE_CFU \
    --template etiss --out outputs/tune_resnet_wca_etiss
```

### Run Tuned Resnet Model via MicroTVM on ETISS (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap \
    --ms-db outputs/tune_resnet_baseline_etiss \
    --template etiss --out outputs/run_resnet_baseline_tuned_etiss
```

### Run Tuned Resnet Model via MicroTVM on ETISS (WCA enabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap \
    --enable-custom --enable-intrin --cfu-mode=MODE_CFU \
    --ms-db outputs/tune_resnet_wca_etiss \
    --template etiss --out outputs/run_resnet_wca_tuned_etiss
```

The results are also printed to the terminal (in seconds):

```
Metrics:
      mode      mean
0    tuned  0.237104
1  untuned  0.882118
2      REL  0.268789
```

### Tuning and benchmarking via MicroTVM on Renode

Replace `--template etiss` with `--template cfu` in the commands above!
Make sure to setup the multilib RISC-V toolchain and CFU Playground repo before and export the relevant paths to the environment:

```sh
export CFU_ROOT=/path/to/cfu-playground
export PATH=/path/to/riscv_gcc_multilib/bin:$PATH
```

### Single layer tuning run with higher verbosity to see errors

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model layers_unpacked/pretrainedResnet_clustered_quant_remap_layer5 \
    --num-trials-per-iter 1 --max-trials-per-task 1 --max-trials-global 1 \
    --enable-custom --enable-intrin --cfu-mode=MODE_CFU \
    --verbose \
    --template etiss --out outputs/debug_resnet_layer_cfu_etiss
```

### Debug auto-tensorization issues by increasing Metascheduler dispatch verbosity


```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model layers_unpacked/pretrainedResnet_clustered_quant_remap_layer5 \
    --num-trials-per-iter 1 --max-trials-per-task 1 --max-trials-global 1 \
    --enable-custom --enable-intrin --cfu-mode=MODE_CFU \
    --ms-dispatch 2 \
    --template --out outputs/debug_resnet_layer_cfu_dispatch_etiss

# (dispatch & 1): unused
# (dispatch & 2): controls whether to print TVMScript for missing TIR
# (dispatch & 4): controls whether to raise fatal errors for missing TIR
```

You may want to check the generated TVM kernels to see if the tensorization was done correctly: `outputs/debug_resnet_layer_cfu_dispatch/project/model/codegen/host/src/default_lib2.c`

It can also be helpful to look into the MS logfiles such as `outputs/tune_resnet_wca/logs/tvm.meta_schedule.logging.task_07_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_subtract_fix_8b7503320bf54f1a__1.log`.

Expected messages for supported layers:

```
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:59] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:59] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:59] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:59] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:62] Tensorizing with cfu_32x
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:62] Tensorizing with cfu_24x
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:62] Tensorizing with cfu_16x
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:52] The workload cannot be tensorized.
2026-01-16 15:38:55 [INFO] [multi_level_tiling_with_intrin.cc:62] Tensorizing with cfu_8x
```

## Details

**Output artifacts:**

```
outputs/tune_resnet_cfu
├── project/   # MicroTVM project directory (tuned)
├── project2/  # MicroTVM project directory (untuned)
├── logs/      # Tuning logfiles
├── database_tuning_record.json  # MetaScheduler DB Tuning Records
├── database_workload.json       # MetaScheduler DB Tuning Workloads
└── metrics.csv
```

**Relevant files:**

```
tests/python/micro/cfu_wca_etiss_script.py    # Tuning and benchmarkming script
python/tvm/tir/tensor_intrin/cfu.py           # tensor intrinsics for WCA
python/tvm/contrib/micro/cfu/wca.py           # Utilities & passes for WCA
python/tvm/contrib/micro/cfu/model_utils.py   # TFLite Model support
python/tvm/contrib/micro/cfu/tuning_utils.py  # Misc Tuning utils
```

## WCA ISA

### Description

TODO

### Constraints

TODO

##  Example flow

In the following the approach for the integration of the WCA in TVM is explained using an example Resnet layer (`task_07_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_subtract_fix_8b7503320bf54f1a__1`)

To enable the weight clustering the data & kernel layout of the model has to be transformed to have the input channels in the innermost axis. This is done automatically in the scripts unless `--no-transform-layout` is used:

```python
if transform_layout:
    with tvm.transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
    ):
        desired_layouts = {"qnn.conv2d": ["NHWC", "HWOI"]}
        seq = transform.Sequential(
            [
                relay.transform.RemoveUnusedFunctions(),
                relay.transform.ConvertLayout(desired_layouts),
                relay.transform.FoldConstant(),
            ]
        )
        mod = seq(mod)
```

As this `NHWC:HWOI` data layout is only supported by TVMs `arm_cpu` Cortex-M relay operator strategies, a non-default TVM target has to be used:

```
target = "c -device=arm_cpu -mcpu=cortex-m7 -num-cores=1"
```

As a worksround to avoid generating unsupported ARM instructions, they are disabled manually. See `python/tvm/topi/arm_cpu/mprofile/dsp/conv2d.py` for example.

Used custom tuning config:

```python
def get_wca_tuning_config(
    enable_intrin: bool = False,
    num_clusters: Optional[int] = None,
    cfu_mode: Optional[str] = None,
    channel_count: Optional[int] = None,
):
    ...
    def _get_sch_rules(
        intrin: Optional[str] = None, num_clusters: Optional[int] = None, channel_count: Optional[int] = None
    ):
        intrins = [
            CFU_64X_INTRIN,
            CFU_56X_INTRIN,
            CFU_48X_INTRIN,
            CFU_40X_INTRIN,
            CFU_32X_INTRIN,
            CFU_24X_INTRIN,
            CFU_16X_INTRIN,
            CFU_8X_INTRIN,
        ]
        structure = "SR"
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
            *(  # CUSTOM
                [
                    ms.schedule_rule.MultiLevelTilingWithIntrin(
                        intrin,
                        structure=structure,
                    )
                    for intrin in intrins
                ]
            ),
            ms.schedule_rule.MultiLevelTiling(
                structure="SSRSRS",
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
            ),
            ms.schedule_rule.RandomComputeLocation(),
        ]

    def _get_postprocs(cfu_mode: Optional[str] = None):
        # print("_get_postprocs", cfu_mode)
        return [
            ms.postproc.DisallowDynamicLoop(),
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            *([ImportCPostprocess(cfu_mode)] if enable_intrin else []),  # CUSTOM
            ms.postproc.RewriteTensorize(),
        ]

     def _get_mutator_probs():
        return {
            ms.mutator.MutateTileSize(): 0.9,
            ms.mutator.MutateComputeLocation(): 0.05,
            ms.mutator.MutateUnroll(): 0.03,
        }
    ...
    sch_rules = _get_sch_rules(intrin, num_clusters, channel_count)
    postprocs = _get_postprocs(cfu_mode)
    mutator_probs = _get_mutator_probs()
    return sch_rules, postprocs, mutator_probs
```

Used tensor intrinsics (see `python/tvm/tir/tensor_intrin/cfu.py`):

```python
def get_cfu_intrin(dtype_a, dtype_b, dtype_c, count):
    global name_supply
    assert dtype_a == "int8"
    assert dtype_b == "int8"
    assert dtype_c == "int32"

    @T.prim_func
    def cfu_desc(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((count,), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:count], B[0:count])
            T.writes(C[0])
            for i in range(0, count):
                with T.block("update"):
                    vi = T.axis.remap("R", [i])
                    C[0] = C[0] + T.cast(A[vi], dtype_c) * T.cast(B[vi], dtype_c)

    @T.prim_func
    def cfu_impl(
        A: T.Buffer((count,), dtype_a, offset_factor=1, align=4),
        B: T.Buffer((count,), dtype_b, offset_factor=1, align=4),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:count], B[0:count])
            T.writes(C[0])
            C[0] += T.call_pure_extern(
                f"cfu_kernel_{count}x",  # TODO: rename
                A.access_ptr("r", offset=0),
                B.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype=dtype_c,
            )
    return cfu_desc, cfu_impl


CFU_64X_INTRIN = "cfu_64x"
TensorIntrin.register(CFU_64X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 64))
CFU_56X_INTRIN = "cfu_56x"
TensorIntrin.register(CFU_56X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 56))
CFU_48X_INTRIN = "cfu_48x"
TensorIntrin.register(CFU_48X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 48))
CFU_40X_INTRIN = "cfu_40x"
TensorIntrin.register(CFU_40X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 40))
CFU_32X_INTRIN = "cfu_32x"
TensorIntrin.register(CFU_32X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 32))
CFU_24X_INTRIN = "cfu_24x"
TensorIntrin.register(CFU_24X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 24))
CFU_16X_INTRIN = "cfu_16x"
TensorIntrin.register(CFU_16X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 16))
CFU_8X_INTRIN = "cfu_8x"
TensorIntrin.register(CFU_8X_INTRIN, *get_cfu_intrin("int8", "int8", "int32", 8)
```

As you can see, the intrinsic istself if not depending on the availability of clustered weights, as it is just looking for the largest possible vectorized MAC over the innermost loop axis.

TIR before auto-tensorization:

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(p0: T.Buffer((T.int64(1), T.int64(18), T.int64(18), T.int64(32)), "int8"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        compile_engine_const = T.alloc_buffer((), "int32")
        compile_engine_const_1 = T.alloc_buffer((), "int32")
        padded_data = T.alloc_buffer((T.int64(1), T.int64(18), T.int64(18), T.int64(32)), "int8")
        conv2d = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        T_add_1 = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        compute = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        T_add_2 = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        compute_1 = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        compute_2 = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
        fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2 = T.allocate_const([36], "int32", [1])
        fused_nn_conv2d_subtract_add_constant_14 = T.allocate_const([10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 9, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 9, 9, 10, 10, 9, 10, 10], "int32", [32])
        fused_nn_conv2d_subtract_add_constant_13 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "int32", [32])
        fused_nn_conv2d_subtract_add_constant_12 = T.allocate_const([2056808628, 1905168670, 1636042109, 1491956100, 2062299949, 2001499105, 1158684220, 1728443867, 1620986542, 1842403241, 1255601325, 1873140342, 1179175324, 1140315871, 1964367871, 1715993266, 1711995180, 1928593422, 1571726082, 1711811142, 1879452133, 1703322293, 1864435071, 1097036940, 1609228058, 1086065751, 1099180528, 2110924669, 1977642693, 1075583783, 1836355265, 1600077502], "int32", [32])
        fused_nn_conv2d_subtract_constant_4 = T.allocate_const([10229, -3487, 6971, -6744, 4211, -2775, 22419, -239, 29246, -13487, 2399, 12899, 8510, 11334, -12014, -8139, 26493, 5022, 2220, 18746, -17465, 11243, 10618, 1220, -1651, 7171, -7218, -6522, 21146, -2016, 12353, 8257], "int32", [1, 1, 1, 32])
        fused_nn_conv2d_constant_4 = T.allocate_const([349056, 146816, 272384, -49792, 269696, 185984, 627712, -50304, 678400, -76416, 134272, 523776, 203008, 261888, -51328, 100096, 491392, 403712, 178176, 456960, 114048, 398848, 344192, 172544, 159616, 220416, 132864, 33024, 513408, 137216, 504704, 305792], "int32", [1, 1, 1, 32])
        fused_constant_5 = T.allocate_const([-127, -54, 53, -6, -54, -54, -6, 53, -6, -6, -54, 53, -6, -54, 53, -6, -54, -54, -6, -54, -54, -127, -54, -54, -127, 53, 53, -54, -54, -54, -54, -54, 53, 53, 53, -6, -6, -54, -6, -127, -6, -54, -54, 53, 53, 53, -54, -6, 53, -6, -54, -6, -54, -6, 53, -54, -54, -54, -54, 53, -6, 53, -54, -6, 53, -54, 53, -54, -127, 53, 53, -6, -127, -54, -54, 53, -54, -54, -127, -54, 53, -54, 53, -127, 53, -54, -54, -6, -54, -54, 53, -127, -127, -54, -6, -54, -127, -6, 53, -6, -6, 53, 53, 53, -6, 53, 53, -54, -6, 53, 53, 53, -6, 53, 53, 53, 53, -6, 53, 53, -127, 53, -54, -6, 53, -6, 53, -127, 53, -6, 53, -54, -54, 53, 53, -54, -127, -54, -54, 53, -6, -6, -6, -54, 53, -6, -6, -6, 53, -6, -54, 53, -6, 53, 53, -6, -127, -6, -6, -6, -6, -54, 53, -54, -54, -6, -6, 53, -54, -54, 53, 53, -6, 53, 53, -6, -54, -6, -6, -6, -6, -54, -127, -6, -6, 53, 53, -54, -54, -6, 53, -54, -54, -6, -6, -54, -54, -54, 53, -127, -54], "int8", [3, 3, 32, 32])
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, T.int64(0))
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = -128
        with T.block("compile_engine_const_1"):
            vi = T.axis.spatial(1, T.int64(0))
            T.reads()
            T.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = 36
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(18), T.int64(18), T.int64(32)):
            with T.block("padded_data"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[v_i0, v_i1, v_i2, v_i3])
                T.writes(padded_data[v_i0, v_i1, v_i2, v_i3])
                padded_data[v_i0, v_i1, v_i2, v_i3] = p0[v_i0, v_i1, v_i2, v_i3]
        for nn, yy, xx, ff, ry, rx, rc in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32), T.int64(3), T.int64(3), T.int64(32)):
            with T.block("conv2d"):
                v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc])
                fused_constant_5_1 = T.Buffer((3, 3, 32, 32), "int8", data=fused_constant_5)
                T.reads(padded_data[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc], fused_constant_5_1[v_ry, v_rx, v_ff, v_rc])
                T.writes(conv2d[v_nn, v_yy, v_xx, v_ff])
                with T.init():
                    conv2d[v_nn, v_yy, v_xx, v_ff] = 0
                conv2d[v_nn, v_yy, v_xx, v_ff] = conv2d[v_nn, v_yy, v_xx, v_ff] + T.Cast("int32", padded_data[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc]) * T.Cast("int32", fused_constant_5_1[v_ry, v_rx, v_ff, v_rc])
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("T_subtract"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                fused_nn_conv2d_constant_4_1 = T.Buffer((1, 1, 1, 32), "int32", data=fused_nn_conv2d_constant_4)
                T.reads(conv2d[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_constant_4_1[v_ax0, T.int64(0), T.int64(0), v_ax3])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = conv2d[v_ax0, v_ax1, v_ax2, v_ax3] - fused_nn_conv2d_constant_4_1[v_ax0, T.int64(0), T.int64(0), v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                fused_nn_conv2d_subtract_constant_4_1 = T.Buffer((1, 1, 1, 32), "int32", data=fused_nn_conv2d_subtract_constant_4)
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_subtract_constant_4_1[v_ax0, T.int64(0), T.int64(0), v_ax3])
                T.writes(T_add_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] + fused_nn_conv2d_subtract_constant_4_1[v_ax0, T.int64(0), T.int64(0), v_ax3]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                fused_nn_conv2d_subtract_add_constant_12_1 = T.Buffer((32,), "int32", data=fused_nn_conv2d_subtract_add_constant_12)
                fused_nn_conv2d_subtract_add_constant_13_1 = T.Buffer((32,), "int32", data=fused_nn_conv2d_subtract_add_constant_13)
                fused_nn_conv2d_subtract_add_constant_14_1 = T.Buffer((32,), "int32", data=fused_nn_conv2d_subtract_add_constant_14)
                T.reads(T_add_1[v_i0, v_i1, v_i2, v_i3], fused_nn_conv2d_subtract_add_constant_12_1[v_i3], fused_nn_conv2d_subtract_add_constant_13_1[v_i3], fused_nn_conv2d_subtract_add_constant_14_1[v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.q_multiply_shift_per_axis(T_add_1[v_i0, v_i1, v_i2, v_i3], fused_nn_conv2d_subtract_add_constant_12_1[v_i3], fused_nn_conv2d_subtract_add_constant_13_1[v_i3], fused_nn_conv2d_subtract_add_constant_14_1[v_i3], 31, T.bool(False), T.bool(True))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(compile_engine_const_1[()], compute[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add_2[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add_2[v_ax0, v_ax1, v_ax2, v_ax3] = compile_engine_const_1[()] + compute[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_2[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_1[v_i0, v_i1, v_i2, v_i3])
                compute_1[v_i0, v_i1, v_i2, v_i3] = T.max(T.min(T_add_2[v_i0, v_i1, v_i2, v_i3], 127), -128)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("T_subtract_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2_1 = T.Buffer((1,), "int32", data=fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2)
                T.reads(compute_1[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2_1[T.int64(0)])
                T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = compute_1[v_ax0, v_ax1, v_ax2, v_ax3] - fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2_1[T.int64(0)]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract_1[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute_2[v_i0, v_i1, v_i2, v_i3])
                compute_2[v_i0, v_i1, v_i2, v_i3] = T.q_multiply_shift(T_subtract_1[v_i0, v_i1, v_i2, v_i3], 1501547619, 31, 2)
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(compile_engine_const[()], compute_2[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = compile_engine_const[()] + compute_2[v_ax0, v_ax1, v_ax2, v_ax3]
```

TIR after auto-tensorization:

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(p0: T.Buffer((T.int64(1), T.int64(18), T.int64(18), T.int64(32)), "int8"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 8})
            conv2d = T.alloc_buffer((T.int64(1), T.int64(16), T.int64(16), T.int64(32)), "int32")
            fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_constant_2 = T.allocate_const([36], "int32", [1])
            fused_nn_conv2d_subtract_add_constant_14 = T.allocate_const([10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 9, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 9, 9, 10, 10, 9, 10, 10], "int32", [32])
            fused_nn_conv2d_subtract_add_constant_13 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "int32", [32])
            fused_nn_conv2d_subtract_add_constant_12 = T.allocate_const([2056808628, 1905168670, 1636042109, 1491956100, 2062299949, 2001499105, 1158684220, 1728443867, 1620986542, 1842403241, 1255601325, 1873140342, 1179175324, 1140315871, 1964367871, 1715993266, 1711995180, 1928593422, 1571726082, 1711811142, 1879452133, 1703322293, 1864435071, 1097036940, 1609228058, 1086065751, 1099180528, 2110924669, 1977642693, 1075583783, 1836355265, 1600077502], "int32", [32])
            fused_nn_conv2d_subtract_constant_4 = T.allocate_const([10229, -3487, 6971, -6744, 4211, -2775, 22419, -239, 29246, -13487, 2399, 12899, 8510, 11334, -12014, -8139, 26493, 5022, 2220, 18746, -17465, 11243, 10618, 1220, -1651, 7171, -7218, -6522, 21146, -2016, 12353, 8257], "int32", [1, 1, 1, 32])
            fused_nn_conv2d_constant_4 = T.allocate_const([349056, 146816, 272384, -49792, 269696, 185984, 627712, -50304, 678400, -76416, 134272, 523776, 203008, 261888, -51328, 100096, 491392, 403712, 178176, 456960, 114048, 398848, 344192, 172544, 159616, 220416, 132864, 33024, 513408, 137216, 504704, 305792], "int32", [1, 1, 1, 32])
            fused_constant_5 = T.allocate_const([-127, -54, 53, -6, -54, -54, -6, 53, -6, -6, -54, 53, -6, -54, 53, -6, -54, -54, -6, -54, -54, -127, -54, -54, -127, 53, 53, -54, -54, -54, -54, -54, 53, 53, 53, -6, -6, -54, -6, -127, -6, -54, -54, 53, 53, 53, -54, -6, 53, -6, -54, -6, -54, -6, 53, -54, -54, -54, -54, 53, -6, 53, -54, -6, 53, -54, 53, -54, -127, 53, 53, -6, -127, -54, -54, 53, -54, -54, -127, -54, 53, -54, 53, -127, 53, -54, -54, -6, -54, -54, 53, -127, -127, -54, -6, -54, -127, -6, 53, -6, -6, 53, 53, 53, -6, 53, 53, -54, -6, 53, 53, 53, -6, 53, 53, 53, 53, -6, 53, 53, -127, 53, -54, -6, 53, -6, 53, -127, 53, -6, 53, -54, -54, 53, 53, -54, -127, -54, -54, 53, -6, -6, -6, -54, 53, -6, -6, -6, 53, -6, -54, 53, -6, 53, 53, -6, -127, -6, -6, -6, -6, -54, 53, -54, -54, -6, -6, 53, -54, -54, 53, 53, -6, 53, 53, -6, -54, -6, -6, -6, -6, -54, -127, -6, -6, 53, 53, -54, -54, -6, 53, -54, -54, -6, -6, -54, -54, -54, 53, -127, -54], "int8", [3, 3, 32, 32])
            for nn, yy, xx, ff, ry, rx, rc_0 in T.grid(T.int64(1), T.int64(16), T.int64(16), T.int64(32), T.int64(3), T.int64(3), T.int64(1)):
                with T.block("conv2d_o"):
                    v_nn_o, v_yy_o, v_xx_o, v_ff_o, v_ry_o, v_rx_o, v_rc_o = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc_0])
                    fused_constant_5_1 = T.Buffer((3, 3, 32, 32), "int8", data=fused_constant_5)
                    T.reads(p0[v_nn_o, v_yy_o + v_ry_o, v_xx_o + v_rx_o, T.int64(0):T.int64(32)], fused_constant_5_1[v_ry_o, v_rx_o, v_ff_o, T.int64(0):T.int64(32)])
                    T.writes(conv2d[v_nn_o, v_yy_o, v_xx_o, v_ff_o])
                    T.block_attr({"meta_schedule.auto_tensorize": "cfu_32x"})
                    with T.init():
                        with T.block("conv2d_init"):
                            T.reads()
                            T.writes(conv2d[v_nn_o, v_yy_o, v_xx_o, v_ff_o])
                            conv2d[v_nn_o, v_yy_o, v_xx_o, v_ff_o] = 0
```

Before the tuning trial is evaluated the defined postprocesses are applied. The custom `ImportCPostprocess` is used to:
- validate the legality of the tensorization (i.e. supported number of clusters and dtype)
- generate layer-specific microkernels (`_gen_cfu_kernel_code`)
- inserts the generated code into the kernel files via `pragma_import_c`
- reverts the tensorization for invalid kernels

```python
...
is_legal = ...  # Check if valid cfu kernel
if has_tensorize:
    if is_legal:
        num_clusters = len(codebook_arr)
        func_name = f"cfu_kernel_{tensorize_count}x_{num_clusters}c"
        code = _gen_cfu_kernel_code(num_clusters, self.mode, tensorize_count, func_name)
        sch.annotate(block, "pragma_import_c", code)
    else:
        block_ = sch.get_block(tensorize_block)
        sch.unannotate(block_, "meta_schedule.auto_tensorize")
```

During TVMs final lowering steps a custom pass called `CompressWeights` is used to detect the clusters and generate the codebook. This has to be enabled during tuning **and** deployment using the TVM PassContext:

```python3
pass_config = {
    "tir.disable_vectorize": True,
    "tir.add_lower_pass": [(3, CompressWeights())],
}
with tvm.transform.PassContext(
    ...
    config=pass_config,
):
    ...
```

*Question:* Why can't we compress the weights in the tuning postprocess? - Postprocesses can not change the TIR code of the tuning task, only the schedule trace can be altered...

The TVM generated kernels should have the imported C code at the top:

```c
#ifndef CFU_KERNEL_CODE_4_16
#define CFU_KERNEL_CODE_4_16
#include <stdint.h>

#ifndef MODE
#define MODE MODE_CFU
#include "cfu_wca.h"
#undef MODE
#else
#include "cfu_wca.h"
#endif


static int32_t __attribute__((always_inline)) inline cfu_kernel_16x_4c(int8_t* data_ptr, int8_t* weights_ptr, int32_t* acc) {
    // COUNT=16, NUM_CLUSTERS=4

    alu_rst();

    uint32_t code_word0 = *((uint32_t*)weights_ptr);
    uint32_t* act_words = (uint32_t*)data_ptr;
    push_weights_4b(code_word0, 0);
    for (int i = 0; i < (16 / 8); i++) {
        alu_mac(act_words[2 * i], act_words[2 * i + 1]);
    }

    return get_acc();
}
#endif  // CFU_KERNEL_CODE_4_16
```

Full example function (random layer):

```c
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2(void* restrict args, int32_t* restrict arg_type_ids, int32_t num_args, void* restrict out_ret_value, int32_t* restrict out_ret_tcode, void* restrict resource_handle) {
  int32_t var_p0_code = arg_type_ids[0];
  int32_t var_T_cast_code = arg_type_ids[1];
  int32_t global_const_workspace_16_var_code = arg_type_ids[2];
  int32_t global_workspace_17_var_code = arg_type_ids[3];
  void* var_p0 = (((TVMValue*)args)[0].v_handle);
  void* var_T_cast = (((TVMValue*)args)[1].v_handle);
  void* global_const_workspace_16_var = (((TVMValue*)args)[2].v_handle);
  void* global_workspace_17_var = (((TVMValue*)args)[3].v_handle);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_p0_shape = (((DLTensor*)var_p0)[0].shape);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_p0_strides = (((DLTensor*)var_p0)[0].strides);
  int32_t dev_id = (((DLTensor*)var_p0)[0].device.device_id);
  void* p0 = (((DLTensor*)var_p0)[0].data);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_T_cast_shape = (((DLTensor*)var_T_cast)[0].shape);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_T_cast_strides = (((DLTensor*)var_T_cast)[0].strides);
  void* T_cast = (((DLTensor*)var_T_cast)[0].data);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_global_const_workspace_16_var_shape = (((DLTensor*)global_const_workspace_16_var)[0].shape);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_global_const_workspace_16_var_strides = (((DLTensor*)global_const_workspace_16_var)[0].strides);
  void* global_const_workspace_16_var_1 = (((DLTensor*)global_const_workspace_16_var)[0].data);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_global_workspace_17_var_shape = (((DLTensor*)global_workspace_17_var)[0].shape);
  void* tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_global_workspace_17_var_strides = (((DLTensor*)global_workspace_17_var)[0].strides);
  void* global_workspace_17_var_1 = (((DLTensor*)global_workspace_17_var)[0].data);
  if (!(tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_p0_strides == NULL)) {
  }
  if (!(tvmgen_default_fused_nn_conv2d_subtract_add_fixed_point_multiply_per_axis_add_clip_cast_2_var_T_cast_strides == NULL)) {
  }
  void* conv2d_let = (&(((uint8_t*)global_workspace_17_var_1)[0]));
  void* fused_nn_conv2d_subtract_add_constant_8_let = (&(((uint8_t*)global_const_workspace_16_var_1)[25616]));
  void* fused_nn_conv2d_subtract_add_constant_7_let = (&(((uint8_t*)global_const_workspace_16_var_1)[25744]));
  void* fused_nn_conv2d_subtract_add_constant_6_let = (&(((uint8_t*)global_const_workspace_16_var_1)[25872]));
  void* fused_nn_conv2d_subtract_constant_2_let = (&(((uint8_t*)global_const_workspace_16_var_1)[25360]));
  void* fused_nn_conv2d_constant_2_let = (&(((uint8_t*)global_const_workspace_16_var_1)[26896]));
  void* fused_constant_2_let = (&(((uint8_t*)global_const_workspace_16_var_1)[18432]));
  void* codebook__1_let = (&(((uint8_t*)global_const_workspace_16_var_1)[28400]));
  set_codebook_4((&(((int8_t*)codebook__1_let)[0])));
  for (int64_t nn = 0; nn < (int64_t)1; ++nn) {
    for (int32_t yy = 0; yy < 16; ++yy) {
      for (int32_t xx = 0; xx < 16; ++xx) {
        for (int32_t ff = 0; ff < 32; ++ff) {
          int32_t cse_var_4 = (xx * 32);
          int32_t cse_var_3 = (ff * 16);
          int32_t cse_var_2 = ((yy * 1056) + cse_var_4);
          int32_t cse_var_1 = (((yy * 512) + cse_var_4) + ff);
          ((int32_t*)conv2d_let)[cse_var_1] = 0;
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[cse_var_2])), (&(((int8_t*)fused_constant_2_let)[(cse_var_3 >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 16)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 512) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 32)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 1024) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 528)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 1536) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 544)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 2048) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 560)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 2560) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 1056)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 3072) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 1072)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 3584) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
          ((int32_t*)conv2d_let)[cse_var_1] = (((int32_t*)conv2d_let)[cse_var_1] + cfu_kernel_16x_4c((&(((int8_t*)p0)[(cse_var_2 + 1088)])), (&(((int8_t*)fused_constant_2_let)[((cse_var_3 + 4096) >> 2)])), (&(((int32_t*)conv2d_let)[cse_var_1]))));
        }
      }
    }
  }
  for (int32_t ax1 = 0; ax1 < 16; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        int32_t cse_var_5 = (((ax1 * 512) + (ax2 * 32)) + ax3);
        int32_t v_ = ((int32_t)(((((int64_t)((((int32_t*)conv2d_let)[cse_var_5] + ((int32_t*)fused_nn_conv2d_subtract_constant_2_let)[ax3]) - ((int32_t*)fused_nn_conv2d_constant_2_let)[ax3])) * ((int64_t)((int32_t*)fused_nn_conv2d_subtract_add_constant_6_let)[ax3])) + ((int64_t)1 << ((int64_t)((((int32_t*)fused_nn_conv2d_subtract_add_constant_8_let)[ax3] + 31) - 1)))) >> ((int64_t)(((int32_t*)fused_nn_conv2d_subtract_add_constant_8_let)[ax3] + 31)))) - 128;
        int32_t v__1 = (v_) < (127) ? (v_) : (127);
        ((int8_t*)T_cast)[cse_var_5] = ((int8_t)((v__1) > (-128) ? (v__1) : (-128)));
      }
    }
  }
  return 0;
}
```

The `default_lib0.c` file also contains the generated codebooks which are configured using i.e. `set_codebook_4((&(((int8_t*)codebook__1_let)[0])));`:

```c
  .codebook__7_let = {
    -0x2e, -0x0d, +0x17, +0x7f
  },
  .codebook__6_let = {
    -0x7f, -0x41, -0x15, +0x2a
  },
  .codebook__5_let = {
    -0x7f, -0x53, -0x18, +0x33
  },
  .codebook__4_let = {
    -0x7f, -0x3c, -0x02, +0x39
  },
  .codebook__3_let = {
    -0x77, +0x00, +0x24, +0x7f
  },
  .codebook__2_let = {
    -0x7f, -0x36, -0x06, +0x35
  },
  .codebook__1_let = {
    -0x68, -0x21, +0x17, +0x7f
  },
  .codebook__let = {
    -0x77, -0x19, +0x42, +0x7f
  },
```


## Supported Targets

Currently only ETISS (Instruction set simulator) is supported.

See https://github.com/PhilippvK/microtvm-etiss-template/tree/cfu for details

## Supported models

Currently only a single model is supported:

### Resnet

See `models/new/pretrainedResnet_clustered_quant_remap.tflite`

### How to train other models?

TODO

## TODOs

- [ ] Support tuning on RTL via CFU Playground (Renode)
- [ ] Integrate tuning with MLonMCU
- [ ] Finish CFUWCA feature for MLonMCU
- [ ] Support more models
- [ ] Implement fallback to normal vector dotproduct if weight clustering is not feasible
- [ ] Document TVMC usage with CFU/WCA
