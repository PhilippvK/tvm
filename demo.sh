#!/bin/bash

## Clone TVM
# git clone https://github.com/PhilippvK/tvm.git -b microtvm_demo --recursive
# cd tvm

## Configure TVM
mkdir -p build/
cp cmake/config.cmake build/
echo "set(USE_MICRO ON)" >> build/config.cmake
echo "set(USE_MICRO_STANDALONE_RUNTIME ON)" >> build/config.cmake
echo "set(USE_LLVM llvm-config-15)" >> build/config.cmake
echo "set(USE_CMSISNN ON)" >> build/config.cmake
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

## Compile TVM
cmake --build build -j`nproc`

## Clone CMSIS-NN
mkdir -p CMSIS
git clone https://github.com/ARM-software/CMSIS_5.git CMSIS_5 -b 5.9.0
git clone https://github.com/ARM-software/CMSIS-NN.git CMSIS_5/CMSIS-NN -b v4.1.0
# bugfix for annoying bug when using non-arm gcc
printf '%s\n%s' "#include <stdint.h>" "$(cat ./CMSIS_5/CMSIS-NN/Include/Internal/arm_nn_compiler.h)" > ./CMSIS_5/CMSIS-NN/Include/Internal/arm_nn_compiler.h

## Setup Python
export PYTHONPATH=$(pwd)/python
virtualenv -p python3.8 venv
source source venv/bin/activate
pip install numpy scipy tflite decorator attrs cloudpickle tornado pyyaml psutil typing_extensions pytest synr

## Run Examples (Model: sine_model -> floating point only)
python3 microtvm_demo.py sine_model --executor graph --mode host-driven --benchmark
# <<<
# TVMError: unknown type = 129 -> Missing rpc serialization of benchmark dict
# >>>
python3 microtvm_demo.py sine_model --executor graph --mode host-driven --profile
# <<<
# tvm.error.RPCError -> ?
# >>>
python3 microtvm_demo.py sine_model --executor graph --mode standalone --benchmark
# <<<
# __nop function is not yet supported.__nop function is not yet supported.__nop function is not yet supported
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    0.0080       0.0080       0.0080       0.0080       0.0000
# Output: [[0.80791104]]
# >>>
python3 microtvm_demo.py sine_model --executor aot --mode host-driven --benchmark
# <<<
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    0.0060       0.0060       0.0060       0.0060       0.0000
# Output: [[0.80791104]]
# >>>

## Run Examples (Model: toycar -> quantized)
python3 microtvm_demo.py toycar --executor graph --mode host-driven --benchmark
# <<<
# TVMError: unknown type = 129 -> Missing rpc serialization of benchmark dict
# >>>
python3 microtvm_demo.py toycar --executor graph --mode host-driven --profile
# <<<
# Node Name                                                                            Ops                                                                                  Time(us)  Time(%)  Shape     Inputs  Outputs  Measurements(us)
# ---------                                                                            ---                                                                                  --------  -------  -----     ------  -------  ----------------
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast    tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast    769.6     31.359   (1, 128)  1       1        [769.6]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_9  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_9  662.6     26.999   (1, 640)  1       1        [662.6]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_2  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_2  327.1     13.329   (1, 128)  1       1        [327.1]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_6  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_6  141.3     5.758    (1, 128)  1       1        [141.3]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_1  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_1  134.2     5.468    (1, 128)  1       1        [134.2]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_3  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_3  130.9     5.334    (1, 128)  1       1        [130.9]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_7  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_7  129.1     5.261    (1, 128)  1       1        [129.1]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_8  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_8  128.9     5.252    (1, 128)  1       1        [128.9]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_5  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_5  11.25     0.458    (1, 128)  1       1        [11.25]
# tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_4  tvmgen_default_fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_4  7.035     0.287    (1, 8)    1       1        [7.035]
# tvmgen_default_fused_reshape_cast_subtract                                           tvmgen_default_fused_reshape_cast_subtract                                           4.541     0.185    (1, 640)  1       1        [4.541]
# tvmgen_default_fused_reshape_cast_subtract_1_2                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.967     0.039    (1, 128)  1       1        [0.967]
# tvmgen_default_fused_reshape_cast_subtract_1_6                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.944     0.038    (1, 128)  1       1        [0.944]
# tvmgen_default_fused_reshape_cast_subtract_1                                         tvmgen_default_fused_reshape_cast_subtract_1                                         0.932     0.038    (1, 128)  1       1        [0.932]
# tvmgen_default_fused_reshape_cast_subtract_1_3                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.932     0.038    (1, 128)  1       1        [0.932]
# tvmgen_default_fused_reshape_cast_subtract_1_5                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.931     0.038    (1, 128)  1       1        [0.931]
# tvmgen_default_fused_reshape_cast_subtract_1_7                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.931     0.038    (1, 128)  1       1        [0.931]
# tvmgen_default_fused_reshape_cast_subtract_1_1                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.929     0.038    (1, 128)  1       1        [0.929]
# tvmgen_default_fused_reshape_cast_subtract_1_4                                       tvmgen_default_fused_reshape_cast_subtract_1                                         0.927     0.038    (1, 128)  1       1        [0.927]
# tvmgen_default_fused_reshape_cast_subtract_2                                         tvmgen_default_fused_reshape_cast_subtract_2                                         0.111     0.005    (1, 8)    1       1        [0.111]
# Total_time                                                                           -                                                                                    2454.129  -        -         -       -        -
# Output: [[-53  ...  -65]]
# >>>
python3 microtvm_demo.py toycar --executor graph --mode standalone --benchmark
# <<<
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    2.2320       2.2320       2.2320       2.2320       0.0000
# Output: [[-53  ... -65]]
# >>>
python3 microtvm_demo.py toycar --executor graph --mode host-driven --cmsisnn --benchmark
# <<<
# TVMError: unknown type = 129 -> Missing rpc serialization of benchmark dict
# >>>
python3 microtvm_demo.py toycar --executor graph --mode host-driven --cmsisnn --profile
# <<<
# InternalError: Check failed: (pf != nullptr) is false: no such function in module: tvmgen_default_cmsis_nn_main_0
# >>>
python3 microtvm_demo.py toycar --executor graph --mode standalone --cmsisnn --benchmark
# <<<
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    0.0080       0.0080       0.0080       0.0080       0.0000
# Output: [[0 ... 0]]
# -> invalid output because of graph executor bug!
# >>>
python3 microtvm_demo.py toycar --executor aot --mode host-driven --benchmark
# <<<
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    2.5860       2.5860       2.5860       2.5860       0.0000
# Output: [[-53  ... -65]]
# >>>
python3 microtvm_demo.py toycar --executor aot --mode host-driven --cmsisnn --benchmark
# <<<
# Execution time summary:
#  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
#    1.3570       1.3570       1.3570       1.3570       0.0000
# Output: [[-53  ... -65]]
# >>>
