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

## Usage

Hint: Don't forget `export PYTHONPATH=$(pwd)/python`

### Run untuned Resnet Model via MicroTVM (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap --skip-tuning --out outputs/bench_resnet_baseline
```

### Tune Resnet Model (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap --num-trials-per-iter 5 --max-trials-per-task 50 --skip-bench --out outputs/tune_resnet_baseline
```

### Tune and Run Resnet Model (WCA enabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap --num-trials-per-iter 5 --max-trials-per-task 50 --enable-custom --enable-intrin --cfu-mode=MODE_CFU --out outputs/tune_resnet_wca
```

### Run Tuned Resnet Model via MicroTVM (WCA disabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap --ms-db ? --out outputs/run_resnet_baseline_tuned
```

### Run Tuned Resnet Model via MicroTVM (WCA enabled)

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model new/pretrainedResnet_clustered_quant_remap --enable-custom --enable-intrin --cfu-mode=MODE_CFU --ms-db ? --out outputs/run_resnet_wca_tuned
```

### Single layer tuning run with higher verbosity to see errors

```sh
python3 tests/python/micro/cfu_wca_etiss_script.py --model layers_unpacked/pretrainedResnet_clustered_quant_remap_layer5 --num-trials-per-iter 1 --max-trials-per-task 1 --max-trials-global 1 --enable-custom --enable-intrin --cfu-mode=MODE_CFU --out outputs/debug_resnet_cfu
```

Output artifacts:

```
outputs/tune_resnet_cfu
├── project/   # MicroTVM project directory (tuned)
├── project2/  # MicroTVM project directory (untuned)
├── logs/      # Tuning logfiles
├── database_tuning_record.json  # MetaScheduler DB Tuning Records
├── database_workload.json       # MetaScheduler DB Tuning Workloads
└── metrics.csv
```
