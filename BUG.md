# TVM Bug Demo with VWW TFLite model and CRT Graph Runtime

See TVM Issue for details: https://github.com/apache/tvm/issues/8953

To reproduce the issue, you need the following:

- TVM build with microTVM, StandaloneCRT and LLVM enabled
- The demo in `apps/tflite_vww_bug_demo`
- The python packages listed in `apps/tflite_vww_bug_demo/requirements.txt`

## Detailed instructions

1. Clone this repo

```
git clone --recursive https://github.com/apache/tvm.git
cd tvm
```

2. Apply `config.cmake` patch

```
git apply tvm_config.patch
```

3. Create build Dir and copy Config

```
mkdir build
cp cmake/config.cmake build/
```

5. Download LLVM if not already available

```
cd ..
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.1.0/clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar xf clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
mv clang+llvm-11.1.0-x86_64-linux-gnu-ubuntu-16.04 llvm
cd tvm
```

6. Edit config to pint to llvm

```
sed -i -- "s/USE_LLVM OFF/USE_LLVM ..\\/..\\/llvm\\/bin\\/llvm-config/g" build/config.cmake
```

7. Configure cmake and build TVM

```
cd build
cmake ..
make -j`nproc`
cd ..
```

9. Set pythonpath

```
export PYTHONPATH=$(pwd)/python
```

10. Go to `apps/tflite_vww_bug_demo`

```
cd apps/tflite_vww_bug_demo
```

11. Create virtualenv

```
python3 -m venv venv
# or
virtualenv venv
```

12. Enter virtualenv

```
source venv/bin/activate
```

13. Install requirments

```
pip install -r requirements.txt
```

14. Build model and run demo

```
make demo_static
```

15. Segfault should happen when executing the build binary

## CI/CD Script

Alternatively you can just have a look at [this](https://github.com/PhilippvK/tvm/blob/bug-demo/.github/workflows/bug.yml) Gitlab Runner workflow.
