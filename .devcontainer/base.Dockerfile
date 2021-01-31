# [Choice] Ubuntu version: bionic, focal
ARG VARIANT=bionic
FROM mcr.microsoft.com/vscode/devcontainers/base:${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu1804_install_python.sh /install/ubuntu1804_install_python.sh
RUN bash /install/ubuntu1804_install_python.sh

COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

COPY install/ubuntu1804_install_llvm.sh /install/ubuntu1804_install_llvm.sh
RUN bash /install/ubuntu1804_install_llvm.sh

COPY install/ubuntu_install_dnnl.sh /install/ubuntu_install_dnnl.sh
RUN bash /install/ubuntu_install_dnnl.sh

# Install MxNet for access to Gluon Model Zoo.
COPY install/ubuntu_install_mxnet.sh /install/ubuntu_install_mxnet.sh
RUN bash /install/ubuntu_install_mxnet.sh

# Rust env (build early; takes a while)
COPY install/ubuntu_install_rust.sh /install/ubuntu_install_rust.sh
RUN bash /install/ubuntu_install_rust.sh
ENV RUSTUP_HOME /opt/rust
ENV CARGO_HOME /opt/rust

# AutoTVM deps
COPY install/ubuntu_install_redis.sh /install/ubuntu_install_redis.sh
RUN bash /install/ubuntu_install_redis.sh

# Golang environment
COPY install/ubuntu_install_golang.sh /install/ubuntu_install_golang.sh
RUN bash /install/ubuntu_install_golang.sh

# NNPACK deps
COPY install/ubuntu_install_nnpack.sh /install/ubuntu_install_nnpack.sh
RUN bash /install/ubuntu_install_nnpack.sh

ENV PATH $PATH:$CARGO_HOME/bin:/usr/lib/go-1.10/bin

# ANTLR deps
COPY install/ubuntu_install_java.sh /install/ubuntu_install_java.sh
RUN bash /install/ubuntu_install_java.sh

# BYODT deps
COPY install/ubuntu_install_universal.sh /install/ubuntu_install_universal.sh
RUN bash /install/ubuntu_install_universal.sh

# Chisel deps for TSIM
COPY install/ubuntu_install_sbt.sh /install/ubuntu_install_sbt.sh
RUN bash /install/ubuntu_install_sbt.sh

# Verilator deps
COPY install/ubuntu_install_verilator.sh /install/ubuntu_install_verilator.sh
RUN bash /install/ubuntu_install_verilator.sh

# TFLite deps
COPY install/ubuntu_install_tflite.sh /install/ubuntu_install_tflite.sh
RUN bash /install/ubuntu_install_tflite.sh

# TensorFlow deps
COPY install/ubuntu_install_tensorflow.sh /install/ubuntu_install_tensorflow.sh
RUN bash /install/ubuntu_install_tensorflow.sh

# Caffe deps
COPY install/ubuntu_install_caffe.sh /install/ubuntu_install_caffe.sh
RUN bash /install/ubuntu_install_caffe.sh

# Github Arm(R) Ethos(TM)-N NPU driver
COPY install/ubuntu_install_ethosn_driver_stack.sh /install/ubuntu_install_ethosn_driver_stack.sh
RUN bash /install/ubuntu_install_ethosn_driver_stack.sh

# Vitis-AI PyXIR CI deps
COPY install/ubuntu_install_vitis_ai_packages_ci.sh /install/ubuntu_install_vitis_ai_packages_ci.sh
RUN bash /install/ubuntu_install_vitis_ai_packages_ci.sh

# Arm(R) Compute Library
#COPY install/ubuntu_install_arm_compute_lib.sh /install/ubuntu_install_arm_compute_lib.sh
#RUN bash /install/ubuntu_install_arm_compute_lib.sh


# Jupyter notebook.
RUN pip3 install matplotlib Image Pillow jupyter[notebook]

# Deep learning frameworks
RUN pip3 install mxnet tensorflow keras gluoncv dgl

# Build TVM
#COPY install/install_tvm_cpu.sh /install/install_tvm_cpu.sh
#RUN bash /install/install_tvm_cpu.sh

# Environment variables
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/vta/python:${PYTHONPATH}
