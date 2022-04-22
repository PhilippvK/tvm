set(MLF_DIR ${CMAKE_SOURCE_DIR}/model/)
set(CRT_DIR ${CMAKE_SOURCE_DIR}/crt/)

 set(TVM_KERNEL_SRCS ${MLF_DIR}/codegen/host/src/default_lib0.c ${MLF_DIR}/codegen/host/src/default_lib1.c)
file(GLOB_RECURSE TVM_KERNEL_SRCS ${MLF_DIR}/codegen/host/src/*.c)

# set(TVM_SRCS ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_server/rpc_server.cc ${MLF_DIR}/runtime/src/runtime/crt/memory/page_allocator.c ${MLF_DIR}/runtime/src/runtime/crt/common/crt_runtime_api.c ${MLF_DIR}/runtime/src/runtime/crt/common/func_registry.c ${MLF_DIR}/runtime/src/runtime/crt/common/ndarray.c ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_common/session.cc ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_common/frame_buffer.cc ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_common/framing.cc ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_common/write_stream.cc ${MLF_DIR}/runtime/src/runtime/crt/microtvm_rpc_common/crcccitt.c)
set(TVM_SRCS ${CRT_DIR}/src/runtime/crt/microtvm_rpc_server/rpc_server.cc ${CRT_DIR}/src/runtime/crt/memory/page_allocator.c ${CRT_DIR}/src/runtime/crt/common/crt_runtime_api.c ${CRT_DIR}/src/runtime/crt/common/func_registry.c ${CRT_DIR}/src/runtime/crt/common/ndarray.c ${CRT_DIR}/src/runtime/crt/microtvm_rpc_common/session.cc ${CRT_DIR}/src/runtime/crt/microtvm_rpc_common/frame_buffer.cc ${CRT_DIR}/src/runtime/crt/microtvm_rpc_common/framing.cc ${CRT_DIR}/src/runtime/crt/microtvm_rpc_common/write_stream.cc ${CRT_DIR}/src/runtime/crt/microtvm_rpc_common/crcccitt.c ${CRT_DIR}//src/runtime/crt/common/crt_backend_api.c)

# set(TVM_INCS ${MLF_DIR}/runtime/include/ ${MLF_DIR}/codegen/host/include/ ${CMAKE_SOURCE_DIR}/crt_config/)
# set(TVM_INCS ${CRT_DIR}/include/ ${MLF_DIR}/codegen/host/include/ ${CMAKE_SOURCE_DIR}/crt_config/)
set(TVM_INCS ${CRT_DIR}/include/ ${CMAKE_SOURCE_DIR}/crt_config/)


idf_component_register(SRCS "main.c" ${TVM_SRCS} ${TVM_KERNEL_SRCS}
                    INCLUDE_DIRS ${TVM_INCS})


target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=unused-but-set-parameter)
