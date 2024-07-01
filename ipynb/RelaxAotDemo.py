#!/usr/bin/env python
# coding: utf-8

# ## Imports

# Imports

# In[2]:


import tvm
import tvm.testing
from tvm.relay import testing
from tvm import relax, relay
from tvm.relax.testing import relay_translator, nn
from tvm.runtime import vm as vm_rt
from tvm.script import relax as R
import numpy as np


# ## Config

# In[3]:


# Target
TARGET = "llvm"
# TARGET = "c"
# TARGET = tvm.target.Target("c", host="c")
target = tvm.target.Target(TARGET, host=TARGET)

# Pipeline (Relax only)
# PIPELINE = "default"
PIPELINE = "micro"
default_pipeline = "default_build"
micro_pipeline = "micro2_build"

# Exec mode (Relax only)
# EXEC_MODE = "bytecode"
# EXEC_MODE = "compiled"
EXEC_MODE = "crt"
bytecode_exec_mode = "bytecode"
compiled_exec_mode = "compiled"
crt_exec_mode = "crt"

# Executor (Relay only)
UNPACKED = False
USMP = False
# EXECUTOR = "graph"
EXECUTOR = "aot"
aot_executor = tvm.relay.backend.Executor("aot", {"interface-api": "c" if UNPACKED else "packed", "unpacked-api": UNPACKED})
graph_executor = tvm.relay.backend.Executor("graph", {"link-params": False})

# Runtime (Relay only?)
RUNTIME = "cpp"
# RUNTIME = "crt"
SYSTEM_LIB = True
cpp_runtime = tvm.relay.backend.Runtime("cpp", {"system-lib": SYSTEM_LIB})
crt_runtime = tvm.relay.backend.Runtime("crt", {"system-lib": SYSTEM_LIB})

# Pass Config (Relay only)
USMP = False
FUSE_DEPTH = 1
VECTORIZE = False
pass_config = {"tir.usmp.enable": USMP, "relay.FuseOps.max_depth": FUSE_DEPTH, "tir.disable_vectorize": not VECTORIZE}


# ## Define Models

# ### Common

# Matmul

# In[4]:


matmul_input_size = 64
matmul_hidden_size = 10
matmul_output_size = 4

matmul_dtype = "float32"

matmul_weights_matrix = np.random.random((matmul_hidden_size, matmul_output_size)).astype(matmul_dtype)
matmul_bias_matrix = np.random.random((matmul_output_size,)).astype(matmul_dtype)

matmul_params = {"weights": tvm.nd.array(matmul_weights_matrix), "bias": tvm.nd.array(matmul_bias_matrix)}
matmul_data = tvm.nd.array(np.random.rand(matmul_input_size, matmul_hidden_size).astype(matmul_dtype))


# Conv2d

# In[5]:


conv2d_input_n = 1
conv2d_input_c = 16
conv2d_input_h = 64
conv2d_input_w = 64
conv2d_kernel_h = 4
conv2d_kernel_w = 4
conv2d_kernel_ci = 16
conv2d_kernel_co = 16
conv2d_output_n = 1
conv2d_output_c = 16
conv2d_output_h = 61
conv2d_output_w = 61

conv2d_dtype = "float32"

conv2d_weights_matrix = np.random.random((conv2d_kernel_h, conv2d_kernel_w, conv2d_kernel_ci, conv2d_kernel_co)).astype(conv2d_dtype)
conv2d_bias_matrix = np.random.random((conv2d_output_w,)).astype(conv2d_dtype)


# ### Relax

# Define Matmul model (Relax)

# In[6]:


def relax_dense():
    builder = relax.BlockBuilder()

    with builder.function("main"):
        input = relax.Var("x", R.Tensor((matmul_input_size, matmul_hidden_size), matmul_dtype))
        weights = relax.Constant(tvm.nd.array(matmul_weights_matrix))
        bias = relax.Constant(tvm.nd.array(matmul_bias_matrix))
        output_matmul = relax.op.matmul(input, weights)
        output_bias = relax.op.add(output_matmul, bias)
        builder.emit_func_output(output_bias, params=[input])
    return builder.get(), matmul_data, matmul_params


# Define Conv2d model (Relax)

# In[7]:


def relax_conv2d():
    builder = relax.BlockBuilder()

    with builder.function("main"):
        input = relax.Var("x", R.Tensor((input_n, input_c, input_h, input_w), conv2d_dtype))
        weights = relax.Constant(tvm.nd.array(weights_matrix))
        bias = relax.Constant(tvm.nd.array(bias_matrix))
        output_conv2d = relax.op.nn.conv2d(input, weights, data_layout="NCHW", kernel_layout="HWIO")
        output_bias = relax.op.add(output_conv2d, bias)
        builder.emit_func_output(output_bias, params=[input])

    return builder.get(), conv2d_data, conv2d_params


# ### Relay

# Define Matmul model (Relay)

# In[8]:


def relay_dense():

    x = relay.var("x", shape=(matmul_input_size, matmul_hidden_size), dtype=matmul_dtype)
    weight = relay.const(tvm.nd.array(matmul_weights_matrix))
    bias = relay.const(tvm.nd.array(matmul_bias_matrix))
    output_matmul = relay.nn.matmul(x, weight)
    output_bias = relay.op.add(output_matmul, bias)
    func = relay.Function(relay.analysis.free_vars(output_bias), output_bias)
    return tvm.IRModule.from_expr(func), matmul_data, matmul_params


# Define Conv2D model (Relay)

# In[9]:


_ = """
def relay_conv2d():
    dtype = "float32"  #  TODO: int32

    weights_matrix = np.random.random((hidden_size, output_size)).astype(dtype)
    bias_matrix = np.random.random((output_size,)).astype(dtype)

    x = relay.var("x", shape=(input_size, hidden_size), dtype=dtype)
    weight = relay.const(tvm.nd.array(weights_matrix))
    bias = relay.const(tvm.nd.array(bias_matrix))
    output_matmul = relay.nn.matmul(x, weight)
    output_bias = relay.op.add(output_matmul, bias)
    func = relay.Function(relay.analysis.free_vars(output_bias), output_bias)
    return tvm.IRModule.from_expr(func)
    return func, conv2d_data, conv2d_params
"""


# ## Show models

# Pick used models

# In[10]:


# -- Relax --
relax_mod, relax_data, relax_params = relax_dense()
# relax_mod = relax_conv2d()

# -- Relay --
relay_mod, relay_data, relay_params = relay_dense()
# relay_mod = relay_conv2d()


# Show Relax module

# In[11]:


relax_mod.show()


# Show Relay module

# In[12]:


relay_mod.show()


# ## Instruments

# Define Pass Instrument to look at intermediate IRs during build

# In[13]:


@tvm.instrument.pass_instrument
class MyInstrument:

    def __init__(self):
        self.skip_pass_name = []
        self.output = []
        self.output_after = []
        self.idx = 0

    def run_before_pass(self, mod, pass_info):
        self.idx += 1
        handle = mod.handle
        g = dict(mod.global_var_map_)
        g_ = list(g.keys())
        if len(g_) == 0:
            return
        # print(self.idx * "  " + ">", self.idx, pass_info.name, g_, len(self.output))
        # print(dir(mod))

        # tmp = (mod.astext(show_meta_data=True), str(pass_info))
        # tmp = (str(mod), str(pass_info))
        # tmp = (mod.script(show_meta=True), str(pass_info))
        tmp = (mod, pass_info)
        self.output.append(tmp)


    def run_after_pass(self, mod, pass_info):
        self.idx -= 1
        handle = mod.handle
        g = dict(mod.global_var_map_)
        g_ = list(g.keys())
        if len(g_) == 0:
            return
        # print((self.idx + 1) * "  " + "<", self.idx + 1, pass_info.name, g_, len(self.output_after))
        # print(dir(mod))
        # tmp = (mod.astext(show_meta_data=True), str(pass_info))
        # tmp = (str(mod), str(pass_info))
        tmp = (mod, pass_info)
        self.output_after.append(tmp)
        pass


# ## Build

# ### Relax

# Relax build (VM Bytecode)

# In[14]:


if PIPELINE == "default" and EXEC_MODE == "bytecode" and TARGET == "llvm":
    relax_instrument_ex_vm_bytecode_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relax_instrument_ex_vm_bytecode_llvm]):
        ex_vm_bytecode_llvm = relax.build(relax_mod, target=target, pipeline=default_pipeline, exec_mode=bytecode_exec_mode)
else:
    relax_instrument_ex_vm_bytecode_llvm = None
    ex_vm_bytecode_llvm = None


# Relax build (VM Compiled)

# In[15]:


if PIPELINE == "default" and EXEC_MODE == "compiled" and TARGET == "llvm":
    relax_instrument_ex_vm_compiled_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relax_instrument_ex_vm_compiled_llvm]):
        ex_vm_compiled_llvm = relax.build(relax_mod, target=target, pipeline=default_pipeline, exec_mode=compiled_exec_mode)
else:
    relax_instrument_ex_vm_compiled_llvm = None
    ex_vm_compiled_llvm = None


# Relax build (AOT CPP LLVM)

# In[ ]:


if PIPELINE == "micro" and EXEC_MODE == "crt" and TARGET == "llvm" and RUNTIME == "cpp":
    relax_instrument_ex_aot_cpp_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relax_instrument_ex_aot_cpp_llvm]):
        ex_aot_cpp_llvm = relax.build(relax_mod, target=target, pipeline=micro_pipeline, exec_mode=crt_exec_mode, executor=aot_executor, runtime=cpp_runtime)
else:
    relax_instrument_ex_aot_cpp_llvm = None
    ex_aot_cpp_llvm = None


# Relax build (AOT CRT LLVM)

# In[ ]:


if PIPELINE == "micro" and EXEC_MODE == "crt" and TARGET == "llvm" and RUNTIME == "crt":
    relax_instrument_ex_aot_crt_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relax_instrument_ex_aot_crt_llvm]):
        ex_aot_crt_llvm = relax.build(relax_mod, target=target, pipeline=micro_pipeline, exec_mode=crt_exec_mode, executor=aot_executor, runtime=crt_runtime)
else:
    relax_instrument_crt_ex_aot = None
    ex_aot_crt_llvm = None


# Relax build (AOT CRT C)

# In[ ]:


if PIPELINE == "micro" and EXEC_MODE == "crt" and TARGET == "c" and RUNTIME == "crt":
    relax_instrument_ex_aot_crt_c = MyInstrument()
    with tvm.transform.PassContext(instruments=[relax_instrument_ex_aot_crt_c]):
        ex_aot_crt_c = relax.build(relax_mod, target=target, pipeline=micro_pipeline, exec_mode=crt_exec_mode, executor=aot_executor, runtime=crt_runtime)
else:
    relax_instrument_crt_ex_c = None
    ex_aot_crt_c = None


# In[ ]:


# In[22]:


# crt_ex_aot, type(crt_ex_aot), dir(crt_ex_aot)
# crt_ex_aot.mod, type(crt_ex_aot.mod), dir(crt_ex_aot.mod), crt_ex_aot.mod.entry_name, crt_ex_aot.mod.imported_modules, crt_ex_aot.mod.is_runnable
# crt_ex_aot.mod.entry_func
# crt_ex_aot.mod.imported_modules[0], type(crt_ex_aot.mod.imported_modules[0]), dir(crt_ex_aot.mod.imported_modules[0])


# ### Relay

# Relay build (Default C++, LLVM)

# In[17]:


if RUNTIME == "cpp" and EXECUTOR == "graph" and TARGET == "llvm":
    relay_instrument_lib_graph_cpp_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_lib_graph_cpp_llvm], config=pass_config):
        lib_graph_cpp_llvm = relay.build(relay_mod, target=target, runtime=cpp_runtime, executor=graph_executor)
else:
    relay_instrument_lib_graph_cpp_llvm = None
    lib_graph_cpp_llvm = None


# Relay build (AoT C++, LLVM)

# In[19]:


if RUNTIME == "cpp" and EXECUTOR == "aot" and TARGET == "llvm" and not UNPACKED and not USMP:
    print("HERE")
    relay_instrument_lib_aot_cpp_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_lib_aot_cpp_llvm], config=pass_config):
        lib_aot_cpp_llvm = relay.build(relay_mod, target=target, runtime=cpp_runtime, executor=aot_executor)
else:
    relay_instrument_lib_aot_cpp_llvm = None
    lib_aot_cpp_llvm = None


# Relay build (Graph CRT, no USMP, Packed API, LLVM)

# In[20]:


if RUNTIME == "crt" and EXECUTOR == "graph" and TARGET == "llvm" and not UNPACKED and not USMP:
    relay_instrument_crt_lib_graph = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_crt_lib_graph], config=pass_config):
        crt_lib_graph = relay.build(relay_mod, target=target, runtime=crt_runtime, executor=graph_executor)
else:
    relay_instrument_crt_lib_graph = None
    crt_lib_graph = None


# Relay build (Graph CRT, no USMP, Packed API, C)

# In[20]:


if RUNTIME == "crt" and EXECUTOR == "graph" and TARGET == "c" and not UNPACKED and not USMP:
    relay_instrument_crt_lib_graph = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_crt_lib_graph], config=pass_config):
        crt_lib_graph = relay.build(relay_mod, target=target, runtime=crt_runtime, executor=graph_executor)
else:
    relay_instrument_crt_lib_graph = None
    crt_lib_graph = None


# Relay build (AOT CRT, no USMP, Packed API, LLVM)

# In[21]:


if RUNTIME == "crt" and EXECUTOR == "aot" and TARGET == "llvm" and not UNPACKED and not USMP:
    relay_instrument_lib_aot_crt_llvm = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_lib_aot_crt_llvm], config=config):
        lib_aot_crt_llvm = relay.build(relay_mod, target=target, runtime=crt_runtime, executor=aot_executor)
else:
    relay_instrument_lib_aot_crt_llvm = None
    lib_aot_crt_llvm = None


# Relay build (AOT CRT, no USMP, Packed API, C)

# In[21]:


if RUNTIME == "crt" and EXECUTOR == "aot" and TARGET == "c" and UNPACKED and not USMP:
    relay_instrument_lib_aot_crt_c = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_lib_aot_crt_c], config=config):
        lib_aot_crt_c = relay.build(relay_mod, target=target, runtime=crt_runtime, executor=aot_executor)
else:
    relay_instrument_lib_aot_crt_c = None
    lib_aot_crt_c = None


# Relay build (AOT CRT, no USMP, Unpacked API, C)

# In[21]:


if RUNTIME == "crt" and EXECUTOR == "aot" and TARGET == "c" and not USMP:
    relay_instrument_lib_aot_crt_c = MyInstrument()
    with tvm.transform.PassContext(instruments=[relay_instrument_lib_aot_crt_c], config=config):
        lib_aot_crt_c = relay.build(relay_mod, target=target, runtime=crt_runtime, executor=aot_executor)
else:
    relay_instrument_lib_aot_crt_c = None
    lib_aot_crt_c = None

# Relay build (AOT CRT, no USMP, Unpacked API, C, USMP)
# TODO


# Pick compiled module

# In[22]:


# -- Relax --
if RUNTIME == "cpp":
    if TARGET == "llvm":
        if PIPELINE == "default" and EXEC_MODE == "bytecode":
            ex, relax_instrument = ex_vm_bytecode_llvm, relax_instrument_ex_vm_bytecode_llvm
        elif PIPELINE == "default" and EXEC_MODE == "compiled":
            ex, relax_instrument = ex_vm_compiled_llvm, relax_instrument_ex_vm_compiled_llvm
        elif PIPELINE == "micro" and EXEC_MODE == "crt":
            ex, relax_instrument = ex_aot_cpp_llvm, relax_instrument_ex_aot_cpp_llvm
        else:
            assert False, f"Invalid PIPELINE ({PIPELINE}) and EXEC_MODE ({EXEC_MODE}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    else:
        assert False, f"Invalid TARGET ({TARGET}) for RUNTIME ({RUNTIME})"
elif RUNTIME == "crt":
    if TARGET == "llvm":
        if PIPELINE == "micro" and EXEC_MODE == "crt":
            ex, relax_instrument = ex_aot_crt_llvm, relax_instrument_ex_aot_crt_llvm
        else:
            assert False, f"Invalid PIPELINE ({PIPELINE}) and EXEC_MODE ({EXEC_MODE}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    elif TARGET == "c":
        if PIPELINE == "micro" and EXEC_MODE == "crt":
            ex, relax_instrument = ex_aot_crt_c, relax_instrument_ex_aot_crt_c
        else:
            assert False, f"Invalid PIPELINE ({PIPELINE}) and EXEC_MODE ({EXEC_MODE}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    else:
        assert False, f"Invalid TARGET ({TARGET}) for RUNTIME ({RUNTIME})"
else:
    assert False, f"Invalid RUNTIME ({RUNTIME})"

# -- Relay --
if RUNTIME == "cpp":
    if TARGET == "llvm":
        if EXECUTOR == "graph" and not UNPACKED and not USMP:
            lib, relay_instrument = lib_graph_cpp_llvm, relay_instrument_lib_graph_cpp_llvm
        elif EXECUTOR == "aot" and not UNPACKED and not USMP:
            lib, relay_instrument = lib_aot_cpp_llvm, relay_instrument_lib_aot_cpp_llvm
        else:
            assert False, f"Invalid EXECUTOR ({EXECUTOR}), UNPACKED ({UNPACKED}) and USMP ({USMP}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    else:
        assert False, f"Invalid TARGET ({TARGET}) for RUNTIME ({RUNTIME})"
elif RUNTIME == "crt":
    if TARGET == "llvm":
        if EXECUTOR == "graph" and not UNPACKED and not USMP:
            lib, relay_instrument = ib_graph_crt_llvm, relay_instrument_lib_graph_crt_llvm
        elif EXECUTOR == "aot" and not UNPACKED and not USMP:
            lib, relay_instrument = lib_aot_crt_llvm, relay_instrument_lib_aot_crt_llvm
        else:
            assert False, f"Invalid EXECUTOR ({EXECUTOR}), UNPACKED ({UNPACKED}) and USMP ({USMP}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    elif TARGET == "c":
        if EXECUTOR == "graph" and not UNPACKED and not USMP:
            lib, relay_instrument = lib_graph_crt_c, relay_instrument_lib_graph_crt_c
        elif EXECUTOR == "aot" and not UNPACKED and not USMP:
            lib, relay_instrument = lib_aot_crt_c, relay_instrument_lib_aot_crt_c
        elif EXECUTOR == "aot" and UNPACKED and not USMP:
            lib, relay_instrument = lib_aot_crt_c_unpacked, relay_instrument_lib_aot_crt_c_unpacked
        elif EXECUTOR == "aot" and UNPACKED and USMP:
            lib, relay_instrument = lib_aot_crt_c_unpacked_usmp, relay_instrument_lib_aot_crt_c_unpacked_usmp
        else:
            assert False, f"Invalid EXECUTOR ({EXECUTOR}), UNPACKED ({UNPACKED}) and USMP ({USMP}) for RUNTIME ({RUNTIME}) and TARGET ({TARGET})"
    else:
        assert False, f"Invalid TARGET ({TARGET}) for RUNTIME ({RUNTIME})"
else:
    assert False, f"Invalid RUNTIME ({RUNTIME})"


# ## Investigate

# Look at generated C code (Relax)

# In[23]:


if TARGET == "c":
    print(ex.mod._collect_dso_modules()[0].get_source())
# print(ex.lib._collect_dso_modules()[0].get_source())
print(ex.lib._collect_dso_modules()[1].get_source())


# Look at generated C code (Relay)

# In[24]:


if TARGET == "c":
    print(lib.lib._collect_dso_modules()[0].get_source())
print(lib.lib._collect_dso_modules()[0].get_source())


# Intermediate IRs (Relax)

# In[25]:


relax_instrument.output[-1][0].show()


# Intermediate IRs (Relay)

# In[26]:


relay_instrument.output[-1][0].show()


# ## Run

# ### Relax

# In[27]:


if TARGET == "llvm":
    if EXEC_MODE in ["bytecode", "compiled"]:
        vm = relax.VirtualMachine(ex, tvm.cpu())
        relax_output = vm["main"](relax_data).numpy()
    elif EXEC_MODE == "crt":
        pass
    else:
        assert False
else:
    relax_output = None
    print("C target does not support execution")


# Show Result

# In[28]:


# relax_output


# In[29]:


# ex, type(ex), dir(ex)
# ex.mod, type(ex.mod), dir(ex.mod), ex.mod.entry_name, ex.mod.imported_modules, ex.mod.is_runnable
# ex.mod.entry_func
# print("1", ex.mod.imported_modules[0])
# print("2", type(ex.mod.imported_modules[0]), dir(ex.mod.imported_modules[0]), ex.mod.entry_name)
# print("3", ex.mod, dir(ex.mod))
# print("4", lib, dir(lib))


# In[30]:


# rt_mod = tvm.contrib.graph_executor.GraphModule(ex.mod["default"])
# fcreate = get_global_func("tvm.aot_executor_factory.create")
# print("fcreate", fcreate)
# args = []
# factory_module = fcreate(ex.mod, "default", *args)
# executor_factory = _executor_factory.AOTExecutorFactoryModule(
#     None,  # ir_mod,
#     None,  # lowered_ir_mods,
#     TARGET,
#     EXECUTOR,
#     RUNTIME,
#     ex.mod,
#     "default",
#     None,  # relax_params,
#     None,  # func_metadata,
#     None,  # executor_codegen_metadata,
#     devices,
# )
# print("factory_module", factory_module)
dev = tvm.device(str(TARGET), dev_id=0)
factory_module = ex
# print("relay lib", lib, type(lib), dir(lib))
# print("relay lib.lib", lib.lib, type(lib.lib), dir(lib.lib))
# print("factory_module", factory_module, type(factory_module), dir(factory_module))
# print("factory_module.lib", factory_module.lib, type(factory_module.lib), dir(factory_module.lib))
rt_mod = tvm.runtime.executor.AotModule(factory_module["default"](dev))
# rt_mod = tvm.runtime.executor.AotModule(lib["default"](dev))
# print("rt_mod", rt_mod)
# print("relax_data", relax_data.shape)
# print("relay_data", relay_data.shape)
rt_mod.set_input("x", relax_data)
rt_mod.run()
relax_output = rt_mod.get_output(0).numpy()


# ### Relay

# In[31]:


# lib, type(lib), dir(lib)
# lib.ir_mod, type(lib.ir_mod), dir(lib.ir_mod)
lib.lib, type(lib.lib), dir(lib.lib)


# In[118]:


if TARGET == "llvm":
    dev = tvm.device(str(TARGET), dev_id=0)
    if EXECUTOR == "graph":
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    elif EXECUTOR == "aot":
        rt_mod = tvm.runtime.executor.AotModule(lib["default"](dev))
    else:
        assert False
    rt_mod.set_input("x", relay_data)
    rt_mod.run()
    relay_output = rt_mod.get_output(0).numpy()
else:
    relay_output = None
    print("C target does not support execution")


# Show Result

# In[56]:


# relay_output


# ## Compare

# In[57]:


assert not(relax_output is None or relay_output is None)
print("relax_output", relax_output)
print("relay_output", relay_output)
tvm.testing.assert_allclose(relax_output, relay_output, rtol=1e-4, atol=1e-4)
