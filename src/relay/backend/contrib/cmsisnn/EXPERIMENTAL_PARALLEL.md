# Experimental two-task elementwise dispatch

This draft adds an opt-in C emitter for CMSIS-NN int8/int16 elementwise add and
multiply. The default remains the existing serial call.

Enable the option in the PassContext surrounding Relay compilation:

```python
with tvm.transform.PassContext(
    opt_level=3,
    config={
        "relay.ext.cmsisnn.options": {
            "experimental_parallel_elementwise": True,
            # Preserve any existing mcpu, mattr, and debug_last_error options here.
        },
    },
):
    factory = relay.build(partitioned_mod, target=target, runtime=runtime, executor=executor)
```

The emitter requests `TVMBackendParallelLaunch(callback, closure, 2)` directly.
It does not depend on `tir.aot_preserve_parallel_for` or create a TIR parallel
loop. This keeps the experiment local to the opaque CMSIS-NN calls.

For N elements, tile 0 processes `[0, ceil(N/2))` and tile 1 processes
`[ceil(N/2), N)`. All three pointers advance in elements of the appropriate
int8/int16 type. Quantization and activation parameters remain unchanged.
Both inputs may refer to the same tensor. No scratch allocation is introduced.
The existing BYOC preparation of constants/broadcasts is unchanged: the emitter
expects the full-length input buffers required by the original CMSIS-NN call.
Calls with fewer than two elements, or nonconstant lengths, remain serial.

## Runtime contract

- The launcher must run callbacks on distinct cores for actual parallelism.
  Two requested tasks alone do not guarantee two-core execution.
- It must wait for every callback to finish before returning, including when a
  callback fails: the closure lives on the caller's stack and the next operator
  may consume the output immediately.
- It must provide consistent `num_task` and task IDs in `[0, num_task)`, and
  propagate nonzero callback results. With one worker, the callback executes
  both tiles; with two workers, each executes one tile. Extra workers do no work.
- The generated caller returns -1 on launch failure. With `debug_last_error`,
  it sets a generic error after joining, avoiding worker writes to error state.
- The default CRT launcher still runs one worker and ignores callback errors.
  For a replacement CRT launcher, the existing `IDF_BACKEND_PARALLEL_LAUNCH`
  guard must be defined when compiling `crt_backend_api.c`. The macro only
  suppresses the stub; the platform must supply the replacement implementation.
- Shared memory must be visible to both cores, with any platform-required
  cache maintenance and synchronization handled by the runtime.

## Validation status and next checks

Host validation completed using the `.env2` environment and `build_spacemit`:

- `make -C build_spacemit -j36` passed. Compilation exposed an overload lookup
  issue in the serial fallback; it now explicitly calls `CodeGenC::VisitStmt_`.
- All 8 focused C++ tests passed (3 parallel-elementwise tests and 5 existing
  compiler-attribute tests). They were linked separately against the built TVM
  library and system GTest because this build configuration disables GTest.
- All 27 existing Python tests for constant extraction, scalar-to-tensor
  conversion, constant generation, and padding fusion passed.
- A standalone host harness generated 128 C modules from 64 Relay variants,
  with the option disabled/enabled. The emitted C compiled with GCC using
  `-Wall -Wextra -Werror` and local CMSIS-NN scalar C kernels.
- All 960 comparisons with the serial CMSIS-NN output were bit-exact using
  pthread launchers with 1, 2, and 3 workers. Cases cover int8/int16 add/multiply,
  lengths 1/2/3/7/8/17/32/257, distinct/identical inputs, clipping, extreme values,
  repeated random inputs, and output guard elements.
- All 112 error-path checks passed for injected launch failures and invalid
  worker counts (callback failure), including error reporting and untouched
  output buffers.

These checks do not validate Arm DSP/MVE kernels or the actual device launcher.
Before enabling on hardware, check those builds and runtime synchronization,
including constant inputs and memory boundaries. No device execution, tuning,
or performance measurement was performed.

Measure end-to-end layer latency with persistent workers. No profitability
threshold is implemented beyond the single-element fallback; small or
memory-bandwidth-bound layers can be slower with dispatch enabled.
