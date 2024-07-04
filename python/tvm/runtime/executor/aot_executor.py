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
"""A Python wrapper for the Module-based Model Runtime Interface for Ahead-of-Time compilation."""

import numpy as np


class AotModule(object):
    """Wraps the AOT executor runtime.Module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the implemented model functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the implemented model functions.

    Examples
    --------

    .. code-block:: python

        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor

        # build the library using graph executor
        lib = relay.build(...)
        lib.export_library("compiled_lib.so")
        # load it back as a runtime
        lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
        # Call the library factory function for default and create
        # a new runtime.Module, wrap with aot module.
        gmod = tvm.runtime.executor.AotModule(lib["default"](dev))
        # use the aot  module.
        gmod.set_input("x", data)
        gmod.run()
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_input_index = module["get_input_index"]
        self._get_num_inputs = module["get_num_inputs"]
        self._get_input_name = module["get_input_name"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if key is not None:
            v = self._get_input(key)
            if v is None:
                raise RuntimeError(f"Could not find '{key}' in model's inputs")
            v.copyfrom(value)

        if params:
            # upload big arrays first to avoid memory issue in rpc mode
            keys = list(params.keys())
            keys.sort(key=lambda x: -np.prod(params[x].shape))
            for k in keys:
                # TODO(zhiics) Skip the weights for submodule in a better way.
                # We should use MetadataModule for initialization and remove
                # params from set_input
                val = self._get_input(k)
                if val:
                    self._get_input(k).copyfrom(params[k])

    def run(self, **input_dict):
        """Run forward execution of the model

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the model

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the model

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_input_index(self, name):
        """Get inputs index via input name.

        Parameters
        ----------
        name : str
           The input key name

        Returns
        -------
        index: int
            The input index. -1 will be returned if the given input name is not found.
        """
        return self._get_input_index(name)

    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)

    def get_input_name(self, index: int) -> str:
        """Return the name of input with index `index`"""
        return self._get_input_name(index)

    def get_input_info(self):
        """Return the 'shape' and 'dtype' dictionaries of the module."""
        self.get_input_name(0)

        shape_dict = dict()
        dtype_dict = dict()
        for ind in range(0, self.get_num_inputs()):
            input_name = self.get_input_name(ind)
            input_tensor = self.get_input(ind)
            shape_dict[input_name] = input_tensor.shape
            dtype_dict[input_name] = input_tensor.dtype

        return shape_dict, dtype_dict

    def benchmark(
        self,
        device,
        func_name="run",
        repeat=5,
        number=5,
        min_repeat_ms=None,
        limit_zero_time_iterations=100,
        end_to_end=False,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
        **kwargs,
    ):
        """Calculate runtime of a function by repeatedly calling it.

        Use this function to get an accurate measurement of the runtime of a function. The function
        is run multiple times in order to account for variability in measurements, processor speed
        or other external factors.  Mean, median, standard deviation, min and max runtime are all
        reported.  On GPUs, CUDA and ROCm specifically, special on-device timers are used so that
        synchonization and data transfer operations are not counted towards the runtime. This allows
        for fair comparison of runtimes across different functions and models. The `end_to_end` flag
        switches this behavior to include data transfer operations in the runtime.

        The benchmarking loop looks approximately like so:

        .. code-block:: python

            for r in range(repeat):
                time_start = now()
                for n in range(number):
                    func_name()
                time_end = now()
                total_times.append((time_end - time_start)/number)


        Parameters
        ----------
        func_name : str
            The function to benchmark. This is ignored if `end_to_end` is true.

        repeat : int
            Number of times to run the outer loop of the timing code (see above). The output will
            contain `repeat` number of datapoints.

        number : int
            Number of times to run the inner loop of the timing code. This inner loop is run in
            between the timer starting and stopping. In order to amortize any timing overhead,
            `number` should be increased when the runtime of the function is small (less than a 1/10
            of a millisecond).

        min_repeat_ms : Optional[int]
            If set, the inner loop will be run until it takes longer than `min_repeat_ms`
            milliseconds. This can be used to ensure that the function is run enough to get an
            accurate measurement.

        limit_zero_time_iterations : Optional[int]
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        end_to_end : bool
            If set, include time to transfer input tensors to the device and time to transfer
            returned tensors in the total runtime. This will give accurate timings for end to end
            workloads.

        cooldown_interval_ms: Optional[int]
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: Optional[int]
            The number of repeats before the cooldown is activated.

        kwargs : Dict[str, Object]
            Named arguments to the function. These are cached before running timing code, so that
            data transfer costs are not counted in the runtime.

        Returns
        -------
        timing_results : BenchmarkResult
            Runtimes of the function. Use `.mean` to access the mean runtime, use `.results` to
            access the individual runtimes (in seconds).
        """
        min_repeat_ms = 0 if min_repeat_ms is None else min_repeat_ms
        if end_to_end:
            # Have to unpack kwargs into a single list
            args = []
            for k, v in kwargs.items():
                args.append(k)
                args.append(v)
            return self.module.time_evaluator(
                "run_from_inputs",
                device,
                repeat=repeat,
                number=number,
                min_repeat_ms=min_repeat_ms,
                limit_zero_time_iterations=limit_zero_time_iterations,
            )(device.device_type % rpc_base.RPC_SESS_MASK, device.device_id, *args)
        if kwargs:
            self.set_input(**kwargs)
        return self.module.time_evaluator(
            func_name,
            device,
            repeat=repeat,
            number=number,
            min_repeat_ms=min_repeat_ms,
            limit_zero_time_iterations=limit_zero_time_iterations,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
            # raw=True,
        )()
