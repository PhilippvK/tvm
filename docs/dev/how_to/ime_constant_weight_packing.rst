.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

IME constant weight packing with MetaSchedule
============================================

``dense_ime_packed_compute`` in ``python/tvm/topi/arm_cpu/dense_gemm.py``
expresses ``B_pack`` as a TE copy from logical weights ``[N, K]`` to
``[NO, KO, KB, NT, 4, 8]`` (with the existing small-shape rank specializations).
It previously became an ordinary TIR allocation and nested copy loop, even
when its input was constant. Relay constant folding cannot see inside this
TOPI computation.

The same mechanism applies to ``conv2d_nhwc_hwoi_ime_packed_compute`` in
``python/tvm/topi/arm_cpu/conv2d_gemm.py``. Its annotated weight copy maps
HWOI weights to ``[NO, KO, KB, NT, 4, 8]``, flattening kernel height, kernel
width, and input channels into K. Activation im2col/packing stays at runtime.
The fixed weight layout does not request ``layout_free_placeholders`` rewriting.

Constant binding and configuration
----------------------------------

Relay weights originate from ``relay.Constant`` or the ``params`` argument
to extraction/build. Parameter binding makes the latter constant in Relay.
By default, FuseOps lifts non-scalar constants into primitive-function inputs;
the resulting TIR no longer knows they are constant. Enable the existing
linked-parameter mechanism consistently during both extraction and compilation::

    executor = relay.backend.Executor("graph", {"link-params": True})
    tasks = ms.relay_integration.extract_tasks(
        mod, target, params=params, executor=executor
    )
    # Tune tasks using the usual MetaSchedule rules and database.
    with database, tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        lib = relay.build(mod, target=target, params=params, executor=executor)

An AOT executor that links parameters can use the same mechanism. The
independent integration test exercises the graph executor above.
``LowerToTECompute`` and ``CreatePrimFuncWithConstants`` then carry these
constants into TIR ``AllocateConst`` nodes, including during task extraction.
A raw TE placeholder is not constant just because a caller always passes the
same NDArray. Direct TIR users must embed the constant before tuning too.

Compiler transformation and schedule compatibility
--------------------------------------------------

``tir.FoldConstantWeightPacking`` runs automatically in the common lowering
pipeline, before block lowering and after MetaSchedule database lookup and
trace replay. It recognizes ``tir.weight_packing`` on bijective copy blocks
reading an embedded constant. It inverts the destination-to-source index map,
uses ``IndexMap.MapNDArray`` to physically rearrange the bytes on the compilation
host, removes the temporary packed allocation and its stores, and redirects
consumers (including tensorized match buffers and pointers) to the transformed
constant. Subsequent lowering eliminates the empty packing loop nest.

The implementation reuses the constant and buffer rewriting infrastructure in
``remove_weight_layout_rewrite_block.cc``. The existing
``RemoveWeightLayoutRewriteBlock`` pass retains its separate semantics for
MetaSchedule-generated layout rewrites; the new folding pass always transforms
the actual constant contents. This also runs in ordinary builds, not just in
a measurement builder.

The tuned workload remains unchanged until after schedule replay. The database
therefore stores and queries the same shapes, blocks, and constant representation.
There is no second workload lookup using packed constants. New annotations do
change the workload relative to versions before this patch, so regenerate old
records rather than assuming their structural keys still match.

The IME layout is fixed by ``NI``, ``K_MIN``, ``K_MAX`` and the logical shape;
``KI`` is selected by the compute, not by a MetaSchedule decision. The pass
uses the copy's index expressions, rather than hardcoding a tile size or a
schedule. The microkernel still sees contiguous 4-by-8 chunks, with byte offsets
``kb * (NT * 32) + nt * 32`` within a macro tile. MetaSchedule continues to
choose tiling/reuse and tensorize the reduction. ``meta_schedule.inline_rule``
keeps the weight copy available for folding during automatic inlining.

``layout_free_placeholders`` is not needed: the active IME reduction blocks
never used the dormant dictionary containing that annotation. Enabling it would
request schedule-dependent layout rewriting, a different mechanism from folding
this already-fixed IME packing. Such layouts normally require RewriteLayout's
recorded index maps and Relay MetaScheduleLayoutRewrite/FoldConstant handling.
This change does not introduce a schedule-dependent packing layout.

IR evidence
-----------

For M=16, N=24, K=32, NI=8, KI=16, the relevant original computation is
(abbreviating block bindings and constant data)::

    B = T.allocate_const(original_bytes, "int8", [24, 32])
    B_pack = T.alloc_buffer((3, 2, 2, 2, 4, 8), "int8")
    for no, ko, kb, nt, ni4, kk in T.grid(3, 2, 2, 2, 4, 8):
        B_pack[no, ko, kb, nt, ni4, kk] = B[no*8 + nt*4 + ni4,
                                                          ko*16 + kb*8 + kk]

After folding, the constant has shape ``[3, 2, 2, 2, 4, 8]`` and packed bytes.
After ordinary lowering the copy loop and allocation are absent; the GEMM
loads directly from that constant (excerpt, abbreviating other operands)::

    B = T.allocate_const(packed_bytes, "int8", [3, 2, 2, 2, 4, 8])
    B_pack = T.Buffer((768,), "int8", data=B)
    C_pack_1[...] += ... * T.Cast("int32", B_pack[
        no*256 + rko*128 + rkb*64 + nt*32 + ni4*8 + rkk])

Activation packing and result unpacking remain necessary at runtime.

Verification and limits
-----------------------

``tests/python/topi/test_topi_dense_ime_prepack.py`` checks exact int32 outputs,
independently enumerated packed bytes, absence of packing stores/allocations,
pass idempotence, dynamic-weight regression, different microtile sizes, and
multiple reduction tiles. It generates MetaSchedule design spaces, stores and
reloads JSON records, replays and tensorizes a candidate, and executes a portable
implementation of the IME intrinsic contract. It also emits C and a RISC-V
object with the IME external microkernel call. The assembly microkernel itself
is not executed on the host. A Relay test checks linked-parameter extraction,
strict database dispatch, folding during normal build, and numerical execution.
No hardware measurements or project experiment scripts are involved.

The pass conservatively leaves a source allocation with multiple readers
unchanged: replacing its layout would invalidate other uses. Dynamic/unlinked
weights also retain their runtime packing. Arbitrary custom schedules that
remove the annotated copy or cease to represent it as a full bijective copy
are outside the pass's contract. Existing IME shape/alignment restrictions
still apply.

``tests/python/topi/test_topi_conv2d_ime_prepack.py`` additionally checks HWOI
packed bytes and convolution outputs across spatial kernels, stride, padding,
dilation, microtile sizes, and reduction tiles, plus dynamic weights and Relay
linked-parameter compilation with MetaSchedule database replay.
