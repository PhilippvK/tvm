import tvm
from tvm import te
from tvm.script import tir as TIR
import re
import os
import ctypes
import pytest

def test_mul(dtype):
    # target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    # target = "llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"
    # target = "llvm -mtriple=aarch64-linux-gnu"
    # target = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+f,+d,+v,+zvl2048b"
    target = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+f,+d,+v,+zvl128b"
    # target = "c"
    # target = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32"

    def check_correct_assembly(type):
        # m = te.var("m")
        # A = te.placeholder(m, dtype=type, name="A")
        # B = te.placeholder(m, dtype=type, name="B")
        # C = te.compute((m), lambda i: A[i] * B[i], name="C")
        # s = te.create_schedule(C.op)
        N = 1
        # K = 16
        K = te.size_var("K")
        A = te.placeholder((N, K), dtype="int8", name="A")
        B = te.placeholder((N, K), dtype="int8", name="B")
        k = te.reduce_axis((0, K))
        C = te.compute(
            (N,),
            lambda n: te.sum(A[n, k].astype("int32") * B[n, k].astype("int32"), axis=[k]),
            # lambda n: te.sum((A[k, n] * B[k, n]).astype("int32"), axis=[k]),
            name="C",
        )
        s = te.create_schedule(C.op)
        # no, ni = ?.apply(s, C, )
        # s[C].reorder()
        s[C].vectorize(s[C].op.axis[0])

        f = tvm.build(s, [A, B, C], target)

        # Verify we see SVE load instructions and mul instructions using z registers
        ll = f.get_source("ll")
        dumps["ll"] = ll
        # print("ll", ll)
        assembly = f.get_source("asm")
        dumps["asm"] = assembly
        # print("assembly", assembly)
        # loads = re.findall("ld1[whdb]	{ z", assembly)
        # matches = re.findall(
        #     r"mul\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        # )
        # print("matches", matches)
        # print("loads", loads)

        # assert len(loads) > 1
        # assert len(matches) > 1

    check_correct_assembly(type=dtype)


dumps = {}

@tvm.tir.transform.prim_func_pass(opt_level=0)
def _dump_tir_pass(tir_func, _, __):
    print("_dump_tir_pass")
    key = "tir"
    if key in dumps:
        dumps[key].append(str(tir_func))
    else:
        dumps[key] = [str(tir_func)]
    return tir_func

cfg = {"tir.add_lower_pass": [[3, _dump_tir_pass]]}
# cfg = {"tir.add_lower_pass": []}

with tvm.transform.PassContext(opt_level=3, config=cfg):
    test_mul("int8")

for key in dumps:
    print(f"======= {key} =======")
    if isinstance(dumps[key], list):
        print("\n".join(dumps[key]))
    else:
        print(dumps[key])
    print(f"------- {key} -------")
