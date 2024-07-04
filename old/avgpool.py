import tvm
from tvm import te
import d2ltvm

size = (64, 64, 3)
# size = (56, 56, 3)
# disabled_pass=["tir.CommonSubexprElimTIR"]

def default_avg(size):
    c, n, k = size[:]
    X, Y, PaddedX = d2ltvm.pool('avg', c, n, n, k, k, 1, 1, 1, 1)
    sch = te.create_schedule(Y.op)
    return sch, (X, Y)


sch, args = default_avg(size)

with tvm.transform.PassContext(opt_level=0, disabled_pass=["tir.CommonSubexprElimTIR"]):
    print(tvm.lower(sch, args, simple_mode=True))
with tvm.transform.PassContext(opt_level=1, disabled_pass=["tir.CommonSubexprElimTIR"]):
    print(tvm.lower(sch, args, simple_mode=True))
with tvm.transform.PassContext(opt_level=2, disabled_pass=["tir.CommonSubexprElimTIR"]):
    print(tvm.lower(sch, args, simple_mode=True))
with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
    print(tvm.lower(sch, args, simple_mode=True))
with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
    print(tvm.lower(sch, args, simple_mode=True))

with tvm.transform.PassContext(opt_level=3):
    print(tvm.lower(sch, args, simple_mode=True))

def schedule_avg(size):
    sch, (X, Y) = default_avg(size)
    te.schedule.AutoInlineInjective(sch)
    c, h, w = Y.op.axis[0:3]
    # fused = sch[Y].fuse(c, h)
    # sch[Y].parallel(fused)
    # sch[Y].vectorize(w)
    PoolSum = Y.op.input_tensors[0]
    sch[PoolSum].compute_at(sch[Y], Y.op.axis[2])
    return sch, (X, Y)

sch, args = schedule_avg(size)
with tvm.transform.PassContext(opt_level=3):
    print(tvm.lower(sch, args, simple_mode=True))

def schedule_avg2(size):
    sch, (X, Y) = default_avg(size)
    te.schedule.AutoInlineInjective(sch)
    c, h, w = Y.op.axis[0:3]
    # ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk)
    ho, hi = sch[Y].split(h, factor=4)
    sch[Y].reorder(ho, c, hi, w)
    fused = sch[Y].fuse(ho, c)
    # sch[Y].parallel(fused)
    # sch[Y].vectorize(w)
    PoolSum = Y.op.input_tensors[0]
    sch[PoolSum].compute_at(sch[Y], Y.op.axis[2])
    print("Y.op.reduce_axis", sch[Y].op.reduce_axis)
    # sch[Y].unroll(Y.op.axis[3])
    # sch[Y].unroll(sch[Y].op.axis[3])
    return sch, (X, Y)

sch, args = schedule_avg2(size)
with tvm.transform.PassContext(opt_level=3):
    print(tvm.lower(sch, args, simple_mode=True))
