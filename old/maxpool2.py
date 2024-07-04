import tvm
from tvm import te
import d2ltvm

# channel, input height and width, kernel height and width
size = (64, 64, 3)

def default_max(size):
    c, n, k = size[:]
    X, Y, PaddedX = d2ltvm.pool('max', c, n, n, k, k, 1, 1, 1, 1)
    sch = te.create_schedule(Y.op)
    return sch, (X, Y)

sch, args = default_max(size)
print(tvm.lower(sch, args, simple_mode=True))

def optimized_max(size):
    sch, (X, Y) = default_max(size)
    # te.schedule.AutoInlineInjective(sch)
    c, h, w = Y.op.axis[0:3]
    rh, rw = sch[Y].op.reduce_axis
    fused = sch[Y].fuse(c, h)
    sch[Y].parallel(fused)
    sch[Y].unroll(rw)
    # sch[Y].vectorize(w)
    return sch, (X, Y)

sch, args = optimized_max(size)
print(tvm.lower(sch, args, simple_mode=True))
