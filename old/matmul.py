import tvm
from tvm import te
import d2ltvm

tx, ty, tk = 32, 32, 4  # tile sizes

def noblock(n):
    A, B, C = d2ltvm.matmul(n, n, n)
    s = te.create_schedule(C.op)
    # Tile by blocks, and then parallelize the computation of each block
    # xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    # xy = s[C].fuse(xo, yo)
    # s[C].parallel(xy)
    # # Optimize the computation of each block
    # ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk)
    # s[C].reorder(ko, xi, ki, yi)
    # s[C].vectorize(yi)
    # s[C].unroll(ki)
    return s, (A, B, C)

def block(n):
    A, B, C = d2ltvm.matmul(n, n, n)
    s = te.create_schedule(C.op)
    # Tile by blocks, and then parallelize the computation of each block
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # # Optimize the computation of each block
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk)
    s[C].reorder(ko, xi, ki, yi)
    s[C].vectorize(yi)
    s[C].unroll(ki)
    return s, (A, B, C)

s, (A, B, C) = noblock(64)
print(tvm.lower(s, [A, B, C], simple_mode=True))
s, (A, B, C) = block(64)
print(tvm.lower(s, [A, B, C], simple_mode=True))
