import heterocl as hcl
import numpy as np
import heterocl.tvm as tvm

def _popcount(num):
    out = hcl.scalar(0, "popcnt")
    with hcl.for_(0, 32) as i:
        # Bit selection operation
        out.v += num[i]
    return out.v

A = hcl.placeholder((3,), dtype=hcl.UInt(8), name="A")
B = hcl.placeholder((3,), dtype=hcl.UInt(8), name="B")
out = hcl.compute((3,),
    lambda x: _popcount(A[x] + B[x]), # tvm.intrin.popcount
    dtype=hcl.UInt(32))
s = hcl.create_schedule([A, B, out])
f = hcl.build(s)
hcl_a = hcl.asarray(np.array([1, 2, 3]), dtype=hcl.UInt(8))
hcl_b = hcl.asarray(np.array([1, 2, 3]), dtype=hcl.UInt(8))
hcl_out = hcl.asarray(np.array([0, 0, 0]), dtype=hcl.UInt(32))
f(hcl_a, hcl_b, hcl_out)
print("Input : {} + {}".format(hcl_a.asnumpy(), hcl_b.asnumpy()))
print("Output : {}".format(hcl_out.asnumpy()))