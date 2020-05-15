import heterocl as hcl
import numpy as np
import heterocl.tvm as tvm

hcl.init()
a = hcl.placeholder((3,), dtype=hcl.UInt(8), name="a")
out = hcl.compute((3,),
    lambda x: tvm.intrin.popcount(a[x]),
    dtype=hcl.UInt(32))
s = hcl.create_schedule([a, out])
f = hcl.build(s)
hcl_a = hcl.asarray(np.array([9, 7, 31]), dtype=hcl.UInt(8))
hcl_out = hcl.asarray(np.array([0, 0, 0]), dtype=hcl.UInt(32))
f(hcl_a, hcl_out)
print("Input : {}".format(hcl_a.asnumpy()))
print("Output : {}".format(hcl_out.asnumpy()))